import pytest

from coconut_lab.data.prepare_reasoning import (
    build,
    extract_final_answer,
    extract_steps,
    format_example,
    load_jsonl,
)

SAMPLE_ANSWER = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72"
)


def test_extract_steps_finds_expr_and_result_in_order():
    steps = extract_steps(SAMPLE_ANSWER)
    assert [s["expr"] for s in steps] == ["48/2", "48+24"]
    assert [s["result"] for s in steps] == ["24", "72"]


def test_extract_steps_spans_point_at_the_real_substring():
    steps = extract_steps(SAMPLE_ANSWER)
    for step in steps:
        start, end = step["span"]
        assert SAMPLE_ANSWER[start:end] == f"<<{step['expr']}={step['result']}>>"


def test_extracted_expr_evaluates_to_the_claimed_result():
    # independent oracle, same spirit as tests/test_reduce.py in Fase B
    steps = extract_steps(SAMPLE_ANSWER)
    for step in steps:
        assert eval(step["expr"]) == pytest.approx(float(step["result"]))  # noqa: S307


def test_extract_final_answer_reads_after_the_marker():
    assert extract_final_answer(SAMPLE_ANSWER) == "72"


def test_format_example_reports_step_count():
    ex = format_example("Some word problem?", SAMPLE_ANSWER)
    assert ex["n_steps"] == 2
    assert ex["final_answer"] == "72"


@pytest.mark.slow
def test_build_produces_expected_splits_and_disjoint_test(tmp_path, monkeypatch):
    monkeypatch.setattr("coconut_lab.data.prepare_reasoning.ARTIFACTS_DIR", tmp_path)
    paths = build(val_fraction=0.05, seed=0)

    train = load_jsonl(paths["train"])
    val = load_jsonl(paths["val"])
    test = load_jsonl(paths["test"])

    assert len(train) + len(val) == 7473  # GSM8K's own train split size
    assert len(test) == 1319  # GSM8K's own held-out test split size, untouched

    train_questions = {ex["question"] for ex in train}
    test_questions = {ex["question"] for ex in test}
    assert train_questions.isdisjoint(test_questions)


@pytest.mark.slow
def test_real_gsm8k_steps_evaluate_correctly_for_a_sample(tmp_path, monkeypatch):
    """Spot-checks that the <<...>> annotations in the real dataset are
    themselves internally consistent (independent oracle via eval), not
    just our parsing regex on a hand-written example."""
    monkeypatch.setattr("coconut_lab.data.prepare_reasoning.ARTIFACTS_DIR", tmp_path)
    paths = build(val_fraction=0.05, seed=0)
    train = load_jsonl(paths["train"])

    checked = 0
    for ex in train[:200]:
        for step in ex["steps"]:
            try:
                claimed = float(step["result"])
            except ValueError:
                continue  # a small number of GSM8K answers have non-numeric results (rare, skip)
            assert eval(step["expr"]) == pytest.approx(claimed, rel=1e-3)  # noqa: S307
            checked += 1
    assert checked > 100  # sanity: we actually checked a meaningful number of steps
