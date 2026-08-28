from coconut_lab.data.prepare_pressed import build_locator_examples, build_replacer_examples, resolve_up_to
from coconut_lab.data.prepare_reasoning import format_example

SAMPLE_ANSWER = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72"
)


def _example():
    return format_example("How many clips did Natalia sell?", SAMPLE_ANSWER)


def test_resolve_up_to_zero_leaves_everything_untouched():
    ex = _example()
    text_state, span = resolve_up_to(ex["answer_text"], ex["steps"], 0)
    assert text_state == ex["answer_text"]
    assert text_state[span[0]:span[1]] == "<<48/2=24>>24"


def test_resolve_up_to_one_collapses_the_first_step_to_its_bare_value():
    ex = _example()
    text_state, span = resolve_up_to(ex["answer_text"], ex["steps"], 1)
    assert "<<48/2=24>>24" not in text_state  # annotation collapsed away
    assert "48/2 = 24 clips in May." in text_state  # only the <<...>> + repeat collapsed to bare "24"
    assert text_state[span[0]:span[1]] == "<<48+24=72>>72"  # step 1 still shows its real annotation


def test_resolve_up_to_the_last_step_returns_no_span():
    ex = _example()
    text_state, span = resolve_up_to(ex["answer_text"], ex["steps"], len(ex["steps"]))
    assert span is None
    assert "<<" not in text_state  # every annotation has been collapsed


def test_build_locator_examples_produces_one_instance_per_step_with_progressively_shrinking_text():
    ex = _example()
    instances = build_locator_examples([ex])
    assert len(instances) == 2
    assert len(instances[0]["text"]) > len(instances[1]["text"])
    for inst in instances:
        start, end = inst["span"]
        assert inst["text"][start:end].startswith("<<")


def test_build_replacer_examples_reverses_expr_and_keeps_result():
    ex = _example()
    instances = build_replacer_examples([ex])
    assert instances == [
        {"prompt": "2/84 => ", "response": "24"},
        {"prompt": "42+84 => ", "response": "72"},
    ]
