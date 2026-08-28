import pytest
import torch
from huggingface_hub import hf_hub_download

from coconut_lab.data.loader import InstructionDataset, PressedLocatorDataset
from coconut_lab.data.prepare_pressed import build_locator_examples, build_replacer_examples
from coconut_lab.data.prepare_reasoning import ARTIFACTS_DIR, load_jsonl
from coconut_lab.models.cracked import build_optimizer, train_steps
from coconut_lab.models.pressed_loop import _extract_expr, run_llr_loop
from depth_lab.models.locator import Locator, LocatorConfig
from depth_lab.models.locator import build_optimizer as build_locator_optimizer
from depth_lab.models.locator import train_steps as locator_train_steps
from depth_lab.models.replacer import Replacer, ReplacerConfig
from mini_llm.tokenizer import BPETokenizer

HF_REPO = "davidmorgado/coconut-mini-llm"
LOC_BLOCK = 384
REP_BLOCK = 48


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    return BPETokenizer(vocab_path, merges_path)


def test_extract_expr_reads_the_expression_before_the_equals_sign():
    assert _extract_expr("<<48/2=24>>24") == "48/2"


def test_extract_expr_returns_none_for_non_annotation_text():
    assert _extract_expr("just some plain text") is None


@pytest.mark.slow
def test_run_llr_loop_resolves_a_draft_to_the_correct_final_answer(tokenizer):
    """Trains tiny locator+replacer on a handful of real GSM8K problems
    (overfit), then runs the locate-and-replace loop alone (no drafter --
    isolates whether the mechanism itself works) on those same problems'
    real answer text, and checks every step correctly recomputes its real
    (expr, result). final_text is the whole resolved paragraph (prose
    intact, only the <<...>> annotations replaced), not a bare number --
    that's the drafter's job to have already written, not the loop's."""
    torch.manual_seed(0)
    device = "cpu"

    examples = load_jsonl(ARTIFACTS_DIR / "gsm8k_train.jsonl")[:10]

    loc_examples = build_locator_examples(examples)
    locator = Locator(LocatorConfig(vocab_size=tokenizer.vocab_size, d_model=64, n_head=4, n_layer=3, d_ff=128))
    loc_ds = PressedLocatorDataset(loc_examples, tokenizer, LOC_BLOCK)
    locator_train_steps(locator, loc_ds, build_locator_optimizer(locator, 3e-3, 0.0), device="cpu",
                         max_steps=800, batch_size=len(loc_ds))

    rep_examples = build_replacer_examples(examples)
    replacer = Replacer(ReplacerConfig(vocab_size=tokenizer.vocab_size, block_size=REP_BLOCK, d_model=64,
                                        n_head=4, n_layer=2, d_ff=128))
    rep_ds = InstructionDataset(rep_examples, tokenizer, REP_BLOCK)
    train_steps(replacer, rep_ds, build_optimizer(replacer, 3e-3, 0.0), device="cpu",
                max_steps=600, batch_size=len(rep_ds))

    correct = 0
    for ex in examples:
        final_text, steps = run_llr_loop(locator, replacer, tokenizer, ex["answer_text"], device="cpu")
        true_results = [s["result"] for s in ex["steps"]]
        predicted_results = [s.predicted_value for s in steps]
        if predicted_results == true_results and "<<" not in final_text:
            correct += 1
    assert correct / len(examples) >= 0.7


@pytest.mark.slow
def test_run_llr_loop_passes_through_text_with_no_annotations_unchanged():
    """General chat text has no <<expr=result>> pattern -- the loop must
    find nothing to fix and leave it untouched, not error out."""
    from depth_lab.models.locator import Locator, LocatorConfig
    from depth_lab.models.replacer import Replacer, ReplacerConfig

    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    tokenizer = BPETokenizer(vocab_path, merges_path)

    locator = Locator(LocatorConfig(vocab_size=tokenizer.vocab_size, d_model=32, n_head=2, n_layer=1, d_ff=64))
    replacer = Replacer(ReplacerConfig(vocab_size=tokenizer.vocab_size, block_size=REP_BLOCK, d_model=32,
                                        n_head=2, n_layer=1, d_ff=64))

    text = "The sky is blue and the grass is green."
    final_text, steps = run_llr_loop(locator, replacer, tokenizer, text, device="cpu")
    assert final_text == text
    assert steps == []
