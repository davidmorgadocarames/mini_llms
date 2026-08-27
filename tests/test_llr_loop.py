import torch

from depth_lab.data.generator import generate_dataset
from depth_lab.data.loader import ExprDataset, LocatorDataset, build_locator_examples, build_replacer_examples
from depth_lab.models import baseline as baseline_mod
from depth_lab.models import locator as locator_mod
from depth_lab.models.llr_loop import evaluate_exact_match, reduce_with_llr
from depth_lab.models.locator import Locator, LocatorConfig
from depth_lab.models.replacer import Replacer, ReplacerConfig
from depth_lab.tokenizer import CharTokenizer

LOC_BLOCK = 48
REP_BLOCK = 32


def _tiny_locator(vocab_size: int) -> Locator:
    return Locator(LocatorConfig(vocab_size=vocab_size, d_model=64, n_head=4, n_layer=3, d_ff=128))


def _tiny_replacer(vocab_size: int) -> Replacer:
    return Replacer(ReplacerConfig(vocab_size=vocab_size, block_size=REP_BLOCK, d_model=64, n_head=4, n_layer=2,
                                    d_ff=128))


def test_reduce_with_llr_short_circuits_on_a_bare_literal():
    """A bare literal needs no model calls at all -- untrained, freshly
    initialized models are fine here since they're never invoked."""
    tokenizer = CharTokenizer()
    locator = _tiny_locator(tokenizer.vocab_size)
    replacer = _tiny_replacer(tokenizer.vocab_size)

    result = reduce_with_llr(locator, replacer, tokenizer, "True", device="cpu")
    assert result.converged
    assert result.final_expr == "True"
    assert result.steps == []


def test_llr_loop_converges_to_the_correct_value_end_to_end():
    """Trains small locator and replacer models to near-overfit on the same
    tiny set of expressions, then runs the full loop and checks it reduces
    each one to the correct final value -- the real integration test that
    the two pieces are wired together correctly, not just individually
    correct."""
    torch.manual_seed(0)
    tokenizer = CharTokenizer()
    examples = [ex.__dict__ for ex in generate_dataset("bool", depths=range(1, 4), n_per_depth=4, seed=2)]

    loc_examples = [ex for ex in build_locator_examples(examples) if len(ex["expr"]) <= LOC_BLOCK]
    locator = _tiny_locator(tokenizer.vocab_size)
    loc_ds = LocatorDataset(loc_examples, tokenizer, LOC_BLOCK)
    locator_mod.train_steps(locator, loc_ds, locator_mod.build_optimizer(locator, 3e-3, 0.0),
                             device="cpu", max_steps=1000, batch_size=len(loc_examples))

    rep_examples = [ex for ex in build_replacer_examples(examples) if len(ex["expr"]) + 6 <= REP_BLOCK]
    replacer = _tiny_replacer(tokenizer.vocab_size)
    rep_ds = ExprDataset(rep_examples, tokenizer, REP_BLOCK)
    baseline_mod.train_steps(replacer, tokenizer, rep_ds, baseline_mod.build_optimizer(replacer, 3e-3, 0.0),
                              device="cpu", max_steps=800, batch_size=len(rep_examples))

    acc = evaluate_exact_match(locator, replacer, tokenizer, examples, device="cpu")
    assert acc >= 0.9

    # every step's recorded span must be a real substring of its expr, so a
    # caller (the eventual Fase B.7 demo) can safely highlight/splice it
    for ex in examples:
        result = reduce_with_llr(locator, replacer, tokenizer, ex["expr"], device="cpu")
        for step in result.steps:
            start, end = step.span
            assert step.expr[start:end] == step.span_text
