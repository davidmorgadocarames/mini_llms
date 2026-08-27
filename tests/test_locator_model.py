import torch

from depth_lab.data.generator import generate_dataset
from depth_lab.data.loader import LocatorDataset, build_locator_examples
from depth_lab.models.locator import (
    Locator,
    LocatorConfig,
    alibi_bias,
    alibi_slopes,
    build_optimizer,
    extract_span_from_probs,
    predict_span,
    train_steps,
)
from depth_lab.tokenizer import CharTokenizer

BLOCK_SIZE = 48


def _tiny_model(vocab_size: int) -> Locator:
    config = LocatorConfig(vocab_size=vocab_size, d_model=64, n_head=4, n_layer=3, d_ff=128)
    return Locator(config)


def test_alibi_slopes_are_decreasing_and_positive():
    slopes = alibi_slopes(4)
    assert slopes.shape == (4,)
    assert torch.all(slopes > 0)
    assert torch.all(slopes[1:] < slopes[:-1])


def test_alibi_bias_is_zero_on_the_diagonal_and_grows_with_distance():
    bias = alibi_bias(seq_len=6, n_head=4)
    assert bias.shape == (4, 6, 6)
    diag = torch.diagonal(bias, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag))
    # farther apart -> more negative bias (for every head)
    assert torch.all(bias[:, 0, 5] < bias[:, 0, 1])


def test_alibi_bias_is_symmetric():
    """No causal masking here -- the locator is bidirectional, so distance
    penalties must be the same in both directions."""
    bias = alibi_bias(seq_len=5, n_head=4)
    assert torch.allclose(bias, bias.transpose(-2, -1))


def test_extract_span_from_probs_picks_the_most_confident_contiguous_run():
    probs = torch.tensor([0.1, 0.9, 0.95, 0.92, 0.2, 0.6, 0.1])
    assert extract_span_from_probs(probs, threshold=0.5) == (1, 4)


def test_extract_span_from_probs_falls_back_to_argmax_when_nothing_clears_threshold():
    probs = torch.tensor([0.1, 0.2, 0.35, 0.15])
    assert extract_span_from_probs(probs, threshold=0.5) == (2, 3)


def test_build_locator_examples_spans_match_the_source_text():
    examples = [ex.__dict__ for ex in generate_dataset("bool", depths=range(1, 3), n_per_depth=3, seed=4)]
    locator_examples = build_locator_examples(examples)
    assert len(locator_examples) > 0
    for inst in locator_examples:
        start, end = inst["span"]
        assert inst["expr"][start:end].startswith("(") or inst["expr"][start:end].startswith("not (")


def test_locator_dataset_labels_mark_exactly_the_span():
    tok = CharTokenizer()
    expr = "(True and False)"
    ds = LocatorDataset([{"expr": expr, "span": (0, len(expr))}], tok, block_size=BLOCK_SIZE)
    ids, labels, pad_mask = ds[0]
    n = len(expr)
    assert labels[:n].sum().item() == n
    assert labels[n:].sum().item() == 0
    assert pad_mask[n:].all()
    assert not pad_mask[:n].any()


def test_locator_overfits_a_tiny_batch():
    torch.manual_seed(0)
    tokenizer = CharTokenizer()
    examples = [ex.__dict__ for ex in generate_dataset("bool", depths=range(1, 4), n_per_depth=4, seed=1)]
    locator_examples = build_locator_examples(examples)
    locator_examples = [ex for ex in locator_examples if len(ex["expr"]) <= BLOCK_SIZE]

    model = _tiny_model(tokenizer.vocab_size)
    train_ds = LocatorDataset(locator_examples, tokenizer, BLOCK_SIZE)
    optimizer = build_optimizer(model, lr=3e-3, weight_decay=0.0)
    train_steps(model, train_ds, optimizer, device="cpu", max_steps=1000, batch_size=len(locator_examples))

    correct = 0
    for inst in locator_examples:
        pred_span = predict_span(model, tokenizer, inst["expr"], device="cpu")
        if pred_span == tuple(inst["span"]):
            correct += 1
    # A couple of examples contain two *literally identical* sub-expressions
    # at different positions (e.g. "((True and True) and (True and True))"),
    # which ALiBi's relative-distance bias alone can't always disambiguate
    # from local content -- 0.85 leaves room for that without masking a real
    # regression.
    assert correct / len(locator_examples) >= 0.85
