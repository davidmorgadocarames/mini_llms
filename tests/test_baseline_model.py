import torch

from depth_lab.data.generator import generate_dataset
from depth_lab.data.loader import ExprDataset
from depth_lab.models.baseline import build_config, build_optimizer, evaluate_exact_match, train_steps
from depth_lab.tokenizer import CharTokenizer
from mini_llm.model import GPT

BLOCK_SIZE = 48


def _tiny_model(vocab_size: int) -> GPT:
    config = build_config(vocab_size, block_size=BLOCK_SIZE, n_layer=2, n_embd=64, n_head=2, n_kv_head=2)
    return GPT(config)


def test_baseline_overfits_a_tiny_batch():
    """Same sanity check used throughout Fase A: a model this size should be
    able to memorize a handful of short examples almost perfectly. If it
    can't, something about the task framing or loss masking is broken."""
    torch.manual_seed(0)
    tokenizer = CharTokenizer()
    examples = [ex.__dict__ for ex in generate_dataset("bool", depths=range(0, 3), n_per_depth=4, seed=1)]

    model = _tiny_model(tokenizer.vocab_size)
    train_ds = ExprDataset(examples, tokenizer, BLOCK_SIZE)
    optimizer = build_optimizer(model, lr=3e-3, weight_decay=0.0)

    train_steps(model, tokenizer, train_ds, optimizer, device="cpu", max_steps=400, batch_size=len(examples))

    acc = evaluate_exact_match(model, tokenizer, examples, device="cpu")
    assert acc >= 0.9


def test_padding_is_excluded_from_the_loss():
    """The dataset pads every sequence to the same block_size; if pad
    positions leaked into the loss (no ignore_index), predicting them would
    dominate the objective and the model would never learn the short,
    variable-length real content."""
    tokenizer = CharTokenizer()
    examples = [{"expr": "True", "value": True, "depth": 0}]
    ds = ExprDataset(examples, tokenizer, BLOCK_SIZE)
    x, y = ds[0]

    content_len = len("True => True")  # excludes bos/eos, matches format_target_text
    # y is seq shifted by one, so the target's pad run starts right after
    # (bos + content), i.e. at the position that predicts eos-then-pad
    assert (y[content_len + 1:] == tokenizer.pad_id).all()
