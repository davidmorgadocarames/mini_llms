import torch

from depth_lab.data.generator import generate_dataset
from depth_lab.data.loader import ExprDataset, build_replacer_examples
from depth_lab.models.baseline import build_optimizer, evaluate_exact_match, train_steps
from depth_lab.models.replacer import Replacer, ReplacerConfig
from depth_lab.tokenizer import CharTokenizer

BLOCK_SIZE = 32


def _tiny_model(vocab_size: int) -> Replacer:
    config = ReplacerConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE, d_model=64, n_head=4, n_layer=2, d_ff=128)
    return Replacer(config)


def test_replacer_has_no_positional_buffers_or_parameters():
    """The whole point of NoPE: no rope/sinusoidal/alibi tensor anywhere --
    contrast with mini_llm.model.GPT (rope_cos/rope_sin buffers) and
    EncoderDecoderTransformer (src_pe/tgt_pe buffers)."""
    tok = CharTokenizer()
    model = _tiny_model(tok.vocab_size)
    assert list(model.buffers()) == []
    names = [n for n, _ in model.named_parameters()]
    assert not any("pos" in n.lower() or "rope" in n.lower() for n in names)


def test_causal_self_attention_is_causal():
    """Even with no positional encoding at all, the causal mask must still
    prevent position i's logits from depending on tokens after it."""
    tok = CharTokenizer()
    model = _tiny_model(tok.vocab_size)
    model.eval()
    ids = torch.tensor([[tok.bos_id, tok.stoi["T"], tok.stoi["r"], tok.stoi["u"], tok.stoi["e"]]])
    logits, _ = model(ids)

    ids_perturbed = ids.clone()
    ids_perturbed[0, -1] = tok.pad_id
    logits_perturbed, _ = model(ids_perturbed)

    assert torch.allclose(logits[0, 0], logits_perturbed[0, 0], atol=1e-5)


def test_build_replacer_examples_reverses_the_span_text():
    examples = [{"expr": "(True and False)", "value": False, "depth": 1}]
    replacer_examples = build_replacer_examples(examples)
    assert replacer_examples == [{"expr": "(True and False)"[::-1], "value": False}]


def test_replacer_overfits_a_tiny_batch():
    torch.manual_seed(0)
    tokenizer = CharTokenizer()
    examples = [ex.__dict__ for ex in generate_dataset("bool", depths=range(1, 4), n_per_depth=4, seed=1)]
    replacer_examples = build_replacer_examples(examples)
    replacer_examples = [ex for ex in replacer_examples if len(ex["expr"]) + 6 <= BLOCK_SIZE]

    model = _tiny_model(tokenizer.vocab_size)
    train_ds = ExprDataset(replacer_examples, tokenizer, BLOCK_SIZE)
    optimizer = build_optimizer(model, lr=3e-3, weight_decay=0.0)
    train_steps(model, tokenizer, train_ds, optimizer, device="cpu", max_steps=600,
                batch_size=len(replacer_examples))

    acc = evaluate_exact_match(model, tokenizer, replacer_examples, device="cpu")
    assert acc >= 0.9
