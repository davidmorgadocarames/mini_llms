import torch

from depth_lab.data.generator import generate_dataset
from depth_lab.data.loader import ExprDataset, build_replacer_examples
from depth_lab.models.baseline import build_optimizer, evaluate_exact_match, train_steps
from depth_lab.models.replacer import KVCache, Replacer, ReplacerConfig
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


def test_kv_cache_produces_identical_logits_to_full_recompute():
    """Same correctness property as mini_llm.model.GPT's cache test: a full
    from-scratch forward pass and one-token-at-a-time cached decoding must
    match numerically at every position."""
    torch.manual_seed(0)
    tok = CharTokenizer()
    model = _tiny_model(tok.vocab_size)
    model.eval()

    full_idx = torch.tensor([tok.encode("(True a")])

    ref_logits = []
    for t in range(1, 7):
        logits, _ = model(full_idx[:, :t])
        ref_logits.append(logits[:, -1, :])

    cache = KVCache(model.config.n_layer)
    logits, _ = model(full_idx[:, :1], kv_cache=cache)
    cached_logits = [logits[:, -1, :]]
    for t in range(1, 6):
        logits, _ = model(full_idx[:, t:t + 1], kv_cache=cache)
        cached_logits.append(logits[:, -1, :])

    for t, (ref, cached) in enumerate(zip(ref_logits, cached_logits)):
        assert torch.allclose(ref, cached, atol=1e-4), f"mismatch at position {t}"


def test_generate_stream_with_cache_matches_full_recompute_given_same_seed():
    tok = CharTokenizer()

    torch.manual_seed(42)
    model = _tiny_model(tok.vocab_size)
    model.eval()
    torch.manual_seed(123)
    idx = torch.tensor([[tok.stoi["("], tok.stoi["T"], tok.stoi["r"]]])

    torch.manual_seed(7)
    out_cached = []
    for grown in model.generate_stream(idx.clone(), max_new_tokens=5, temperature=1.0, top_k=5):
        out_cached = grown

    torch.manual_seed(42)
    model_b = _tiny_model(tok.vocab_size)
    model_b.eval()
    torch.manual_seed(123)
    idx_b = torch.tensor([[tok.stoi["("], tok.stoi["T"], tok.stoi["r"]]])
    torch.manual_seed(7)
    out_fallback = idx_b.clone()
    with torch.no_grad():
        for _ in range(5):
            logits, _ = model_b(out_fallback)
            idx_next = model_b._sample(logits[:, -1, :], 1.0, 5)
            out_fallback = torch.cat((out_fallback, idx_next), dim=1)

    assert torch.equal(out_cached, out_fallback)
