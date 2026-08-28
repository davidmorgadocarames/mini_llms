import torch

from mini_llm.model import GPT, GPTConfig
from mini_llm.model.layers import GroupedQueryAttention, KVCache, precompute_rope


def tiny_config(**overrides) -> GPTConfig:
    defaults = dict(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def test_rope_shapes():
    head_dim = 8
    seq_len = 10
    cos, sin = precompute_rope(head_dim, seq_len)
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)


def test_gqa_output_shape_matches_input():
    config = tiny_config()
    attn = GroupedQueryAttention(config)
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rope(head_dim, config.block_size)

    x = torch.randn(2, config.block_size, config.n_embd)
    out = attn(x, cos, sin)
    assert out.shape == x.shape


def test_forward_shapes_and_loss():
    config = tiny_config()
    model = GPT(config)

    B, T = 3, 10
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, config.vocab_size)
    assert loss.ndim == 0
    assert loss.item() > 0

    logits_no_targets, loss_none = model(idx)
    assert logits_no_targets.shape == (B, T, config.vocab_size)
    assert loss_none is None


def test_generate_extends_sequence():
    config = tiny_config()
    model = GPT(config)
    model.eval()

    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=5)
    assert out.shape == (1, 6)


def test_generate_crops_context_beyond_block_size():
    config = tiny_config()
    model = GPT(config)
    model.eval()

    idx = torch.zeros((1, config.block_size), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape == (1, config.block_size + 3)


def test_overfits_tiny_batch():
    """Sanity check before touching real data: with enough capacity relative to a
    tiny fixed batch, the model should be able to drive next-token loss close to
    zero. If this fails, something is broken in the architecture."""
    torch.manual_seed(0)
    config = tiny_config(vocab_size=16, block_size=16, n_layer=2, n_embd=64, n_head=4, n_kv_head=2)
    model = GPT(config)

    data = torch.randint(0, config.vocab_size, (4, config.block_size + 1))
    idx = data[:, :-1]
    targets = data[:, 1:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    _, initial_loss = model(idx, targets)
    initial_loss = initial_loss.item()

    for _ in range(300):
        optimizer.zero_grad()
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.1, (
        f"expected overfit: initial={initial_loss:.3f} final={final_loss:.3f}"
    )


def test_kv_cache_produces_identical_logits_to_full_recompute():
    """The correctness property the whole cache rests on: at every growing
    sequence length, the logits from a full from-scratch forward pass and
    from one-token-at-a-time cached decoding must match numerically -- if
    RoPE's absolute-position offsetting or the cache concatenation were
    wrong, this would drift or diverge outright."""
    torch.manual_seed(0)
    config = tiny_config()
    model = GPT(config)
    model.eval()

    full_idx = torch.randint(0, config.vocab_size, (1, 6))

    ref_logits = []
    for t in range(1, 7):
        logits, _ = model(full_idx[:, :t])
        ref_logits.append(logits[:, -1, :])

    cache = KVCache(config.n_layer)
    logits, _ = model(full_idx[:, :1], kv_cache=cache)
    cached_logits = [logits[:, -1, :]]
    for t in range(1, 6):
        logits, _ = model(full_idx[:, t:t + 1], kv_cache=cache)
        cached_logits.append(logits[:, -1, :])

    for t, (ref, cached) in enumerate(zip(ref_logits, cached_logits)):
        assert torch.allclose(ref, cached, atol=1e-4), f"mismatch at position {t}"


def test_kv_cache_length_tracks_tokens_fed_in():
    config = tiny_config()
    model = GPT(config)
    model.eval()
    cache = KVCache(config.n_layer)
    assert cache.length == 0

    model(torch.zeros((1, 3), dtype=torch.long), kv_cache=cache)
    assert cache.length == 3
    model(torch.zeros((1, 1), dtype=torch.long), kv_cache=cache)
    assert cache.length == 4


def test_generate_stream_with_cache_matches_generate_stream_without_cache_given_same_seed():
    """End-to-end version of the logits-equivalence check: with the same
    random seed, sampling should pick the exact same tokens whether or not
    the fast (short-prompt) cached path or the fallback path is taken."""
    config = tiny_config()

    torch.manual_seed(42)
    model_a = GPT(config)
    model_a.eval()
    torch.manual_seed(123)
    idx_a = torch.randint(0, config.vocab_size, (1, 4))
    torch.manual_seed(7)
    out_cached = model_a.generate(idx_a.clone(), max_new_tokens=5, temperature=1.0, top_k=5)

    torch.manual_seed(42)
    model_b = GPT(config)
    model_b.eval()
    torch.manual_seed(123)
    idx_b = torch.randint(0, config.vocab_size, (1, 4))
    # force the fallback (no-cache) path by requesting more tokens than fit
    torch.manual_seed(7)
    out_fallback_equivalent = idx_b.clone()
    with torch.no_grad():
        for _ in range(5):
            logits, _ = model_b(out_fallback_equivalent)
            idx_next = model_b._sample(logits[:, -1, :], 1.0, 5)
            out_fallback_equivalent = torch.cat((out_fallback_equivalent, idx_next), dim=1)

    assert torch.equal(out_cached, out_fallback_equivalent)


def test_generate_stream_falls_back_when_prompt_plus_new_tokens_exceeds_block_size():
    """Regression guard: this exact scenario (idx already at block_size) is
    covered by test_generate_crops_context_beyond_block_size above, which
    must keep passing unchanged -- it exercises the fallback path, not the
    cached one, since the cache can't slide its window."""
    config = tiny_config()
    model = GPT(config)
    model.eval()

    idx = torch.zeros((1, config.block_size), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape == (1, config.block_size + 3)
