import torch

from mini_llm.model import GPT, GPTConfig
from mini_llm.model.layers import GroupedQueryAttention, precompute_rope


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
