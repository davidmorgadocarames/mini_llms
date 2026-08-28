import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_llm.model.config import GPTConfig


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (Zhang & Sennrich 2019) — cheaper than LayerNorm
    because it skips re-centering to zero mean, only rescales by RMS."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0,
                     device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos/sin tables for Rotary Position Embeddings (Su et al. 2021)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim / 2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim). cos/sin: (T, head_dim)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class KVCache:
    """Accumulated key/value tensors per layer, for incremental (one new
    token at a time) autoregressive generation -- avoids recomputing
    attention over the whole growing sequence at every decoding step, the
    classic quadratic-vs-linear generation cost fix.

    Stores k/v *before* GroupedQueryAttention's repeat_interleave (i.e. at
    n_kv_head, not n_head) -- that's the whole point of GQA: a smaller
    cache. Purely additive to the model: every forward()/Block.forward()
    call defaults kv_cache=None, which reproduces the exact prior behavior
    (no cache, full recompute) bit-for-bit -- existing callers (training,
    and any code not yet updated to pass a cache) are unaffected."""

    def __init__(self, n_layer: int):
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer

    @property
    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].size(2)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = k, v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=2)
        return self.k[layer_idx], self.v[layer_idx]


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with Grouped Query Attention (Ainslie et al. 2023):
    fewer key/value heads than query heads, shared across groups of query heads,
    to shrink the KV-cache at inference time."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: KVCache | None = None, layer_idx: int | None = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # cos/sin already correspond to this chunk's absolute positions
        # (transformer.py slices them starting at kv_cache.length when a
        # cache is in use), so RoPE stays correct whether this is a full
        # from-scratch pass or a single cached decoding step.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Only need an explicit causal mask when query and key positions
        # line up 1:1 (a full/prefill pass, or no cache at all) -- a single
        # new token (T_q=1) attending to the full cached history (T_k>1)
        # has nothing "future" to mask by construction.
        is_causal = q.size(2) == k.size(2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward (Shazeer 2020), used in LLaMA in place of ReLU-MLP."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = int(config.ffn_mult * config.n_embd)
        m = config.ffn_multiple_of
        hidden_dim = ((hidden_dim + m - 1) // m) * m

        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm Transformer block: x = x + Attn(RMSNorm(x)); x = x + FFN(RMSNorm(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: KVCache | None = None, layer_idx: int | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache, layer_idx)
        x = x + self.ffn(self.ffn_norm(x))
        return x
