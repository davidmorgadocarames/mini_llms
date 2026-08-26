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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x
