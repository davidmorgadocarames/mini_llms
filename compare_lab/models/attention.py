"""Grouped-Query Attention that Sliced-D reuses for its three attention roles:
encoder self-attention (bidirectional), decoder self-attention (causal) and
decoder cross-attention (queries from the decoder, keys/values from the encoder
memory).

The self-attention math mirrors mini_llm.model.layers.GroupedQueryAttention
exactly (same q/k/v/o projection shapes, same repeat_interleave, same RoPE via
the reused apply_rope) so Cracked-D and Sliced-D share genuinely the same block;
the only additions are the two things an encoder-decoder needs and a decoder-only
does not: an explicit key-padding mask (the encoder is bidirectional, so padding
positions WOULD corrupt every real token without one -- unlike a right-padded
causal decoder), and a cross-attention mode.

Cross-attention uses NO positional encoding (use_rope=False): the decoder query
positions and the encoder key positions live in different coordinate systems, so
rotating both by RoPE would be ill-defined. The positional information is already
baked into Q (from the decoder's RoPE self-attention) and into K/V (from the
encoder's RoPE self-attention); this is the same choice T5 makes (relative bias
in self-attention only). Documented here because the plan calls it out explicitly.

Masks are built from lengths / key-padding masks, never by comparing token ids to
pad_id -- that comparison is exactly what broke Fase C (see AUDITORIA_FASE_C.md).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_llm.model.layers import apply_rope


class GQAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_kv_head: int, dropout: float = 0.0,
                 use_rope: bool = True):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        self.n_rep = n_head // n_kv_head
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, kv_source: torch.Tensor | None = None,
                cos: torch.Tensor | None = None, sin: torch.Tensor | None = None,
                causal: bool = False,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, Tq, C) queries. kv_source: (B, Tk, C) for cross-attention;
        None means self-attention (keys/values come from x).
        key_padding_mask: (B, Tk) boolean, True at padding positions to block."""
        kv = x if kv_source is None else kv_source
        B, Tq, C = x.shape
        Tk = kv.shape[1]

        q = self.q_proj(x).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, Tk, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, Tk, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            # Only valid for self-attention, where q and k share positions.
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn_mask = None
        is_causal = False
        if key_padding_mask is None and causal:
            # Fast path: let sdpa build the causal mask (flash kernels).
            is_causal = True
        else:
            keep = torch.ones(B, 1, Tq, Tk, dtype=torch.bool, device=x.device)
            if causal:
                causal_keep = torch.tril(torch.ones(Tq, Tk, dtype=torch.bool, device=x.device))
                keep = keep & causal_keep
            if key_padding_mask is not None:
                keep = keep & (~key_padding_mask[:, None, None, :])
            attn_mask = keep

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.n_head * self.head_dim)
        return self.o_proj(out)
