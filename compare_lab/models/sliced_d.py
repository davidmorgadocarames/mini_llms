"""Sliced-D: bidirectional encoder + causal decoder with cross-attention, built
from the SAME modern blocks as Cracked-D (RoPE, RMSNorm pre-norm, SwiGLU, GQA)
so the only difference from the decoder-only baseline is the topology -- unlike
Fase C's Sliced, which used the classic Vaswani recipe (sinusoidal PE, LayerNorm,
ReLU) and was much smaller. Parameter count is matched to Cracked-D within 5%.

Reuses RMSNorm / SwiGLU / precompute_rope from mini_llm.model.layers unchanged;
attention is compare_lab.models.attention.GQAttention (self math identical to
mini_llm's GQA, plus padding masks and a no-RoPE cross-attention mode).
"""

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_llm.model.config import GPTConfig
from mini_llm.model.layers import RMSNorm, SwiGLU, precompute_rope
from compare_lab.models.attention import GQAttention


@dataclass
class SlicedDConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_head: int = 8
    n_kv_head: int = 2
    n_enc_layer: int = 3
    n_dec_layer: int = 4
    ffn_mult: float = 8 / 3
    ffn_multiple_of: int = 32
    max_src_len: int = 1024
    max_tgt_len: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

    def _ffn_config(self) -> GPTConfig:
        """A GPTConfig whose FFN fields feed the reused SwiGLU, so the hidden
        dim is rounded identically to Cracked-D (fair FFN size)."""
        return GPTConfig(
            vocab_size=self.vocab_size, block_size=max(self.max_src_len, self.max_tgt_len),
            n_layer=self.n_enc_layer, n_embd=self.d_model, n_head=self.n_head,
            n_kv_head=self.n_kv_head, ffn_mult=self.ffn_mult,
            ffn_multiple_of=self.ffn_multiple_of, dropout=self.dropout,
            rope_theta=self.rope_theta, norm_eps=self.norm_eps,
        )


class EncoderLayer(nn.Module):
    """Pre-norm: x = x + SelfAttn(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x)).
    Self-attention is bidirectional (no causal mask)."""

    def __init__(self, config: SlicedDConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GQAttention(config.d_model, config.n_head, config.n_kv_head, config.dropout, use_rope=True)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(config._ffn_config())

    def forward(self, x, cos, sin, src_key_padding_mask=None):
        x = x + self.attn(self.attn_norm(x), cos=cos, sin=sin, causal=False,
                          key_padding_mask=src_key_padding_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderLayer(nn.Module):
    """Pre-norm: causal self-attn, then cross-attn to encoder memory (no RoPE),
    then SwiGLU."""

    def __init__(self, config: SlicedDConfig):
        super().__init__()
        self.self_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.self_attn = GQAttention(config.d_model, config.n_head, config.n_kv_head, config.dropout, use_rope=True)
        self.cross_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.cross_attn = GQAttention(config.d_model, config.n_head, config.n_kv_head, config.dropout, use_rope=False)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(config._ffn_config())

    def forward(self, x, memory, cos, sin, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = x + self.self_attn(self.self_norm(x), cos=cos, sin=sin, causal=True,
                               key_padding_mask=tgt_key_padding_mask)
        x = x + self.cross_attn(self.cross_norm(x), kv_source=memory, causal=False,
                                key_padding_mask=memory_key_padding_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SlicedD(nn.Module):
    def __init__(self, config: SlicedDConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_enc_layer)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_dec_layer)])
        self.norm_enc = RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        head_dim = config.d_model // config.n_head
        cos, sin = precompute_rope(head_dim, max(config.max_src_len, config.max_tgt_len), config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        import math
        self.apply(self._init_weights)
        n_layer = config.n_enc_layer + config.n_dec_layer
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, src_idx, src_key_padding_mask=None):
        T = src_idx.size(1)
        assert T <= self.config.max_src_len, f"src len {T} exceeds max_src_len {self.config.max_src_len}"
        x = self.tok_emb(src_idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for layer in self.encoder_layers:
            x = layer(x, cos, sin, src_key_padding_mask)
        return self.norm_enc(x)

    def decode(self, tgt_idx, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        T = tgt_idx.size(1)
        assert T <= self.config.max_tgt_len, f"tgt len {T} exceeds max_tgt_len {self.config.max_tgt_len}"
        x = self.tok_emb(tgt_idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for layer in self.decoder_layers:
            x = layer(x, memory, cos, sin, tgt_key_padding_mask, memory_key_padding_mask)
        x = self.norm_f(x)
        return self.lm_head(x)

    def forward(self, src_idx, tgt_idx, targets=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None):
        memory = self.encode(src_idx, src_key_padding_mask)
        logits = self.decode(tgt_idx, memory, src_key_padding_mask, tgt_key_padding_mask)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate_stream(self, src_idx, bos_id: int, max_new_tokens: int,
                        src_key_padding_mask=None, temperature: float = 0.0, top_k: int | None = None):
        """Encode once, then decode autoregressively; yields the growing target
        sequence after each new token. temperature=0.0 is greedy argmax."""
        memory = self.encode(src_idx, src_key_padding_mask)
        tgt = torch.full((src_idx.size(0), 1), bos_id, dtype=torch.long, device=src_idx.device)
        for _ in range(max_new_tokens):
            logits = self.decode(tgt, memory, src_key_padding_mask)
            last = logits[:, -1, :]
            if temperature > 0.0:
                last = last / temperature
                if top_k is not None:
                    v, _ = torch.topk(last, min(top_k, last.size(-1)))
                    last[last < v[:, [-1]]] = -float("inf")
                probs = F.softmax(last, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = last.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_id], dim=1)
            yield tgt

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def generate_response(model: SlicedD, tokenizer, messages: list[dict],
                      max_new_tokens: int = 256, temperature: float = 0.8,
                      top_k: int | None = 50, device: str = "cpu") -> str:
    """Sliced-D's counterpart to cracked_d.generate_response, so the demo, the
    harness and any eval all stop and sample identically for both architectures.

    generate_stream deliberately does NOT stop by itself (it just yields the
    growing sequence, like every generator in this project); the caller decides.
    Without a caller that breaks on <eos> the model would ramble to the token
    budget every time -- the exact failure the plan calls out ("la generacion se
    detiene en <eos>") and the one Fase C carried for a while.

    The encoder input is built with the same canonical chat template used in
    fine-tuning (history ending at the <|assistant|> marker), so training and
    inference cannot drift apart.
    """
    model.eval()
    ids, _ = tokenizer.build_chat(messages, add_generation_prompt=True)
    if len(ids) > model.config.max_src_len:
        ids = ids[-model.config.max_src_len:]
    src = torch.tensor([ids], dtype=torch.long, device=device)
    # single unpadded sequence -> no padding mask needed; built from the length,
    # never by comparing against pad_id
    budget = min(max_new_tokens, model.config.max_tgt_len - 1)

    out_ids: list[int] = []
    for grown in model.generate_stream(src, tokenizer.bos_id, budget,
                                       temperature=temperature, top_k=top_k):
        next_id = grown[0, -1].item()
        if next_id == tokenizer.eos_id:
            break
        out_ids.append(next_id)
    return tokenizer.decode(out_ids)
