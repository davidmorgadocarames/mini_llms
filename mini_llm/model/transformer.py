import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_llm.model.config import GPTConfig
from mini_llm.model.layers import Block, KVCache, RMSNorm, precompute_rope


class GPT(nn.Module):
    """Decoder-only Transformer with RoPE, RMSNorm, SwiGLU and Grouped Query
    Attention — the "modern" building blocks used in LLaMA-family models, in place
    of the original GPT-2 recipe (learned position embeddings, LayerNorm, ReLU-MLP,
    standard multi-head attention)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying (Press & Wolf 2016)

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for the projections that feed directly into the residual
        # stream, so residual variance doesn't grow with depth (GPT-2 recipe)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                kv_cache: KVCache | None = None):
        B, T = idx.shape
        start_pos = kv_cache.length if kv_cache is not None else 0
        assert start_pos + T <= self.config.block_size, (
            f"sequence length {start_pos + T} exceeds block_size {self.config.block_size}"
        )

        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, kv_cache, i)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def generate_stream(self, idx: torch.Tensor, max_new_tokens: int,
                         temperature: float = 1.0, top_k: int | None = None):
        """Yields the growing sequence tensor after each newly sampled token, so
        callers (e.g. the Coconut CLI) can print/decode as tokens arrive instead
        of waiting for the whole completion.

        Uses a KV cache so each step only runs the new token through the
        network instead of recomputing attention over the whole growing
        sequence -- except in the (rare, e.g. a very long conversation)
        case where the prompt plus every possible generated token wouldn't
        fit in block_size even before we start, where the cache can't help
        without evicting/rebasing positions; that case falls back to the
        original recompute-every-step behavior unchanged."""
        with torch.no_grad():
            if idx.size(1) + max_new_tokens > self.config.block_size:
                for _ in range(max_new_tokens):
                    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                    logits, _ = self(idx_cond)
                    idx = torch.cat((idx, self._sample(logits[:, -1, :], temperature, top_k)), dim=1)
                    yield idx
                return

            kv_cache = KVCache(self.config.n_layer)
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond, kv_cache=kv_cache)
            for _ in range(max_new_tokens):
                idx_next = self._sample(logits[:, -1, :], temperature, top_k)
                idx = torch.cat((idx, idx_next), dim=1)
                yield idx
                logits, _ = self(idx_next, kv_cache=kv_cache)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        out = idx
        for out in self.generate_stream(idx, max_new_tokens, temperature, top_k):
            pass
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
