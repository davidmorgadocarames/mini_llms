"""Cracked-D: the decoder-only baseline. It IS mini_llm.model.GPT (reused
unchanged) with the Fase D context length; this module just holds the canonical
config factory and the chat-generation helper the demo and eval share, so no
Streamlit or eval code re-implements prompting/stopping.

Cracked-D-full uses the exact same architecture -- only its pretraining loss
differs (all tokens vs continuation-only), which lives in the training code, not
here.
"""

import torch

from mini_llm.model.config import GPTConfig
from mini_llm.model.transformer import GPT
from compare_lab.data.tokenizer import Tokenizer

VOCAB_SIZE = 8192
BLOCK_SIZE = 1024


def build_config(vocab_size: int = VOCAB_SIZE, block_size: int = BLOCK_SIZE) -> GPTConfig:
    """Same modern stack as Fase A (RoPE/RMSNorm/SwiGLU/GQA, ffn_mult=8/3), resized
    from the original 26.35M to ~80.63M params (n_layer=12, n_embd=768, n_head=12,
    n_kv_head=3) to reach the Chinchilla floor (~20 tokens/param) at 1.6B
    pretraining tokens -- the 26M/1200M combination was too undertrained for basic
    chat coherence. MUST stay numerically identical to compare_lab.config.cracked_config()
    (this is what actually trains; config.py's version is only what eval/tables.py
    and the freeze tests compare against)."""
    return GPTConfig(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=12, n_embd=768, n_head=12, n_kv_head=3,
    )


def build_model(config: GPTConfig | None = None) -> GPT:
    return GPT(config or build_config())


@torch.no_grad()
def generate_response(model: GPT, tokenizer: Tokenizer, messages: list[dict],
                      max_new_tokens: int = 256, temperature: float = 0.8,
                      top_k: int | None = 50, device: str = "cpu") -> str:
    """Build the chat prompt (canonical template, add_generation_prompt=True),
    generate until <eos>, and decode only the newly generated assistant tokens.
    The prompt is truncated from the left to fit block_size if needed."""
    model.eval()
    ids, _ = tokenizer.build_chat(messages, add_generation_prompt=True)
    max_prompt = model.config.block_size - max_new_tokens
    if max_prompt < 1:
        max_prompt = model.config.block_size // 2
    if len(ids) > max_prompt:
        ids = ids[-max_prompt:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out_ids: list[int] = []
    for grown in model.generate_stream(idx, max_new_tokens, temperature=temperature, top_k=top_k):
        next_id = grown[0, -1].item()
        if next_id == tokenizer.eos_id:
            break
        out_ids.append(next_id)
    return tokenizer.decode(out_ids)
