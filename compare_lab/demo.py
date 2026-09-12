"""Streamlit-free load + generate logic for the Fase D demo, so the page is pure
UI and this is unit-testable. Mirrors the Fase A/C loading pattern (same HF repo,
under a compare_lab/ prefix), but uses Fase D's tokenizer (distinct special ids)
and the canonical chat template.
"""

from pathlib import Path

import torch

from mini_llm.model.config import GPTConfig
from mini_llm.model.transformer import GPT
from compare_lab.data.tokenizer import Tokenizer
from compare_lab.models.cracked_d import generate_response

HF_REPO = "davidmorgado/coconut-mini-llm"


def _build_from_ckpt(ckpt_path: str, device: str) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_from_local(ckpt_path: str | Path, tokenizer_dir: str | Path,
                    device: str = "cpu") -> tuple[GPT, Tokenizer]:
    tok = Tokenizer.from_dir(tokenizer_dir)
    model = _build_from_ckpt(str(ckpt_path), device)
    return model, tok


def load_from_hf(repo: str = HF_REPO, device: str | None = None) -> tuple[GPT, Tokenizer, str]:
    from huggingface_hub import hf_hub_download

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab = hf_hub_download(repo, "compare_lab/tokenizer/vocab.json")
    merges = hf_hub_download(repo, "compare_lab/tokenizer/merges.txt")
    tok = Tokenizer(vocab, merges)
    ckpt = hf_hub_download(repo, "compare_lab/cracked_d_final.pt")
    model = _build_from_ckpt(ckpt, device)
    return model, tok, device


def chat(model: GPT, tok: Tokenizer, messages: list[dict], device: str = "cpu",
         temperature: float = 0.8, top_k: int = 50, max_new_tokens: int = 256) -> str:
    """messages: [{"role": "user"|"assistant"|"system", "content": str}]."""
    return generate_response(model, tok, messages, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k, device=device)
