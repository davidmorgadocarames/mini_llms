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


_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "cracked"
# Prefer the slim, inference-only export: same weights, but without AdamW's two
# moment tensors it is 100MB instead of 301MB, which is a third of the download
# and ~200MB less peak memory on Streamlit Cloud (see export_for_demo.py).
LOCAL_CKPT_CANDIDATES = (_CKPT_DIR / "finetune_slim.pt", _CKPT_DIR / "finetune_final.pt")
LOCAL_TOKENIZER = Path(__file__).resolve().parent / "data" / "artifacts" / "tokenizer"


def _local_checkpoint() -> Path | None:
    return next((p for p in LOCAL_CKPT_CANDIDATES if p.exists()), None)


def load_model(device: str | None = None) -> tuple[GPT, Tokenizer, str]:
    """Load Cracked-D, preferring a local checkpoint over the Hub.

    On the training machine the weights are already on disk, so downloading
    300MB again would be wasteful -- and before the first upload it is the only
    way to try the demo at all. Deployed on Streamlit Cloud there is no local
    checkpoint, so it falls through to the Hub, which is the normal path there."""
    import torch as _torch

    device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
    local = _local_checkpoint()
    if local is not None and (LOCAL_TOKENIZER / "vocab.json").exists():
        model, tok = load_from_local(local, LOCAL_TOKENIZER, device)
        return model, tok, device
    return load_from_hf(device=device)


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
         temperature: float | None = None, top_k: int | None = None,
         max_new_tokens: int | None = None) -> str:
    """messages: [{"role": "user"|"assistant"|"system", "content": str}].

    Sampling defaults come from the frozen config, never from literals here. The
    comparison requires all three architectures to generate under identical
    settings, so a second copy of these numbers is a way for them to drift apart
    silently -- which already happened once in this phase (the fine-tuning CLI
    defaulted to batch 8 / 2000 steps while the frozen config said 16 / 20000).
    """
    from compare_lab.config import GENERATION

    return generate_response(
        model, tok, messages,
        max_new_tokens=GENERATION["max_new_tokens"] if max_new_tokens is None else max_new_tokens,
        temperature=GENERATION["temperature"] if temperature is None else temperature,
        top_k=GENERATION["top_k"] if top_k is None else top_k,
        device=device)
