"""Measures real generation throughput (tokens/sec) for Fase A and Fase D
(Cracked-D) on this machine's GPU, with and without the KV cache, so the
README's architecture comparison stops saying "not measured" for this row.

Both models share the exact same GPT class (mini_llm.model.transformer.GPT) and
generate_stream, so the only difference the benchmark can be measuring is
block_size / training data / vocab, not divergent inference code paths.

Follows the project convention (see compare_lab/eval/tables.py): writes the raw
numbers to a .json and a rendered .md side by side.

Usage:
    python -m compare_lab.eval.inference_speed
"""

import json
import time
from pathlib import Path

import torch

from mini_llm.model.config import GPTConfig
from mini_llm.model.transformer import GPT
from mini_llm.tokenizer import BPETokenizer
from compare_lab.data.tokenizer import Tokenizer as CompareTokenizer

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MINI_LLM_CKPT = Path(__file__).resolve().parent.parent.parent / "mini_llm" / "checkpoints" / "ckpt.pt"
MINI_LLM_TOKENIZER_DIR = Path(__file__).resolve().parent.parent.parent / "mini_llm" / "data" / "artifacts" / "tokenizer"
CRACKED_D_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "cracked" / "finetune_slim.pt"
CRACKED_D_TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts" / "tokenizer"

PROMPTS = ["The history of", "In 1943, the", "Once upon a time"]
SEEDS = (0, 1, 2)
MAX_NEW_TOKENS = 200
WARMUP_TOKENS = 32


def _load_mini_llm(device: str) -> tuple[GPT, BPETokenizer]:
    ckpt = torch.load(MINI_LLM_CKPT, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.from_dir(MINI_LLM_TOKENIZER_DIR)
    return model, tok


def _load_cracked_d(device: str) -> tuple[GPT, CompareTokenizer]:
    ckpt = torch.load(CRACKED_D_CKPT, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CompareTokenizer.from_dir(CRACKED_D_TOKENIZER_DIR)
    return model, tok


@torch.no_grad()
def _generate_cached(model: GPT, idx: torch.Tensor, n: int) -> int:
    last = idx
    for last in model.generate_stream(idx, n, temperature=0.8, top_k=50):
        pass
    return last.size(1) - idx.size(1)


@torch.no_grad()
def _generate_uncached(model: GPT, idx: torch.Tensor, n: int) -> int:
    """Mirrors generate_stream's own no-cache fallback path (full recompute of
    the growing sequence every step) instead of a second sampling
    implementation, so this measures the same code the model actually falls
    back to for long contexts, not a hand-rolled stand-in."""
    produced = 0
    for _ in range(n):
        logits, _ = model(idx)
        idx_next = model._sample(logits[:, -1, :], temperature=0.8, top_k=50)
        idx = torch.cat((idx, idx_next), dim=1)
        produced += 1
    return produced


def _time_tokens_per_sec(fn, model, idx, n, device: str) -> float:
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    produced = fn(model, idx, n)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return produced / dt


def _bench_one(model, tok, device: str) -> dict:
    device_t = device
    cached_rates, uncached_rates = [], []
    for prompt in PROMPTS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            ids = tok.encode(prompt)
            idx = torch.tensor([ids], dtype=torch.long, device=device_t)
            # warmup (CUDA kernel compilation / allocator caching)
            _generate_cached(model, idx, WARMUP_TOKENS)
            torch.manual_seed(seed)
            cached_rates.append(_time_tokens_per_sec(_generate_cached, model, idx, MAX_NEW_TOKENS, device))
            torch.manual_seed(seed)
            uncached_rates.append(_time_tokens_per_sec(_generate_uncached, model, idx, MAX_NEW_TOKENS, device))

    def _stats(xs: list[float]) -> dict:
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        return {"mean_tok_s": round(mean, 1), "std_tok_s": round(var ** 0.5, 1), "n": len(xs)}

    return {"kv_cache": _stats(cached_rates), "no_cache": _stats(uncached_rates)}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    model_a, tok_a = _load_mini_llm(device)
    model_d, tok_d = _load_cracked_d(device)

    results = {
        "device": gpu_name,
        "prompts": PROMPTS, "seeds": list(SEEDS), "max_new_tokens": MAX_NEW_TOKENS,
        "fase_a": {"block_size": model_a.config.block_size, "params": model_a.num_parameters(),
                   **_bench_one(model_a, tok_a, device)},
        "fase_d_cracked": {"block_size": model_d.config.block_size, "params": model_d.num_parameters(),
                           **_bench_one(model_d, tok_d, device)},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "inference_speed.json").write_text(json.dumps(results, indent=2))

    def _row(label: str, r: dict) -> str:
        return (f"| {label} | {r['block_size']} | {r['params']:,} | "
                f"{r['kv_cache']['mean_tok_s']} +/- {r['kv_cache']['std_tok_s']} | "
                f"{r['no_cache']['mean_tok_s']} +/- {r['no_cache']['std_tok_s']} |")

    md = [
        "# Velocidad de inferencia: Fase A vs Fase D (Cracked-D)", "",
        f"Medido en {gpu_name}, {len(PROMPTS)} prompts x {len(SEEDS)} semillas, "
        f"{MAX_NEW_TOKENS} tokens generados por corrida (tras {WARMUP_TOKENS} tokens de warmup).", "",
        "| Modelo | Contexto | Parametros | tok/s (KV cache) | tok/s (sin cache) |",
        "| --- | --- | --- | --- | --- |",
        _row("Fase A", results["fase_a"]),
        _row("Fase D - Cracked-D", results["fase_d_cracked"]),
        "",
        "Ambos modelos comparten literalmente la misma clase `GPT` y el mismo "
        "`generate_stream` (mini_llm/model/transformer.py); la unica diferencia de "
        "codigo entre las dos filas es la config (block_size, vocab) y los pesos "
        "entrenados, no una ruta de inferencia distinta.",
    ]
    (RESULTS_DIR / "inference_speed.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))
    print(f"\nescrito en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
