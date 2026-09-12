"""Export a slim, inference-only checkpoint for the demo and the Hub.

A training checkpoint carries everything needed to RESUME: weights, AdamW's two
moment tensors per parameter, RNG state, the loss curve. That is 301 MB, of
which 201 MB is optimizer state that inference never touches. Publishing the
training checkpoint would mean a 3x bigger download and 3x the peak memory
during load on Streamlit Cloud, where the free tier's memory is the binding
constraint for how many phases' models can be resident at once.

This writes just the weights plus the config needed to rebuild the model.

Usage:
    python -m compare_lab.export_for_demo --arch cracked
"""

import argparse
from pathlib import Path

import torch

from compare_lab.train.checkpoint import save_checkpoint  # noqa: F401  (kept for symmetry)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def export(arch: str = "cracked", stage: str = "finetune") -> dict:
    src = CHECKPOINT_DIR / arch / f"{stage}_final.pt"
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist -- train it first")
    dst = CHECKPOINT_DIR / arch / f"{stage}_slim.pt"

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    slim = {"model": ckpt["model"], "config": ckpt["config"], "step": ckpt.get("step")}

    tmp = dst.with_suffix(".pt.tmp")
    torch.save(slim, tmp)
    tmp.replace(dst)

    src_mb = src.stat().st_size / 1048576
    dst_mb = dst.stat().st_size / 1048576
    return {"src": str(src), "dst": str(dst), "src_mb": round(src_mb, 1),
            "dst_mb": round(dst_mb, 1), "saved_pct": round(100 * (1 - dst_mb / src_mb), 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="cracked", choices=["cracked", "cracked_full", "sliced"])
    p.add_argument("--stage", default="finetune", choices=["pretrain", "finetune"])
    args = p.parse_args()
    info = export(args.arch, args.stage)
    print(f"{info['src_mb']} MB -> {info['dst_mb']} MB  ({info['saved_pct']}% menos)")
    print(f"escrito en {info['dst']}")


if __name__ == "__main__":
    main()
