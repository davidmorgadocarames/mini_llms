"""Rebuild a loss curve (.json + .png) from a checkpoint's stored history.

The training loop writes the curve at the end of a run, but the history also
travels inside every checkpoint (that is what makes a resume extend the curve
instead of restarting it). So the checkpoint is a second, independent copy of the
same data -- and the reason the pretrain curve was recoverable after a test
overwrote its .json with "[]".

This is also the repo's stated convention in practice: every plot ships the raw
data that produced it so the plot can be regenerated without re-running the
training. Here the checkpoint is that raw data.

Usage:
    python -m compare_lab.train.regen_curves --arch cracked --stage pretrain
"""

import argparse
from pathlib import Path

import torch

from compare_lab.train.pretrain import CHECKPOINT_DIR, RESULTS_DIR, _save_curves


def regen(arch: str = "cracked", stage: str = "pretrain") -> dict:
    # prefer the _final checkpoint; fall back to the rolling one mid-run
    candidates = [CHECKPOINT_DIR / arch / f"{stage}_final.pt",
                  CHECKPOINT_DIR / arch / f"{stage}.pt"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(f"no {stage} checkpoint for {arch} in {CHECKPOINT_DIR / arch}")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    history = list(ckpt.get("history") or [])
    if not history:
        raise ValueError(f"{src} carries no loss history -- nothing to rebuild from")

    base = RESULTS_DIR / f"{stage}_loss_{arch}"
    _save_curves(base, history, f"{stage} {arch}")
    return {"src": str(src), "base": str(base), "points": len(history),
            "first_step": history[0]["step"], "last_step": history[-1]["step"],
            "last_train_loss": history[-1].get("train_loss")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="cracked", choices=["cracked", "cracked_full", "sliced"])
    p.add_argument("--stage", default="pretrain", choices=["pretrain", "finetune"])
    args = p.parse_args()
    info = regen(args.arch, args.stage)
    print(f"reconstruido desde {info['src']}")
    print(f"  {info['points']} puntos, pasos {info['first_step']}..{info['last_step']}, "
          f"train_loss final {info['last_train_loss']:.4f}")
    print(f"  escrito en {info['base']}.json y .png")


if __name__ == "__main__":
    main()
