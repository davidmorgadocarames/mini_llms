"""Periodic checkpoint + auto-resume for long training runs. Motivation:
Fase C's k-fold stability check (25 individual training runs, several hours
unattended) is the first point in this project where losing a run to a
crash/power loss/dropped connection midway would be genuinely costly.

save_checkpoint saves model + optimizer state (never omitted before this --
every other checkpoint in this project only ever saved model weights, which
is fine for a one-shot final save but means "resuming" from one of those
would silently reset Adam's momentum/variance estimates) + the current step,
written atomically (write to a temp file, then rename) so an interruption
mid-write can never leave a corrupt checkpoint that later fails to load.
"""

from pathlib import Path

import torch


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int,
                     config, extra: dict | None = None) -> None:
    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": config}
    if extra:
        ckpt.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
                     device: str = "cpu") -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)
