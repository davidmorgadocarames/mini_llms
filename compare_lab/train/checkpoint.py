"""Checkpoint with everything needed for a *reproducible* resume -- the gap
AUDITORIA_FASE_C.md flags in mini_llm.train.checkpoint (which saves only
model+optimizer+step, so a resumed run gets a different batch order and reset
scheduler/scaler state).

Beyond weights + optimizer we persist: the LR scheduler position (implicit via
`step`), the GradScaler state, the exact data position (which micro-batch to
resume at), and the RNG states of Python, NumPy, torch CPU and torch CUDA. A
resume = relaunch the same command; it continues with the same data in the same
order and the same randomness.

DURABILITY -- the invariant this module guarantees is:

    if a checkpoint file exists, it is complete and loadable.

That needs more than a temp file plus a rename. `os.replace` gives *atomicity*
(the destination is either the old file or the new one, never a mix) but NOT
*durability*: without an fsync the 300+MB may still be sitting in the OS page
cache when the rename returns, so a power cut right then can leave a file that
exists but is truncated. Since this project's only stop mechanism is a hard
failure, that window is exactly the one we cannot afford. So every save:

  1. writes to a temp file and **fsyncs** it (bytes on the physical device),
  2. **re-loads and validates the temp file** before promoting it, so a bad write
     can never replace a good checkpoint,
  3. keeps the previous generation as `<name>.prev` (an instant metadata rename),
  4. promotes the temp file to the real name.

and every load tries the current file first, then `.prev`, so even a power cut
landing between the two renames in (3)/(4) still finds a valid checkpoint.
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# Keys every checkpoint must contain to be considered usable.
REQUIRED_KEYS = ("model", "optimizer", "step", "data_pos")


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte(t) -> torch.Tensor:
    """RNG states must be handed back as CPU ByteTensors.

    load_checkpoint passes map_location=device, which moves EVERY tensor in the
    checkpoint to that device -- including these RNG states. Restoring them then
    fails with "RNG state must be a torch.ByteTensor" on a GPU run, while a
    CPU run works fine, so this only ever breaks in production. Coerce back
    explicitly instead of relying on where the checkpoint happened to be loaded."""
    return t.cpu().to(torch.uint8) if isinstance(t, torch.Tensor) else t


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_cpu_byte(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [_as_cpu_byte(s) for s in state["cuda"]]
        # only restore if the device count still matches what was saved
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)


def _replace_with_retry(tmp_path: Path, path: Path, attempts: int = 25,
                        base_delay: float = 0.5, max_delay: float = 3.0) -> bool:
    """os.replace is all-or-nothing, but on Windows it also FAILS OUTRIGHT with
    PermissionError if any process holds a handle on the destination -- a
    real-time antivirus scan of a freshly written 300+MB .pt, or simply the user
    copying/inspecting the checkpoint from another window. Verified on this
    machine: an open read handle is enough to get WinError 5.

    Returns True if the replace succeeded, False if it never did. It does NOT
    raise, because a checkpoint that cannot be written must not kill the run:
    the previous checkpoint is still on disk and intact, so the worst case is
    losing this one write. (A first version retried for ~22s and then raised --
    a 45s handle was enough to kill a real run at the exact moment the
    checkpoint was meant to protect it.)

    On failure the temp file is left in place: it is a complete, valid
    checkpoint, just under the wrong name, so nothing is actually lost."""
    for i in range(attempts):
        try:
            tmp_path.replace(path)
            return True
        except PermissionError:
            if i == attempts - 1:
                return False
            time.sleep(min(base_delay * (i + 1), max_delay))
    return False


def prev_path(path: Path) -> Path:
    """Where the previous generation is kept."""
    path = Path(path)
    return path.with_suffix(path.suffix + ".prev")


def _write_and_fsync(ckpt: dict, tmp_path: Path) -> None:
    """Serialize and force the bytes onto the physical device.

    Without the fsync, torch.save only guarantees the data reached the OS page
    cache; a power cut before the OS flushes leaves a file that exists but is
    truncated. os.replace would still be atomic -- and still promote a truncated
    file."""
    with open(tmp_path, "wb") as f:
        torch.save(ckpt, f)
        f.flush()
        os.fsync(f.fileno())


def verify_checkpoint(path: Path) -> bool:
    """Load a checkpoint back and confirm it is complete and usable.

    Called on the temp file BEFORE it is promoted, so a bad write can never
    replace a good checkpoint. Reading ~300MB back costs a couple of seconds
    every 15 minutes -- cheap insurance for the one guarantee that matters here:
    if the file is there, the information is there."""
    try:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"WARNING: checkpoint {Path(path).name} failed to load back: {e!r}")
        return False
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        print(f"WARNING: checkpoint {Path(path).name} is missing keys {missing}")
        return False
    return True


def save_checkpoint(path: Path, model, optimizer, step: int, data_pos: int, config,
                    scaler=None, extra: dict | None = None, verify: bool = True) -> bool:
    """Durably write a checkpoint. Returns True if it landed at `path`.

    Sequence (see the module docstring for why each step is needed):
      write temp -> fsync -> verify it loads -> keep the old one as .prev ->
      promote temp to `path`.

    Never raises: a checkpoint that cannot be written must not kill a
    multi-hour run, since the previous one is still on disk and intact."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "data_pos": data_pos,
        "config": config,
        "rng": _rng_state(),
    }
    if extra:
        ckpt.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        _write_and_fsync(ckpt, tmp_path)
    except OSError as e:
        print(f"WARNING: could not write {tmp_path.name}: {e!r}. Previous checkpoint intact; "
              f"training continues.")
        return False

    if verify and not verify_checkpoint(tmp_path):
        print(f"WARNING: refusing to promote an unverifiable checkpoint; {path.name} left as is.")
        return False

    # Keep the previous generation. Both renames are metadata-only and microseconds
    # apart; if a power cut lands between them, `path` is missing but `.prev` holds
    # a valid checkpoint, and load_checkpoint falls back to it.
    if path.exists():
        _replace_with_retry(path, prev_path(path), attempts=3)

    ok = _replace_with_retry(tmp_path, path)
    if not ok:
        print(f"WARNING: could not rename {tmp_path.name} -> {path.name} (file held open by "
              f"another process). The complete, verified checkpoint is at {tmp_path}; the "
              f"previous one is still available. Training continues; the next checkpoint "
              f"will retry.")
    return ok


def _load_raw(path: Path, device: str):
    """Load and validate one candidate file; returns None if unusable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"WARNING: {path.name} exists but failed to load ({e!r})")
        return None
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        print(f"WARNING: {path.name} is missing keys {missing}")
        return None
    return ckpt


def load_checkpoint(path: Path, model, optimizer=None, scaler=None, device: str = "cpu",
                    restore_rng: bool = True) -> dict:
    """Load the newest usable checkpoint: `path` first, then `<path>.prev`.

    The fallback matters because the only stop mechanism here is a hard failure.
    A power cut between the two renames in save_checkpoint leaves `path` missing
    while `.prev` still holds a valid checkpoint, and a torn file (which the
    fsync + verify should prevent, but belt and braces) would otherwise abort the
    resume entirely."""
    path = Path(path)
    ckpt = _load_raw(path, device)
    if ckpt is None:
        ckpt = _load_raw(prev_path(path), device)
        if ckpt is not None:
            print(f"note: {path.name} was unusable; resumed from {prev_path(path).name} "
                  f"(step {ckpt.get('step')})")
    if ckpt is None:
        raise FileNotFoundError(
            f"no usable checkpoint at {path} or {prev_path(path)}. If a "
            f"{path.name}.tmp exists it is a complete checkpoint whose rename was "
            f"blocked -- rename it by hand to recover that work.")

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        # AdamW deliberately keeps its per-parameter `step` counter on the CPU
        # (kernel launches are costly), but map_location=device migrates it to
        # the GPU along with everything else. load_state_dict keeps it there, and
        # the non-capturable foreach path then does a GPU->CPU sync per parameter
        # per step. Harmless in results but ~1.5x slower optimizer steps, and it
        # makes a resumed run take a different code path than a fresh one.
        opt_state = ckpt["optimizer"]
        for st in opt_state.get("state", {}).values():
            if isinstance(st.get("step"), torch.Tensor):
                st["step"] = st["step"].cpu()
        optimizer.load_state_dict(opt_state)
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and ckpt.get("rng") is not None:
        _restore_rng(ckpt["rng"])
    known = {"model", "optimizer", "scaler", "step", "data_pos", "config", "rng"}
    extra = {k: v for k, v in ckpt.items() if k not in known}
    return {"step": ckpt.get("step", 0), "data_pos": ckpt.get("data_pos", 0),
            "config": ckpt.get("config"), **extra}
