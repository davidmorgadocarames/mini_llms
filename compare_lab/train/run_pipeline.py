"""Single entry point for one architecture's full run: data prep (shared, once)
-> pretrain -> finetune. Meant to be launched once as an independent process
(plan section 10) and left to run for hours; relaunching the same command
resumes wherever it stopped, at both the stage level and, within
pretrain/finetune, at the exact batch (see compare_lab.train.checkpoint).

Stage bookkeeping (compare_lab/runs/pipeline_<arch>/stages.json) records the
outcome of every stage, so a pause is never mistaken for completion. Two rules
make that safe:

  1. If a training stage returns state "paused", the pipeline stops right there
     and does NOT start the next stage. Without this, pausing the pretrain would
     fall through to fine-tuning, which (finding its pretrained checkpoint
     missing) would silently train from random weights and then write
     finetune_final.pt -- so the relaunch would skip fine-tuning entirely and
     the deliverable would be a model fine-tuned from noise.
  2. Fine-tuning treats a missing pretrained checkpoint as a hard error, never
     as "start from scratch".

"Done" is decided by the artifacts themselves (stats.json / *_final.pt), which
cannot lie about what is on disk; stages.json is the human-readable view of it.

To stop: create a STOP file in the CURRENT active stage's run directory (shown
in the log and by `python -m compare_lab.status`). A STOP file in the pipeline's
own run directory is checked between stages and prevents starting the next one.

Usage:
    python -m compare_lab.train.run_pipeline --arch cracked
"""

import argparse
import json
import time
from pathlib import Path

import torch

from compare_lab import config
from compare_lab.data import prepare_pretrain, prepare_smoltalk
from compare_lab.train import pretrain as pretrain_mod
from compare_lab.train import finetune as finetune_mod
from compare_lab.train.tasks import StatusWriter, task_dir, atomic_write_json, clear_stop
from compare_lab.data.tokenizer import DEFAULT_TOKENIZER_DIR

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
STAGES = ("data_prep_pretrain", "data_prep_finetune", "pretrain", "finetune")


def _pipeline_stopped(arch: str) -> bool:
    return (task_dir(f"pipeline_{arch}") / "STOP").exists()


def _stages_path(arch: str) -> Path:
    return task_dir(f"pipeline_{arch}") / "stages.json"


def read_stages(arch: str) -> dict:
    path = _stages_path(arch)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {s: "pending" for s in STAGES}


def _write_stage(arch: str, name: str, state: str) -> None:
    stages = read_stages(arch)
    stages[name] = state
    stages["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(_stages_path(arch), stages)


def run(arch: str, device: str, pretrain_max_steps: int = 0,
        finetune_max_steps: int | None = None, tokenizer_docs: int | None = None,
        finetune_max_conversations: int | None = None) -> str:
    """Returns the final pipeline state: "finished" or "paused"."""
    finetune_max_steps = config.FINETUNE_MAX_STEPS if finetune_max_steps is None else finetune_max_steps
    tokenizer_docs = config.TOKENIZER_DOCS if tokenizer_docs is None else tokenizer_docs
    finetune_max_conversations = (config.FINETUNE_MAX_CONVERSATIONS
                                  if finetune_max_conversations is None else finetune_max_conversations)

    # Clear our own STOP on launch, exactly like the training stages do. Without
    # this the pipeline STOP was sticky: creating it (the natural thing to do,
    # since it matches the process you launched) made every relaunch exit
    # immediately as "paused" forever, while a training STOP self-cleared -- the
    # same gesture behaving differently in the two cases.
    clear_stop(f"pipeline_{arch}")

    pipeline_status = StatusWriter(f"pipeline_{arch}", arch, total_steps=len(STAGES),
                                  total_tokens=len(STAGES))
    pipeline_status.update(0, 0, force=True)
    data_dir = Path(DEFAULT_TOKENIZER_DIR).parent

    # Write our own log from Python instead of having the launcher pipe the whole
    # console through Tee-Object. That pipe used to destroy the tqdm progress bar:
    # tqdm redraws itself on stderr with carriage returns, and PowerShell turns
    # each redraw into a separate line (wrapped in an ErrorRecord), so the bar was
    # invisible for an 8-hour run. Logging here keeps the console clean for tqdm.
    log_path = task_dir(f"pipeline_{arch}") / "pipeline.log"

    def say(msg: str) -> None:
        print(msg, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        except OSError:
            pass  # logging must never break the run

    def stage(name: str, n: int, fn) -> bool:
        """Runs one stage. Returns False if the pipeline must stop here."""
        if _pipeline_stopped(arch):
            pipeline_status.set_state("paused")
            _write_stage(arch, name, "pending")
            say(f"[pipeline {arch}] STOP found -- not starting stage {name}")
            return False
        pipeline_status.update(n - 1, n - 1, force=True)
        pipeline_status._state["model"] = f"{arch}:{name}"
        _write_stage(arch, name, "running")
        say(f"\n=== [pipeline {arch}] stage {n}/{len(STAGES)}: {name} ===")
        t0 = time.time()
        try:
            result = fn()
        except KeyboardInterrupt:
            # Ctrl+C, or a STOP during data prep (which raises this deliberately,
            # since that stage cannot resume mid-way and has to restart).
            _write_stage(arch, name, "pending")
            pipeline_status.set_state("paused")
            say(f"=== [pipeline {arch}] stage {name} INTERRUPTED -- recorded as pending ===")
            raise
        except Exception as e:
            # Without this, any failure left stages.json and the pipeline status
            # reading "running" forever, so from the phone you could not tell a
            # crashed pipeline from a working one.
            _write_stage(arch, name, "error")
            pipeline_status.error(f"{name}: {e!r}")
            say(f"=== [pipeline {arch}] stage {name} FAILED: {e!r} ===")
            raise
        elapsed = time.time() - t0
        # A training stage that was paused must NOT be treated as done.
        state = (result or {}).get("state") if isinstance(result, dict) else None
        if state == "paused":
            _write_stage(arch, name, "paused")
            pipeline_status.update(n - 1, n - 1, force=True)
            pipeline_status.set_state("paused")
            say(f"=== [pipeline {arch}] stage {name} PAUSED after {elapsed:.0f}s -- "
                f"relaunch the same command to continue it; later stages not started ===")
            return False
        _write_stage(arch, name, "done")
        say(f"=== [pipeline {arch}] stage {name} done in {elapsed:.0f}s ===")
        return True

    # --- stage 1: shared pretraining sample (tokenizer + bins + prefix-LM plan) ---
    if not stage("data_prep_pretrain", 1, lambda: prepare_pretrain.build(
            tokens=config.PRETRAIN_TOKENS, val_tokens=config.PRETRAIN_VAL_TOKENS,
            block_size=config.BLOCK_SIZE, batch_size=config.MICRO_BATCH, seed=config.SEED,
            tokenizer_dir=Path(DEFAULT_TOKENIZER_DIR), retrain_tokenizer=True,
            tokenizer_docs=tokenizer_docs, vocab_size=config.VOCAB_SIZE, skip_if_done=True)):
        return "paused"

    # --- stage 2: shared fine-tuning set ---
    def do_data_prep_finetune():
        # Tokenizer loaded lazily inside the stage: building it eagerly as an
        # argument would read from disk even when the stage is skipped.
        from compare_lab.data.tokenizer import Tokenizer

        return prepare_smoltalk.prepare(
            Tokenizer.from_dir(DEFAULT_TOKENIZER_DIR),
            block_size=config.BLOCK_SIZE, max_conversations=finetune_max_conversations,
            test_frac=config.FINETUNE_TEST_FRAC, seed=config.SEED, skip_if_done=True)

    if not stage("data_prep_finetune", 2, do_data_prep_finetune):
        return "paused"

    pretrain_final = CHECKPOINT_DIR / arch / "pretrain_final.pt"

    def do_pretrain():
        if pretrain_final.exists():
            print(f"{pretrain_final} exists -- skipping pretrain")
            return {"state": "finished"}
        args = argparse.Namespace(
            arch=arch, grad_accum=config.GRAD_ACCUM, max_optimizer_steps=pretrain_max_steps,
            lr=config.PRETRAIN["lr"], min_lr=config.PRETRAIN["min_lr"],
            warmup_steps=config.PRETRAIN["warmup_steps"], weight_decay=config.PRETRAIN["weight_decay"],
            dtype=config.PRETRAIN["dtype"], eval_interval=config.PRETRAIN_EVAL_INTERVAL,
            log_interval=config.LOG_INTERVAL, ckpt_interval_min=config.CKPT_INTERVAL_MIN,
            seed=config.SEED, no_resume=False, device=device,
            tokenizer_dir=str(DEFAULT_TOKENIZER_DIR), pretrain_dir=str(data_dir / "pretrain"))
        return pretrain_mod.train(args)

    if not stage("pretrain", 3, do_pretrain):
        return "paused"

    finetune_final = CHECKPOINT_DIR / arch / "finetune_final.pt"

    def do_finetune():
        if finetune_final.exists():
            print(f"{finetune_final} exists -- skipping finetune")
            return {"state": "finished"}
        if not pretrain_final.exists():
            # Never silently fall back to random weights (see module docstring).
            raise FileNotFoundError(
                f"pretrained checkpoint {pretrain_final} is missing -- refusing to fine-tune "
                f"from random weights. Finish the pretrain stage first.")
        args = argparse.Namespace(
            arch=arch, block_size=config.BLOCK_SIZE, batch_size=config.FINETUNE_BATCH_SIZE,
            max_steps=finetune_max_steps,
            lr=config.FINETUNE["lr"], min_lr=config.FINETUNE["min_lr"],
            warmup_steps=config.FINETUNE["warmup_steps"], weight_decay=config.FINETUNE["weight_decay"],
            dtype=config.FINETUNE["dtype"], log_interval=config.LOG_INTERVAL,
            eval_interval=config.FINETUNE_EVAL_INTERVAL, ckpt_interval_min=config.CKPT_INTERVAL_MIN,
            seed=config.SEED, no_resume=False, pretrained=str(pretrain_final), device=device,
            tokenizer_dir=str(DEFAULT_TOKENIZER_DIR), finetune_dir=str(data_dir / "finetune"))
        return finetune_mod.train(args)

    if not stage("finetune", 4, do_finetune):
        return "paused"

    pipeline_status.update(len(STAGES), len(STAGES), force=True)
    pipeline_status.finished()
    say(f"\n[pipeline {arch}] ALL STAGES DONE")
    return "finished"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=["cracked", "cracked_full", "sliced"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrain-max-steps", type=int, default=0, help="0 = full single epoch")
    p.add_argument("--finetune-max-steps", type=int, default=None)
    p.add_argument("--tokenizer-docs", type=int, default=None)
    p.add_argument("--finetune-max-conversations", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    state = run(a.arch, a.device, a.pretrain_max_steps, a.finetune_max_steps,
                a.tokenizer_docs, a.finetune_max_conversations)
    raise SystemExit(0 if state in ("finished", "paused") else 1)
