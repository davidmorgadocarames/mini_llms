"""Runs the REAL pretrain as an independent OS process with paths redirected to
a test root, so a STOP file created from outside behaves exactly as in
production. Paths are monkeypatched here rather than added as CLI flags, to
avoid growing the production CLI surface just for a test.

Usage: python verify_launcher.py <root_dir> <max_optimizer_steps> [ckpt_interval_min]
"""
import sys
import types
from pathlib import Path

root = Path(sys.argv[1])
max_steps = int(sys.argv[2])
# Default well under the test's duration so the PERIODIC checkpoint path runs
# several times. With production's 15 min it would never fire in a 5-minute
# test, leaving the most-used save path (and the resume-from-it path) untested.
ckpt_interval_min = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

import compare_lab.train.tasks as tasks
import compare_lab.train.pretrain as PT

tasks.RUNS_DIR = root / "runs"
PT.CHECKPOINT_DIR = root / "ckpt"
PT.RESULTS_DIR = root / "results"

args = types.SimpleNamespace(
    arch="cracked",
    grad_accum=16,          # production value
    max_optimizer_steps=max_steps,
    lr=6e-4, min_lr=6e-5, warmup_steps=5, weight_decay=0.1,
    dtype="bfloat16",       # production value
    eval_interval=10,       # more often than production so the curve has points
    log_interval=2,
    ckpt_interval_min=ckpt_interval_min,
    seed=1337,
    no_resume=False,
    device="cuda",
    tokenizer_dir=str(root / "tok"),
    pretrain_dir=str(root / "pre"),
)
result = PT.train(args)
print(f"LAUNCHER_RESULT state={result['state']} step={result['step']} "
      f"curve_points={len(result['history'])}")
