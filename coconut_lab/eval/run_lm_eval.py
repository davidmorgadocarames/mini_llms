"""Fase C.6 item 3: evaluates Cracked, Sliced, and Pressed against
lm-evaluation-harness (EleutherAI, the de-facto standard eval harness) via
the custom LM adapters in coconut_lab/eval/lm_eval_adapter.py.

Tasks: lambada_openai (last-word prediction, no world knowledge needed --
the classic sanity-check task even for GPT-2-small-scale models) and piqa
(physical commonsense, 2-way choice, random baseline = 50%). Both are
purely loglikelihood-based -- neither exercises generate_until.

No retraining here: loads the already-trained "final" checkpoints from
run_eval.py (skip_training=True).

IMPORTANT (Windows): this script MUST be run with the `if __name__ ==
"__main__":` guard intact. lm-eval-harness's stderr bootstrap uses
multiprocessing, and Windows' "spawn" start method re-imports this module
in every worker process -- without the guard, each worker would re-run the
entire script (including model loading and evaluation) from scratch,
recursively. Confirmed the hard way during development.

Usage:
    python -m coconut_lab.eval.run_lm_eval
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from lm_eval.evaluator import simple_evaluate

from coconut_lab.eval.lm_eval_adapter import GPTFamilyAdapter, SlicedAdapter
from coconut_lab.eval.run_eval import RESULTS_DIR, train_or_load_cracked, train_or_load_pressed, train_or_load_sliced
from coconut_lab.models import pressed as pressed_mod

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TASKS = ["lambada_openai", "piqa"]
RANDOM_BASELINE = {"lambada_openai": None, "piqa": 0.5}  # piqa is 2-way; lambada has no fixed chance rate


def plot_lm_eval(results: dict, out_path: Path) -> None:
    architectures = ["cracked", "sliced", "pressed"]
    labels = ["Cracked", "Sliced", "Pressed (drafter)"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, axes = plt.subplots(1, len(TASKS), figsize=(5 * len(TASKS), 5))
    for ax, task in zip(axes, TASKS):
        metric_key = "acc,none"
        accs = [results[a][task].get(metric_key, float("nan")) for a in architectures]
        errs = [results[a][task].get("acc_stderr,none", 0.0) for a in architectures]
        ax.bar(labels, accs, yerr=errs, capsize=8, color=colors)
        if RANDOM_BASELINE[task] is not None:
            ax.axhline(RANDOM_BASELINE[task], color="gray", linestyle="--", linewidth=1, label="random baseline")
            ax.legend()
        ax.set_title(task)
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fase C: lm-evaluation-harness (lambada_openai + piqa)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"chart saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None, help="cap examples per task (for smoke testing)")
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = pressed_mod.load_tokenizer()
    cracked = train_or_load_cracked(tokenizer, args.device, max_steps=0, batch_size=1, skip_training=True)
    sliced = train_or_load_sliced(tokenizer, args.device, max_steps=0, batch_size=1, skip_training=True)
    drafter, _locator, _replacer = train_or_load_pressed(tokenizer, args.device, drafter_steps=0, locator_steps=0,
                                                           replacer_steps=0, batch_size=1, skip_training=True)

    adapters = {
        "cracked": GPTFamilyAdapter(cracked, tokenizer, args.device, "cracked"),
        "sliced": SlicedAdapter(sliced, tokenizer, args.device),
        # Pressed's locator/replacer only correct <<expr=result>> spans in a
        # fresh draft; a loglikelihood request scores a fixed continuation
        # with nothing left to generate or correct, so Pressed here is, by
        # construction, identical to scoring with its drafter alone.
        "pressed": GPTFamilyAdapter(drafter, tokenizer, args.device, "pressed_drafter"),
    }

    results: dict[str, dict] = {}
    for arch, adapter in adapters.items():
        print(f"\n--- {arch} ---")
        out = simple_evaluate(model=adapter, tasks=TASKS, limit=args.limit, verbosity="ERROR",
                               confirm_run_unsafe_code=True, bootstrap_iters=args.bootstrap_iters)
        results[arch] = out["results"]
        for task in TASKS:
            print(f"  {task}: {out['results'][task]}")

    print("\n--- summary (accuracy) ---")
    print(f"{'task':20s} " + " ".join(f"{a:>12s}" for a in adapters))
    for task in TASKS:
        row = " ".join(f"{results[a][task].get('acc,none', float('nan')):>12.3f}" for a in adapters)
        print(f"{task:20s} {row}")
    print("\nNote: at ~8-26M parameters these models will perform near chance on "
          "knowledge-intensive tasks -- the useful signal here is the *relative* "
          "comparison between architectures, not the absolute numbers.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "lm_eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_lm_eval(results, out_dir / "lm_eval_results.png")


if __name__ == "__main__":
    main()
