"""Fase C.6 item 2: runs the custom domain eval set (coconut_lab/eval/
domain_eval_set.jsonl, built by build_domain_eval_set.py) against the
already-trained "final" checkpoints (coconut_lab/checkpoints/*_final.pt,
produced by run_eval.py) -- no retraining here, this is pure evaluation.

Usage:
    python -m coconut_lab.eval.build_domain_eval_set   # once, or after editing the set
    python -m coconut_lab.eval.run_domain_eval
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from coconut_lab.eval.build_domain_eval_set import OUT_PATH as DOMAIN_EVAL_SET_PATH
from coconut_lab.eval.domain_eval import run_domain_eval
from coconut_lab.eval.run_eval import RESULTS_DIR, train_or_load_cracked, train_or_load_pressed, train_or_load_sliced
from coconut_lab.models import pressed as pressed_mod

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_domain_eval_set(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_domain_eval(results: dict, out_path: Path) -> None:
    architectures = ["cracked", "sliced", "pressed"]
    labels = ["Cracked", "Sliced", "Pressed"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    means = [results["overall"][a]["mean"] for a in architectures]
    stds = [results["overall"][a]["std"] for a in architectures]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, means, yerr=stds, capsize=8, color=colors)
    ax.set_ylabel("Accuracy on custom domain eval set")
    ax.set_title(f"Fase C: domain eval ({results['n_examples']} examples, {len(results['seeds'])} seeds)")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"chart saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = pressed_mod.load_tokenizer()
    cracked = train_or_load_cracked(tokenizer, args.device, max_steps=0, batch_size=1, skip_training=True)
    sliced = train_or_load_sliced(tokenizer, args.device, max_steps=0, batch_size=1, skip_training=True)
    drafter, locator, replacer = train_or_load_pressed(tokenizer, args.device, drafter_steps=0, locator_steps=0,
                                                         replacer_steps=0, batch_size=1, skip_training=True)
    models = {"cracked": cracked, "sliced": sliced, "pressed": (drafter, locator, replacer)}

    examples = load_domain_eval_set(DOMAIN_EVAL_SET_PATH)
    print(f"domain eval set: {len(examples)} examples, seeds={args.seeds}, temperature={args.temperature}")

    results = run_domain_eval(models, tokenizer, examples, args.device, seeds=tuple(args.seeds),
                               temperature=args.temperature, top_k=args.top_k, max_new_tokens=args.max_new_tokens)

    print("\n--- domain eval summary (mean +- std across seeds) ---")
    for arch, stats in results["overall"].items():
        print(f"  {arch:8s} {stats['mean']:.3f} +- {stats['std']:.3f}  (seeds: {stats['seed_accuracies']})")

    print("\n--- by category (averaged over seeds) ---")
    for arch, cats in results["by_category"].items():
        print(f"  {arch}:")
        for cat, acc in sorted(cats.items()):
            print(f"    {cat:20s} {acc:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "domain_eval.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_domain_eval(results, out_dir / "domain_eval.png")


if __name__ == "__main__":
    main()
