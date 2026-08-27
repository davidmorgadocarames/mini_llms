"""Fase B.6: trains all three architectures and produces the
accuracy-vs-depth comparison chart -- the "proof" that the depth-
generalization failure the He 2026 paper describes shows up here too, and
how much each architecture mitigates it.

Split discipline (see the Fase B plan): training ever only touches
bool_train.jsonl (depths 0-5). bool_val.jsonl (depths 0-5, a different seed)
is used purely to monitor training health -- never for model selection.
bool_test_depth{6..12}.jsonl (depths 6-12, entirely out-of-distribution, yet
other seeds) is read exactly once, after every model has already finished
training, purely to produce the final numbers. No hyperparameter or
checkpoint choice in this script is made by looking at test accuracy.

Usage:
    python -m depth_lab.eval.run_eval
    python -m depth_lab.eval.run_eval --skip-training   # reload existing checkpoints instead
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from depth_lab.data.build_dataset import ARTIFACTS_DIR, TEST_DEPTHS, TRAIN_DEPTHS, load_jsonl
from depth_lab.data.loader import ExprDataset, LocatorDataset, Seq2SeqDataset, build_locator_examples, \
    build_replacer_examples
from depth_lab.models import baseline as baseline_mod
from depth_lab.models import encoder_decoder as encdec_mod
from depth_lab.models import locator as locator_mod
from depth_lab.models import llr_loop
from depth_lab.models import replacer as replacer_mod
from depth_lab.tokenizer import CharTokenizer

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def train_or_load_baseline(tokenizer, train_examples, val_examples, device, max_steps, batch_size, skip_training):
    ckpt_path = CHECKPOINT_DIR / "baseline_bool.pt"
    config = baseline_mod.build_config(tokenizer.vocab_size)
    model = baseline_mod.GPT(config).to(device)

    if skip_training and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[baseline] loaded existing checkpoint ({ckpt_path})")
        return model

    train_ds = ExprDataset(train_examples, tokenizer, config.block_size)
    optimizer = baseline_mod.build_optimizer(model, lr=3e-4, weight_decay=0.1)
    baseline_mod.train_steps(model, tokenizer, train_ds, optimizer, device, max_steps, batch_size, log_interval=300)

    val_acc = baseline_mod.evaluate_exact_match(model, tokenizer, val_examples[:200], device)
    print(f"[baseline] in-distribution val exact-match: {val_acc:.3f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    import dataclasses
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config)}, ckpt_path)
    return model


def train_or_load_encdec(tokenizer, train_examples, val_examples, device, max_steps, batch_size, skip_training):
    ckpt_path = CHECKPOINT_DIR / "encdec_bool.pt"
    config = encdec_mod.build_config(tokenizer.vocab_size)
    model = encdec_mod.EncoderDecoderTransformer(config).to(device)

    if skip_training and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[encdec] loaded existing checkpoint ({ckpt_path})")
        return model

    train_ds = Seq2SeqDataset(train_examples, tokenizer, encdec_mod.SRC_BLOCK_SIZE, encdec_mod.TGT_BLOCK_SIZE)
    optimizer = encdec_mod.build_optimizer(model, lr=3e-4, weight_decay=0.1)
    encdec_mod.train_steps(model, tokenizer, train_ds, optimizer, device, max_steps, batch_size, log_interval=300)

    val_acc = encdec_mod.evaluate_exact_match(model, tokenizer, val_examples[:200], device)
    print(f"[encdec] in-distribution val exact-match: {val_acc:.3f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    import dataclasses
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config)}, ckpt_path)
    return model


def train_or_load_llr(tokenizer, train_examples, device, locator_steps, replacer_steps, batch_size, skip_training):
    import dataclasses

    loc_ckpt_path = CHECKPOINT_DIR / "locator_bool.pt"
    loc_config = locator_mod.build_config(tokenizer.vocab_size)
    locator = locator_mod.Locator(loc_config).to(device)

    rep_ckpt_path = CHECKPOINT_DIR / "replacer_bool.pt"
    rep_config = replacer_mod.build_config(tokenizer.vocab_size)
    replacer = replacer_mod.Replacer(rep_config).to(device)

    if skip_training and loc_ckpt_path.exists() and rep_ckpt_path.exists():
        loc_ckpt = torch.load(loc_ckpt_path, weights_only=False, map_location=device)
        locator.load_state_dict(loc_ckpt["model"])
        rep_ckpt = torch.load(rep_ckpt_path, weights_only=False, map_location=device)
        replacer.load_state_dict(rep_ckpt["model"])
        print(f"[llr] loaded existing checkpoints ({loc_ckpt_path}, {rep_ckpt_path})")
        return locator, replacer

    locator_examples = build_locator_examples(train_examples)
    loc_ds = LocatorDataset(locator_examples, tokenizer, locator_mod.BLOCK_SIZE)
    loc_optimizer = locator_mod.build_optimizer(locator, lr=3e-4, weight_decay=0.1)
    locator_mod.train_steps(locator, loc_ds, loc_optimizer, device, locator_steps, batch_size, log_interval=300)

    replacer_examples = build_replacer_examples(train_examples)
    rep_ds = ExprDataset(replacer_examples, tokenizer, replacer_mod.BLOCK_SIZE)
    rep_optimizer = baseline_mod.build_optimizer(replacer, lr=3e-4, weight_decay=0.1)
    baseline_mod.train_steps(replacer, tokenizer, rep_ds, rep_optimizer, device, replacer_steps, batch_size,
                              log_interval=300)
    rep_acc = baseline_mod.evaluate_exact_match(replacer, tokenizer, replacer_examples[:200], device)
    print(f"[llr] replacer in-distribution exact-match: {rep_acc:.3f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": locator.state_dict(), "config": dataclasses.asdict(loc_config)}, loc_ckpt_path)
    torch.save({"model": replacer.state_dict(), "config": dataclasses.asdict(rep_config)}, rep_ckpt_path)
    return locator, replacer


def evaluate_all_depths(models: dict, tokenizer: CharTokenizer, device: str) -> dict:
    """Reads bool_test_depth{d}.jsonl only here, at the very end -- this is
    the one and only place test data is touched."""
    results: dict[str, dict[int, float]] = {name: {} for name in models}
    for d in TEST_DEPTHS:
        test_examples = load_jsonl(ARTIFACTS_DIR / f"bool_test_depth{d}.jsonl")

        baseline_acc = baseline_mod.evaluate_exact_match(models["baseline"], tokenizer, test_examples, device)
        encdec_acc = encdec_mod.evaluate_exact_match(models["encoder-decoder"], tokenizer, test_examples, device)
        locator, replacer = models["llr"]
        llr_acc = llr_loop.evaluate_exact_match(locator, replacer, tokenizer, test_examples, device)

        results["baseline"][d] = baseline_acc
        results["encoder-decoder"][d] = encdec_acc
        results["llr"][d] = llr_acc
        print(f"depth {d:2d} | baseline {baseline_acc:.3f} | encoder-decoder {encdec_acc:.3f} | llr {llr_acc:.3f}")
    return results


def plot_accuracy_vs_depth(results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {
        "baseline": dict(color="#e74c3c", marker="o", label="Decoder-only (baseline)"),
        "encoder-decoder": dict(color="#3498db", marker="s", label="Encoder-decoder"),
        "llr": dict(color="#2ecc71", marker="^", label="Looped Locate-and-Replace"),
    }
    for name, per_depth in results.items():
        depths = sorted(per_depth)
        accs = [per_depth[d] for d in depths]
        ax.plot(depths, accs, linewidth=2, markersize=7, **styles[name])

    ax.axvline(x=5.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(5.55, 0.02, "train depth ≤ 5", fontsize=8, color="gray", rotation=90, va="bottom")
    ax.set_xlabel("Expression depth (out-of-distribution: never seen in training)")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title("Depth generalization: accuracy vs. expression depth")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(sorted(TEST_DEPTHS))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"chart saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-steps", type=int, default=3000)
    p.add_argument("--encdec-steps", type=int, default=3000)
    p.add_argument("--locator-steps", type=int, default=3000)
    p.add_argument("--replacer-steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skip-training", action="store_true",
                    help="reload existing checkpoints from depth_lab/checkpoints instead of retraining")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)

    tokenizer = CharTokenizer()
    train_examples = load_jsonl(ARTIFACTS_DIR / "bool_train.jsonl")
    val_examples = load_jsonl(ARTIFACTS_DIR / "bool_val.jsonl")
    print(f"train: {len(train_examples)} examples (depths {min(TRAIN_DEPTHS)}-{max(TRAIN_DEPTHS)})")

    baseline_model = train_or_load_baseline(tokenizer, train_examples, val_examples, args.device,
                                             args.baseline_steps, args.batch_size, args.skip_training)
    encdec_model = train_or_load_encdec(tokenizer, train_examples, val_examples, args.device,
                                         args.encdec_steps, args.batch_size, args.skip_training)
    locator, replacer = train_or_load_llr(tokenizer, train_examples, args.device,
                                           args.locator_steps, args.replacer_steps, args.batch_size,
                                           args.skip_training)

    models = {"baseline": baseline_model, "encoder-decoder": encdec_model, "llr": (locator, replacer)}

    print("\n--- final OOD evaluation (test depths 6-12, touched only here) ---")
    results = evaluate_all_depths(models, tokenizer, args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "accuracy_vs_depth.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_accuracy_vs_depth(results, out_dir / "accuracy_vs_depth.png")


if __name__ == "__main__":
    main()
