"""Fase C.6 item 5: k-fold=5 training-stability diagnostic. Different
question from every other C.6 piece -- not "does it generalize?" (the
held-out GSM8K test split already measures that), but "does the result
depend on which slice of GSM8K training data the model happened to see, or
is it stable?". Only GSM8K's **train** split is partitioned (never val/test,
which stay held out exactly as everywhere else in C.6); Alpaca+oasst1 stay
full and identical across folds -- the diagnostic is specifically about
sensitivity to GSM8K's own partition, the smallest and most fold-sensitive
dataset in the mixture.

Per fold: 4/5 of GSM8K train + full Alpaca+oasst1 trains Cracked, Sliced,
and Pressed's three components (drafter/locator/replacer); the held-out 1/5
of GSM8K train (never used for training in that fold) is the eval set for
that fold's variant. 5 folds x 5 components = 25 individual training runs --
the most expensive piece of C.6 in compute, so every training call here is
checkpointed (mini_llm.train.checkpoint) and auto-resumes: re-running this
exact command after any interruption (crash, power loss, dropped
connection) picks back up from the last completed step of whichever
fold/component was in progress, not from scratch.

Step budget is deliberately the *lighter* "first pass" budget from C.6's
main run (not the heavier "final" budget) -- this is a stability check, not
a push for max accuracy, and 25 runs at the "final" budget would cost
8+ hours vs. ~2-2.5 hours at this one (both estimates from real throughput
measured earlier this session).

Usage:
    python -m coconut_lab.eval.run_kfold
"""

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from coconut_lab.data.loader import ConversationDataset, InstructionDataset, PressedLocatorDataset, Seq2SeqDataset
from coconut_lab.data.prepare_conversations import ARTIFACTS_DIR as CONV_DIR
from coconut_lab.data.prepare_conversations import load_jsonl as load_conv_jsonl
from coconut_lab.data.prepare_instructions import ARTIFACTS_DIR as INSTR_DIR
from coconut_lab.data.prepare_instructions import load_jsonl as load_instr_jsonl
from coconut_lab.data.prepare_pressed import build_locator_examples, build_replacer_examples
from coconut_lab.data.prepare_reasoning import ARTIFACTS_DIR as GSM8K_DIR
from coconut_lab.data.prepare_reasoning import gsm8k_to_instructions, load_jsonl as load_gsm8k_jsonl
from coconut_lab.eval.run_eval import RESULTS_DIR, answers_match, extract_final_number
from coconut_lab.models import cracked as cracked_mod
from coconut_lab.models import pressed as pressed_mod
from coconut_lab.models import sliced as sliced_mod
from coconut_lab.models.pressed_loop import reduce_with_pressed
from depth_lab.models.locator import Locator, LocatorConfig
from depth_lab.models.locator import build_config as build_locator_config
from depth_lab.models.locator import build_optimizer as build_locator_optimizer
from depth_lab.models.locator import train_steps as locator_train_steps
from depth_lab.models.replacer import Replacer, ReplacerConfig
from depth_lab.models.replacer import build_config as build_replacer_config
from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer.bpe import EOT_TOKEN

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KFOLD_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "kfold"
MAX_EVAL_EXAMPLES_PER_FOLD = 100  # keeps 5 folds x 3 architectures tractable, same spirit as run_eval.py's cap


def build_folds(examples: list[dict], k: int, seed: int) -> list[list[dict]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    return [shuffled[i::k] for i in range(k)]  # round-robin -> near-equal fold sizes


def fold_train_examples(folds: list[list[dict]], holdout_idx: int) -> list[dict]:
    return [ex for i, fold in enumerate(folds) if i != holdout_idx for ex in fold]


def train_cracked_fold(tokenizer, device, fold_train, alpaca_train, conv_train, block_size, max_steps, batch_size,
                        checkpoint_interval, ckpt_path: Path) -> GPT:
    model, _ = cracked_mod.load_base_model(device)
    train_ds = torch.utils.data.ConcatDataset([
        InstructionDataset(alpaca_train, tokenizer, block_size),
        ConversationDataset(conv_train, tokenizer, block_size),
        InstructionDataset(gsm8k_to_instructions(fold_train), tokenizer, block_size),
    ])
    optimizer = cracked_mod.build_optimizer(model, lr=5e-5, weight_decay=0.01)
    cracked_mod.train_steps(model, train_ds, optimizer, device, max_steps, batch_size, log_interval=500,
                             dtype="bfloat16", checkpoint_path=ckpt_path, checkpoint_interval=checkpoint_interval)
    return model


def train_sliced_fold(tokenizer, device, fold_train, alpaca_train, conv_train, max_steps, batch_size,
                       checkpoint_interval, ckpt_path: Path) -> "EncoderDecoderTransformer":
    config = sliced_mod.build_config(tokenizer.vocab_size)
    from depth_lab.models.encoder_decoder import EncoderDecoderTransformer
    model = EncoderDecoderTransformer(config).to(device)
    alpaca = sliced_mod.alpaca_to_seq2seq(alpaca_train)
    conv = sliced_mod.conversations_to_seq2seq(conv_train)
    gsm8k = sliced_mod.alpaca_to_seq2seq(gsm8k_to_instructions(fold_train))
    train_ds = torch.utils.data.ConcatDataset([
        Seq2SeqDataset(alpaca, tokenizer, sliced_mod.SRC_BLOCK_SIZE, sliced_mod.TGT_BLOCK_SIZE),
        Seq2SeqDataset(conv, tokenizer, sliced_mod.SRC_BLOCK_SIZE, sliced_mod.TGT_BLOCK_SIZE),
        Seq2SeqDataset(gsm8k, tokenizer, sliced_mod.SRC_BLOCK_SIZE, sliced_mod.TGT_BLOCK_SIZE),
    ])
    pad_id = tokenizer.encode(EOT_TOKEN)[0]
    optimizer = sliced_mod.build_optimizer(model, lr=3e-4, weight_decay=0.01)
    sliced_mod.train_steps(model, train_ds, optimizer, device, max_steps, batch_size, pad_id, log_interval=500,
                            dtype="bfloat16", checkpoint_path=ckpt_path, checkpoint_interval=checkpoint_interval)
    return model


def train_pressed_fold(tokenizer, device, fold_train, alpaca_train, conv_train, drafter_steps, locator_steps,
                        replacer_steps, batch_size, checkpoint_interval, drafter_ckpt: Path, locator_ckpt: Path,
                        replacer_ckpt: Path) -> tuple[GPT, Locator, Replacer]:
    drafter_config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=pressed_mod.DRAFTER_BLOCK_SIZE,
                                n_layer=6, n_embd=384, n_head=6, n_kv_head=2)
    drafter = GPT(drafter_config).to(device)
    drafter_train_ds = torch.utils.data.ConcatDataset([
        InstructionDataset(alpaca_train, tokenizer, pressed_mod.DRAFTER_BLOCK_SIZE),
        ConversationDataset(conv_train, tokenizer, pressed_mod.DRAFTER_BLOCK_SIZE),
        InstructionDataset(gsm8k_to_instructions(fold_train), tokenizer, pressed_mod.DRAFTER_BLOCK_SIZE),
    ])
    drafter_optimizer = cracked_mod.build_optimizer(drafter, lr=3e-4, weight_decay=0.1)
    cracked_mod.train_steps(drafter, drafter_train_ds, drafter_optimizer, device, drafter_steps, batch_size,
                             log_interval=500, dtype="bfloat16", checkpoint_path=drafter_ckpt,
                             checkpoint_interval=checkpoint_interval)

    locator_examples = build_locator_examples(fold_train)
    locator_config = build_locator_config(tokenizer.vocab_size, n_layer=3, d_model=128, n_head=4, d_ff=512)
    locator = Locator(locator_config).to(device)
    locator_train_ds = PressedLocatorDataset(locator_examples, tokenizer, pressed_mod.LOCATOR_BLOCK_SIZE)
    locator_optimizer = build_locator_optimizer(locator, lr=3e-4, weight_decay=0.1)
    locator_train_steps(locator, locator_train_ds, locator_optimizer, device, locator_steps, batch_size,
                         log_interval=500, checkpoint_path=locator_ckpt, checkpoint_interval=checkpoint_interval)

    replacer_examples = build_replacer_examples(fold_train)
    replacer_config = build_replacer_config(tokenizer.vocab_size, pressed_mod.REPLACER_BLOCK_SIZE, n_layer=3,
                                             d_model=128, n_head=4, d_ff=512)
    replacer = Replacer(replacer_config).to(device)
    replacer_train_ds = InstructionDataset(replacer_examples, tokenizer, pressed_mod.REPLACER_BLOCK_SIZE)
    replacer_optimizer = cracked_mod.build_optimizer(replacer, lr=3e-4, weight_decay=0.1)
    cracked_mod.train_steps(replacer, replacer_train_ds, replacer_optimizer, device, replacer_steps, batch_size,
                             log_interval=500, dtype="bfloat16", checkpoint_path=replacer_ckpt,
                             checkpoint_interval=checkpoint_interval)
    return drafter, locator, replacer


@torch.no_grad()
def evaluate_fold(cracked, sliced, drafter, locator, replacer, tokenizer, device, holdout: list[dict]) -> dict:
    subset = holdout[:MAX_EVAL_EXAMPLES_PER_FOLD]
    correct = {"cracked": 0, "sliced": 0, "pressed": 0}
    for ex in subset:
        prompt = f"Question: {ex['question']}\n\nAnswer: "

        cracked_pred = extract_final_number(
            cracked_mod.generate_response(cracked, tokenizer, prompt, device, max_new_tokens=200))
        correct["cracked"] += answers_match(cracked_pred, ex["final_answer"])

        sliced_pred = extract_final_number(
            sliced_mod.generate_response(sliced, tokenizer, prompt, device, max_new_tokens=150))
        correct["sliced"] += answers_match(sliced_pred, ex["final_answer"])

        pressed_result = reduce_with_pressed(drafter, locator, replacer, tokenizer, prompt, device,
                                              max_new_tokens_draft=200)
        pressed_pred = extract_final_number(pressed_result.final_text)
        correct["pressed"] += answers_match(pressed_pred, ex["final_answer"])

    return {arch: correct[arch] / len(subset) for arch in correct}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cracked-steps", type=int, default=6800)
    p.add_argument("--sliced-steps", type=int, default=6000)
    p.add_argument("--drafter-steps", type=int, default=8000)
    p.add_argument("--locator-steps", type=int, default=3000)
    p.add_argument("--replacer-steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)

    tokenizer = pressed_mod.load_tokenizer()
    alpaca_train = load_instr_jsonl(INSTR_DIR / "alpaca_train.jsonl")
    conv_train = load_conv_jsonl(CONV_DIR / "oasst1_train.jsonl")
    gsm8k_train = load_gsm8k_jsonl(GSM8K_DIR / "gsm8k_train.jsonl")

    folds = build_folds(gsm8k_train, args.k, args.seed)
    print(f"{len(gsm8k_train)} GSM8K train examples -> {args.k} folds of ~{len(gsm8k_train) // args.k} each")

    per_fold_results: dict[int, dict] = {}
    for i in range(args.k):
        print(f"\n=== fold {i} ===")
        fold_train = fold_train_examples(folds, i)
        fold_holdout = folds[i]

        cracked = train_cracked_fold(tokenizer, args.device, fold_train, alpaca_train, conv_train,
                                      block_size=512, max_steps=args.cracked_steps, batch_size=args.batch_size,
                                      checkpoint_interval=args.checkpoint_interval,
                                      ckpt_path=KFOLD_CHECKPOINT_DIR / f"fold{i}_cracked.pt")
        sliced = train_sliced_fold(tokenizer, args.device, fold_train, alpaca_train, conv_train,
                                    max_steps=args.sliced_steps, batch_size=args.batch_size,
                                    checkpoint_interval=args.checkpoint_interval,
                                    ckpt_path=KFOLD_CHECKPOINT_DIR / f"fold{i}_sliced.pt")
        drafter, locator, replacer = train_pressed_fold(
            tokenizer, args.device, fold_train, alpaca_train, conv_train, args.drafter_steps, args.locator_steps,
            args.replacer_steps, args.batch_size, args.checkpoint_interval,
            drafter_ckpt=KFOLD_CHECKPOINT_DIR / f"fold{i}_drafter.pt",
            locator_ckpt=KFOLD_CHECKPOINT_DIR / f"fold{i}_locator.pt",
            replacer_ckpt=KFOLD_CHECKPOINT_DIR / f"fold{i}_replacer.pt")

        fold_acc = evaluate_fold(cracked, sliced, drafter, locator, replacer, tokenizer, args.device, fold_holdout)
        per_fold_results[i] = fold_acc
        print(f"fold {i} accuracy: {fold_acc}")

    summary = {}
    for arch in ("cracked", "sliced", "pressed"):
        accs = [per_fold_results[i][arch] for i in range(args.k)]
        mean = sum(accs) / len(accs)
        std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        summary[arch] = {"fold_accuracies": accs, "mean": mean, "std": std}

    print("\n--- k-fold stability summary (mean +- std across folds) ---")
    for arch, stats in summary.items():
        print(f"  {arch:8s} {stats['mean']:.3f} +- {stats['std']:.3f}  (folds: {stats['fold_accuracies']})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"per_fold": per_fold_results, "summary": summary, "k": args.k}
    with (out_dir / "kfold_stability.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    architectures = ["cracked", "sliced", "pressed"]
    labels = ["Cracked", "Sliced", "Pressed"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    means = [summary[a]["mean"] for a in architectures]
    stds = [summary[a]["std"] for a in architectures]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, means, yerr=stds, capsize=8, color=colors)
    ax.set_ylabel("GSM8K accuracy on held-out fold")
    ax.set_title(f"Fase C: k-fold={args.k} training stability")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "kfold_stability.png", dpi=150)
    print(f"chart saved to {out_dir / 'kfold_stability.png'}")


if __name__ == "__main__":
    main()
