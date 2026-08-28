"""Fase C.6: trains a "final" comparison-ready checkpoint for each
architecture (Cracked, Sliced, Pressed) on a *consistent* data mixture that
includes GSM8K -- not just Alpaca+oasst1. Without this, Cracked/Sliced would
be asked to solve GSM8K test problems zero-shot while Pressed's drafter has
already been fine-tuned on GSM8K's format (see C.5), which wouldn't be an
apples-to-apples comparison.

Evaluates GSM8K test accuracy bucketed by number of reasoning steps -- the
natural-language analog of depth_lab/eval/run_eval.py's
accuracy-vs-depth.png -- and measures efficiency (params, generation
latency, memory).

Split discipline: GSM8K's own test split (1319 problems, never touched
during training) is read exactly once, here, for the final comparison --
same discipline as depth_lab/eval/run_eval.py.

Usage:
    python -m coconut_lab.eval.run_eval
    python -m coconut_lab.eval.run_eval --skip-training
"""

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from coconut_lab.data.loader import ConversationDataset, InstructionDataset, Seq2SeqDataset
from coconut_lab.data.prepare_conversations import ARTIFACTS_DIR as CONV_DIR
from coconut_lab.data.prepare_conversations import load_jsonl as load_conv_jsonl
from coconut_lab.data.prepare_instructions import ARTIFACTS_DIR as INSTR_DIR
from coconut_lab.data.prepare_instructions import load_jsonl as load_instr_jsonl
from coconut_lab.data.prepare_reasoning import ARTIFACTS_DIR as GSM8K_DIR
from coconut_lab.data.prepare_reasoning import gsm8k_to_instructions, load_jsonl as load_gsm8k_jsonl
from coconut_lab.models import cracked as cracked_mod
from coconut_lab.models import pressed as pressed_mod
from coconut_lab.models import sliced as sliced_mod
from coconut_lab.models.pressed_loop import reduce_with_pressed
from depth_lab.models.encoder_decoder import EncoderDecoderTransformer
from depth_lab.models.locator import Locator, LocatorConfig
from depth_lab.models.replacer import Replacer, ReplacerConfig
from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer import BPETokenizer
from mini_llm.tokenizer.bpe import EOT_TOKEN

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

MAX_EXAMPLES_PER_STEP_BUCKET = 40  # keeps total eval cost tractable across 9 step buckets

_FINAL_MARKER_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_ANY_NUMBER_PATTERN = re.compile(r"-?[\d,]+(?:\.\d+)?")


def extract_final_number(text: str) -> str | None:
    """Standard GSM8K answer-extraction heuristic (matches the paper's own
    and lm-evaluation-harness's gsm8k task): prefer the number right after a
    "####" marker if the model produced one, else fall back to the last
    number anywhere in the generated text."""
    m = _FINAL_MARKER_PATTERN.search(text)
    if m:
        return m.group(1).replace(",", "")
    numbers = _ANY_NUMBER_PATTERN.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def answers_match(predicted: str | None, true_answer: str) -> bool:
    if predicted is None:
        return False
    try:
        return float(predicted) == float(true_answer.replace(",", ""))
    except ValueError:
        return False


def build_final_chat_datasets(tokenizer: BPETokenizer, block_size: int):
    """The consistent Alpaca + oasst1 + GSM8K mixture used to train the
    "final" Cracked and Sliced checkpoints for this comparison."""
    alpaca_train = load_instr_jsonl(INSTR_DIR / "alpaca_train.jsonl")
    alpaca_val = load_instr_jsonl(INSTR_DIR / "alpaca_val.jsonl")
    conv_train = load_conv_jsonl(CONV_DIR / "oasst1_train.jsonl")
    conv_val = load_conv_jsonl(CONV_DIR / "oasst1_val.jsonl")
    gsm8k_train = gsm8k_to_instructions(load_gsm8k_jsonl(GSM8K_DIR / "gsm8k_train.jsonl"))
    gsm8k_val = gsm8k_to_instructions(load_gsm8k_jsonl(GSM8K_DIR / "gsm8k_val.jsonl"))
    return {"alpaca_train": alpaca_train, "alpaca_val": alpaca_val,
            "conv_train": conv_train, "conv_val": conv_val,
            "gsm8k_train": gsm8k_train, "gsm8k_val": gsm8k_val}


def train_or_load_cracked(tokenizer: BPETokenizer, device: str, max_steps: int, batch_size: int,
                           skip_training: bool) -> GPT:
    ckpt_path = CHECKPOINT_DIR / "cracked_final.pt"
    model, _ = cracked_mod.load_base_model(device)

    if skip_training and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[cracked] loaded existing final checkpoint ({ckpt_path})")
        return model

    data = build_final_chat_datasets(tokenizer, model.config.block_size)
    train_ds = torch.utils.data.ConcatDataset([
        InstructionDataset(data["alpaca_train"], tokenizer, model.config.block_size),
        ConversationDataset(data["conv_train"], tokenizer, model.config.block_size),
        InstructionDataset(data["gsm8k_train"], tokenizer, model.config.block_size),
    ])
    print(f"[cracked] final train set: {len(train_ds)} examples")
    optimizer = cracked_mod.build_optimizer(model, lr=5e-5, weight_decay=0.01)
    cracked_mod.train_steps(model, train_ds, optimizer, device, max_steps, batch_size, log_interval=300,
                             grad_accum_steps=1, dtype="bfloat16")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(model.config)}, ckpt_path)
    return model


def train_or_load_sliced(tokenizer: BPETokenizer, device: str, max_steps: int, batch_size: int,
                          skip_training: bool) -> EncoderDecoderTransformer:
    ckpt_path = CHECKPOINT_DIR / "sliced_final.pt"
    config = sliced_mod.build_config(tokenizer.vocab_size)
    model = EncoderDecoderTransformer(config).to(device)

    if skip_training and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[sliced] loaded existing final checkpoint ({ckpt_path})")
        return model

    data = build_final_chat_datasets(tokenizer, sliced_mod.SRC_BLOCK_SIZE)
    alpaca = sliced_mod.alpaca_to_seq2seq(data["alpaca_train"])
    gsm8k = sliced_mod.alpaca_to_seq2seq(data["gsm8k_train"])
    conv = sliced_mod.conversations_to_seq2seq(data["conv_train"])
    train_ds = torch.utils.data.ConcatDataset([
        Seq2SeqDataset(alpaca, tokenizer, sliced_mod.SRC_BLOCK_SIZE, sliced_mod.TGT_BLOCK_SIZE),
        Seq2SeqDataset(conv, tokenizer, sliced_mod.SRC_BLOCK_SIZE, sliced_mod.TGT_BLOCK_SIZE),
        Seq2SeqDataset(gsm8k, tokenizer, sliced_mod.SRC_BLOCK_SIZE, sliced_mod.TGT_BLOCK_SIZE),
    ])
    print(f"[sliced] final train set: {len(train_ds)} examples")
    pad_id = tokenizer.encode(EOT_TOKEN)[0]
    optimizer = sliced_mod.build_optimizer(model, lr=3e-4, weight_decay=0.01)
    sliced_mod.train_steps(model, train_ds, optimizer, device, max_steps, batch_size, pad_id, log_interval=300,
                            grad_accum_steps=1, dtype="bfloat16")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config)}, ckpt_path)
    return model


def train_or_load_pressed(tokenizer: BPETokenizer, device: str, drafter_steps: int, locator_steps: int,
                           replacer_steps: int, batch_size: int, skip_training: bool):
    drafter_ckpt_path = CHECKPOINT_DIR / "pressed_drafter_final.pt"
    locator_ckpt_path = CHECKPOINT_DIR / "pressed_locator.pt"
    replacer_ckpt_path = CHECKPOINT_DIR / "pressed_replacer.pt"

    drafter_config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=pressed_mod.DRAFTER_BLOCK_SIZE,
                                n_layer=6, n_embd=384, n_head=6, n_kv_head=2)
    drafter = GPT(drafter_config).to(device)

    locator_ckpt = torch.load(locator_ckpt_path, map_location=device, weights_only=False)
    locator = Locator(LocatorConfig(**locator_ckpt["config"])).to(device)
    locator.load_state_dict(locator_ckpt["model"])

    replacer_ckpt = torch.load(replacer_ckpt_path, map_location=device, weights_only=False)
    replacer = Replacer(ReplacerConfig(**replacer_ckpt["config"])).to(device)
    replacer.load_state_dict(replacer_ckpt["model"])
    print("[pressed] loaded existing locator/replacer (already validated in C.5, not retrained here)")

    if skip_training and drafter_ckpt_path.exists():
        ckpt = torch.load(drafter_ckpt_path, map_location=device, weights_only=False)
        drafter.load_state_dict(ckpt["model"])
        print(f"[pressed] loaded existing final drafter checkpoint ({drafter_ckpt_path})")
        return drafter, locator, replacer

    train_ds, _ = pressed_mod._drafter_datasets(tokenizer, pressed_mod.DRAFTER_BLOCK_SIZE)
    print(f"[pressed] final drafter train set: {len(train_ds)} examples")
    optimizer = cracked_mod.build_optimizer(drafter, lr=3e-4, weight_decay=0.1)
    cracked_mod.train_steps(drafter, train_ds, optimizer, device, drafter_steps, batch_size, log_interval=300,
                             grad_accum_steps=1, dtype="bfloat16")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": drafter.state_dict(), "config": dataclasses.asdict(drafter_config)}, drafter_ckpt_path)
    return drafter, locator, replacer


@torch.no_grad()
def evaluate_by_steps(models: dict, tokenizer: BPETokenizer, device: str) -> dict:
    """Reads bool_test... no, gsm8k_test.jsonl only here, at the very end --
    this is the one and only place test data is touched."""
    test_examples = load_gsm8k_jsonl(GSM8K_DIR / "gsm8k_test.jsonl")
    by_step: dict[int, list[dict]] = {}
    for ex in test_examples:
        by_step.setdefault(ex["n_steps"], []).append(ex)

    cracked, sliced, (drafter, locator, replacer) = models["cracked"], models["sliced"], models["pressed"]
    results: dict[str, dict[int, float]] = {"cracked": {}, "sliced": {}, "pressed": {}}

    for n_steps in sorted(by_step):
        subset = by_step[n_steps][:MAX_EXAMPLES_PER_STEP_BUCKET]
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

        for arch in results:
            results[arch][n_steps] = correct[arch] / len(subset)
        print(f"n_steps {n_steps} (n={len(subset)}) | cracked {results['cracked'][n_steps]:.3f} | "
              f"sliced {results['sliced'][n_steps]:.3f} | pressed {results['pressed'][n_steps]:.3f}")

    return results


@torch.no_grad()
def measure_efficiency(models: dict, tokenizer: BPETokenizer, device: str) -> dict:
    cracked, sliced, (drafter, locator, replacer) = models["cracked"], models["sliced"], models["pressed"]
    prompt = "Question: What is 12 plus 15?\n\nAnswer: "
    efficiency = {}

    for name, model in [("cracked", cracked), ("sliced_encoder_decoder", sliced),
                         ("pressed_drafter", drafter), ("pressed_locator", locator),
                         ("pressed_replacer", replacer)]:
        efficiency[name] = {"params": model.num_parameters()}

    for name, fn in [
        ("cracked", lambda: cracked_mod.generate_response(cracked, tokenizer, prompt, device, max_new_tokens=100)),
        ("sliced", lambda: sliced_mod.generate_response(sliced, tokenizer, "What is 12 plus 15?", device,
                                                          max_new_tokens=50)),
        ("pressed", lambda: reduce_with_pressed(drafter, locator, replacer, tokenizer, prompt, device,
                                                 max_new_tokens_draft=100)),
    ]:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        efficiency.setdefault(name, {})["latency_s"] = time.time() - t0

    return efficiency


def plot_accuracy_vs_steps(results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {
        "cracked": dict(color="#e74c3c", marker="o", label="Cracked (decoder-only)"),
        "sliced": dict(color="#3498db", marker="s", label="Sliced (encoder-decoder)"),
        "pressed": dict(color="#2ecc71", marker="^", label="Pressed (LLR)"),
    }
    for name, per_step in results.items():
        steps = sorted(per_step)
        accs = [per_step[s] for s in steps]
        ax.plot(steps, accs, linewidth=2, markersize=7, **styles[name])

    ax.set_xlabel("Reasoning steps in the GSM8K problem")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title("Fase C: accuracy vs. reasoning steps (GSM8K test set)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"chart saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cracked-steps", type=int, default=6800)
    p.add_argument("--sliced-steps", type=int, default=6000)
    p.add_argument("--drafter-steps", type=int, default=8000)
    p.add_argument("--locator-steps", type=int, default=3000)
    p.add_argument("--replacer-steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)

    tokenizer = pressed_mod.load_tokenizer()

    cracked = train_or_load_cracked(tokenizer, args.device, args.cracked_steps, args.batch_size, args.skip_training)
    sliced = train_or_load_sliced(tokenizer, args.device, args.sliced_steps, args.batch_size, args.skip_training)
    drafter, locator, replacer = train_or_load_pressed(tokenizer, args.device, args.drafter_steps,
                                                         args.locator_steps, args.replacer_steps, args.batch_size,
                                                         args.skip_training)

    models = {"cracked": cracked, "sliced": sliced, "pressed": (drafter, locator, replacer)}

    print("\n--- final evaluation (GSM8K test set, touched only here) ---")
    results = evaluate_by_steps(models, tokenizer, args.device)

    print("\n--- efficiency ---")
    efficiency = measure_efficiency(models, tokenizer, args.device)
    for name, stats in efficiency.items():
        print(f"  {name}: {stats}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "accuracy_vs_steps.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with (out_dir / "efficiency.json").open("w", encoding="utf-8") as f:
        json.dump(efficiency, f, indent=2)

    plot_accuracy_vs_steps(results, out_dir / "accuracy_vs_steps.png")


if __name__ == "__main__":
    main()
