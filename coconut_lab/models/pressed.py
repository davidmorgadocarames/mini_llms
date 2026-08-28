"""Fase C.5: "Pressed" -- Looped Locate-and-Replace adapted to natural
language. See coconut_lab/data/prepare_pressed.py for why this needs a
"drafter" stage that Fase B's B.5 didn't: unlike a synthetic expression,
GSM8K reasoning (or a chat response) doesn't exist ahead of time for a new
question -- something has to write it first, before locator/replacer can
find and correct its calculations.

Three components, trained independently here and tied together at
inference by coconut_lab.models.pressed_loop:

  - drafter: mini_llm.model.GPT, trained from scratch (no pretrained
    checkpoint -- like Sliced) on GSM8K + Alpaca + oasst1 combined. Reuses
    coconut_lab.models.cracked's generic training/generation utilities,
    which only depend on GPT's forward/generate_stream contract, not on
    which checkpoint was loaded.
  - locator: depth_lab.models.locator.Locator, reused *unmodified* -- only
    the data loader changes (BPE-aware PressedLocatorDataset instead of
    depth_lab's char-level LocatorDataset).
  - replacer: depth_lab.models.replacer.Replacer, reused *unmodified* --
    trained via coconut_lab.models.cracked's generic training loop, since
    Replacer deliberately mirrors GPT's forward/generate_stream interface
    (that was Fase B's own design choice), over InstructionDataset-framed
    (expr, result) pairs from prepare_pressed.build_replacer_examples.

Usage:
    python -m coconut_lab.models.pressed drafter --max-steps 4000
    python -m coconut_lab.models.pressed locator --max-steps 3000
    python -m coconut_lab.models.pressed replacer --max-steps 1500
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from coconut_lab.data.loader import ConversationDataset, InstructionDataset, PressedLocatorDataset
from coconut_lab.data.prepare_conversations import ARTIFACTS_DIR as CONV_DIR
from coconut_lab.data.prepare_conversations import load_jsonl as load_conv_jsonl
from coconut_lab.data.prepare_instructions import ARTIFACTS_DIR as INSTR_DIR
from coconut_lab.data.prepare_instructions import load_jsonl as load_instr_jsonl
from coconut_lab.data.prepare_pressed import build_locator_examples, build_replacer_examples
from coconut_lab.data.prepare_reasoning import ARTIFACTS_DIR as GSM8K_DIR
from coconut_lab.data.prepare_reasoning import load_jsonl as load_gsm8k_jsonl
from coconut_lab.models.cracked import build_optimizer as build_gpt_optimizer
from coconut_lab.models.cracked import generate_response, masked_loss
from coconut_lab.models.cracked import train_steps as gpt_train_steps
from depth_lab.models.locator import Locator, LocatorConfig
from depth_lab.models.locator import build_config as build_locator_config
from depth_lab.models.locator import build_optimizer as build_locator_optimizer
from depth_lab.models.locator import train_steps as locator_train_steps
from depth_lab.models.replacer import Replacer, ReplacerConfig
from depth_lab.models.replacer import build_config as build_replacer_config
from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer import BPETokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HF_REPO = "davidmorgado/coconut-mini-llm"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

DRAFTER_BLOCK_SIZE = 512
LOCATOR_BLOCK_SIZE = 384
REPLACER_BLOCK_SIZE = 48


def load_tokenizer() -> BPETokenizer:
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    return BPETokenizer(vocab_path, merges_path)


def gsm8k_to_instructions(examples: list[dict]) -> list[dict]:
    return [{"prompt": f"Question: {ex['question']}\n\nAnswer: ", "response": ex["answer_text"]} for ex in examples]


def _drafter_datasets(tokenizer: BPETokenizer, block_size: int):
    alpaca_train = load_instr_jsonl(INSTR_DIR / "alpaca_train.jsonl")
    alpaca_val = load_instr_jsonl(INSTR_DIR / "alpaca_val.jsonl")
    conv_train = load_conv_jsonl(CONV_DIR / "oasst1_train.jsonl")
    conv_val = load_conv_jsonl(CONV_DIR / "oasst1_val.jsonl")
    gsm8k_train = gsm8k_to_instructions(load_gsm8k_jsonl(GSM8K_DIR / "gsm8k_train.jsonl"))
    gsm8k_val = gsm8k_to_instructions(load_gsm8k_jsonl(GSM8K_DIR / "gsm8k_val.jsonl"))

    train_ds = torch.utils.data.ConcatDataset([
        InstructionDataset(alpaca_train, tokenizer, block_size),
        ConversationDataset(conv_train, tokenizer, block_size),
        InstructionDataset(gsm8k_train, tokenizer, block_size),
    ])
    val_ds = torch.utils.data.ConcatDataset([
        InstructionDataset(alpaca_val, tokenizer, block_size),
        ConversationDataset(conv_val, tokenizer, block_size),
        InstructionDataset(gsm8k_val, tokenizer, block_size),
    ])
    return train_ds, val_ds


def _save(model: torch.nn.Module, config, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config)}, out_dir / f"{name}.pt")
    print(f"done. checkpoint saved to {out_dir / f'{name}.pt'}")


def train_drafter(args: argparse.Namespace) -> None:
    torch.manual_seed(1337)
    tokenizer = load_tokenizer()
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=DRAFTER_BLOCK_SIZE,
                       n_layer=args.n_layer, n_embd=args.d_model, n_head=args.n_head, n_kv_head=args.n_kv_head)
    model = GPT(config).to(args.device)
    print(f"drafter: {model.num_parameters():,} parameters, config={config}")

    train_ds, val_ds = _drafter_datasets(tokenizer, DRAFTER_BLOCK_SIZE)
    print(f"combined train: {len(train_ds)} | combined val: {len(val_ds)}")

    optimizer = build_gpt_optimizer(model, args.lr, args.weight_decay)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gpt_train_steps(model, train_ds, optimizer, args.device, args.max_steps, args.batch_size, args.log_interval,
                     args.grad_accum_steps, args.dtype)
    elapsed = time.time() - t0
    print(f"\ntraining took {elapsed:.1f}s for {args.max_steps} steps ({elapsed / args.max_steps:.3f}s/step)")
    if args.device == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    _save(model, config, "pressed_drafter", Path(args.out_dir))

    print("\nsample generations:")
    for prompt in ["Question: What is 2 plus 2?\n\nAnswer: ",
                   "Below is an instruction that describes a task. Write a response that appropriately "
                   "completes the request.\n\n### Instruction:\nName three colors.\n\n### Response:\n"]:
        print(f"  {generate_response(model, tokenizer, prompt, args.device)!r}")


def train_locator(args: argparse.Namespace) -> None:
    torch.manual_seed(1337)
    tokenizer = load_tokenizer()
    gsm8k_train = load_gsm8k_jsonl(GSM8K_DIR / "gsm8k_train.jsonl")
    locator_examples = build_locator_examples(gsm8k_train)
    print(f"{len(locator_examples)} locator training instances from {len(gsm8k_train)} problems")

    config = build_locator_config(tokenizer.vocab_size, args.n_layer, args.d_model, args.n_head, args.d_ff)
    model = Locator(config).to(args.device)
    print(f"locator: {model.num_parameters():,} parameters, config={config}")

    train_ds = PressedLocatorDataset(locator_examples, tokenizer, LOCATOR_BLOCK_SIZE)
    print(f"{len(train_ds)}/{len(locator_examples)} instances fit block_size={LOCATOR_BLOCK_SIZE}")
    optimizer = build_locator_optimizer(model, args.lr, args.weight_decay)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    locator_train_steps(model, train_ds, optimizer, args.device, args.max_steps, args.batch_size, args.log_interval)
    elapsed = time.time() - t0
    print(f"\ntraining took {elapsed:.1f}s for {args.max_steps} steps ({elapsed / args.max_steps:.3f}s/step)")
    if args.device == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    _save(model, config, "pressed_locator", Path(args.out_dir))


def train_replacer(args: argparse.Namespace) -> None:
    torch.manual_seed(1337)
    tokenizer = load_tokenizer()
    gsm8k_train = load_gsm8k_jsonl(GSM8K_DIR / "gsm8k_train.jsonl")
    replacer_examples = build_replacer_examples(gsm8k_train)
    print(f"{len(replacer_examples)} replacer training instances from {len(gsm8k_train)} problems")

    config = build_replacer_config(tokenizer.vocab_size, REPLACER_BLOCK_SIZE, args.n_layer, args.d_model,
                                    args.n_head, args.d_ff)
    model = Replacer(config).to(args.device)
    print(f"replacer: {model.num_parameters():,} parameters, config={config}")

    train_ds = InstructionDataset(replacer_examples, tokenizer, REPLACER_BLOCK_SIZE)
    print(f"{len(train_ds)}/{len(replacer_examples)} instances fit block_size={REPLACER_BLOCK_SIZE}")
    optimizer = build_gpt_optimizer(model, args.lr, args.weight_decay)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gpt_train_steps(model, train_ds, optimizer, args.device, args.max_steps, args.batch_size, args.log_interval,
                     args.grad_accum_steps, args.dtype)
    elapsed = time.time() - t0
    print(f"\ntraining took {elapsed:.1f}s for {args.max_steps} steps ({elapsed / args.max_steps:.3f}s/step)")
    if args.device == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    _save(model, config, "pressed_replacer", Path(args.out_dir))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="component", required=True)

    for name, defaults in [
        ("drafter", dict(n_layer=6, d_model=384, n_head=6, max_steps=4000, lr=3e-4, weight_decay=0.1)),
        ("locator", dict(n_layer=3, d_model=128, n_head=4, max_steps=3000, lr=3e-4, weight_decay=0.1)),
        ("replacer", dict(n_layer=3, d_model=128, n_head=4, max_steps=1500, lr=3e-4, weight_decay=0.1)),
    ]:
        sp = sub.add_parser(name)
        sp.add_argument("--n-layer", type=int, default=defaults["n_layer"])
        sp.add_argument("--d-model", type=int, default=defaults["d_model"])
        sp.add_argument("--n-head", type=int, default=defaults["n_head"])
        sp.add_argument("--d-ff", type=int, default=512)
        sp.add_argument("--n-kv-head", type=int, default=2)
        sp.add_argument("--batch-size", type=int, default=16)
        sp.add_argument("--max-steps", type=int, default=defaults["max_steps"])
        sp.add_argument("--lr", type=float, default=defaults["lr"])
        sp.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
        sp.add_argument("--log-interval", type=int, default=200)
        sp.add_argument("--grad-accum-steps", type=int, default=1)
        sp.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--out-dir", default=str(CHECKPOINT_DIR))

    return p.parse_args()


def main() -> None:
    args = parse_args()
    {"drafter": train_drafter, "locator": train_locator, "replacer": train_replacer}[args.component](args)


if __name__ == "__main__":
    main()
