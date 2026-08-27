"""Fase C.1: prepares the Alpaca instruction-tuning dataset for fine-tuning
Coconut ("Cracked" -- see the Fase C plan) into something that follows
instructions instead of only continuing WikiText-style prose.

Splits each example into a (prompt, response) pair using the standard Alpaca
prompt template (Taori et al. 2023) rather than inventing our own -- this is
a well-known, widely reused format, no reason to deviate from it. Saved as
JSONL, not pre-tokenized: at ~52K short examples this is small enough to
tokenize on the fly (see coconut_lab/data/loader.py), unlike Fase A's
WikiText corpus which needed the binary-mmap pipeline in mini_llm/data/.

Usage:
    python -m coconut_lab.data.prepare_instructions
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# Standard Alpaca prompt template (Taori et al. 2023, tatsu-lab/stanford_alpaca)
_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_example(instruction: str, input_: str, output: str) -> dict:
    template = _PROMPT_WITH_INPUT if input_.strip() else _PROMPT_NO_INPUT
    prompt = template.format(instruction=instruction, input=input_)
    return {"prompt": prompt, "response": output}


def _write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(val_fraction: float = 0.02, seed: int = 0) -> dict[str, Path]:
    dataset = load_dataset("tatsu-lab/alpaca")["train"]
    examples = [
        format_example(row["instruction"], row["input"], row["output"])
        for row in dataset
    ]

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = int(len(examples) * val_fraction)
    val, train = examples[:n_val], examples[n_val:]

    paths = {
        "train": ARTIFACTS_DIR / "alpaca_train.jsonl",
        "val": ARTIFACTS_DIR / "alpaca_val.jsonl",
    }
    _write_jsonl(train, paths["train"])
    _write_jsonl(val, paths["val"])
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = build(args.val_fraction, args.seed)
    for name, path in paths.items():
        n = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"{name:8s} {n:6d} examples -> {path}")


if __name__ == "__main__":
    main()
