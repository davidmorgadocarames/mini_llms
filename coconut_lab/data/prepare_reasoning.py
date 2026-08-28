"""Fase C.3: prepares GSM8K (grade-school math word problems in natural
language) as structured step-by-step supervision for Pressed's locator and
replacer (C.5) -- the natural-language analog of Fase B's paren-matching
supervision.

GSM8K's own answer format already annotates every intermediate calculation
inline, e.g. "Natalia sold 48/2 = <<48/2=24>>24 clips in May." -- each
<<expr=result>> is, in effect, a pre-labeled "reducible step" (compare to
depth_lab.data.reduce, which had to compute that labeling itself for
synthetic expressions). We just have to parse it out, not invent it.

Number of steps per problem (len(steps)) is this domain's analog of
"depth" in Fase B: the thing C.6 will bucket accuracy by, to see whether the
same OOD generalization pattern shows up in real reasoning problems.

GSM8K's own test split is used, untouched, as the held-out set for the
final C.6 comparison -- same "touch it exactly once" discipline as Fase B.

Usage:
    python -m coconut_lab.data.prepare_reasoning
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

_STEP_PATTERN = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
_FINAL_ANSWER_MARKER = "#### "


def extract_steps(answer_text: str) -> list[dict]:
    steps = []
    for m in _STEP_PATTERN.finditer(answer_text):
        steps.append({"expr": m.group(1), "result": m.group(2), "span": [m.start(), m.end()]})
    return steps


def extract_final_answer(answer_text: str) -> str:
    idx = answer_text.rindex(_FINAL_ANSWER_MARKER)
    return answer_text[idx + len(_FINAL_ANSWER_MARKER):].strip()


def format_example(question: str, answer_text: str) -> dict:
    steps = extract_steps(answer_text)
    return {
        "question": question,
        "answer_text": answer_text,
        "final_answer": extract_final_answer(answer_text),
        "steps": steps,
        "n_steps": len(steps),
    }


def _write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(val_fraction: float = 0.05, seed: int = 0) -> dict[str, Path]:
    dataset = load_dataset("openai/gsm8k", "main")

    train_and_val = [format_example(row["question"], row["answer"]) for row in dataset["train"]]
    test = [format_example(row["question"], row["answer"]) for row in dataset["test"]]

    rng = random.Random(seed)
    rng.shuffle(train_and_val)
    n_val = int(len(train_and_val) * val_fraction)
    val, train = train_and_val[:n_val], train_and_val[n_val:]

    paths = {
        "train": ARTIFACTS_DIR / "gsm8k_train.jsonl",
        "val": ARTIFACTS_DIR / "gsm8k_val.jsonl",
        "test": ARTIFACTS_DIR / "gsm8k_test.jsonl",
    }
    _write_jsonl(train, paths["train"])
    _write_jsonl(val, paths["val"])
    _write_jsonl(test, paths["test"])
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-fraction", type=float, default=0.05)
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
