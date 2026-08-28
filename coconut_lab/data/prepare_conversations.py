"""Fase C.1b: prepares OpenAssistant/oasst1 (real multi-turn conversation
trees) so Cracked/Sliced/Pressed can be fine-tuned for actual back-and-forth
dialogue, not just single instruction-response pairs like Alpaca (C.1a).

oasst1 stores each conversation as a tree: a root "prompter" message, with
one or more candidate "assistant" replies at each node (siblings, ranked by
quality), which themselves may have further replies, and so on. We turn each
tree into one or more root-to-leaf *paths* by always following the
best-ranked child (rank 0) at every branch -- the same "pick the preferred
continuation" idea used to build any SFT conversation dataset from ranked
data, just without a separate reward model.

Usage:
    python -m coconut_lab.data.prepare_conversations
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

USER_MARKER = "<|user|>\n"
ASSISTANT_MARKER = "<|assistant|>\n"


def _build_trees(rows: list[dict]) -> list[list[dict]]:
    """Returns one root-to-leaf path per conversation tree, always following
    the lowest-rank (best) child at each branch. rows must already be
    filtered to a single language and to approved, non-deleted messages."""
    by_parent: dict[str | None, list[dict]] = defaultdict(list)
    by_id: dict[str, dict] = {}
    for row in rows:
        by_parent[row["parent_id"]].append(row)
        by_id[row["message_id"]] = row

    roots = by_parent.get(None, [])
    paths: list[list[dict]] = []
    for root in roots:
        path = [root]
        current = root
        while True:
            children = by_parent.get(current["message_id"], [])
            if not children:
                break
            children = sorted(children, key=lambda c: (c["rank"] if c["rank"] is not None else 0))
            current = children[0]
            path.append(current)
        paths.append(path)
    return paths


def _path_to_turns(path: list[dict]) -> list[dict]:
    return [{"role": "user" if row["role"] == "prompter" else "assistant", "text": row["text"]} for row in path]


def build(val_fraction: float = 0.02, seed: int = 0, min_turns: int = 2) -> dict[str, Path]:
    dataset = load_dataset("OpenAssistant/oasst1")["train"]
    rows = [
        row for row in dataset
        if row["lang"] == "en" and not row["deleted"] and row["review_result"]
    ]
    paths = _build_trees(rows)

    conversations = []
    for path in paths:
        turns = _path_to_turns(path)
        # every conversation must start with a user turn and end with an
        # assistant turn (a trailing unanswered user message has nothing to
        # train the loss on)
        if turns and turns[0]["role"] != "user":
            turns = turns[1:]
        if turns and turns[-1]["role"] != "assistant":
            turns = turns[:-1]
        if len(turns) >= min_turns:
            conversations.append({"turns": turns})

    rng = random.Random(seed)
    rng.shuffle(conversations)
    n_val = int(len(conversations) * val_fraction)
    val, train = conversations[:n_val], conversations[n_val:]

    paths_out = {
        "train": ARTIFACTS_DIR / "oasst1_train.jsonl",
        "val": ARTIFACTS_DIR / "oasst1_val.jsonl",
    }
    for split_name, split_data in (("train", train), ("val", val)):
        path = paths_out[split_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
    return paths_out


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_turns(turns: list[dict]) -> str:
    """Renders a list of {"role", "text"} turns with the <|user|>/
    <|assistant|> markers, in order -- shared by the loader (for
    tokenization) and anything that needs a human-readable transcript."""
    marker = {"user": USER_MARKER, "assistant": ASSISTANT_MARKER}
    return "".join(f"{marker[t['role']]}{t['text']}\n" for t in turns)


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
        print(f"{name:8s} {n:6d} conversations -> {path}")


if __name__ == "__main__":
    main()
