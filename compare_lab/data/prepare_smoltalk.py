"""Prepare the fine-tuning set from HuggingFaceTB/smol-smoltalk (the <1B-param
SmolTalk subset). Applies the lessons from AUDITORIA_FASE_C.md:

- One canonical chat template (compare_lab.data.tokenizer.build_chat), identical
  in training, eval and demo.
- Conversations that don't fit the context are DISCARDED, never truncated (a
  truncated answer would train the model on half a response).
- A fixed test split (~2%) is set aside and never used in training or k-fold.
- Data is written in a fixed shuffled order so all three models fine-tune on the
  exact same conversations in the same order (the loader must not reshuffle).

Each stored record is {"messages": [{"role","content"}, ...]} that already passed
the length filter, so the fine-tuning loader can tokenize freely.

Usage:
    python -m compare_lab.data.prepare_smoltalk --max-conversations 150000
"""

import argparse
import json
import time
from pathlib import Path

from compare_lab.data.tokenizer import Tokenizer, DEFAULT_TOKENIZER_DIR
from compare_lab.train.tasks import StatusWriter

DATASET = "HuggingFaceTB/smol-smoltalk"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts" / "finetune"
VALID_ROLES = {"system", "user", "assistant"}
TASK_NAME = "data_prep_finetune"
# smol-smoltalk train has 460,341 conversations (~970MB download); fine-tuning
# only needs a fraction of that relative to the multi-hour pretrainings.
DEFAULT_MAX_CONVERSATIONS = 150_000


def normalize_messages(raw_messages) -> list[dict] | None:
    """Coerce a dataset row's messages into our {role, content} list, or None if
    it isn't a usable user->assistant conversation."""
    msgs = []
    for m in raw_messages:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value")
        if role == "human":
            role = "user"
        if role == "gpt":
            role = "assistant"
        if role not in VALID_ROLES or not content:
            return None
        msgs.append({"role": role, "content": content})
    if len(msgs) < 2:
        return None
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    return msgs


def sample_smoltalk_texts(max_docs: int, seed: int = 1337):
    """Yield rendered conversation text for tokenizer training (streaming)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    n = 0
    for row in ds:
        msgs = normalize_messages(row.get("messages", []))
        if not msgs:
            continue
        yield "\n".join(m["content"] for m in msgs)
        n += 1
        if n >= max_docs:
            break


def prepare(tokenizer: Tokenizer, block_size: int, out_dir: Path | None = None,
            max_conversations: int | None = DEFAULT_MAX_CONVERSATIONS, test_frac: float = 0.02,
            seed: int = 1337, skip_if_done: bool = False) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats.json"
    if skip_if_done and stats_path.exists():
        print(f"{stats_path} already exists -- skipping (pass skip_if_done=False to rebuild)")
        return json.loads(stats_path.read_text())

    from datasets import load_dataset

    status = StatusWriter(TASK_NAME, "shared", total_steps=1,
                          total_tokens=max_conversations or 460_341)
    t0 = time.time()
    print(f"Downloading {DATASET} (train split, ~970MB) ...")
    ds = load_dataset(DATASET, split="train")
    ds = ds.shuffle(seed=seed)

    kept, dropped_bad, dropped_long = [], 0, 0
    for i, row in enumerate(ds):
        msgs = normalize_messages(row.get("messages", []))
        if not msgs:
            dropped_bad += 1
            continue
        ids, _ = tokenizer.build_chat(msgs)
        if len(ids) > block_size:
            dropped_long += 1
            continue
        kept.append(msgs)
        if i % 5000 == 0:
            status.update(0, len(kept), tokens_per_sec=len(kept) / max(1e-9, time.time() - t0))
        if max_conversations and len(kept) >= max_conversations:
            break

    n_test = max(1, int(len(kept) * test_frac))
    test, train = kept[:n_test], kept[n_test:]

    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "test.jsonl", test)

    stats = {"kept": len(kept), "train": len(train), "test": len(test),
             "dropped_bad": dropped_bad, "dropped_too_long": dropped_long,
             "block_size": block_size, "max_conversations": max_conversations,
             "elapsed_seconds": round(time.time() - t0, 1)}
    stats_path.write_text(json.dumps(stats, indent=2))
    status.update(1, len(kept), force=True)
    status.finished()
    return stats


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for msgs in rows:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--max-conversations", type=int, default=DEFAULT_MAX_CONVERSATIONS)
    p.add_argument("--test-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    p.add_argument("--skip-if-done", action="store_true")
    args = p.parse_args()

    tok = Tokenizer.from_dir(args.tokenizer_dir)
    stats = prepare(tok, args.block_size, max_conversations=args.max_conversations,
                    test_frac=args.test_frac, seed=args.seed, skip_if_done=args.skip_if_done)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
