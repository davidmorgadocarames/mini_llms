"""Build the fixed pretraining sample once and freeze it (plan sections 1-2).

Steps:
  1. (optional) Train the shared BPE tokenizer on a streamed ~90/10 mix of
     pretraining text and conversations.
  2. Stream SmolLM-Corpus (cosmopedia-v2 + fineweb-edu-dedup, 50/50 by tokens),
     tokenize on the fly, and write val.bin (disjoint, ~5M tokens) then train.bin.
  3. Build and save the prefix-LM plan (cut points + visit order) for train.bin.

The same sample is reused by all three models; nothing here reads the internet at
training time. Sizes are configurable; the default 1B-token sample fits the 24h
GPU budget across the three pretrainings (finalize after the throughput probe).

Usage (short smoke run):
    python -m compare_lab.data.prepare_pretrain --tokens 2_000_000 --val-tokens 100_000 \
        --block-size 256 --batch-size 8 --retrain-tokenizer --tokenizer-docs 300
"""

import argparse
import json
import time
from pathlib import Path

from compare_lab.data.bindata import write_bin_from_token_iter, load_bin
from compare_lab.data.prefix_lm import build_plan, save_plan
from compare_lab.data.streaming import mixed_token_lists, sample_texts_for_tokenizer, SUBSETS
from compare_lab.data.tokenizer import Tokenizer, train_tokenizer, DEFAULT_TOKENIZER_DIR
from compare_lab.train.tasks import StatusWriter, stop_requested, clear_stop

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
PRETRAIN_DIR = ARTIFACTS_DIR / "pretrain"
TASK_NAME = "data_prep_pretrain"


def _tokenizer_text_sample(tokenizer_docs: int, seed: int):
    """~90% pretraining text + ~10% conversations for tokenizer training."""
    from compare_lab.data.prepare_smoltalk import sample_smoltalk_texts

    yield from sample_texts_for_tokenizer(max_docs_per_subset=tokenizer_docs // 2, seed=seed)
    yield from sample_smoltalk_texts(max_docs=max(1, tokenizer_docs // 9), seed=seed)


def build(tokens: int, val_tokens: int, block_size: int, batch_size: int, seed: int,
          tokenizer_dir: Path, out_dir: Path | None = None,
          retrain_tokenizer: bool = False, tokenizer_docs: int = 20_000,
          vocab_size: int = 8192, skip_if_done: bool = False) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else PRETRAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats.json"
    if skip_if_done and stats_path.exists():
        print(f"{stats_path} already exists -- skipping (pass skip_if_done=False to rebuild)")
        return json.loads(stats_path.read_text())

    total_target = tokens + val_tokens
    clear_stop(TASK_NAME)
    status = StatusWriter(TASK_NAME, "shared", total_steps=1, total_tokens=total_target)
    t0 = time.time()

    # Train the tokenizer when asked to, and also whenever it simply isn't there
    # yet: loading a missing one fails deep inside the `tokenizers` library with
    # an opaque error, which is what running this module standalone used to do
    # (the pipeline always passes retrain_tokenizer=True, so only the documented
    # standalone command hit it).
    # Require BOTH files: an interrupted tokenizer training leaves a half-written
    # directory, and a dir with only vocab.json would take the "reuse" branch and
    # fail on the missing merges.txt with the same opaque error.
    tokenizer_exists = all((Path(tokenizer_dir) / f).exists()
                           for f in ("vocab.json", "merges.txt"))
    if retrain_tokenizer or not tokenizer_exists:
        why = "as requested" if retrain_tokenizer else f"({tokenizer_dir} has none yet)"
        print(f"Training tokenizer from streamed sample {why} ...")
        status.update(0, 0, force=True)
        tok = train_tokenizer(_tokenizer_text_sample(tokenizer_docs, seed),
                              vocab_size=vocab_size, out_dir=tokenizer_dir)
    else:
        print(f"Reusing the tokenizer in {tokenizer_dir}")
        tok = Tokenizer.from_dir(tokenizer_dir)

    # One mixed stream; val is pulled first, then train continues from it (disjoint).
    # half_budget is HALF the total target, so the 50/50-by-tokens balancing in
    # interleave_token_lists actually has the right per-subset target to aim at.
    total_target = tokens + val_tokens
    produced = [0, 0]  # real per-subset token counts, filled by the stream
    raw_stream = mixed_token_lists(tok.encode, half_budget=total_target // 2, seed=seed,
                                   produced_out=produced)

    # Append <eos> after every document. Without a separator the fixed-size
    # fragments straddle document boundaries with no signal, so many prefix-LM
    # pairs would be "end of doc A -> start of unrelated doc B" (pure noise);
    # and the model would never see <eos> during pretraining, only in
    # fine-tuning, despite being asked to learn to stop. Same convention as
    # nanoGPT/GPT-2 (an end-of-text token between documents).
    def with_eos(token_lists):
        for ids in token_lists:
            yield list(ids) + [tok.eos_id]

    stream = with_eos(raw_stream)

    def progress(base: int):
        def cb(written: int):
            done = base + written
            tok_s = done / max(1e-9, time.time() - t0)
            status.update(0, done, tokens_per_sec=tok_s)
            # This stage is NOT resumable mid-way (the bins are written from
            # scratch and the HF stream cannot be rewound), so a STOP here means
            # "abandon and start over next time" -- but it should at least exit
            # cleanly and say so, instead of leaving the task stuck at "running".
            if stop_requested(TASK_NAME):
                status.set_state("paused")
                raise KeyboardInterrupt(
                    "STOP requested during data prep; this stage restarts from "
                    "scratch on the next launch (it is not resumable mid-way)")
        return cb

    print(f"Writing val.bin (target {val_tokens:,} tokens) ...")
    n_val = write_bin_from_token_iter(stream, out_dir / "val.bin", val_tokens, on_progress=progress(0))
    print(f"Writing train.bin (target {tokens:,} tokens) ...")
    n_train = write_bin_from_token_iter(stream, out_dir / "train.bin", tokens, on_progress=progress(n_val))

    # The writer returns whatever it managed to get. If a stream ran dry early,
    # everything downstream would still "succeed" -- build_plan would scale
    # total_opt_steps to the short corpus and the pretraining would report
    # finished on a truncated sample, with nothing to flag it.
    shortfall = 0.02  # allow the small overshoot/undershoot of whole documents
    if n_train < tokens * (1 - shortfall) or n_val < val_tokens * (1 - shortfall):
        raise RuntimeError(
            f"data prep came up short: got {n_train:,} train (target {tokens:,}) and "
            f"{n_val:,} val (target {val_tokens:,}) tokens. Refusing to build a plan over a "
            f"truncated corpus -- check the network/stream and rerun.")

    print("Building prefix-LM plan ...")
    plan = build_plan(n_train, block_size=block_size, batch_size=batch_size, seed=seed)
    save_plan(plan, out_dir / "plan")

    total_produced = sum(produced) or 1
    stats = {"n_train_tokens": int(n_train), "n_val_tokens": int(n_val),
             "block_size": block_size, "batch_size": batch_size, "seed": seed,
             "n_fragments": int(plan.n_fragments), "n_batches": int(plan.n_batches),
             "train_bin_bytes": int(n_train * 2), "val_bin_bytes": int(n_val * 2),
             "vocab_size": tok.vocab_size, "elapsed_seconds": round(time.time() - t0, 1),
             # achieved mix, so the real composition is reportable (target: 50/50)
             "subsets": list(SUBSETS),
             "tokens_per_subset": [int(p) for p in produced],
             "pct_per_subset": [round(100.0 * p / total_produced, 2) for p in produced],
             "eos_between_documents": True}
    stats_path.write_text(json.dumps(stats, indent=2))
    status.update(1, n_train + n_val, force=True)
    status.finished()
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=1_200_000_000)
    p.add_argument("--val-tokens", type=int, default=5_000_000)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    p.add_argument("--retrain-tokenizer", action="store_true")
    p.add_argument("--tokenizer-docs", type=int, default=20_000)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--skip-if-done", action="store_true")
    args = p.parse_args()

    stats = build(args.tokens, args.val_tokens, args.block_size, args.batch_size,
                  args.seed, Path(args.tokenizer_dir), retrain_tokenizer=args.retrain_tokenizer,
                  tokenizer_docs=args.tokenizer_docs, vocab_size=args.vocab_size,
                  skip_if_done=args.skip_if_done)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
