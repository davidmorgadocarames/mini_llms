"""Download a real text corpus, train a BPE tokenizer on it, and binarize the
train/val splits into uint16 token-id arrays for fast memory-mapped loading
during training (same trick nanoGPT uses: avoid re-tokenizing every epoch).

Usage:
    python -m mini_llm.data.prepare_data --dataset wikitext-2-raw-v1   # quick smoke test
    python -m mini_llm.data.prepare_data --dataset wikitext-103-raw-v1 --vocab-size 8192
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

from mini_llm.tokenizer.bpe import EOT_TOKEN

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"


def load_corpus_text(dataset_config: str) -> tuple[str, str]:
    """Returns (train_text, val_text) joined with the EOT token as a document
    separator, from the HuggingFace `wikitext` dataset."""
    ds = load_dataset("Salesforce/wikitext", dataset_config)
    sep = f" {EOT_TOKEN} "
    train_text = sep.join(t for t in ds["train"]["text"] if t.strip())
    val_text = sep.join(t for t in ds["validation"]["text"] if t.strip())
    return train_text, val_text


def train_bpe_tokenizer(train_text_path: Path, vocab_size: int) -> ByteLevelBPETokenizer:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        [str(train_text_path)],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOT_TOKEN],
    )
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(TOKENIZER_DIR))
    return tokenizer


def encode_to_bin(tokenizer: ByteLevelBPETokenizer, text: str, out_path: Path,
                   chunk_chars: int = 5_000_000) -> int:
    """Encode in chunks to keep peak memory bounded on large corpora, and write
    incrementally to a flat uint16 binary file."""
    ids_chunks = []
    for i in tqdm(range(0, len(text), chunk_chars), desc=f"encoding -> {out_path.name}"):
        chunk = text[i:i + chunk_chars]
        ids_chunks.append(np.array(tokenizer.encode(chunk).ids, dtype=np.uint16))
    ids = np.concatenate(ids_chunks) if ids_chunks else np.array([], dtype=np.uint16)
    ids.tofile(out_path)
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wikitext-103-raw-v1",
                         choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()

    assert args.vocab_size < 65536, "vocab_size must fit in uint16 for the binary format"

    print(f"Loading dataset Salesforce/wikitext/{args.dataset} ...")
    train_text, val_text = load_corpus_text(args.dataset)
    print(f"train chars: {len(train_text):,} | val chars: {len(val_text):,}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    train_txt_path = ARTIFACTS_DIR / "train_raw.txt"
    train_txt_path.write_text(train_text, encoding="utf-8")

    print(f"Training BPE tokenizer (vocab_size={args.vocab_size}) ...")
    tokenizer = train_bpe_tokenizer(train_txt_path, args.vocab_size)
    train_txt_path.unlink()  # only needed transiently for tokenizer.train()

    n_train = encode_to_bin(tokenizer, train_text, ARTIFACTS_DIR / "train.bin")
    n_val = encode_to_bin(tokenizer, val_text, ARTIFACTS_DIR / "val.bin")

    print(f"train.bin: {n_train:,} tokens | val.bin: {n_val:,} tokens")
    print(f"Tokenizer saved to {TOKENIZER_DIR}")


if __name__ == "__main__":
    main()
