from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

EOT_TOKEN = "<|endoftext|>"


class BPETokenizer:
    """Thin wrapper around a HuggingFace byte-level BPE tokenizer. We train the
    BPE merges ourselves (see mini_llm/data/prepare_data.py) rather than
    reimplementing the BPE algorithm — that part is a solved problem; the
    architecture is where we want to spend our own engineering effort."""

    def __init__(self, vocab_file: str, merges_file: str):
        self._tok = ByteLevelBPETokenizer(vocab_file, merges_file)

    @classmethod
    def from_dir(cls, directory: str | Path) -> "BPETokenizer":
        directory = Path(directory)
        return cls(str(directory / "vocab.json"), str(directory / "merges.txt"))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()
