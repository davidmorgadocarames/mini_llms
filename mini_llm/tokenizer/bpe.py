import re
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

EOT_TOKEN = "<|endoftext|>"

# EOT_TOKEN was inserted between concatenated WikiText articles as a document
# separator during data prep (see mini_llm/data/prepare_data.py). The model
# can emit it mid-generation when it "jumps" to a new, unrelated article --
# decode() faithfully returns the literal token text, which reads as noise,
# so UIs should render it as a paragraph break instead of the raw string.
_EOT_DISPLAY_PATTERN = re.compile(r"\s*" + re.escape(EOT_TOKEN) + r"\s*")


def clean_for_display(text: str) -> str:
    return _EOT_DISPLAY_PATTERN.sub("\n\n", text)


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
