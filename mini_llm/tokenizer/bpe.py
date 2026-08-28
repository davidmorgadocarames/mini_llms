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

# WikiText marks section headers as e.g. "= = = Siege of Cape Town = = ="
# (the tokenized source has "=" as its own space-separated symbol, repeated
# once per heading level). Render it as an actual heading -- its own line,
# uppercased -- instead of leaving it stuck mid-paragraph.
_HEADER_PATTERN = re.compile(r"(?:=\s)*=\s+([^=\n]+?)\s+=(?:\s=)*")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
_SPACE_AROUND_BREAK = re.compile(r"[ \t]*\n\n[ \t]*")


def clean_for_display(text: str) -> str:
    text = _EOT_DISPLAY_PATTERN.sub("\n\n", text)
    text = _HEADER_PATTERN.sub(lambda m: f"\n\n{m.group(1).strip().upper()}\n\n", text)
    text = _SPACE_AROUND_BREAK.sub("\n\n", text)
    return _EXTRA_BLANK_LINES.sub("\n\n", text)


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

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Like encode(), but also returns each token's (start, end) character
        offset into `text` -- needed to convert a character-level span into a
        token-level span (BPE merges multiple characters per token, so token
        index != character index, unlike depth_lab's char-level tokenizer)."""
        enc = self._tok.encode(text)
        return enc.ids, enc.offsets

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()
