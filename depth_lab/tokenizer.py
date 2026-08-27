"""Character-level tokenizer for Fase B. The vocabulary is small and fixed
enough (~30 symbols, covering both the boolean and arithmetic domains) that
BPE buys nothing here -- this matches the paper's own choice of
character-level tokenization "to ensure exact positional matching for
pattern localization in our locate-and-replace pipeline"."""

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

SPECIAL_TOKENS = [PAD, BOS, EOS]

# Covers: True/False, and/or/not, parens, digits (arithmetic domain), the
# arithmetic operators, and "=>" (the input/output separator used to frame
# "evaluate this expression" as next-token prediction in Fase B.3+).
_CHARS = sorted(set("TrueFalse andornot()0123456789+-*=>"))


class CharTokenizer:
    def __init__(self):
        vocab = SPECIAL_TOKENS + _CHARS
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"character {exc} not in vocabulary: {text!r}") from exc

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        chars = []
        for i in ids:
            token = self.itos[i]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            chars.append(token)
        return "".join(chars)
