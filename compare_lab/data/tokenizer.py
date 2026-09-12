"""Shared BPE tokenizer for Fase D, with *distinct* ids for every special
token -- the single most important fix over Fase A/C, where <bos>, <eos> and
padding all collapsed onto the id of <|endoftext|> and caused the Sliced
masking bug (see AUDITORIA_FASE_C.md, section 3).

Design choices:

- Byte-level BPE (via the `tokenizers` library, same as Fase A -- we don't
  reinvent BPE). vocab_size 8192 fits in uint16 for the binary data format.
- Six reserved special tokens at the front of the vocab, ids 0..5:
      <pad>=0  <bos>=1  <eos>=2  <|system|>=3  <|user|>=4  <|assistant|>=5
  Normal document text never produces these ids (its bytes map to ids >= 6),
  and chat sequences are assembled *by id* (see build_chat), never by encoding
  the literal marker strings -- so raw text can never hijack a control token.
- One canonical chat template, used identically in training, evaluation and the
  demo (Fase C had two templates and a train/inference mismatch):

      <bos> [<|system|> {sys} <eos>] (<|user|> {user} <eos> <|assistant|> {asst} <eos>)+

  Role markers are single atomic ids that fully delimit turns; there are no
  literal newline tokens to keep the framing unambiguous and token-efficient.

- The loss mask returned by build_chat is 1 on assistant content tokens *and*
  the <eos> that closes each assistant turn, 0 everywhere else -- so the model
  learns to answer and to stop, but is never trained on the prompt.

Masks elsewhere in the project are built from lengths / attention_mask, never
by comparing against pad_id (that comparison is exactly what broke Fase C).
"""

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

# Order matters: these land on ids 0..5 in the trained vocab.
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN]

ROLE_TOKEN = {"system": SYSTEM_TOKEN, "user": USER_TOKEN, "assistant": ASSISTANT_TOKEN}

DEFAULT_TOKENIZER_DIR = Path(__file__).resolve().parent / "artifacts" / "tokenizer"


def train_tokenizer(text_iterator, vocab_size: int = 8192, out_dir: Path = DEFAULT_TOKENIZER_DIR,
                    min_frequency: int = 2) -> "Tokenizer":
    """Train a byte-level BPE from an iterator of text strings and save it.

    The iterator is expected to be a small representative sample (the plan
    mixes ~90% pretraining text with ~10% conversations), obtained by streaming
    so we never materialize the full corpus."""
    assert vocab_size < 65536, "vocab_size must fit in uint16 for the binary data format"
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        text_iterator,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bpe.save_model(str(out_dir))
    tok = Tokenizer.from_dir(out_dir)
    tok._assert_special_ids()
    return tok


class Tokenizer:
    """Byte-level BPE with dedicated special-token ids and one chat template."""

    def __init__(self, vocab_file: str, merges_file: str):
        self._tok = ByteLevelBPETokenizer(vocab_file, merges_file)
        # Cache special-token ids resolved from the trained vocab.
        self.pad_id = self._id(PAD_TOKEN)
        self.bos_id = self._id(BOS_TOKEN)
        self.eos_id = self._id(EOS_TOKEN)
        self.system_id = self._id(SYSTEM_TOKEN)
        self.user_id = self._id(USER_TOKEN)
        self.assistant_id = self._id(ASSISTANT_TOKEN)
        self.role_id = {"system": self.system_id, "user": self.user_id, "assistant": self.assistant_id}

    @classmethod
    def from_dir(cls, directory: str | Path) -> "Tokenizer":
        directory = Path(directory)
        return cls(str(directory / "vocab.json"), str(directory / "merges.txt"))

    def _id(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise ValueError(f"special token {token!r} missing from tokenizer vocab")
        return tid

    def _assert_special_ids(self) -> None:
        ids = [self.pad_id, self.bos_id, self.eos_id, self.system_id, self.user_id, self.assistant_id]
        assert len(set(ids)) == len(ids), f"special tokens must have distinct ids, got {ids}"
        assert ids == [0, 1, 2, 3, 4, 5], f"expected special ids 0..5, got {ids}"

    def encode(self, text: str) -> list[int]:
        """Encode raw text. Never emits special-token ids (0..5)."""
        return self._tok.encode(text).ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if skip_special:
            special = {self.pad_id, self.bos_id, self.eos_id, self.system_id, self.user_id, self.assistant_id}
            ids = [i for i in ids if i not in special]
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def build_chat(self, messages: list[dict], add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
        """Render a conversation to (token_ids, loss_mask) with the canonical
        template. `messages` is a list of {"role": "system|user|assistant",
        "content": str}.

        loss_mask[i] == 1 iff token i is an assistant content token or the
        <eos> closing an assistant turn -- these are the only tokens trained on.

        If add_generation_prompt is True, the sequence ends with the
        <|assistant|> marker and no assistant content, ready for the model to
        generate a reply (used at inference; the returned mask is all zeros)."""
        ids: list[int] = [self.bos_id]
        mask: list[int] = [0]
        for msg in messages:
            role = msg["role"]
            if role not in self.role_id:
                raise ValueError(f"unknown role {role!r}")
            content_ids = self.encode(msg["content"])
            is_asst = role == "assistant"
            ids.append(self.role_id[role]);            mask.append(0)
            ids.extend(content_ids);                   mask.extend([1 if is_asst else 0] * len(content_ids))
            ids.append(self.eos_id);                   mask.append(1 if is_asst else 0)
        if add_generation_prompt:
            ids.append(self.assistant_id);             mask.append(0)
        return ids, mask
