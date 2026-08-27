"""Turns Fase B JSONL examples into padded token tensors for next-token
prediction, framed as "<expr> => <value>" (see depth_lab/models/baseline.py
docstring for why this framing). Shared by every architecture in Fase B, so
the task framing stays identical across the baseline/encoder-decoder/LLR
comparison.
"""

import torch
from torch.utils.data import Dataset

from depth_lab.data.reduce import reduction_trace
from depth_lab.tokenizer import CharTokenizer


def render_value(value: bool | int) -> str:
    return str(value)


def format_target_text(expr: str, value: bool | int) -> str:
    return f"{expr} => {render_value(value)}"


class ExprDataset(Dataset):
    """Each item is a fixed-length (block_size) pair (x, y) for next-token
    prediction over "<bos><expr> => <value><eos><pad>...". Padding beyond the
    content is filled with pad_id in both x and y; callers should pass
    ignore_index=tokenizer.pad_id to the loss so padding never contributes."""

    def __init__(self, examples: list[dict], tokenizer: CharTokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        text = format_target_text(ex["expr"], ex["value"])
        ids = [self.tokenizer.bos_id, *self.tokenizer.encode(text), self.tokenizer.eos_id]

        if len(ids) > self.block_size + 1:
            raise ValueError(
                f"example needs {len(ids)} tokens, exceeds block_size+1={self.block_size + 1}: {text!r}"
            )
        pad_id = self.tokenizer.pad_id
        ids = ids + [pad_id] * (self.block_size + 1 - len(ids))

        seq = torch.tensor(ids, dtype=torch.long)
        return seq[:-1], seq[1:]

    def prompt_ids(self, idx: int) -> list[int]:
        """The "<bos><expr> => " prefix alone, for autoregressive generation
        at eval time (exact-match accuracy)."""
        ex = self.examples[idx]
        prefix = f"{ex['expr']} => "
        return [self.tokenizer.bos_id, *self.tokenizer.encode(prefix)]


class Seq2SeqDataset(Dataset):
    """Encoder/decoder counterpart of ExprDataset: the encoder side gets the
    raw expression (no bos/eos needed -- it isn't autoregressive), and the
    decoder side gets "<bos><value><eos>" so training can teach next-token
    prediction of just the value, cross-attending to the encoded expression.
    Both sides are padded to a fixed length so batches stack into plain
    tensors; pad_id is excluded from the loss via ignore_index, same as
    ExprDataset."""

    def __init__(self, examples: list[dict], tokenizer: CharTokenizer, src_block_size: int, tgt_block_size: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.src_block_size = src_block_size
        self.tgt_block_size = tgt_block_size

    def __len__(self) -> int:
        return len(self.examples)

    def _pad(self, ids: list[int], length: int, what: str) -> list[int]:
        if len(ids) > length:
            raise ValueError(f"{what} needs {len(ids)} tokens, exceeds block size {length}")
        return ids + [self.tokenizer.pad_id] * (length - len(ids))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        src_ids = self._pad(self.tokenizer.encode(ex["expr"]), self.src_block_size, "src")
        tgt_ids = self._pad(
            [self.tokenizer.bos_id, *self.tokenizer.encode(render_value(ex["value"])), self.tokenizer.eos_id],
            self.tgt_block_size + 1,
            "tgt",
        )
        src = torch.tensor(src_ids, dtype=torch.long)
        tgt = torch.tensor(tgt_ids, dtype=torch.long)
        return src, tgt[:-1], tgt[1:]


def build_locator_examples(examples: list[dict]) -> list[dict]:
    """Every intermediate state of every expression's reduction trace becomes
    one locator training instance: {"expr": <state>, "span": (start, end)} --
    the locator's job is to point at the next innermost sub-expression given
    *any* partially-reduced state, not just the original expression."""
    out = []
    for ex in examples:
        for step in reduction_trace(ex["expr"]):
            out.append({"expr": step.expr, "span": step.span})
    return out


def build_replacer_examples(examples: list[dict]) -> list[dict]:
    """Every reduction step's isolated span becomes one replacer training
    instance: {"expr": reversed(span_text), "value": step.value}. The span
    text is reversed before framing as "<expr> => <value>" -- the paper's
    trick for helping a no-positional-encoding model localize the operator;
    see depth_lab/models/replacer.py's docstring for the fuller rationale."""
    out = []
    for ex in examples:
        for step in reduction_trace(ex["expr"]):
            out.append({"expr": step.span_text[::-1], "value": step.value})
    return out


class LocatorDataset(Dataset):
    """Each item is a fixed-length (block_size) pair (ids, labels, pad_mask)
    for per-character binary classification: labels[i] = 1.0 if character i
    is part of the target span, else 0.0. pad_mask[i] = True at padding
    positions, so callers can exclude them from the loss. Character-level
    tokenization means token index == character index directly, so span
    boundaries from depth_lab.data.reduce need no translation."""

    def __init__(self, examples: list[dict], tokenizer: CharTokenizer, block_size: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        expr, (start, end) = ex["expr"], ex["span"]
        if len(expr) > self.block_size:
            raise ValueError(f"expr needs {len(expr)} tokens, exceeds block_size={self.block_size}: {expr!r}")

        ids = self.tokenizer.encode(expr)
        pad_len = self.block_size - len(ids)
        ids = ids + [self.tokenizer.pad_id] * pad_len

        labels = [1.0 if start <= i < end else 0.0 for i in range(len(expr))] + [0.0] * pad_len
        pad_mask = [False] * len(expr) + [True] * pad_len

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(pad_mask, dtype=torch.bool),
        )
