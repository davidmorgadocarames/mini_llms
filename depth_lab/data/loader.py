"""Turns Fase B JSONL examples into padded token tensors for next-token
prediction, framed as "<expr> => <value>" (see depth_lab/models/baseline.py
docstring for why this framing). Shared by every architecture in Fase B, so
the task framing stays identical across the baseline/encoder-decoder/LLR
comparison.
"""

import torch
from torch.utils.data import Dataset

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
