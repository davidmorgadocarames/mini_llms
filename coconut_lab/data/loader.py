"""Turns Fase C.1 (prompt, response) JSONL examples into padded token
tensors for instruction fine-tuning, with the loss masked to the response
tokens only -- unlike Fase B's baseline (which trained on the whole
"<expr> => <value>" sequence deliberately, since predicting the expression
itself was harmless there), here the prompt is a fixed template plus
arbitrary instruction text: training the model to "predict" it back would
waste capacity on a task that isn't the actual objective. Standard SFT
practice.

No dedicated pad token exists in Fase A's BPE vocab (only the EOT_TOKEN
document separator was trained in) -- reusing EOT's id as padding is safe
here specifically because padding positions are excluded from the loss via
loss_mask, so the model never has to distinguish "real EOT" from "padding"
by value alone.
"""

import torch
from torch.utils.data import Dataset

from mini_llm.tokenizer import BPETokenizer
from mini_llm.tokenizer.bpe import EOT_TOKEN


class InstructionDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer: BPETokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = tokenizer.encode(EOT_TOKEN)[0]

        self.examples: list[tuple[list[int], list[int]]] = []
        for ex in examples:
            prompt_ids = tokenizer.encode(ex["prompt"])
            if len(prompt_ids) >= block_size:
                continue  # no room left for any response -- drop, don't truncate the instruction
            response_ids = tokenizer.encode(ex["response"]) + [self.pad_id]  # EOT terminates the response
            max_response_len = block_size - len(prompt_ids)
            response_ids = response_ids[:max_response_len]
            self.examples.append((prompt_ids, response_ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids, response_ids = self.examples[idx]
        ids = prompt_ids + response_ids
        pad_len = self.block_size + 1 - len(ids)
        ids = ids + [self.pad_id] * pad_len

        # is_response[j] = 1 iff ids[j] is a response token (to predict), not
        # prompt or padding
        is_response = [0] * len(prompt_ids) + [1] * len(response_ids) + [0] * pad_len

        seq = torch.tensor(ids, dtype=torch.long)
        mask = torch.tensor(is_response, dtype=torch.float32)
        x, y = seq[:-1], seq[1:]
        y_mask = mask[1:]  # predicting y[i] = ids[i+1]; include it iff ids[i+1] is a response token
        return x, y, y_mask


class ConversationDataset(Dataset):
    """Fase C.1b counterpart of InstructionDataset: each example is a
    multi-turn oasst1 conversation (see prepare_conversations.py) instead of
    a single (prompt, response) pair. The loss is masked to *every*
    assistant turn in the conversation, not just the last one -- training on
    the whole path teaches the model to keep responding across turns, not
    just to answer a single trailing question. If a conversation doesn't fit
    block_size, it's truncated from the *start* (oldest turns dropped first),
    matching how a chat UI would truncate context -- the most recent turns
    matter most."""

    def __init__(self, examples: list[dict], tokenizer: BPETokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = tokenizer.encode(EOT_TOKEN)[0]

        self.examples: list[list[int]] = []
        self.masks: list[list[int]] = []
        for ex in examples:
            ids, is_response = self._encode_turns(ex["turns"])
            if not ids:
                continue
            # keep the most recent turns if the conversation overflows block_size
            ids = ids[-block_size:]
            is_response = is_response[-block_size:]
            if sum(is_response) == 0:
                continue  # nothing left to train on after truncation
            self.examples.append(ids)
            self.masks.append(is_response)

    def _encode_turns(self, turns: list[dict]) -> tuple[list[int], list[int]]:
        marker = {"user": "<|user|>\n", "assistant": "<|assistant|>\n"}
        ids: list[int] = []
        is_response: list[int] = []
        for turn in turns:
            turn_ids = self.tokenizer.encode(f"{marker[turn['role']]}{turn['text']}\n")
            if turn["role"] == "assistant":
                turn_ids = turn_ids + [self.pad_id]  # EOT marks the end of this assistant turn
            ids.extend(turn_ids)
            is_response.extend([1 if turn["role"] == "assistant" else 0] * len(turn_ids))
        return ids, is_response

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = self.examples[idx]
        is_response = self.masks[idx]
        pad_len = self.block_size + 1 - len(ids)
        ids = ids + [self.pad_id] * pad_len
        is_response = is_response + [0] * pad_len

        seq = torch.tensor(ids, dtype=torch.long)
        mask = torch.tensor(is_response, dtype=torch.float32)
        x, y = seq[:-1], seq[1:]
        y_mask = mask[1:]
        return x, y, y_mask
