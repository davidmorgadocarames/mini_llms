from pathlib import Path

import numpy as np
import torch


class BinaryTokenDataset:
    """Memory-maps a flat uint16 token-id file (produced by prepare_data.py) and
    samples random contiguous windows for next-token-prediction training, the
    same approach nanoGPT uses to avoid holding the whole tokenized corpus in
    RAM."""

    def __init__(self, bin_path: str | Path, block_size: int):
        self.bin_path = Path(bin_path)
        self.block_size = block_size

    def get_batch(self, batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        # re-open the memmap every call: cheap, and avoids a memory leak from
        # keeping the mmap alive across DataLoader worker forks
        data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        max_start = len(data) - self.block_size - 1
        starts = np.random.randint(0, max_start, size=batch_size)

        x = np.stack([data[s:s + self.block_size].astype(np.int64) for s in starts])
        y = np.stack([data[s + 1:s + 1 + self.block_size].astype(np.int64) for s in starts])

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if device != "cpu":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y
