"""Write and read the flat uint16 token binaries (same format as Fase A's
train.bin/val.bin). The writer takes an *iterator of token-id lists* and a token
budget, so the HuggingFace streaming path and the tests share the exact same
binarization code -- tests inject a synthetic iterator and never touch the network.
"""

from pathlib import Path

import numpy as np


def write_bin_from_token_iter(token_lists, out_path: Path, budget_tokens: int,
                              dtype=np.uint16, flush_every: int = 4_000_000,
                              on_progress=None) -> int:
    """Consume `token_lists` (iterable of lists/arrays of token ids), appending to
    a flat binary at out_path until `budget_tokens` is reached (the document that
    crosses the budget is written whole, so the final count may slightly exceed
    it). Returns the number of tokens written.

    on_progress(written_so_far), if given, is called each time the buffer is
    flushed -- lets long-running callers (prepare_pretrain) update a status.json
    without this function knowing anything about the task-tracking system."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    buf: list[np.ndarray] = []
    buf_len = 0
    with open(out_path, "wb") as f:
        for ids in token_lists:
            arr = np.asarray(ids, dtype=dtype)
            buf.append(arr)
            buf_len += arr.size
            written += arr.size
            if buf_len >= flush_every:
                np.concatenate(buf).tofile(f)
                buf, buf_len = [], 0
                if on_progress is not None:
                    on_progress(written)
            if written >= budget_tokens:
                break
        if buf:
            np.concatenate(buf).tofile(f)
    if on_progress is not None:
        on_progress(written)
    return written


def load_bin(path: Path, dtype=np.uint16) -> np.ndarray:
    """Memory-mapped read-only view of the token binary."""
    return np.memmap(Path(path), dtype=dtype, mode="r")
