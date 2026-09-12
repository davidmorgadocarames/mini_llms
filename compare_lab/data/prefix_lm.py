"""Prefix-LM data plan shared by all three models.

The pretraining objective (plan section 4): each fixed-size fragment of the token
stream is cut at a random point into a prefix and a continuation; every model
must predict the continuation. To make the comparison fair the three models must
see *exactly the same* (prefix, continuation) pairs in the *same order*. We
guarantee that by making the batch identity architecture-independent:

- Fragments are the consecutive, non-overlapping block_size-token slices of the
  bin (deterministic, no randomness).
- Each fragment gets one cut point drawn once from a seeded RNG and saved.
- Fragments are visited in one seeded permutation, saved.
- Batches are consecutive groups of batch_size fragments in that order.

So a "batch" is identified by (fragment_indices, cut_points) -- identical for
Cracked-D, Cracked-D-full and Sliced-D. Cracked-D writes a hash of that identity
per step; the Etapa 2 models recompute and verify it. The per-architecture
collate functions turn the same identity into that architecture's tensors:

- Cracked-D: [prefix||continuation] as one causal sequence, loss only on the
  continuation positions (Cracked-D-full: loss on all positions).
- Sliced-D: encoder reads the prefix, decoder predicts the continuation.

Both put loss on the SAME continuation tokens (verified by a test). Masks are
built from lengths, never by comparing to pad_id.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class PrefixLMPlan:
    block_size: int
    batch_size: int
    seed: int
    n_fragments: int
    cuts: np.ndarray      # (n_fragments,) int32, cut point per fragment
    order: np.ndarray     # (n_fragments,) int32, permutation of fragment indices

    @property
    def n_batches(self) -> int:
        return self.n_fragments // self.batch_size  # drop_last


def build_plan(n_tokens: int, block_size: int, batch_size: int, seed: int) -> PrefixLMPlan:
    """Deterministically build cut points and visit order for a bin of n_tokens.
    Cut c is in [1, block_size-1] so both prefix and continuation are non-empty."""
    n_fragments = n_tokens // block_size
    assert n_fragments >= batch_size, f"need >= batch_size fragments, got {n_fragments}"
    rng = np.random.default_rng(seed)
    cuts = rng.integers(1, block_size, size=n_fragments, dtype=np.int64).astype(np.int32)
    order = rng.permutation(n_fragments).astype(np.int32)
    return PrefixLMPlan(block_size, batch_size, seed, n_fragments, cuts, order)


def save_plan(plan: PrefixLMPlan, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cuts.npy", plan.cuts)
    np.save(out_dir / "order.npy", plan.order)
    meta = {"block_size": plan.block_size, "batch_size": plan.batch_size,
            "seed": plan.seed, "n_fragments": plan.n_fragments}
    (out_dir / "plan_meta.json").write_text(json.dumps(meta, indent=2))


def load_plan(out_dir: Path) -> PrefixLMPlan:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "plan_meta.json").read_text())
    cuts = np.load(out_dir / "cuts.npy")
    order = np.load(out_dir / "order.npy")
    return PrefixLMPlan(meta["block_size"], meta["batch_size"], meta["seed"],
                        meta["n_fragments"], cuts, order)


def iter_batch_ids(plan: PrefixLMPlan, start_batch: int = 0):
    """Yield (batch_index, fragment_indices, cut_points) for each batch, in the
    frozen order. start_batch supports resuming from an exact data position."""
    for b in range(start_batch, plan.n_batches):
        sl = slice(b * plan.batch_size, (b + 1) * plan.batch_size)
        frag_idx = plan.order[sl]
        cuts = plan.cuts[frag_idx]
        yield b, frag_idx, cuts


def batch_hash(frag_idx: np.ndarray, cuts: np.ndarray) -> str:
    """Architecture-independent hash of a batch's identity."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(frag_idx, dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(cuts, dtype=np.int32).tobytes())
    return h.hexdigest()


def _fragments(bin_data: np.ndarray, frag_idx: np.ndarray, block_size: int) -> np.ndarray:
    """Gather the block_size-token slices for the given fragment indices -> (B, block_size)."""
    starts = frag_idx.astype(np.int64) * block_size
    rows = np.stack([bin_data[s:s + block_size] for s in starts]).astype(np.int64)
    return rows


def collate_cracked(bin_data, frag_idx, cuts, block_size, full_loss: bool = False):
    """Decoder-only batch. Returns (input_ids, target_ids, loss_mask), all
    (B, block_size-1). Loss on continuation positions only, unless full_loss
    (Cracked-D-full), which puts loss on every position."""
    rows = _fragments(bin_data, frag_idx, block_size)          # (B, block_size)
    x = torch.from_numpy(rows[:, :-1]).long()
    y = torch.from_numpy(rows[:, 1:]).long()
    B, T = x.shape                                             # T = block_size-1
    # position j predicts token j+1; it's a continuation prediction iff j+1 >= cut
    pos_plus_1 = torch.arange(1, T + 1).unsqueeze(0).expand(B, T)
    cut_t = torch.from_numpy(np.asarray(cuts)).long().unsqueeze(1)
    loss_mask = torch.ones(B, T, dtype=torch.float32) if full_loss \
        else (pos_plus_1 >= cut_t).float()
    return x, y, loss_mask


def collate_sliced(bin_data, frag_idx, cuts, block_size, bos_id: int, pad_id: int):
    """Encoder-decoder batch. Encoder reads the prefix, decoder predicts the
    continuation. Returns (src, src_pad_mask, tgt_in, tgt_out, loss_mask).
    Padding masks are built from lengths, never from pad_id comparison."""
    rows = _fragments(bin_data, frag_idx, block_size)          # (B, block_size)
    cuts = np.asarray(cuts).astype(np.int64)
    B = rows.shape[0]
    prefix_lens = cuts                                         # length of each prefix
    cont_lens = block_size - cuts                              # length of each continuation
    max_src = int(prefix_lens.max())
    max_cont = int(cont_lens.max())

    src = np.full((B, max_src), pad_id, dtype=np.int64)
    tgt_in = np.full((B, max_cont), pad_id, dtype=np.int64)
    tgt_out = np.full((B, max_cont), pad_id, dtype=np.int64)
    src_pad = np.ones((B, max_src), dtype=bool)
    loss_mask = np.zeros((B, max_cont), dtype=np.float32)

    for i in range(B):
        c = int(cuts[i])
        prefix = rows[i, :c]
        cont = rows[i, c:]
        src[i, :c] = prefix
        src_pad[i, :c] = False
        # decoder input: bos + continuation[:-1]; target: continuation
        tgt_in[i, 0] = bos_id
        tgt_in[i, 1:len(cont)] = cont[:-1]
        tgt_out[i, :len(cont)] = cont
        loss_mask[i, :len(cont)] = 1.0

    return (torch.from_numpy(src), torch.from_numpy(src_pad),
            torch.from_numpy(tgt_in), torch.from_numpy(tgt_out),
            torch.from_numpy(loss_mask))
