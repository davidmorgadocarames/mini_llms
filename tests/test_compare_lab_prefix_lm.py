"""Prefix-LM: deterministic plan, stable architecture-independent batch hashes,
and Cracked-D vs Sliced-D computing loss over the SAME continuation tokens."""

import numpy as np

from compare_lab.data import prefix_lm as P


def _bin(n_tokens):
    return np.arange(n_tokens, dtype=np.int64) % 8000


def test_plan_is_deterministic_and_cuts_in_range():
    n, block = 32 * 50, 32
    p1 = P.build_plan(n, block, 8, seed=7)
    p2 = P.build_plan(n, block, 8, seed=7)
    p3 = P.build_plan(n, block, 8, seed=8)
    assert np.array_equal(p1.cuts, p2.cuts) and np.array_equal(p1.order, p2.order)
    assert not np.array_equal(p1.cuts, p3.cuts)  # different seed -> different cuts
    assert p1.cuts.min() >= 1 and p1.cuts.max() <= block - 1


def test_batch_hash_is_stable_and_order_dependent():
    p = P.build_plan(32 * 50, 32, 8, seed=1)
    _, frag, cuts = next(P.iter_batch_ids(p))
    assert P.batch_hash(frag, cuts) == P.batch_hash(frag, cuts)
    assert P.batch_hash(frag, cuts) != P.batch_hash(frag[::-1], cuts)


def test_cracked_and_sliced_loss_over_identical_tokens():
    block, bs = 16, 4
    n = block * 20
    bin_data = _bin(n)
    plan = P.build_plan(n, block, bs, seed=123)
    _, frag, cuts = next(P.iter_batch_ids(plan))

    x, y, lm_c = P.collate_cracked(bin_data, frag, cuts, block)
    src, src_pad, tin, tout, lm_s = P.collate_sliced(bin_data, frag, cuts, block, bos_id=1, pad_id=0)

    for i in range(len(frag)):
        c = int(cuts[i])
        frag_tokens = bin_data[int(frag[i]) * block:(int(frag[i]) + 1) * block]
        continuation = list(frag_tokens[c:])
        cracked_loss_tokens = y[i][lm_c[i] > 0].tolist()
        sliced_loss_tokens = tout[i][lm_s[i] > 0].tolist()
        assert cracked_loss_tokens == continuation
        assert sliced_loss_tokens == continuation
        # loss token counts match exactly
        assert int((lm_c[i] > 0).sum()) == int((lm_s[i] > 0).sum()) == len(continuation)


def test_cracked_full_puts_loss_on_all_positions():
    block, bs, n = 16, 4, 16 * 20
    bin_data = _bin(n)
    plan = P.build_plan(n, block, bs, seed=1)
    _, frag, cuts = next(P.iter_batch_ids(plan))
    _, _, lm_full = P.collate_cracked(bin_data, frag, cuts, block, full_loss=True)
    assert (lm_full == 1).all()


def test_sliced_prefix_not_padding_and_bos_kept():
    block, bs, n = 16, 4, 16 * 20
    bin_data = _bin(n)
    plan = P.build_plan(n, block, bs, seed=2)
    _, frag, cuts = next(P.iter_batch_ids(plan))
    src, src_pad, tin, tout, lm_s = P.collate_sliced(bin_data, frag, cuts, block, bos_id=1, pad_id=0)
    for i in range(len(frag)):
        c = int(cuts[i])
        assert not src_pad[i, :c].any()      # prefix region is never masked as padding
        assert tin[i, 0].item() == 1         # decoder starts from a real <bos>
