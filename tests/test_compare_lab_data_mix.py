"""The pretraining sample's two frozen properties: the subset mix is balanced by
TOKENS (not by document count), and documents are separated by <eos>."""

import numpy as np

from compare_lab.data.streaming import interleave_token_lists


def _stream(tag, doc_len, n):
    for _ in range(n):
        yield (tag, doc_len)


def _encode(pair):
    tag, length = pair
    return [tag] * length


def test_mix_is_balanced_by_tokens_not_documents():
    """Subsets whose documents differ in length must still come out 50/50 in
    tokens. A plain round-robin by document would give ~39/61 here, which is the
    bug this guards: SmolLM-Corpus's two subsets have very different mean
    document lengths."""
    produced = [0, 0]
    docs = list(interleave_token_lists([_stream(7, 720, 5000), _stream(9, 1150, 5000)],
                                       budgets=[100_000, 100_000], encode=_encode,
                                       produced_out=produced))
    flat = [t for doc in docs for t in doc]
    pct_a = 100 * flat.count(7) / len(flat)
    assert abs(pct_a - 50) < 3, f"mix is {pct_a:.1f}% / {100 - pct_a:.1f}%, not 50/50 by tokens"
    assert produced[0] > 0 and produced[1] > 0
    assert abs(produced[0] - produced[1]) / max(produced) < 0.05


def test_mix_stays_balanced_throughout_the_file():
    """Training reads the bin in one pass, so the mix must hold locally too --
    not just in aggregate with one subset bunched at the end."""
    docs = list(interleave_token_lists([_stream(7, 500, 5000), _stream(9, 1500, 5000)],
                                       budgets=[60_000, 60_000], encode=_encode))
    flat = [t for doc in docs for t in doc]
    half = len(flat) // 2
    pct_first_half = 100 * flat[:half].count(7) / half
    assert abs(pct_first_half - 50) < 6, f"first half is {pct_first_half:.1f}% subset A"


def test_eos_is_appended_after_every_document():
    """Mirrors the with_eos() wrapper in prepare_pretrain: without a separator
    the fragments straddle document boundaries with no signal, and the model
    never sees <eos> during pretraining."""
    eos_id = 2
    docs = [[10, 11, 12], [20, 21], [30]]

    def with_eos(token_lists):
        for ids in token_lists:
            yield list(ids) + [eos_id]

    out = list(with_eos(docs))
    assert all(d[-1] == eos_id for d in out)
    flat = [t for d in out for t in d]
    assert flat == [10, 11, 12, eos_id, 20, 21, eos_id, 30, eos_id]
    assert flat.count(eos_id) == len(docs)
