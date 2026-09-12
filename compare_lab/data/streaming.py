"""Stream a small, fixed sample of SmolLM-Corpus without downloading the whole
dataset (it is hundreds of GB / ~248B tokens; Fase D needs <1%). We read with
streaming=True, tokenize on the fly, and stop as soon as the token budget is met.

Two subsets, mixed 50/50 by tokens (plan section 1): cosmopedia-v2 (synthetic
textbooks/stories) and fineweb-edu-dedup (filtered educational web). python-edu
is not used.

Everything is driven by a fixed seed so the same sample is reproducible and shared
by all models. The heavy binarization lives in bindata.py; this module only
produces text/token iterators, so it is easy to reason about and the pure logic
(interleaving, budget) is testable with fake streams.
"""

from typing import Iterator

DATASET = "HuggingFaceTB/smollm-corpus"
SUBSETS = ("cosmopedia-v2", "fineweb-edu-dedup")


def stream_subset_texts(subset: str, seed: int = 1337, shuffle_buffer: int = 10_000,
                        split: str = "train") -> Iterator[str]:
    """Yield document texts from one SmolLM-Corpus subset, streaming and lightly
    shuffled with a fixed seed for representativeness."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, subset, split=split, streaming=True)
    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for row in ds:
        text = row.get("text")
        if text and text.strip():
            yield text


def interleave_token_lists(text_streams: list[Iterator[str]], budgets: list[int],
                           encode, produced_out: list[int] | None = None) -> Iterator[list[int]]:
    """Interleave several text streams, tokenizing each document with `encode`,
    so the mix is balanced **by tokens** (the plan's "50/50 en tokens"), not by
    document count.

    A plain round-robin (one document per stream per turn) would NOT give a
    50/50 token mix, because the two SmolLM-Corpus subsets have very different
    mean document lengths (fineweb-edu-dedup documents are markedly longer than
    cosmopedia-v2's) -- alternating 1:1 by document would silently skew the
    token mix. So at each step we pull from whichever stream is furthest behind
    its share, i.e. argmin(produced[i] / budgets[i]). That keeps the mix at the
    target ratio *throughout* the file, not just in aggregate (which matters
    because training reads it as one pass).

    Yields token-id lists (one per document). A document that crosses its
    stream's budget is yielded whole. Deterministic given deterministic inputs.
    If `produced_out` is given, per-stream token totals are written into it so
    callers can record the real split.
    """
    n = len(text_streams)
    produced = produced_out if produced_out is not None else [0] * n
    produced[:] = [0] * n
    done = [False] * n
    while not all(done):
        # pick the stream furthest behind its token share
        candidates = [i for i in range(n) if not done[i]]
        i = min(candidates, key=lambda j: produced[j] / max(1, budgets[j]))
        try:
            text = next(text_streams[i])
        except StopIteration:
            done[i] = True
            continue
        ids = encode(text)
        if not ids:
            continue
        produced[i] += len(ids)
        yield ids
        if produced[i] >= budgets[i]:
            done[i] = True


def mixed_token_lists(encode, half_budget: int, seed: int = 1337,
                      shuffle_buffer: int = 10_000,
                      produced_out: list[int] | None = None) -> Iterator[list[int]]:
    """The full pretraining token stream: cosmopedia + fineweb-edu interleaved
    50/50 **by tokens**, each contributing `half_budget` tokens.

    `half_budget` must be half of the total token target (not the total), or the
    per-stream budgets are never reached and the balancing target is wrong.
    `produced_out` receives the real per-subset token counts (same order as
    SUBSETS) so the caller can record the achieved split."""
    streams = [stream_subset_texts(s, seed=seed, shuffle_buffer=shuffle_buffer) for s in SUBSETS]
    yield from interleave_token_lists(streams, [half_budget, half_budget], encode,
                                      produced_out=produced_out)


def sample_texts_for_tokenizer(max_docs_per_subset: int, seed: int = 1337) -> Iterator[str]:
    """A small text sample (both subsets) to TRAIN the tokenizer, before any
    binaries exist. Mixed with conversations by the caller (plan: ~90/10)."""
    streams = [stream_subset_texts(s, seed=seed) for s in SUBSETS]
    counts = [0, 0]
    done = [False, False]
    while not all(done):
        for i, st in enumerate(streams):
            if done[i]:
                continue
            try:
                text = next(st)
            except StopIteration:
                done[i] = True
                continue
            counts[i] += 1
            yield text
            if counts[i] >= max_docs_per_subset:
                done[i] = True
