"""Shared fixtures for the compare_lab (Fase D) tests. Other phases' tests don't
reference these, so this conftest is inert for them."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory):
    """Train a small real BPE tokenizer with the Fase D special tokens once and
    return its directory (so tests that need the on-disk tokenizer can point to it)."""
    from compare_lab.data.tokenizer import train_tokenizer

    corpus = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "Machine learning models predict the next token in a sequence. " * 20,
        "Small language models are trained on filtered educational web text. " * 20,
        "Hola, esto es una conversacion de prueba para el tokenizador. " * 20,
    ] * 40
    out = tmp_path_factory.mktemp("tok")
    train_tokenizer(iter(corpus), vocab_size=1024, out_dir=out)
    return out


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_tokenizer_dir):
    from compare_lab.data.tokenizer import Tokenizer

    return Tokenizer.from_dir(tiny_tokenizer_dir)


@pytest.fixture(scope="session")
def tiny_bins(tmp_path_factory):
    """A synthetic pretraining bin + plan at a small block size, plus a val bin."""
    from compare_lab.data.bindata import write_bin_from_token_iter
    from compare_lab.data.prefix_lm import build_plan, save_plan

    d = tmp_path_factory.mktemp("pretrain")
    block = 32
    rng = np.random.default_rng(0)
    # ids kept < 256 so any real byte-level BPE vocab (>= 262) covers them
    train_ids = rng.integers(6, 256, size=block * 200, dtype=np.uint16)
    val_ids = rng.integers(6, 256, size=block * 20, dtype=np.uint16)
    write_bin_from_token_iter([train_ids], d / "train.bin", budget_tokens=len(train_ids))
    write_bin_from_token_iter([val_ids], d / "val.bin", budget_tokens=len(val_ids))
    plan = build_plan(len(train_ids), block_size=block, batch_size=8, seed=1337)
    save_plan(plan, d / "plan")
    return {"dir": d, "block": block, "n_train": len(train_ids)}


@pytest.fixture(autouse=True)
def _results_dir_is_never_the_real_one(tmp_path, monkeypatch):
    """Keep every test's curve writes out of compare_lab/eval/results/.

    That directory holds committed deliverables (the loss curves of the real
    training runs). Two resume tests called pretrain.train() without redirecting
    RESULTS_DIR, so each `pytest` run overwrote the real pretrain curve .json --
    silently, because the write succeeded and nothing asserts on it. Redirecting
    here instead of per-test means a future test cannot reintroduce the leak.
    Tests that assert on the file still patch RESULTS_DIR themselves; monkeypatch
    applies their value after this one, so they win.
    """
    import sys

    # Only touch modules the suite already imported -- no cost for other phases.
    for name in ("compare_lab.train.pretrain", "compare_lab.train.finetune"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "RESULTS_DIR"):
            monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results")
