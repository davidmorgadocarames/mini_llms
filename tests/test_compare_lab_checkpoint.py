"""Checkpoint save/restore, including the GPU path.

The RNG-restore test below exists because a CPU-only test suite CANNOT catch the
failure it guards: load_checkpoint passes map_location=device, so on a GPU run
every tensor in the checkpoint -- including the saved RNG states -- comes back as
a CUDA tensor, and torch.set_rng_state() rejects anything that is not a CPU
ByteTensor. On CPU the same code path is perfectly happy. It surfaced only when
a real GPU run was paused and resumed.
"""

import random

import numpy as np
import pytest
import torch

from compare_lab.train.checkpoint import save_checkpoint, load_checkpoint, _restore_rng, _rng_state


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


def _roundtrip(tmp_path, device):
    model = _Tiny().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # take a step so the optimizer has real state to restore
    model(torch.randn(2, 4, device=device)).sum().backward()
    opt.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, step=5, data_pos=80, config={"x": 1})

    model2 = _Tiny().to(device)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    return load_checkpoint(path, model2, opt2, device=device), opt, opt2


def test_roundtrip_restores_step_data_pos_and_optimizer_cpu(tmp_path):
    info, opt, opt2 = _roundtrip(tmp_path, "cpu")
    assert info["step"] == 5 and info["data_pos"] == 80
    # Adam's moment estimates must come back, not be silently reset
    s1 = list(opt.state.values())[0]
    s2 = list(opt2.state.values())[0]
    assert torch.allclose(s1["exp_avg"], s2["exp_avg"])
    assert torch.allclose(s1["exp_avg_sq"], s2["exp_avg_sq"])
    assert int(s2["step"]) == int(s1["step"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_roundtrip_on_gpu_restores_rng(tmp_path):
    """The exact production path: save on GPU, load with map_location='cuda'."""
    info, opt, opt2 = _roundtrip(tmp_path, "cuda")
    assert info["step"] == 5 and info["data_pos"] == 80
    s1 = list(opt.state.values())[0]
    s2 = list(opt2.state.values())[0]
    assert torch.allclose(s1["exp_avg"], s2["exp_avg"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_restore_rng_accepts_cuda_resident_states(tmp_path):
    """Directly reproduces the crash: RNG states that came back on the GPU."""
    state = _rng_state()
    on_gpu = dict(state)
    on_gpu["torch"] = state["torch"].cuda()
    if "cuda" in state:
        on_gpu["cuda"] = [s.cuda() for s in state["cuda"]]
    _restore_rng(on_gpu)  # must not raise "RNG state must be a torch.ByteTensor"


def test_rng_restore_makes_subsequent_draws_reproducible(tmp_path):
    """Restoring RNG must actually reproduce the same random stream -- otherwise
    a resumed run would take a different data/dropout path than an uninterrupted
    one, which is the whole point of saving it."""
    model, opt = _Tiny(), None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "c.pt"

    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    save_checkpoint(path, model, opt, step=1, data_pos=0, config={})
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    # burn the generators, then restore
    random.random(); np.random.rand(); torch.rand(5)
    load_checkpoint(path, _Tiny(), torch.optim.AdamW(_Tiny().parameters(), lr=1e-3), device="cpu")
    got = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert got == pytest.approx(expected)


def test_extra_fields_survive_the_roundtrip(tmp_path):
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, opt, step=2, data_pos=3, config={},
                    extra={"active_seconds": 12.5, "history": [{"step": 2, "train_loss": 1.0}]})
    info = load_checkpoint(path, _Tiny(), device="cpu")
    assert info["active_seconds"] == 12.5
    assert info["history"] == [{"step": 2, "train_loss": 1.0}]


# ------------------------------------------------------------- durability ------
# The project's only stop mechanism is a hard failure, so the one invariant that
# matters is: if a checkpoint file exists, it is complete and loadable.

def test_save_fsyncs_before_promoting(tmp_path, monkeypatch):
    """Without an fsync, torch.save only guarantees the bytes reached the OS page
    cache; a power cut before the flush leaves a file that exists but is
    truncated, and os.replace would promote it anyway (atomic != durable)."""
    import compare_lab.train.checkpoint as C

    synced = []
    real_fsync = C.os.fsync
    monkeypatch.setattr(C.os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])

    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert C.save_checkpoint(tmp_path / "c.pt", model, opt, step=1, data_pos=2, config={})
    assert synced, "the checkpoint was promoted without being fsynced"


def test_a_torn_write_never_replaces_a_good_checkpoint(tmp_path, monkeypatch):
    """Simulates a truncated write (what a power cut mid-save produces). The
    verification step must catch it and leave the previous checkpoint alone."""
    import compare_lab.train.checkpoint as C

    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dst = tmp_path / "c.pt"
    assert C.save_checkpoint(dst, model, opt, step=1, data_pos=2, config={})
    good = dst.read_bytes()

    # next save produces a truncated temp file
    def torn(ckpt, tmp_path_):
        tmp_path_.write_bytes(b"\x80\x02}q\x00truncated")

    monkeypatch.setattr(C, "_write_and_fsync", torn)
    assert C.save_checkpoint(dst, model, opt, step=99, data_pos=99, config={}) is False
    assert dst.read_bytes() == good, "a torn write replaced a good checkpoint"


def test_load_falls_back_to_the_previous_generation(tmp_path):
    """A power cut can land between 'keep the old one as .prev' and 'promote the
    new one', leaving the main file missing. The resume must still find work."""
    import compare_lab.train.checkpoint as C

    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dst = tmp_path / "c.pt"
    C.save_checkpoint(dst, model, opt, step=5, data_pos=80, config={})
    C.save_checkpoint(dst, model, opt, step=10, data_pos=160, config={})
    assert C.prev_path(dst).exists(), "the previous generation was not kept"

    dst.unlink()  # simulate the crash window
    info = C.load_checkpoint(dst, _Tiny(), device="cpu")
    assert info["step"] == 5, "did not fall back to the previous generation"


def test_load_falls_back_when_the_main_file_is_corrupt(tmp_path):
    import compare_lab.train.checkpoint as C

    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dst = tmp_path / "c.pt"
    C.save_checkpoint(dst, model, opt, step=5, data_pos=80, config={})
    C.save_checkpoint(dst, model, opt, step=10, data_pos=160, config={})

    dst.write_bytes(b"garbage")
    info = C.load_checkpoint(dst, _Tiny(), device="cpu")
    assert info["step"] == 5


def test_load_raises_a_clear_error_when_nothing_is_usable(tmp_path):
    import compare_lab.train.checkpoint as C

    with pytest.raises(FileNotFoundError, match="no usable checkpoint"):
        C.load_checkpoint(tmp_path / "missing.pt", _Tiny(), device="cpu")


def test_verify_rejects_a_checkpoint_missing_required_keys(tmp_path):
    import compare_lab.train.checkpoint as C

    bad = tmp_path / "bad.pt"
    torch.save({"model": {}, "step": 1}, bad)  # no optimizer, no data_pos
    assert C.verify_checkpoint(bad) is False
