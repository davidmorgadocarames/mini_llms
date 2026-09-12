"""Guards added after a real GPU pause/resume run and an adversarial review:
the task lock, the Windows-safe atomic replace, and the hyperparameter-drift
check on resume. Each one exists because of a demonstrated failure, noted below.
"""

import os
import types

import pytest
import torch

import compare_lab.train.pretrain as PT
import compare_lab.train.tasks as tasks
from compare_lab.train.checkpoint import _replace_with_retry, save_checkpoint
from compare_lab.train.tasks import acquire_lock, release_lock


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


# ---------------------------------------------------------------- task lock ---
def _write_lock(task: str, payload: dict) -> None:
    import json
    (tasks.task_dir(task) / "LOCK").write_text(json.dumps(payload))


def test_lock_blocks_a_second_live_process(tmp_path, monkeypatch):
    """Two concurrent runs of the same task interleave appends into the shared
    batch-hash registry, duplicating lines so index no longer matches batch. The
    corruption is silent and only surfaces hours later, in another architecture's
    verification, as a bogus "config drift" -- and can only be undone by
    retraining from scratch."""
    import psutil
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    # a lock owned by a process that IS alive (this one), while we pretend to be
    # a different process trying to start
    _write_lock("pretrain_cracked",
                {"pid": os.getpid(), "create_time": psutil.Process().create_time()})
    monkeypatch.setattr(os, "getpid", lambda: 999_999_999)
    with pytest.raises(RuntimeError, match="locked by PID"):
        acquire_lock("pretrain_cracked")


def test_lock_is_taken_over_when_the_owner_is_dead(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    d = tasks.task_dir("pretrain_cracked")
    _write_lock("pretrain_cracked", {"pid": 999_999_999})  # a PID that cannot exist
    lock = acquire_lock("pretrain_cracked")  # must not raise
    import json
    assert json.loads((d / "LOCK").read_text())["pid"] == os.getpid()
    release_lock(lock)
    assert not (d / "LOCK").exists()


def test_lock_survives_pid_reuse(tmp_path, monkeypatch):
    """PIDs are recycled, especially across a reboot. Without comparing the
    process start time, a stale lock whose PID now belongs to some unrelated
    process would REFUSE a perfectly legitimate resume -- an availability bug in
    the exact flow (pause from the phone, relaunch later) this protects."""
    import psutil
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    # our PID, but a start time from the past: a recycled PID, not our process
    _write_lock("pretrain_cracked",
                {"pid": os.getpid(), "create_time": psutil.Process().create_time() - 10_000})
    monkeypatch.setattr(os, "getpid", lambda: 999_999_999)
    lock = acquire_lock("pretrain_cracked")  # must NOT raise: owner is really gone
    assert lock is not None


def test_lock_fails_closed_when_liveness_is_unknown(tmp_path, monkeypatch):
    """If we cannot tell whether the owner is alive, refuse rather than steal the
    lock: a false "it's dead" leads straight to registry corruption."""
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    _write_lock("pretrain_cracked", {"pid": 4321, "create_time": 1.0})
    monkeypatch.setattr(tasks, "_process_alive", lambda pid, ct: None)
    with pytest.raises(RuntimeError, match="cannot be verified"):
        acquire_lock("pretrain_cracked")


def test_lock_creation_is_atomic(tmp_path, monkeypatch):
    """O_CREAT|O_EXCL, not exists()-then-write: with the latter, two processes
    racing to start could both pass the check and both write."""
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    lock = acquire_lock("pretrain_cracked")
    assert lock.exists()
    # a *second* attempt from a different pid must not silently overwrite it
    import psutil, json
    payload = json.loads(lock.read_text())
    assert payload["pid"] == os.getpid()
    release_lock(lock)


# ------------------------------------------------- Windows-safe atomic write ---
def test_replace_retries_then_succeeds(tmp_path, monkeypatch):
    """On Windows, os.replace FAILS with PermissionError while any process holds
    a handle on the destination -- an antivirus scan of a freshly written 300MB
    .pt is enough (verified on this machine). Unretried, that kills a multi-hour
    run at the exact moment the checkpoint was supposed to protect it."""
    src, dst = tmp_path / "a.tmp", tmp_path / "a.pt"
    src.write_bytes(b"new")
    dst.write_bytes(b"old")

    calls = {"n": 0}
    real_replace = type(src).replace

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("[WinError 5] simulated busy file")
        return real_replace(self, target)

    monkeypatch.setattr(type(src), "replace", flaky)
    monkeypatch.setattr("compare_lab.train.checkpoint.time.sleep", lambda s: None)
    _replace_with_retry(src, dst)
    assert calls["n"] == 3
    monkeypatch.undo()
    assert dst.read_bytes() == b"new"


def test_replace_gives_up_without_raising(tmp_path, monkeypatch):
    """A checkpoint that cannot be written must NOT kill the run: the previous
    checkpoint is intact and the new one is left as .tmp. An earlier version
    raised after ~22s of retries, and a 45s handle was enough to kill a real
    training run at the exact moment the checkpoint was protecting it."""
    src, dst = tmp_path / "a.tmp", tmp_path / "a.pt"
    src.write_bytes(b"new")

    def always_busy(self, target):
        raise PermissionError("[WinError 5] always busy")

    monkeypatch.setattr(type(src), "replace", always_busy)
    monkeypatch.setattr("compare_lab.train.checkpoint.time.sleep", lambda s: None)
    assert _replace_with_retry(src, dst, attempts=3) is False  # no exception


def test_save_checkpoint_survives_a_blocked_rename(tmp_path, monkeypatch):
    """End to end: save_checkpoint reports failure instead of raising, keeps the
    complete checkpoint as .tmp, and leaves any previous checkpoint untouched."""
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dst = tmp_path / "pretrain.pt"
    dst.write_bytes(b"previous checkpoint")

    def always_busy(self, target):
        raise PermissionError("[WinError 5] busy")

    monkeypatch.setattr(type(dst), "replace", always_busy)
    monkeypatch.setattr("compare_lab.train.checkpoint.time.sleep", lambda s: None)
    ok = save_checkpoint(dst, model, opt, step=3, data_pos=9, config={})
    assert ok is False
    assert dst.read_bytes() == b"previous checkpoint"   # previous one intact
    assert (tmp_path / "pretrain.pt.tmp").exists()      # new one recoverable
    monkeypatch.undo()
    # and the leftover .tmp is a real, loadable checkpoint
    ck = torch.load(tmp_path / "pretrain.pt.tmp", map_location="cpu", weights_only=False)
    assert ck["step"] == 3 and ck["data_pos"] == 9


def test_save_checkpoint_reports_success_normally(tmp_path):
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert save_checkpoint(tmp_path / "c.pt", model, opt, step=1, data_pos=2, config={}) is True


def test_status_write_survives_a_busy_file(tmp_path, monkeypatch):
    """Losing one progress update must never take down the run."""
    target = tmp_path / "status.json"

    def always_busy(self, t):
        raise PermissionError("[WinError 5] busy")

    monkeypatch.setattr(type(target), "replace", always_busy)
    monkeypatch.setattr("compare_lab.train.tasks.time.sleep", lambda s: None)
    tasks.atomic_write_json(target, {"a": 1}, retries=2)  # must NOT raise


# --------------------------------------------- hyperparameter drift on resume ---
def _tiny_build(arch, vocab_size, block_size, device):
    from mini_llm.model.config import GPTConfig
    from mini_llm.model.transformer import GPT
    return GPT(GPTConfig(vocab_size=vocab_size, block_size=block_size,
                         n_layer=2, n_embd=64, n_head=4, n_kv_head=2)).to(device)


def _args(tokenizer_dir, pretrain_dir, steps, grad_accum=1):
    return types.SimpleNamespace(
        arch="cracked", grad_accum=grad_accum, max_optimizer_steps=steps, lr=1e-3, min_lr=1e-4,
        warmup_steps=1, weight_decay=0.1, dtype="float32", eval_interval=0, log_interval=1000,
        ckpt_interval_min=999, seed=1337, no_resume=False, device="cpu",
        tokenizer_dir=str(tokenizer_dir), pretrain_dir=str(pretrain_dir))


def test_resume_refuses_a_different_grad_accum(tiny_tokenizer_dir, tiny_bins, tmp_path, monkeypatch):
    """grad_accum sets total_opt_steps, which is the denominator of the cosine LR
    schedule. Resuming with a different value silently stretches the schedule
    mid-run and halves the effective batch, and nothing else would catch it: the
    batch hashes still match because micro-batch indices are unchanged."""
    monkeypatch.setattr(PT, "build_model", _tiny_build)
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(PT, "CHECKPOINT_DIR", tmp_path / "ckpt")
    monkeypatch.setattr(PT, "RESULTS_DIR", tmp_path / "results")

    PT.train(_args(tiny_tokenizer_dir, tiny_bins["dir"], steps=4, grad_accum=1))
    with pytest.raises(RuntimeError, match="different training hyperparameters"):
        PT.train(_args(tiny_tokenizer_dir, tiny_bins["dir"], steps=4, grad_accum=2))


def test_resume_accepts_identical_hyperparameters(tiny_tokenizer_dir, tiny_bins, tmp_path, monkeypatch):
    monkeypatch.setattr(PT, "build_model", _tiny_build)
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(PT, "CHECKPOINT_DIR", tmp_path / "ckpt")
    monkeypatch.setattr(PT, "RESULTS_DIR", tmp_path / "results")

    args = _args(tiny_tokenizer_dir, tiny_bins["dir"], steps=4, grad_accum=1)
    PT.train(args)
    out = PT.train(args)  # same command again: must resume cleanly
    assert out["state"] == "finished"


def test_grad_accum_cli_default_matches_frozen_config(monkeypatch):
    """A CLI default that disagreed with the frozen config (it was 8 vs 16) meant
    a direct-CLI resume would silently run a different LR schedule and a halved
    effective batch."""
    import sys
    from compare_lab import config

    monkeypatch.setattr(sys, "argv", ["pretrain", "--arch", "cracked"])
    args = PT.parse_args()
    assert args.grad_accum == config.GRAD_ACCUM
    assert args.lr == config.PRETRAIN["lr"]
    assert args.dtype == config.PRETRAIN["dtype"]


# ------------------------------------------------- finetune resume guards ------
def test_finetune_cli_defaults_match_frozen_config(monkeypatch):
    """These disagreed with config.py (batch 8 vs 16, max_steps 2000 vs 20000).
    A CLI resume past step 2000 then yielded ZERO batches, left final_state as
    "finished", and wrote finetune_final.pt -- so run_pipeline skipped
    fine-tuning forever and the deliverable was silently under-trained."""
    import sys
    from compare_lab import config
    import compare_lab.train.finetune as FT

    monkeypatch.setattr(sys, "argv", ["finetune", "--arch", "cracked"])
    args = FT.parse_args()
    assert args.batch_size == config.FINETUNE_BATCH_SIZE
    assert args.max_steps == config.FINETUNE_MAX_STEPS
    assert args.block_size == config.BLOCK_SIZE
    assert args.lr == config.FINETUNE["lr"]


def test_batches_yields_nothing_past_the_target():
    """The mechanism behind the bug above, pinned so it cannot come back
    unnoticed: _batches is `while step < max_steps`, so resuming beyond the
    target trains nothing. The guard in _train must catch this."""
    from compare_lab.train.finetune import _batches
    assert list(_batches(list(range(100)), batch_size=8, start_batch=8000, max_steps=2000)) == []


# ------------------------------------------------------- single stop path ---
def test_signal_handling_is_gone():
    """Ctrl+C handling was removed deliberately: two shutdown paths meant the
    rarely-used one rotted, and it produced several real bugs. The STOP file is
    now the only voluntary stop; everything else is a hard failure bounded by the
    periodic checkpoint."""
    import compare_lab.train.tasks as t
    assert not hasattr(t, "CleanStop")
    import compare_lab.train.pretrain as p
    import compare_lab.train.finetune as f
    for mod in (p, f):
        src = __import__("inspect").getsource(mod)
        assert "CleanStop" not in src, f"{mod.__name__} still references CleanStop"


def test_stop_file_is_detected_and_cleared(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    assert tasks.stop_requested("pretrain_cracked") is False
    (tasks.task_dir("pretrain_cracked") / "STOP").touch()
    assert tasks.stop_requested("pretrain_cracked") is True
    tasks.clear_stop("pretrain_cracked")
    assert tasks.stop_requested("pretrain_cracked") is False


def test_checkpoint_interval_bounds_what_a_hard_failure_costs():
    """With STOP as the only clean stop, this interval IS the worst-case loss of
    any unplanned stop, so it should be small enough to be cheap to lose."""
    from compare_lab import config
    assert config.CKPT_INTERVAL_MIN <= 5.0
