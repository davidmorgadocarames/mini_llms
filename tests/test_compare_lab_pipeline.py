"""The pipeline's stage bookkeeping: a paused stage is never mistaken for a
finished one, and fine-tuning refuses to start from random weights.

This guards the worst failure mode found in review: pausing the pretrain used to
fall through to fine-tuning, which -- finding no pretrained checkpoint -- trained
from random weights and wrote finetune_final.pt, so the relaunch then SKIPPED
fine-tuning and the final deliverable was a model fine-tuned from noise, with
nothing in the logs to show it.
"""

import types

import pytest

import compare_lab.train.run_pipeline as RP
import compare_lab.train.tasks as tasks


def _patch_runs(monkeypatch, tmp_path, tokenizer_dir=None):
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(RP, "CHECKPOINT_DIR", tmp_path / "ckpt")
    if tokenizer_dir is not None:
        monkeypatch.setattr(RP, "DEFAULT_TOKENIZER_DIR", tokenizer_dir)


def test_paused_stage_stops_the_pipeline(monkeypatch, tmp_path, tiny_tokenizer_dir):
    _patch_runs(monkeypatch, tmp_path, tiny_tokenizer_dir)
    calls = []

    monkeypatch.setattr(RP.prepare_pretrain, "build", lambda **kw: calls.append("data_pre") or {})
    monkeypatch.setattr(RP.prepare_smoltalk, "prepare", lambda *a, **kw: calls.append("data_ft") or {})
    # pretrain reports it was paused
    monkeypatch.setattr(RP.pretrain_mod, "train",
                        lambda args: calls.append("pretrain") or {"state": "paused", "step": 7})
    monkeypatch.setattr(RP.finetune_mod, "train",
                        lambda args: calls.append("finetune") or {"state": "finished"})

    state = RP.run("cracked", device="cpu")

    assert state == "paused"
    assert "finetune" not in calls, "fine-tuning must NOT start after a paused pretrain"
    stages = RP.read_stages("cracked")
    assert stages["pretrain"] == "paused"
    assert stages["finetune"] == "pending", "finetune must still be recorded as pending"


def test_finished_stages_are_recorded_done(monkeypatch, tmp_path, tiny_tokenizer_dir):
    _patch_runs(monkeypatch, tmp_path, tiny_tokenizer_dir)
    monkeypatch.setattr(RP.prepare_pretrain, "build", lambda **kw: {})
    monkeypatch.setattr(RP.prepare_smoltalk, "prepare", lambda *a, **kw: {})

    def fake_pretrain(args):
        # a real successful pretrain leaves pretrain_final.pt behind
        final = RP.CHECKPOINT_DIR / "cracked" / "pretrain_final.pt"
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"stub")
        return {"state": "finished", "step": 10}

    monkeypatch.setattr(RP.pretrain_mod, "train", fake_pretrain)
    monkeypatch.setattr(RP.finetune_mod, "train", lambda args: {"state": "finished", "step": 10})

    state = RP.run("cracked", device="cpu")

    assert state == "finished"
    stages = RP.read_stages("cracked")
    assert all(stages[s] == "done" for s in RP.STAGES)


def test_finetune_refuses_to_start_without_pretrained_checkpoint(monkeypatch, tmp_path, tiny_tokenizer_dir):
    """Even if the pretrain stage somehow reports success without leaving a
    checkpoint, fine-tuning must fail loudly rather than train from noise."""
    _patch_runs(monkeypatch, tmp_path, tiny_tokenizer_dir)
    monkeypatch.setattr(RP.prepare_pretrain, "build", lambda **kw: {})
    monkeypatch.setattr(RP.prepare_smoltalk, "prepare", lambda *a, **kw: {})
    monkeypatch.setattr(RP.pretrain_mod, "train", lambda args: {"state": "finished"})
    monkeypatch.setattr(RP.finetune_mod, "train",
                        lambda args: pytest.fail("finetune must not be reached"))

    with pytest.raises(FileNotFoundError, match="refusing to fine-tune"):
        RP.run("cracked", device="cpu")


def test_stop_created_during_the_run_prevents_the_next_stage(monkeypatch, tmp_path, tiny_tokenizer_dir):
    """A STOP created WHILE the pipeline runs stops it at the next stage boundary."""
    _patch_runs(monkeypatch, tmp_path, tiny_tokenizer_dir)
    started = []

    def build_then_stop(**kw):
        started.append("data_pre")
        (tasks.task_dir("pipeline_cracked") / "STOP").touch()
        return {}

    monkeypatch.setattr(RP.prepare_pretrain, "build", build_then_stop)
    monkeypatch.setattr(RP.prepare_smoltalk, "prepare",
                        lambda *a, **kw: pytest.fail("stage 2 must not start after a STOP"))

    assert RP.run("cracked", device="cpu") == "paused"
    assert started == ["data_pre"]


def test_stale_stop_does_not_block_a_relaunch(monkeypatch, tmp_path, tiny_tokenizer_dir):
    """A STOP left over from a previous pause must NOT wedge the pipeline forever.

    The pipeline STOP used to be sticky (never cleared), so creating it -- the
    natural gesture, since it names the process you launched -- made every later
    relaunch exit instantly as "paused". Training STOPs self-clear; this one has
    to behave the same way or the two gestures mean different things."""
    _patch_runs(monkeypatch, tmp_path, tiny_tokenizer_dir)
    (tasks.task_dir("pipeline_cracked") / "STOP").touch()

    monkeypatch.setattr(RP.prepare_pretrain, "build", lambda **kw: {})
    monkeypatch.setattr(RP.prepare_smoltalk, "prepare", lambda *a, **kw: {})

    def fake_pretrain(args):
        final = RP.CHECKPOINT_DIR / "cracked" / "pretrain_final.pt"
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"stub")
        return {"state": "finished"}

    monkeypatch.setattr(RP.pretrain_mod, "train", fake_pretrain)
    monkeypatch.setattr(RP.finetune_mod, "train", lambda args: {"state": "finished"})

    assert RP.run("cracked", device="cpu") == "finished"
    assert not (tasks.task_dir("pipeline_cracked") / "STOP").exists()


def test_crashed_stage_is_recorded_not_left_running(monkeypatch, tmp_path, tiny_tokenizer_dir):
    """A failure used to leave stages.json and the pipeline status reading
    "running" forever, so from the phone a crashed pipeline looked like a working
    one."""
    _patch_runs(monkeypatch, tmp_path, tiny_tokenizer_dir)

    def boom(**kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(RP.prepare_pretrain, "build", boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        RP.run("cracked", device="cpu")

    assert RP.read_stages("cracked")["data_prep_pretrain"] == "error"
    import json
    status = json.loads((tasks.task_dir("pipeline_cracked") / "status.json").read_text())
    assert status["state"] == "error"
    assert "disk on fire" in status["error"]


def test_interrupted_stage_is_recorded_pending(monkeypatch, tmp_path, tiny_tokenizer_dir):
    """Data prep raises KeyboardInterrupt on STOP (it cannot resume mid-way, so it
    restarts next time). The stage must go back to 'pending', not stay 'running'."""
    _patch_runs(monkeypatch, tmp_path, tiny_tokenizer_dir)

    def interrupted(**kw):
        raise KeyboardInterrupt("STOP during data prep")

    monkeypatch.setattr(RP.prepare_pretrain, "build", interrupted)
    with pytest.raises(KeyboardInterrupt):
        RP.run("cracked", device="cpu")

    assert RP.read_stages("cracked")["data_prep_pretrain"] == "pending"
    import json
    status = json.loads((tasks.task_dir("pipeline_cracked") / "status.json").read_text())
    assert status["state"] == "paused"
