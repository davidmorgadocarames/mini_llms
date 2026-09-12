"""Resume: an interrupted-and-resumed run receives exactly the same batches, and
produces the same weights, as an uninterrupted run."""

import json
import time
import types

import numpy as np
import torch

from compare_lab.data import prefix_lm as P
import compare_lab.train.pretrain as PT
import compare_lab.train.tasks as tasks
from mini_llm.model.config import GPTConfig
from mini_llm.model.transformer import GPT


def test_resume_sees_same_batch_stream():
    """iter_batch_ids from a resume point equals the tail of a full run."""
    plan = P.build_plan(32 * 80, 32, 8, seed=99)
    full = [(b, tuple(f), tuple(c)) for b, f, c in P.iter_batch_ids(plan)]
    resumed = [(b, tuple(f), tuple(c)) for b, f, c in P.iter_batch_ids(plan, start_batch=5)]
    assert resumed == full[5:]


def _tiny_build(arch, vocab_size, block_size, device):
    return GPT(GPTConfig(vocab_size=vocab_size, block_size=block_size,
                         n_layer=2, n_embd=64, n_head=4, n_kv_head=2)).to(device)


def _args(tokenizer_dir, pretrain_dir, steps):
    return types.SimpleNamespace(
        arch="cracked", grad_accum=1, max_optimizer_steps=steps, lr=1e-3, min_lr=1e-4,
        warmup_steps=1, weight_decay=0.1, dtype="float32", eval_interval=0, log_interval=1000,
        ckpt_interval_min=999, seed=1337, no_resume=False, device="cpu",
        tokenizer_dir=str(tokenizer_dir), pretrain_dir=str(pretrain_dir))


def _flat_weights(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    return torch.cat([v.flatten() for v in ckpt["model"].values()])


def test_interrupted_resume_matches_uninterrupted(tiny_tokenizer_dir, tiny_bins, tmp_path, monkeypatch):
    monkeypatch.setattr(PT, "build_model", _tiny_build)
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")

    # Run A: uninterrupted 6 steps
    monkeypatch.setattr(PT, "CHECKPOINT_DIR", tmp_path / "ckpt_A")
    PT.train(_args(tiny_tokenizer_dir, tiny_bins["dir"], steps=6))
    weights_A = _flat_weights(tmp_path / "ckpt_A" / "cracked" / "pretrain_final.pt")

    # Run B: SAME command (steps=6) but interrupted after 3 steps, then resumed.
    # Interrupting (not lowering the step target) keeps the LR schedule identical.
    monkeypatch.setattr(PT, "CHECKPOINT_DIR", tmp_path / "ckpt_B")
    calls = {"n": 0}

    def stop_after_3(task_name):
        calls["n"] += 1
        return calls["n"] >= 3
    monkeypatch.setattr(PT, "stop_requested", stop_after_3)
    PT.train(_args(tiny_tokenizer_dir, tiny_bins["dir"], steps=6))  # pauses at step 3

    monkeypatch.setattr(PT, "stop_requested", lambda task_name: False)
    PT.train(_args(tiny_tokenizer_dir, tiny_bins["dir"], steps=6))  # resumes to 6
    weights_B = _flat_weights(tmp_path / "ckpt_B" / "cracked" / "pretrain_final.pt")

    # deterministic CPU float32 -> should match closely (kernels aren't bit-exact)
    assert torch.allclose(weights_A, weights_B, atol=1e-5, rtol=1e-4)


def test_loss_curve_continues_across_a_pause(tiny_tokenizer_dir, tiny_bins, tmp_path, monkeypatch):
    """The curve is a deliverable: a resume must EXTEND it, not restart it and
    overwrite the earlier segment on the next save."""
    monkeypatch.setattr(PT, "build_model", _tiny_build)
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(PT, "CHECKPOINT_DIR", tmp_path / "ckpt")
    monkeypatch.setattr(PT, "RESULTS_DIR", tmp_path / "results")

    args = _args(tiny_tokenizer_dir, tiny_bins["dir"], steps=8)
    args.eval_interval = 2
    args.log_interval = 2

    # segment 1: pause after a few steps
    calls = {"n": 0}

    def stop_after_3(task_name):
        calls["n"] += 1
        return calls["n"] >= 3
    monkeypatch.setattr(PT, "stop_requested", stop_after_3)
    first = PT.train(args)
    assert first["state"] == "paused"
    steps_1 = [h["step"] for h in first["history"]]
    assert steps_1, "expected some curve points before the pause"

    # segment 2: resume to the end
    monkeypatch.setattr(PT, "stop_requested", lambda task_name: False)
    second = PT.train(_args_with(args, eval_interval=2, log_interval=2))
    steps_2 = [h["step"] for h in second["history"]]

    # the resumed history CONTAINS the pre-pause points and extends beyond them
    assert steps_2[:len(steps_1)] == steps_1, "earlier curve points were lost on resume"
    assert steps_2[-1] > steps_1[-1], "curve did not extend past the pause"

    saved = json.loads((tmp_path / "results" / "pretrain_loss_cracked.json").read_text())
    assert [h["step"] for h in saved] == steps_2


def _args_with(base, **overrides):
    new = types.SimpleNamespace(**vars(base))
    for k, v in overrides.items():
        setattr(new, k, v)
    return new


def test_active_training_seconds_excludes_pause_time(tiny_tokenizer_dir, tiny_bins, tmp_path, monkeypatch):
    """A real wall-clock pause between two segments must not inflate the
    reported work time -- active_training_seconds only ticks while a training
    process is actually running."""
    monkeypatch.setattr(PT, "build_model", _tiny_build)
    monkeypatch.setattr(tasks, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(PT, "CHECKPOINT_DIR", tmp_path / "ckpt")

    # Same step target in both segments (a different one would -- correctly --
    # trip the hyperparameter-drift guard); the first is cut short by a STOP.
    args = _args(tiny_tokenizer_dir, tiny_bins["dir"], steps=6)
    calls = {"n": 0}

    def stop_after_3(task_name):
        calls["n"] += 1
        return calls["n"] >= 3
    monkeypatch.setattr(PT, "stop_requested", stop_after_3)
    PT.train(args)
    ckpt_1 = torch.load(tmp_path / "ckpt" / "cracked" / "pretrain.pt", weights_only=False)
    active_after_segment_1 = ckpt_1["active_seconds"]
    assert active_after_segment_1 > 0

    time.sleep(2.0)  # simulate the user pausing to use the PC for something else

    monkeypatch.setattr(PT, "stop_requested", lambda task_name: False)
    PT.train(args)
    ckpt_2 = torch.load(tmp_path / "ckpt" / "cracked" / "pretrain_final.pt", weights_only=False)
    active_after_segment_2 = ckpt_2["active_seconds"]

    # cumulative active time grew by roughly segment 2's own duration, NOT by
    # the ~2s pause -- so it must stay well under active_after_segment_1 + 2s.
    assert active_after_segment_2 > active_after_segment_1
    assert active_after_segment_2 < active_after_segment_1 + 1.5

    status = json.loads((tmp_path / "runs" / "pretrain_cracked" / "status.json").read_text())
    assert status["active_training_seconds"] == round(active_after_segment_2, 1)
