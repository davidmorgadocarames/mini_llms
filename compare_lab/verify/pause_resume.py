"""FULL-FIDELITY pause/resume verification on REAL data before committing ~9h of GPU.

Shape (as requested): ~5 minutes of real training, paused halfway for 1 minute,
then resumed for the rest. Everything that production uses is used here --
real SmolLM-Corpus streaming, the real tokenizer, production grad_accum/micro_batch/
block_size/bf16, the real pretrain loop, and the process launched as an
INDEPENDENT OS process with the pause triggered by a STOP file from outside.

What it proves (each assertion prints PASS/FAIL):
  1. The pause is clean and fast, with checkpoint + curve + status + log written.
  2. NOTHING from before the pause is lost when merged with the new work:
     - every pre-pause curve point survives, in order, with identical values
     - weights CONTINUE (they differ from the pre-pause snapshot, i.e. training
       carried on, and they are not a reinitialization)
     - the optimizer/scaler/RNG/data position are restored, so the resumed run
       picks up at exactly the next batch with no repeats and no gaps
  3. The batch-hash registry stays consistent (no duplicates, no gaps) across
     the interruption -- the property the Etapa 2 models will verify against.
  4. active_training_seconds excludes the 1-minute pause (work time, not wall time).
  5. The final artifacts exist and are loadable: weights + curve .json/.png.
  6. The run SURVIVES an outside process holding a handle on the checkpoint file
     while it is being rewritten (on Windows that makes os.replace fail, which
     unretried would kill a multi-hour run at a checkpoint).

Everything it produces is collected under _run/report/ for inspection: the loss
curve as .json and .png, a second plot with the pause marked so continuity is
visible at a glance, the full training log, the status.json at pause and at the
end, and a human-readable REPORT.md.

Usage:
    python -m compare_lab.verify.pause_resume
    python -m compare_lab.verify.pause_resume --reuse-data   # skip the data prep
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT = str(PROJECT_ROOT)
LAUNCHER = Path(__file__).resolve().parent / "_launcher.py"
# The rehearsal's own data/checkpoints/runs live here, never in the real
# artifacts directories, so they can't be confused with the frozen run.
ROOT = PROJECT_ROOT / "compare_lab" / "verify" / "_run"

results = []


def check(name: str, ok, detail: str = ""):
    # coerce to a plain bool: numpy bools (e.g. from np.isfinite) are not JSON
    # serializable, and writing checks.json is the last thing this script does --
    # a TypeError there would throw away the whole report after a 10-minute run
    ok = bool(ok)
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def weights_fingerprint(ckpt_path: Path) -> tuple[str, float]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    flat = torch.cat([v.flatten().float() for v in ck["model"].values()])
    h = hashlib.sha1(flat.numpy().tobytes()).hexdigest()[:16]
    return h, float(flat.norm())


def flat_weights(ckpt_path: Path) -> torch.Tensor:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return torch.cat([v.flatten().float() for v in ck["model"].values()])


def adam_step_counts(ckpt_path: Path) -> list[int]:
    """Adam's own step counter per parameter tensor. If the optimizer state were
    silently reset on resume, these would restart from the resumed segment's
    length instead of matching the global step -- and no weights/curve check
    would notice."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["optimizer"]["state"]
    return [int(s["step"]) for s in st.values() if "step" in s]


def adam_moments_nonzero(ckpt_path: Path) -> bool:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["optimizer"]["state"]
    return all(float(s["exp_avg_sq"].abs().sum()) > 0 for s in st.values() if "exp_avg_sq" in s)


def hold_file_handle(path: Path, start_after: float, hold_for: float, flag: dict) -> threading.Thread:
    """Open a read handle on the checkpoint mid-run, exactly like a real-time
    antivirus scan of a freshly written .pt (or the user copying it). On Windows
    that makes os.replace fail with PermissionError; unretried it would kill a
    multi-hour run at a checkpoint. With ckpt_interval_min small, holding the
    handle across an interval guarantees the retry path is exercised."""
    def worker():
        time.sleep(start_after)
        if not path.exists():
            flag["held"] = False
            return
        try:
            with open(path, "rb") as fh:
                fh.read(1024)
                flag["held"] = True
                time.sleep(hold_for)
        except OSError as e:
            flag["held"] = False
            flag["error"] = repr(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=70, help="total optimizer steps (~5 min on a 4060)")
    ap.add_argument("--pause-at", type=float, default=150.0, help="seconds of training before STOP")
    ap.add_argument("--pause-for", type=float, default=60.0, help="seconds to stay paused")
    ap.add_argument("--tokens", type=int, default=20_000_000)
    ap.add_argument("--tokenizer-docs", type=int, default=400)
    ap.add_argument("--reuse-data", action="store_true",
                    help="skip data prep if the bins are already there (faster iteration)")
    args = ap.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT
    env["HF_HUB_DISABLE_XET"] = "1"

    have_data = (ROOT / "pre" / "stats.json").exists() and (ROOT / "tok" / "vocab.json").exists()
    if args.reuse_data and have_data:
        print(f"reusing existing data in {ROOT / 'pre'}")
        for sub in ("ckpt", "runs", "results", "report"):
            shutil.rmtree(ROOT / sub, ignore_errors=True)
    else:
        shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True, exist_ok=True)

    # ---------------- stage A: real data prep (small slice of the real corpus) ----
    print("\n=== A. Real data prep (streaming SmolLM-Corpus) ===")
    import compare_lab.train.tasks as tasks
    # keep the test hermetic: data prep runs IN THIS PROCESS and would otherwise
    # write its status.json into the real repo's runs/, so `compare_lab.status`
    # would show test noise as if it were the production pipeline
    tasks.RUNS_DIR = ROOT / "runs"

    from compare_lab.data import prepare_pretrain
    from compare_lab import config

    t0 = time.time()
    stats = prepare_pretrain.build(
        tokens=args.tokens, val_tokens=500_000,
        block_size=config.BLOCK_SIZE, batch_size=config.MICRO_BATCH, seed=config.SEED,
        tokenizer_dir=ROOT / "tok", out_dir=ROOT / "pre",
        retrain_tokenizer=not (args.reuse_data and have_data),
        tokenizer_docs=args.tokenizer_docs,
        vocab_size=config.VOCAB_SIZE, skip_if_done=args.reuse_data)
    print(f"  data prep took {time.time() - t0:.0f}s")
    print(f"  train tokens: {stats['n_train_tokens']:,}  val: {stats['n_val_tokens']:,}")
    print(f"  subset mix: {stats['pct_per_subset']} % over {stats['subsets']}")
    check("mix is ~50/50 by tokens on REAL data",
          all(abs(p - 50) < 6 for p in stats["pct_per_subset"]),
          f"{stats['pct_per_subset']}")
    check("eos separator flag recorded", stats.get("eos_between_documents") is True)
    needed = args.steps * config.MICRO_BATCH * config.GRAD_ACCUM * config.BLOCK_SIZE
    check("bin large enough for the planned steps", stats["n_train_tokens"] >= needed,
          f"{stats['n_train_tokens']:,} >= {needed:,}")

    ckpt = ROOT / "ckpt" / "cracked" / "pretrain.pt"
    curve_json = ROOT / "results" / "pretrain_loss_cracked.json"
    status_json = ROOT / "runs" / "pretrain_cracked" / "status.json"
    log_file = ROOT / "runs" / "pretrain_cracked" / "pretrain_cracked.log"
    registry = ROOT / "pre" / "plan" / "batch_hashes.txt"

    # ---------------- stage B: train, then pause from outside --------------------
    print(f"\n=== B. Training as an independent process, STOP after {args.pause_at:.0f}s ===")
    proc = subprocess.Popen([sys.executable, str(LAUNCHER), str(ROOT), str(args.steps)],
                            cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    time.sleep(args.pause_at)
    if proc.poll() is not None:
        out, _ = proc.communicate()
        print(out[-3000:])
        raise SystemExit("process finished before the pause -- lower --steps or raise --pause-at")

    stop_file = ROOT / "runs" / "pretrain_cracked" / "STOP"
    t_stop = time.time()
    stop_file.touch()
    print(f"  STOP created at t={args.pause_at:.0f}s")
    try:
        out1, _ = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise SystemExit("process did not exit within 120s of STOP")
    pause_latency = time.time() - t_stop
    print(f"  process exited {pause_latency:.2f}s after STOP")
    check("pause is clean and prompt (<20s)", pause_latency < 20, f"{pause_latency:.2f}s")
    check("exit code 0 on pause", proc.returncode == 0, f"rc={proc.returncode}")

    # --- snapshot EVERYTHING before the pause ---
    check("checkpoint written on pause", ckpt.exists())
    check("curve json written on pause", curve_json.exists())
    check("status.json written on pause", status_json.exists())
    check("log written on pause", log_file.exists())
    if not (ckpt.exists() and curve_json.exists() and status_json.exists()):
        print(out1[-3000:])
        raise SystemExit("missing artifacts after pause")

    pre_weights = flat_weights(ckpt)
    pre_ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    pre_step = pre_ck["step"]
    pre_data_pos = pre_ck["data_pos"]
    pre_active = pre_ck["active_seconds"]
    pre_hash, pre_norm = weights_fingerprint(ckpt)
    pre_curve = json.loads(curve_json.read_text())
    pre_status = json.loads(status_json.read_text())
    pre_registry_lines = registry.read_text().split() if registry.exists() else []
    pre_log_lines = log_file.read_text(encoding="utf-8").splitlines()

    print(f"  paused at step={pre_step} data_pos={pre_data_pos} active={pre_active:.1f}s "
          f"curve_points={len(pre_curve)} weights={pre_hash} |w|={pre_norm:.2f}")
    check("status.json says paused", pre_status["state"] == "paused", pre_status["state"])
    check("status step matches checkpoint step", pre_status["step"] == pre_step)
    # Adam state: not just "the key exists" (which is unconditionally true), but
    # that its own step counter matches the training step and its moments are real
    pre_adam_steps = adam_step_counts(ckpt)
    check("Adam step counter matches the training step at pause",
          bool(pre_adam_steps) and all(s == pre_step for s in pre_adam_steps),
          f"adam steps {set(pre_adam_steps)} vs step {pre_step}")
    check("Adam second-moment estimates are populated", adam_moments_nonzero(ckpt))
    check("checkpoint has RNG state", pre_ck.get("rng") is not None)
    check("checkpoint records the training hyperparameters",
          isinstance(pre_ck.get("train_hparams"), dict),
          str(pre_ck.get("train_hparams")))
    check("checkpoint records data position", pre_data_pos > 0, f"data_pos={pre_data_pos}")
    check("some curve points exist before the pause", len(pre_curve) > 0, f"{len(pre_curve)}")
    check("batch-hash registry non-empty", len(pre_registry_lines) > 0, f"{len(pre_registry_lines)} lines")
    check("registry has no duplicate entries",
          len(set(pre_registry_lines)) == len(pre_registry_lines))
    check("registry length matches data position",
          len(pre_registry_lines) == pre_data_pos,
          f"{len(pre_registry_lines)} vs data_pos {pre_data_pos}")

    # ---------------- stage C: the pause itself ---------------------------------
    print(f"\n=== C. Staying paused for {args.pause_for:.0f}s (PC free for other work) ===")
    time.sleep(args.pause_for)
    check("artifacts intact after the pause window",
          ckpt.exists() and curve_json.exists() and weights_fingerprint(ckpt)[0] == pre_hash)

    # ---------------- stage D: resume, then HARD KILL partway ---------------------
    # A clean STOP is the easy case. Llama 3 (Dubey et al. 2024) reports 419 of
    # 466 training interruptions as *unplanned*, so recovery from an ungraceful
    # death is the common case, not the edge -- and it is the only reason the
    # periodic checkpoint exists. Killing here also exercises the
    # registry-ahead-of-checkpoint path, which a clean pause never reaches.
    # The step target stays the same across all segments, so the
    # hyperparameter-drift guard stays satisfied (as it should).
    print("\n=== D. Resume, then HARD KILL partway (power-cut simulation) ===")
    t_resume = time.time()
    proc2 = subprocess.Popen([sys.executable, str(LAUNCHER), str(ROOT), str(args.steps)],
                             cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    resume_line_seen = None
    time.sleep(50.0)  # past the resume and at least one periodic checkpoint
    if proc2.poll() is None:
        killed_mid_run = True
        proc2.kill()
    else:
        killed_mid_run = False
    out2, _ = proc2.communicate()
    resume_line = [l for l in out2.splitlines() if "resumed cracked" in l]
    resume_line_seen = resume_line[0] if resume_line else None
    check("resume message reports the right step and restored curve",
          bool(resume_line_seen) and f"step {pre_step}" in resume_line_seen,
          resume_line_seen or "no resume line")
    check("STOP file was cleared on relaunch", not stop_file.exists())
    check("process was still running when killed (hard-kill really tested)",
          killed_mid_run,
          "" if killed_mid_run else "it had already finished; raise --steps")

    killed_ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    killed_registry = registry.read_text().split()
    print(f"  killed at step={killed_ck['step']} data_pos={killed_ck['data_pos']}, "
          f"registry={len(killed_registry)} lines")
    check("checkpoint survived the hard kill and is loadable",
          killed_ck["step"] >= pre_step, f"step {pre_step} -> {killed_ck['step']}")
    check("registry is ahead of (or equal to) the checkpoint after the kill",
          len(killed_registry) >= killed_ck["data_pos"],
          f"registry {len(killed_registry)} vs data_pos {killed_ck['data_pos']}")
    check("registry has no duplicates after the hard kill",
          len(set(killed_registry)) == len(killed_registry))

    # ---------------- stage D2: resume again, finish, with a held handle ----------
    print("\n=== D2. Resuming after the hard kill, running to completion ===")
    # Simulate an antivirus scan / file copy holding the checkpoint open across a
    # checkpoint interval (30s here). Before the retry+non-fatal fix this made
    # os.replace raise PermissionError and killed the run outright.
    # hold_for must sit clearly INSIDE the retry tolerance, or the check becomes a
    # coin flip: too long and the rename never lands, too short and it may not
    # overlap a checkpoint at all, so the assertion would pass without the retry
    # path having run.
    handle_flag: dict = {}
    proc3 = subprocess.Popen([sys.executable, str(LAUNCHER), str(ROOT), str(args.steps)],
                             cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    print("  holding a read handle on the checkpoint for 10s (antivirus simulation)")
    hold_file_handle(ckpt, start_after=20.0, hold_for=10.0, flag=handle_flag)
    out3, _ = proc3.communicate(timeout=1800)
    resume_wall = time.time() - t_resume
    print(f"  final segment finished, rc={proc3.returncode}")
    check("recovered cleanly from the hard-killed checkpoint", proc3.returncode == 0,
          f"rc={proc3.returncode}")
    check("no bogus hash mismatch when revisiting already-registered batches",
          "hash mismatch" not in out3 and "registry looks corrupt" not in out3)
    check("run SURVIVED an outside handle held on the checkpoint",
          proc3.returncode == 0 and handle_flag.get("held") is True,
          f"handle held={handle_flag.get('held')} {handle_flag.get('error', '')}")

    # ---------------- stage E: nothing lost, everything merged -------------------
    print("\n=== E. Verifying NOTHING from before the pause was lost ===")
    post_ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    post_hash, post_norm = weights_fingerprint(ckpt)
    post_curve = json.loads(curve_json.read_text())
    post_status = json.loads(status_json.read_text())
    post_registry = registry.read_text().split()
    post_log = log_file.read_text(encoding="utf-8").splitlines()

    print(f"  final step={post_ck['step']} active={post_ck['active_seconds']:.1f}s "
          f"curve_points={len(post_curve)} weights={post_hash} |w|={post_norm:.2f}")

    # curve: every pre-pause point survives, identical, in order, and extended
    pre_steps = [h["step"] for h in pre_curve]
    post_steps = [h["step"] for h in post_curve]
    check("curve keeps ALL pre-pause points, in order",
          post_steps[:len(pre_steps)] == pre_steps,
          f"{pre_steps} vs {post_steps[:len(pre_steps)]}")
    check("pre-pause curve VALUES are byte-identical (not recomputed)",
          post_curve[:len(pre_curve)] == pre_curve)
    check("curve was extended past the pause",
          len(post_steps) > len(pre_steps) and post_steps[-1] > pre_steps[-1],
          f"{len(pre_steps)} -> {len(post_steps)} points, last step {post_steps[-1]}")
    check("curve steps are strictly increasing (no duplicates from the restart)",
          all(b > a for a, b in zip(post_steps, post_steps[1:])), f"{post_steps}")
    check("every curve point has loss and tok/s",
          all(h.get("train_loss") is not None and h.get("tokens_per_sec") is not None
              for h in post_curve))

    # weights: training CONTINUED from the paused state, rather than restarting.
    # "changed" alone proves nothing -- a reinitialization also changes them --
    # so measure the relative distance. Threshold calibrated by MEASUREMENT on
    # this model, not guessed: two independent initializations sit at 0.967
    # relative distance, while continuing ~57 early steps moves the weights by
    # ~0.124 (early training with warmup moves fast). 0.5 separates the two by a
    # wide margin in both directions.
    # NOTE: the decisive proof of continuation is the Adam step-counter check
    # below -- it reads the global step (70), where a reset would read the
    # resumed segment's length (57). This norm check is the cheap corroboration.
    post_weights = flat_weights(ckpt)
    rel_delta = float((post_weights - pre_weights).norm() / pre_weights.norm())
    check("weights CONTINUED from the paused state (not reinitialized)",
          0.0 < rel_delta < 0.5,
          f"relative change {rel_delta:.4f} (measured: reinit = 0.967)")
    check("weights are finite (no NaN/inf across the interruption)",
          bool(post_weights.isfinite().all()))

    # Adam continued too: its step counter must reach the final training step,
    # not restart from the length of the resumed segment
    post_adam_steps = adam_step_counts(ckpt)
    check("Adam step counter continued across the pause",
          bool(post_adam_steps) and all(s == post_ck["step"] for s in post_adam_steps),
          f"adam steps {set(post_adam_steps)} vs final step {post_ck['step']} "
          f"(a reset would show {post_ck['step'] - pre_step})")
    check("Adam moments still populated after resume", adam_moments_nonzero(ckpt))

    # step/data continuity: no repeats, no gaps
    check("final step is beyond the pause step", post_ck["step"] > pre_step,
          f"{pre_step} -> {post_ck['step']}")
    # ...and it actually REACHED the target. Without this, a second segment cut
    # short for any reason would pass the whole rehearsal, "finished" included.
    check("final step reached the requested target", post_ck["step"] == args.steps,
          f"{post_ck['step']} vs target {args.steps}")
    expected_micro = args.steps * config.GRAD_ACCUM
    check("data position reached the expected micro-batch count",
          post_ck["data_pos"] == expected_micro,
          f"{post_ck['data_pos']} vs expected {expected_micro}")
    check("data position advanced", post_ck["data_pos"] > pre_data_pos,
          f"{pre_data_pos} -> {post_ck['data_pos']}")
    check("registry extends the earlier one with no rewrite",
          post_registry[:len(pre_registry_lines)] == pre_registry_lines,
          f"{len(pre_registry_lines)} -> {len(post_registry)} lines")
    check("registry still has no duplicates after the interruption",
          len(set(post_registry)) == len(post_registry),
          f"{len(post_registry)} lines, {len(set(post_registry))} unique")
    # On a clean exit these are equal; after an unclean one the registry
    # legitimately runs AHEAD of the checkpoint (it is appended per micro-batch,
    # the checkpoint only at accumulation boundaries). Ahead is fine and handled
    # -- the resumed run revisits those batches and verifies them. BEHIND would
    # mean lost bookkeeping.
    check("registry is at or ahead of the checkpoint's data position (never behind)",
          len(post_registry) >= post_ck["data_pos"],
          f"{len(post_registry)} vs {post_ck['data_pos']}")

    # INDEPENDENT ORACLE. Every check above derives from data_pos, the very value
    # being validated, so an off-by-one in the resume point would pass them all:
    # the revisited batch's hash would match (same plan), it would simply be
    # trained twice. Recompute the expected hashes straight from the frozen plan
    # and compare element by element -- this is the only check here that can
    # catch a repeated or skipped batch.
    from compare_lab.data import prefix_lm as P
    plan = P.load_plan(ROOT / "pre" / "plan")
    expected = [P.batch_hash(frag, cuts) for b, frag, cuts in P.iter_batch_ids(plan)
                if b < len(post_registry)]
    first_bad = next((i for i, (a, b) in enumerate(zip(expected, post_registry)) if a != b), None)
    check("registry matches the frozen plan hash-for-hash (independent oracle)",
          first_bad is None and len(expected) == len(post_registry),
          f"first mismatch at index {first_bad}" if first_bad is not None
          else f"all {len(post_registry)} entries match the plan")

    # log is append-only across runs
    check("log keeps all pre-pause lines (append-only)",
          post_log[:len(pre_log_lines)] == pre_log_lines)
    # the PERIODIC checkpoint path (the one that protects against a hard kill)
    # must actually have run -- with production's 15min interval it never would
    n_periodic = sum(1 for l in post_log if "checkpoint @ step" in l)
    check("periodic checkpoint path exercised", n_periodic > 0,
          f"{n_periodic} periodic checkpoints")
    check("log records the pause and the restart",
          any("paused" in l for l in post_log) and
          sum(1 for l in post_log if "start arch=" in l) >= 2)

    # work time excludes the pause
    active = post_ck["active_seconds"]
    expected_max = args.pause_at + resume_wall + 15  # generous, but well under +pause_for
    check("active_training_seconds EXCLUDES the pause",
          active < args.pause_at + resume_wall + args.pause_for * 0.5,
          f"active={active:.1f}s vs wall-with-pause="
          f"{args.pause_at + args.pause_for + resume_wall:.1f}s")
    check("active_training_seconds accumulated across both segments",
          active > pre_active, f"{pre_active:.1f}s -> {active:.1f}s")

    # final artifacts
    final_ckpt = ROOT / "ckpt" / "cracked" / "pretrain_final.pt"
    check("final checkpoint written", final_ckpt.exists())
    check("curve png written", (ROOT / "results" / "pretrain_loss_cracked.png").exists())
    check("final status is finished", post_status["state"] == "finished", post_status["state"])
    if final_ckpt.exists():
        fh, fnorm = weights_fingerprint(final_ckpt)
        check("final checkpoint loads and has finite weights", np.isfinite(fnorm), f"|w|={fnorm:.2f}")

    # loss actually moved (sanity: real training on real data)
    losses = [h["train_loss"] for h in post_curve]
    check("train loss decreased over the run", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    vals = [h["val_loss"] for h in post_curve if h.get("val_loss") is not None]
    check("validation loss recorded", len(vals) > 0, f"{len(vals)} points: {vals}")

    # ---------------- stage F: collect everything for inspection ------------------
    print("\n=== F. Collecting artifacts for inspection ===")
    report_dir = ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (curve_json, "loss_curve.json"),
        (ROOT / "results" / "pretrain_loss_cracked.png", "loss_curve.png"),
        (log_file, "training.log"),
        (status_json, "status_final.json"),
        (registry, "batch_hashes.txt"),
        (ROOT / "pre" / "stats.json", "data_stats.json"),
    ]:
        if Path(src).exists():
            shutil.copy2(src, report_dir / dst)
    (report_dir / "status_at_pause.json").write_text(json.dumps(pre_status, indent=2))
    (report_dir / "checks.json").write_text(json.dumps(
        [{"check": n, "passed": ok, "detail": d} for n, ok, d in results], indent=2))

    # Throughput per segment: if the two differ a lot, the GPU was shared with
    # something else (a game, another job) rather than the code being slow.
    seg1_tok_s = pre_status.get("tokens_per_sec")
    seg2_tok_s = post_status.get("tokens_per_sec")

    curve_with_pause = report_dir / "loss_curve_with_pause.png"
    _plot_with_pause(post_curve, pre_step, curve_with_pause)
    check("annotated curve written", curve_with_pause.exists())

    failed = [(n, d) for n, ok, d in results if not ok]
    _write_report(report_dir / "REPORT.md", args, stats, results, failed,
                  pre_step, post_ck["step"], pre_curve, post_curve,
                  pre_active, active, pause_latency, resume_wall,
                  seg1_tok_s, seg2_tok_s, n_periodic, handle_flag)

    # ---------------- summary ---------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULT: {len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("\nFAILED CHECKS:")
        for n, d in failed:
            print(f"  - {n}: {d}")
    else:
        print("ALL CHECKS PASSED -- pause/resume is safe on real data")
    print(f"\nthroughput: segment 1 = {seg1_tok_s} tok/s, segment 2 = {seg2_tok_s} tok/s")
    print("=" * 70)
    print(f"\nREPORT + graphs + logs + json for inspection:\n  {report_dir}")
    return 1 if failed else 0


def _plot_with_pause(curve: list[dict], pause_step: int, out_path: Path) -> None:
    """Loss curve with the pause marked, so continuity across the interruption is
    visible at a glance rather than only assertable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [h["step"] for h in curve]
        train = [h.get("train_loss") for h in curve]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, train, marker="o", ms=3, label="train loss", color="tab:blue")
        val_pts = [(h["step"], h["val_loss"]) for h in curve if h.get("val_loss") is not None]
        if val_pts:
            vs, vl = zip(*val_pts)
            ax.plot(vs, vl, marker="s", ms=3, label="val loss", color="tab:orange")
        ax.axvline(pause_step, color="tab:red", ls="--", lw=2,
                   label=f"pause + resume @ step {pause_step}")
        ax.axvspan(min(steps), pause_step, alpha=0.06, color="tab:green")
        ax.annotate("before the pause", xy=(min(steps), max(train)), fontsize=9, color="gray")
        ax.annotate("after resuming", xy=(pause_step + 1, max(train)), fontsize=9, color="gray")
        ax.set_xlabel("optimizer step"); ax.set_ylabel("loss")
        ax.set_title("Loss curve across a pause/resume (one continuous curve)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"(annotated plot skipped: {e})")


def _write_report(path: Path, args, stats, results, failed, pre_step, post_step,
                  pre_curve, post_curve, pre_active, active, pause_latency,
                  resume_wall, seg1_tok_s, seg2_tok_s, n_periodic, handle_flag) -> None:
    passed = len(results) - len(failed)
    lines = [
        "# Ensayo de pausa y reanudacion con datos reales",
        "",
        f"**Resultado: {passed}/{len(results)} comprobaciones superadas.**",
        "",
        "Este ensayo ejecuta el mismo codigo que el entrenamiento real (mismo bucle,",
        "mismo tokenizer, mismos `grad_accum`/`micro_batch`/`block_size`/bf16, datos",
        "reales de SmolLM-Corpus), como **proceso independiente**, y lo pausa desde",
        "fuera con un fichero `STOP` igual que lo harias tu desde el movil.",
        "",
        "## Lo que ha pasado",
        "",
        f"| | |", "| --- | --- |",
        f"| Tokens de entrenamiento | {stats['n_train_tokens']:,} |",
        f"| Mezcla real de subsets | {stats['pct_per_subset']} % {stats['subsets']} |",
        f"| Pausa solicitada en | {args.pause_at:.0f}s de entrenamiento |",
        f"| Tardo en pausar | **{pause_latency:.2f}s** |",
        f"| Paso al pausar | {pre_step} |",
        f"| Duracion de la pausa | {args.pause_for:.0f}s |",
        f"| Paso final | {post_step} |",
        f"| Puntos de curva antes -> despues | {len(pre_curve)} -> {len(post_curve)} |",
        f"| Tiempo de trabajo acumulado | {pre_active:.1f}s -> **{active:.1f}s** |",
        f"| Tiempo de reloj con pausa | {args.pause_at + args.pause_for + resume_wall:.1f}s |",
        f"| Checkpoints periodicos ejercitados | {n_periodic} |",
        f"| Handle externo sobre el checkpoint | {'si, y el run sobrevivio' if handle_flag.get('held') else 'no se pudo simular'} |",
        f"| Throughput tramo 1 / tramo 2 | {seg1_tok_s} / {seg2_tok_s} tok/s |",
        "",
        "El **tiempo de trabajo acumulado** es menor que el tiempo de reloj: la pausa",
        "no se cuenta, que es exactamente lo que se queria medir.",
        "",
        "Si los dos throughputs difieren mucho, la GPU estaba compartida con otra",
        "cosa (un juego, otro proceso) durante uno de los tramos; no es el codigo.",
        "",
        "## Ficheros para revisar",
        "",
        "- `loss_curve_with_pause.png` — la curva con la pausa marcada en rojo. Es",
        "  **una sola curva continua**: los puntos anteriores a la pausa siguen ahi y",
        "  los nuevos continuan a partir de ellos.",
        "- `loss_curve.png` / `loss_curve.json` — la curva tal y como la genera el",
        "  entrenamiento (con tokens/s en el eje derecho) y sus datos crudos.",
        "- `training.log` — el log completo, incluyendo la linea de la pausa y el",
        "  arranque del segundo tramo (es append-only: no se pierde nada).",
        "- `status_at_pause.json` / `status_final.json` — el estado en el momento de",
        "  pausar y al terminar.",
        "- `batch_hashes.txt` — el registro de identidad de lotes, sin duplicados ni",
        "  huecos a pesar de la interrupcion.",
        "- `data_stats.json` — composicion real de la muestra de datos.",
        "- `checks.json` — las comprobaciones una por una, en bruto.",
        "",
        "## Comprobaciones",
        "",
    ]
    for n, ok, d in results:
        lines.append(f"- {'OK  ' if ok else 'FALLO'} {n}" + (f" — {d}" if d else ""))
    if failed:
        lines += ["", "## Fallos", ""] + [f"- **{n}**: {d}" for n, d in failed]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
