"""Prefix-LM pretraining for the three models, sharing one data plan so they see
identical (prefix, continuation) pairs in identical order (plan section 4).

  --arch cracked       decoder-only, loss on the continuation only (Cracked-D)
  --arch cracked_full  decoder-only, loss on ALL tokens (Cracked-D-full control)
  --arch sliced        encoder reads prefix, decoder predicts continuation (Sliced-D)

Cracked writes a per-micro-batch hash registry; sliced/cracked_full verify their
batches against it and stop on any mismatch (guards the freeze rule). Resume =
relaunch the same command: it restores weights, optimizer, scaler, RNG and the
exact data position, and continues in the same order.

The ONLY voluntary stop is the STOP file: creating it makes the loop finish the
current accumulation cycle, save, and exit as "paused", losing nothing. Anything
else (Ctrl+C, closing the window, a power cut) is a hard failure and falls back
to the last periodic checkpoint. Keeping a single stop path is deliberate: the
signal-handling one was rarely exercised and produced several real bugs.

Usage (short smoke run):
    python -m compare_lab.train.pretrain --arch cracked --grad-accum 1 \
        --max-optimizer-steps 20 --eval-interval 10 --ckpt-interval-min 999
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from compare_lab import config
from compare_lab.data import prefix_lm as P
from compare_lab.data.bindata import load_bin
from compare_lab.data.tokenizer import Tokenizer, DEFAULT_TOKENIZER_DIR
from compare_lab.models.cracked_d import build_model as build_cracked, build_config as cracked_config
from compare_lab.models.sliced_d import SlicedD, SlicedDConfig
from compare_lab.train.checkpoint import save_checkpoint, load_checkpoint
from compare_lab.train.lr import lr_at_step
from compare_lab.train.tasks import (StatusWriter, stop_requested, clear_stop,
                                     mark_error, acquire_lock, release_lock, atomic_write_json)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
PRETRAIN_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts" / "pretrain"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"


def per_token_ce(logits, targets):
    """Per-token cross-entropy, (B, T). Computing this once lets several masks
    be applied to the same tensor for free (train loss + prefix diagnostic),
    instead of a second full cross_entropy over (B, T, 8192) logits."""
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none").view(B, T)


def apply_mask(ce, mask):
    ntok = mask.sum()
    return (ce * mask).sum() / ntok.clamp(min=1), ntok


def masked_ce(logits, targets, mask):
    return apply_mask(per_token_ce(logits, targets), mask)


def build_model(arch: str, vocab_size: int, block_size: int, device: str):
    if arch in ("cracked", "cracked_full"):
        model = build_cracked(cracked_config(vocab_size=vocab_size, block_size=block_size))
    elif arch == "sliced":
        model = SlicedD(SlicedDConfig(vocab_size=vocab_size, max_src_len=block_size, max_tgt_len=block_size))
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return model.to(device)


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def forward_loss(arch, model, bin_data, frag, cuts, block_size, bos_id, pad_id, device,
                 continuation_only: bool = False):
    """Returns (loss, n_loss_tokens, prefix_diag_loss_or_None).

    continuation_only forces the continuation-only mask even for cracked_full:
    used for VALIDATION so the control model's val_loss is measured on the same
    tokens as Cracked-D's and the two curves are directly comparable (otherwise
    one reports perplexity over all tokens and the other over the continuation,
    and the control loses most of its value)."""
    if arch in ("cracked", "cracked_full"):
        full_loss = (arch == "cracked_full") and not continuation_only
        x, y, lm = P.collate_cracked(bin_data, frag, cuts, block_size, full_loss=full_loss)
        x, y, lm = x.to(device), y.to(device), lm.to(device)
        logits, _ = model(x)
        ce = per_token_ce(logits, y)
        loss, ntok = apply_mask(ce, lm)
        diag = None
        if arch == "cracked":
            # prefix loss as a diagnostic only (never in the backward); reuses the
            # per-token ce already computed, so this costs essentially nothing
            with torch.no_grad():
                diag, _ = apply_mask(ce.detach(), 1.0 - lm)
        return loss, ntok, diag
    else:
        src, src_pad, tin, tout, lm = P.collate_sliced(bin_data, frag, cuts, block_size, bos_id, pad_id)
        src, src_pad, tin, tout, lm = (src.to(device), src_pad.to(device), tin.to(device),
                                       tout.to(device), lm.to(device))
        logits, _ = model(src, tin, src_key_padding_mask=src_pad)
        loss, ntok = masked_ce(logits, tout, lm)
        return loss, ntok, None


class HashRegistry:
    """Per-micro-batch batch-identity hashes. Cracked appends; the others verify."""

    def __init__(self, path: Path, write: bool):
        self.path = Path(path)
        self.write = write
        self.hashes: list[str] = []
        if self.path.exists():
            self.hashes = self.path.read_text().split()
        if not write and not self.hashes:
            raise FileNotFoundError(f"batch-hash registry {path} missing; run Cracked-D first")

    def check_or_append(self, i: int, h: str) -> None:
        if i < len(self.hashes):
            saved = self.hashes[i]
            if saved != h:
                # Show the FULL hashes: truncating to 12 chars can print two
                # identical-looking values when they differ further along, which
                # is unreadable at hour five of a run. Also distinguish a
                # corrupt registry from a genuine config change.
                if len(saved) != len(h):
                    raise RuntimeError(
                        f"batch-hash registry looks corrupt at entry {i}: expected a "
                        f"{len(h)}-char hash, found {len(saved)} chars ({saved!r}). A run "
                        f"likely died mid-append. Truncate the file to {i} lines and resume.")
                raise RuntimeError(f"batch {i} hash mismatch: config drifted since Cracked-D "
                                   f"({saved} != {h}). See the freeze rule.")
            return
        if not self.write:
            raise RuntimeError(f"registry has {len(self.hashes)} entries but batch {i} requested; "
                               "Cracked-D did not finish this run")
        with open(self.path, "a") as f:
            f.write(h + "\n")
        self.hashes.append(h)


@torch.no_grad()
def evaluate(arch, model, val_bin, val_plan, block_size, bos_id, pad_id, device, max_batches=20,
             amp_dtype=torch.bfloat16, use_amp: bool = False):
    """Validation loss, always over the CONTINUATION tokens only (even for
    cracked_full) so all three models report the same metric.

    Runs under the same autocast as training: it keeps the metric consistent
    with the training loss and avoids an fp32 logits tensor of
    (batch, block_size, vocab) -- at 16x1023x8192 that is 537 MB in fp32, a
    needless OOM risk 38 minutes into a run."""
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for b, frag, cuts in P.iter_batch_ids(val_plan):
        if b >= max_batches:
            break
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            loss, ntok, _ = forward_loss(arch, model, val_bin, frag, cuts, block_size,
                                         bos_id, pad_id, device, continuation_only=True)
        tot_loss += loss.item() * int(ntok)
        tot_tok += int(ntok)
    model.train()
    return tot_loss / max(1, tot_tok)


def train(args) -> dict:
    """Wrapper that records a crash in status.json before re-raising, so a run
    that dies (OOM, hash mismatch, disk error) is visible as "error" with its
    message instead of silently staying "running" until it looks merely stale."""
    task_name = f"pretrain_{args.arch}"
    lock = acquire_lock(task_name)
    try:
        return _train(args)
    except Exception as e:
        mark_error(task_name, repr(e))
        raise
    finally:
        release_lock(lock)


def _train(args) -> dict:
    device = args.device
    torch.manual_seed(args.seed)
    tok = Tokenizer.from_dir(args.tokenizer_dir)

    train_bin = load_bin(Path(args.pretrain_dir) / "train.bin")
    val_bin = load_bin(Path(args.pretrain_dir) / "val.bin")
    plan = P.load_plan(Path(args.pretrain_dir) / "plan")
    block_size = plan.block_size
    val_plan = P.build_plan(len(val_bin), block_size, plan.batch_size, plan.seed + 1)

    model = build_model(args.arch, tok.vocab_size, block_size, device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    use_amp = device == "cuda" and args.dtype != "float32"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    task_name = f"pretrain_{args.arch}"
    ckpt_path = CHECKPOINT_DIR / args.arch / "pretrain.pt"
    registry = HashRegistry(Path(args.pretrain_dir) / "plan" / "batch_hashes.txt",
                            write=(args.arch == "cracked"))

    total_micro = plan.n_batches
    if args.max_optimizer_steps:
        total_micro = min(total_micro, args.max_optimizer_steps * args.grad_accum)
    total_opt_steps = total_micro // args.grad_accum
    total_tokens = total_micro * plan.batch_size * block_size

    step, data_pos, active_seconds_before = 0, 0, 0.0
    history: list[dict] = []
    resumed_running_loss = None
    if ckpt_path.exists() and not args.no_resume:
        info = load_checkpoint(ckpt_path, model, optimizer, scaler, device=device)
        step, data_pos = info["step"], info["data_pos"]
        active_seconds_before = info.get("active_seconds", 0.0)
        # Restore the loss curve so a resume CONTINUES it. Without this, the
        # fresh empty history overwrites loss_curve.json at the next save and
        # every point before the pause is lost -- and that curve is a deliverable.
        history = list(info.get("history") or [])
        resumed_running_loss = info.get("running_loss")

        # Refuse to silently resume under different training hyperparameters.
        prev = info.get("train_hparams")
        if prev:
            now = {"grad_accum": args.grad_accum, "lr": args.lr, "min_lr": args.min_lr,
                   "warmup_steps": args.warmup_steps, "total_opt_steps": total_opt_steps}
            drift = {k: (prev[k], now[k]) for k in prev if k in now and prev[k] != now[k]}
            if drift:
                raise RuntimeError(
                    f"refusing to resume {args.arch} with different training hyperparameters: "
                    f"{drift} (saved vs now). Resuming with a different grad_accum or step "
                    f"target silently rescales the cosine LR schedule mid-run. Relaunch with "
                    f"the original values, or start fresh with --no-resume.")

        print(f"resumed {args.arch} at optimizer step {step}, micro-batch {data_pos}, "
              f"{active_seconds_before/3600:.2f}h of prior active work time, "
              f"{len(history)} curve points restored")

    clear_stop(task_name)
    status = StatusWriter(task_name, args.arch, total_opt_steps, total_tokens,
                          active_seconds_before=active_seconds_before)
    log_path = status.dir / f"{task_name}.log"

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"start arch={args.arch} device={device} amp={use_amp}/{args.dtype} "
        f"micro-batches={total_micro} opt-steps={total_opt_steps} block={block_size} bs={plan.batch_size}")

    last_ckpt_time = time.time()
    tokens_done = data_pos * plan.batch_size * block_size
    tokens_at_segment_start = tokens_done  # only THIS segment's tokens/elapsed feed tok/s below
    t0 = time.time()
    micro_in_accum = 0
    running_loss = resumed_running_loss  # keep the EMA warm across a pause
    final_state = "finished"

    def checkpoint_extra() -> dict:
        return {
            "active_seconds": status.active_seconds,
            "history": history,
            # running_loss is an EMA; restarting it from None after every pause
            # leaves one artificially noisy point in the curve per pause
            "running_loss": running_loss,
            # The training hyperparameters that change the RESULT but are not in
            # the model config. grad_accum in particular sets total_opt_steps,
            # which is the denominator of the cosine LR schedule -- resuming with
            # a different value silently stretches the schedule mid-run with
            # nothing to flag it (the batch hashes still match, since micro-batch
            # indices are unchanged).
            "train_hparams": {"grad_accum": args.grad_accum, "lr": args.lr,
                              "min_lr": args.min_lr, "warmup_steps": args.warmup_steps,
                              "total_opt_steps": total_opt_steps},
        }

    pbar = tqdm(total=total_opt_steps, initial=step, desc=f"pretrain {args.arch}")
    optimizer.zero_grad(set_to_none=True)
    for b, frag, cuts in P.iter_batch_ids(plan, start_batch=data_pos):
        if b >= total_micro:
            break
        registry.check_or_append(b, P.batch_hash(frag, cuts))

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            loss, ntok, diag = forward_loss(args.arch, model, train_bin, frag, cuts,
                                            block_size, tok.bos_id, tok.pad_id, device)
        scaler.scale(loss / args.grad_accum).backward()
        running_loss = loss.item() if running_loss is None else 0.9 * running_loss + 0.1 * loss.item()
        micro_in_accum += 1
        tokens_done += plan.batch_size * block_size
        data_pos = b + 1  # next micro-batch to process; kept current for checkpoints

        if micro_in_accum == args.grad_accum:
            lr = lr_at_step(step, args.lr, args.min_lr, args.warmup_steps, total_opt_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            micro_in_accum = 0
            step += 1
            pbar.update(1)

            tok_s = (tokens_done - tokens_at_segment_start) / max(1e-9, time.time() - t0)
            status.update(step, tokens_done, train_loss=running_loss, tokens_per_sec=tok_s)

            is_eval_step = bool(args.eval_interval and step % args.eval_interval == 0)
            if step % args.log_interval == 0:
                extra = f" prefix_diag={diag.item():.4f}" if diag is not None else ""
                log(f"step {step}/{total_opt_steps} loss {running_loss:.4f} lr {lr:.2e} "
                    f"{tok_s:.0f} tok/s{extra}")
                if not is_eval_step:  # avoid a duplicate row for the same step below
                    history.append({"step": step, "train_loss": running_loss, "val_loss": None,
                                    "prefix_loss": round(diag.item(), 4) if diag is not None else None,
                                    "tokens_per_sec": round(tok_s, 1)})

            if is_eval_step:
                val = evaluate(args.arch, model, val_bin, val_plan, block_size,
                               tok.bos_id, tok.pad_id, device,
                               amp_dtype=amp_dtype, use_amp=use_amp)
                history.append({"step": step, "train_loss": running_loss, "val_loss": val,
                                "prefix_loss": round(diag.item(), 4) if diag is not None else None,
                                "tokens_per_sec": round(tok_s, 1)})
                status.update(step, tokens_done, train_loss=running_loss, val_loss=val,
                              tokens_per_sec=tok_s, force=True)
                log(f"  [eval] step {step} val_loss {val:.4f}")

            # checkpoint only at accumulation boundaries -> data_pos consistent
            if (time.time() - last_ckpt_time) / 60.0 >= args.ckpt_interval_min:
                ok = save_checkpoint(ckpt_path, model, optimizer, step, data_pos,
                                     _config_dict(args.arch, model), scaler=scaler,
                                     extra=checkpoint_extra())
                last_ckpt_time = time.time()
                status.update(step, tokens_done, last_ckpt_step=step if ok else None, force=True)
                log(f"  checkpoint @ step {step}" if ok else
                    f"  checkpoint @ step {step} DEFERRED (destination busy; kept as .tmp, "
                    f"previous checkpoint intact) -- training continues")

            if stop_requested(task_name):
                final_state = "paused"
                log(f"stop requested -> paused @ step {step}")
                break  # the unconditional save below covers this case

            if step >= total_opt_steps:
                break
    pbar.close()

    # Single final save covering every exit path (finished, paused, step
    # target reached) -- saving again inside the pause branch would write
    # these ~300MB twice per pause for nothing.
    if not save_checkpoint(ckpt_path, model, optimizer, step, data_pos,
                           _config_dict(args.arch, model), scaler=scaler,
                           extra=checkpoint_extra()):
        log(f"WARNING: final checkpoint could not be renamed into place; it is at "
            f"{ckpt_path}.tmp. Rename it manually before resuming, or the run will "
            f"restart from the previous checkpoint.")
    # Only publish *_final.pt if the step target was actually reached: that file
    # is what makes run_pipeline consider the stage done and skip it forever.
    if final_state == "finished" and step < total_opt_steps:
        final_state = "paused"
        log(f"WARNING: loop ended at step {step} of {total_opt_steps} without a stop "
            f"request; NOT writing pretrain_final.pt. Reporting 'paused' so the pipeline "
            f"resumes it.")
    if final_state == "finished":
        final_path = CHECKPOINT_DIR / args.arch / "pretrain_final.pt"
        if not save_checkpoint(final_path, model, optimizer, step, data_pos,
                               _config_dict(args.arch, model), scaler=scaler,
                               extra=checkpoint_extra()):
            log(f"WARNING: {final_path.name} could not be renamed into place; it is at "
                f"{final_path}.tmp")
    # Curves live in eval/results/ (a deliverable, versioned) rather than runs/
    # (gitignored working state), following the project convention of keeping a
    # .png next to the .json that feeds it.
    _save_curves(RESULTS_DIR / f"pretrain_loss_{args.arch}", history, f"pretrain {args.arch}")
    status.update(step, tokens_done, force=True)
    status.set_state(final_state)
    log(f"done state={final_state} step={step} active_work={status.active_seconds/3600:.2f}h")
    # Release the memory-mapped bins explicitly. On Windows an open mapping keeps
    # a handle on the file, so a leftover reference would block deleting or
    # regenerating train.bin (2.4GB) until the process exits.
    del train_bin, val_bin
    return {"state": final_state, "step": step, "history": history}


def _config_dict(arch, model):
    import dataclasses
    return dataclasses.asdict(model.config) if hasattr(model.config, "__dataclass_fields__") else vars(model.config)


def _save_curves(base: Path, history: list[dict], arch: str) -> None:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    # atomic like every other write in this package: the curve json is a
    # deliverable, and a crash mid-write would leave it truncated
    atomic_write_json(base.with_suffix(".json"), history)
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [h["step"] for h in history]
        fig, ax1 = plt.subplots()
        ax1.plot(steps, [h.get("train_loss") for h in history], label="train loss", color="tab:blue")
        val_pts = [(h["step"], h["val_loss"]) for h in history if h.get("val_loss") is not None]
        if val_pts:
            vs, vl = zip(*val_pts)
            ax1.plot(vs, vl, label="val loss", color="tab:orange")
        ax1.set_xlabel("step"); ax1.set_ylabel("loss")

        tok_pts = [(h["step"], h["tokens_per_sec"]) for h in history if h.get("tokens_per_sec") is not None]
        if tok_pts:
            ts, tps = zip(*tok_pts)
            ax2 = ax1.twinx()
            ax2.plot(ts, tps, label="tokens/s", color="tab:green", alpha=0.4, linewidth=1)
            ax2.set_ylabel("tokens/s")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2)
        else:
            ax1.legend()

        plt.title(arch)
        fig.savefig(base.with_suffix(".png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # plotting must never crash a training run
        print(f"(curve plot skipped: {e})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=["cracked", "cracked_full", "sliced"])
    # defaults come from the frozen config so a direct CLI resume cannot
    # accidentally run under different hyperparameters than the pipeline
    p.add_argument("--grad-accum", type=int, default=config.GRAD_ACCUM)
    p.add_argument("--max-optimizer-steps", type=int, default=0, help="0 = full single epoch")
    p.add_argument("--lr", type=float, default=config.PRETRAIN["lr"])
    p.add_argument("--min-lr", type=float, default=config.PRETRAIN["min_lr"])
    p.add_argument("--warmup-steps", type=int, default=config.PRETRAIN["warmup_steps"])
    p.add_argument("--weight-decay", type=float, default=config.PRETRAIN["weight_decay"])
    p.add_argument("--dtype", default=config.PRETRAIN["dtype"],
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--eval-interval", type=int, default=config.PRETRAIN_EVAL_INTERVAL)
    p.add_argument("--log-interval", type=int, default=config.LOG_INTERVAL)
    p.add_argument("--ckpt-interval-min", type=float, default=config.CKPT_INTERVAL_MIN)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    p.add_argument("--pretrain-dir", default=str(PRETRAIN_DIR))
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
