"""Chat fine-tuning on smol-smoltalk for the three models (plan section 5).

Loss is computed only on assistant tokens (including the closing <eos>), using
the canonical chat template identically to eval and the demo. The two topologies
package multi-turn conversations differently but put loss on the SAME assistant
tokens:

  - Cracked / Cracked-D-full: the whole conversation is one causal sequence;
    everything that is not an assistant token is masked out of the loss.
  - Sliced: one example per assistant turn -- the encoder reads the conversation
    history up to and including the <|assistant|> marker, the decoder generates
    that turn's response and its <eos>.

Data is read in the fixed order written by prepare_smoltalk (no reshuffle), and
all models run the same number of optimizer steps. Generation stops at <eos>.
Same long-task plumbing as pretraining (resume, STOP file, durable checkpoints).

Usage (short smoke run):
    python -m compare_lab.train.finetune --arch cracked --max-steps 20 --batch-size 4
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from compare_lab import config
from compare_lab.data.prepare_smoltalk import load_jsonl, ARTIFACTS_DIR
from compare_lab.data.tokenizer import Tokenizer, DEFAULT_TOKENIZER_DIR
from compare_lab.models.cracked_d import build_model as build_cracked, build_config as cracked_config
from compare_lab.models.sliced_d import SlicedD, SlicedDConfig
from compare_lab.train.checkpoint import save_checkpoint, load_checkpoint
from compare_lab.train.lr import lr_at_step
from compare_lab.train.pretrain import build_optimizer, masked_ce, _config_dict, _save_curves
from compare_lab.train.tasks import (StatusWriter, stop_requested, clear_stop,
                                     mark_error, acquire_lock, release_lock)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"


# ---- example builders (fixed order, loss only on assistant tokens) ----

def iter_assistant_turns(convs, tok, block_size):
    """The single source of truth for WHICH training examples exist: one per
    assistant turn, yielding (history_ids_ending_in_assistant_marker, response_ids).

    Both architectures derive their examples from this, so they get exactly the
    same examples in the same order -- the fairness requirement. The length
    filter is the stricter of the two (history + response must fit block_size,
    which also implies each part fits on its own), applied to BOTH so neither
    keeps a turn the other drops. Turns that don't fit are discarded, never
    truncated (a truncated answer would train the model on half a response).
    """
    for row in convs:
        msgs = row["messages"]
        for t, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            history_ids, _ = tok.build_chat(msgs[:t], add_generation_prompt=True)
            resp = tok.encode(m["content"]) + [tok.eos_id]
            if len(resp) <= 1:  # empty response, only <eos>
                continue
            if len(history_ids) + len(resp) > block_size:
                continue
            yield history_ids, resp


def build_cracked_examples(convs, tok, block_size):
    """Decoder-only packing of the shared turns: [history || response] as one
    causal sequence, loss only on the response and its <eos>.

    Note this is ONE EXAMPLE PER ASSISTANT TURN, not per conversation. Packing a
    whole conversation at once would be the natural decoder-only choice, but
    Sliced-D structurally needs one example per turn (its decoder generates a
    single response), so with the same number of optimizer steps the decoder-only
    model would get ~50% more gradient over assistant tokens and run more epochs.
    The price of fairness is some redundant recomputation of shared history.
    """
    out = []
    for history_ids, resp in iter_assistant_turns(convs, tok, block_size):
        ids = history_ids + resp
        mask = [0] * len(history_ids) + [1] * len(resp)
        out.append((ids, mask))
    return out


def build_sliced_examples(convs, tok, block_size):
    """Encoder-decoder packing of the same shared turns: the encoder reads the
    history (ending in the <|assistant|> marker), the decoder generates the
    response and its <eos>."""
    return [(history_ids, resp) for history_ids, resp in iter_assistant_turns(convs, tok, block_size)]


def collate_cracked(batch, pad_id, device):
    T = max(len(ids) for ids, _ in batch)
    B = len(batch)
    x = torch.full((B, T - 1), pad_id, dtype=torch.long)
    y = torch.full((B, T - 1), pad_id, dtype=torch.long)
    lm = torch.zeros((B, T - 1), dtype=torch.float32)
    for i, (ids, mask) in enumerate(batch):
        n = len(ids)
        x[i, :n - 1] = torch.tensor(ids[:-1])
        y[i, :n - 1] = torch.tensor(ids[1:])
        lm[i, :n - 1] = torch.tensor(mask[1:], dtype=torch.float32)
    return x.to(device), y.to(device), lm.to(device)


def collate_sliced(batch, bos_id, pad_id, device):
    Ts = max(len(s) for s, _ in batch)
    Tt = max(len(r) for _, r in batch)
    B = len(batch)
    src = torch.full((B, Ts), pad_id, dtype=torch.long)
    src_pad = torch.ones((B, Ts), dtype=torch.bool)
    tin = torch.full((B, Tt), pad_id, dtype=torch.long)
    tout = torch.full((B, Tt), pad_id, dtype=torch.long)
    lm = torch.zeros((B, Tt), dtype=torch.float32)
    for i, (s, r) in enumerate(batch):
        src[i, :len(s)] = torch.tensor(s)
        src_pad[i, :len(s)] = False
        tin[i, 0] = bos_id
        tin[i, 1:len(r)] = torch.tensor(r[:-1])
        tout[i, :len(r)] = torch.tensor(r)
        lm[i, :len(r)] = 1.0
    return (src.to(device), src_pad.to(device), tin.to(device), tout.to(device), lm.to(device))


def _batches(examples, batch_size, start_batch, max_steps):
    """Yield fixed-order batches, cycling through the data until max_steps."""
    n = len(examples)
    nb = max(1, n // batch_size)
    step = start_batch
    while step < max_steps:
        b = step % nb
        yield step, examples[b * batch_size:(b + 1) * batch_size]
        step += 1


@torch.no_grad()
def evaluate(arch, model, examples, batch_size, tok, device, max_batches: int = 12):
    """Mean loss over assistant tokens on held-out conversations -- without this
    there is no way to tell whether the fine-tuning is overfitting, and no
    evidence for the chosen number of steps."""
    if not examples:
        return None
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for b in range(min(max_batches, max(1, len(examples) // batch_size))):
        batch = examples[b * batch_size:(b + 1) * batch_size]
        if not batch:
            break
        if arch in ("cracked", "cracked_full"):
            x, y, lm = collate_cracked(batch, tok.pad_id, device)
            logits, _ = model(x)
            loss, ntok = masked_ce(logits, y, lm)
        else:
            src, src_pad, tin, tout, lm = collate_sliced(batch, tok.bos_id, tok.pad_id, device)
            logits, _ = model(src, tin, src_key_padding_mask=src_pad)
            loss, ntok = masked_ce(logits, tout, lm)
        tot_loss += loss.item() * int(ntok)
        tot_tok += int(ntok)
    model.train()
    return tot_loss / max(1, tot_tok)


def train(args) -> dict:
    """Wrapper that records a crash in status.json before re-raising (see the
    same wrapper in pretrain.py)."""
    task_name = f"finetune_{args.arch}"
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
    convs = load_jsonl(Path(args.finetune_dir) / "train.jsonl")

    test_path = Path(args.finetune_dir) / "test.jsonl"
    test_convs = load_jsonl(test_path) if test_path.exists() else []

    builder = (build_cracked_examples if args.arch in ("cracked", "cracked_full")
               else build_sliced_examples)
    if args.arch in ("cracked", "cracked_full"):
        model = build_cracked(cracked_config(vocab_size=tok.vocab_size, block_size=args.block_size)).to(device)
    else:
        model = SlicedD(SlicedDConfig(vocab_size=tok.vocab_size, max_src_len=args.block_size,
                                      max_tgt_len=args.block_size)).to(device)
    examples = builder(convs, tok, args.block_size)
    test_examples = builder(test_convs, tok, args.block_size)

    if args.pretrained:
        # A missing pretrained checkpoint is a hard error, never a silent
        # fallback to random weights: that failure mode produced a
        # fine-tuned-from-noise deliverable with nothing in the logs to show it
        # (it is what happened when a paused pretrain fell through to this stage).
        pretrained_path = Path(args.pretrained)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"pretrained checkpoint {pretrained_path} not found -- refusing to fine-tune from "
                f"random weights. Pass --pretrained '' explicitly if that is really what you want.")
        load_checkpoint(pretrained_path, model, device=device, restore_rng=False)
        print(f"loaded pretrained weights from {args.pretrained}")
    else:
        print("WARNING: no --pretrained given, fine-tuning from RANDOM weights")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    use_amp = device == "cuda" and args.dtype != "float32"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    task_name = f"finetune_{args.arch}"
    ckpt_path = CHECKPOINT_DIR / args.arch / "finetune.pt"
    step, active_seconds_before = 0, 0.0
    history: list[dict] = []
    resumed_running_loss = None
    if ckpt_path.exists() and not args.no_resume:
        info = load_checkpoint(ckpt_path, model, optimizer, scaler, device=device)
        step = info["step"]
        active_seconds_before = info.get("active_seconds", 0.0)
        # Restore the loss curve so a resume CONTINUES it instead of starting a
        # fresh one that overwrites the earlier segment on the next save.
        history = list(info.get("history") or [])
        resumed_running_loss = info.get("running_loss")

        prev = info.get("train_hparams")
        if prev:
            now = {"batch_size": args.batch_size, "max_steps": args.max_steps,
                   "lr": args.lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps,
                   "block_size": args.block_size}
            drift = {k: (prev[k], now[k]) for k in prev if k in now and prev[k] != now[k]}
            if drift:
                raise RuntimeError(
                    f"refusing to resume finetune {args.arch} with different hyperparameters: "
                    f"{drift} (saved vs now). batch_size and max_steps in particular change "
                    f"which examples each step sees and the LR schedule; a lower max_steps "
                    f"than the saved step would train nothing at all while still reporting "
                    f"success. Relaunch with the original values, or use --no-resume.")
        if step >= args.max_steps:
            raise RuntimeError(
                f"checkpoint is already at step {step} but --max-steps is {args.max_steps}: "
                f"there is nothing left to train. This would silently write "
                f"{args.arch}/finetune_final.pt with zero steps of work. Raise --max-steps "
                f"or use --no-resume to start over.")

        print(f"resumed finetune {args.arch} at step {step}, "
              f"{active_seconds_before/3600:.2f}h of prior active work time, "
              f"{len(history)} curve points restored")

    clear_stop(task_name)
    status = StatusWriter(task_name, args.arch, args.max_steps, args.max_steps * args.batch_size,
                          active_seconds_before=active_seconds_before)
    log_path = status.dir / f"{task_name}.log"
    running_loss = resumed_running_loss  # keep the EMA warm across a pause
    last_ckpt_time = time.time()
    t0 = time.time()
    tokens_at_segment_start = step * args.batch_size
    final_state = "finished"

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def checkpoint_extra() -> dict:
        return {
            "active_seconds": status.active_seconds,
            "history": history,
            "running_loss": running_loss,
            "train_hparams": {"batch_size": args.batch_size, "max_steps": args.max_steps,
                              "lr": args.lr, "min_lr": args.min_lr,
                              "warmup_steps": args.warmup_steps, "block_size": args.block_size},
        }

    log(f"start arch={args.arch} device={device} amp={use_amp}/{args.dtype} "
        f"examples={len(examples)} test_examples={len(test_examples)} "
        f"max_steps={args.max_steps} batch_size={args.batch_size}")

    pbar = tqdm(total=args.max_steps, initial=step, desc=f"finetune {args.arch}")
    for st, batch in _batches(examples, args.batch_size, step, args.max_steps):
        if not batch:
            continue
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            if args.arch in ("cracked", "cracked_full"):
                x, y, lm = collate_cracked(batch, tok.pad_id, device)
                logits, _ = model(x)
                loss, _ = masked_ce(logits, y, lm)
            else:
                src, src_pad, tin, tout, lm = collate_sliced(batch, tok.bos_id, tok.pad_id, device)
                logits, _ = model(src, tin, src_key_padding_mask=src_pad)
                loss, _ = masked_ce(logits, tout, lm)

        lr = lr_at_step(st, args.lr, args.min_lr, args.warmup_steps, args.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss = loss.item() if running_loss is None else 0.9 * running_loss + 0.1 * loss.item()
        step = st + 1
        pbar.update(1)
        tokens_done = step * args.batch_size
        tok_s = (tokens_done - tokens_at_segment_start) / max(1e-9, time.time() - t0)
        status.update(step, tokens_done, train_loss=running_loss, tokens_per_sec=tok_s)
        eval_interval = getattr(args, "eval_interval", 0)
        is_eval_step = bool(eval_interval and step % eval_interval == 0)
        val = None
        if is_eval_step:
            val = evaluate(args.arch, model, test_examples, args.batch_size, tok, device)
            status.update(step, tokens_done, train_loss=running_loss, val_loss=val,
                          tokens_per_sec=tok_s, force=True)
            log(f"  [eval] step {step} val_loss {val:.4f}" if val is not None
                else f"  [eval] step {step} (no test examples)")
        if step % args.log_interval == 0 or is_eval_step:
            history.append({"step": step, "train_loss": running_loss, "val_loss": val,
                            "tokens_per_sec": round(tok_s, 1)})
            if step % args.log_interval == 0:
                log(f"step {step}/{args.max_steps} loss {running_loss:.4f} lr {lr:.2e} {tok_s:.0f} tok/s")

        if (time.time() - last_ckpt_time) / 60.0 >= args.ckpt_interval_min:
            ok = save_checkpoint(ckpt_path, model, optimizer, step, step,
                                 _config_dict(args.arch, model), scaler=scaler,
                                 extra=checkpoint_extra())
            last_ckpt_time = time.time()
            status.update(step, tokens_done, last_ckpt_step=step if ok else None, force=True)
            log(f"  checkpoint @ step {step}" if ok else
                f"  checkpoint @ step {step} DEFERRED (destination busy) -- continuing")

        if stop_requested(task_name):
            final_state = "paused"
            log(f"stop requested -> paused @ step {step}")
            break  # the single save below covers this path too
    pbar.close()

    if not save_checkpoint(ckpt_path, model, optimizer, step, step,
                           _config_dict(args.arch, model), scaler=scaler,
                           extra=checkpoint_extra()):
        log(f"WARNING: final checkpoint could not be renamed into place; it is at "
            f"{ckpt_path}.tmp -- rename it manually before resuming.")
    # Only publish *_final.pt if the step target was actually reached. Writing it
    # otherwise makes run_pipeline skip fine-tuning forever, so an under-trained
    # model would become the permanent deliverable.
    if final_state == "finished" and step >= args.max_steps:
        save_checkpoint(CHECKPOINT_DIR / args.arch / "finetune_final.pt", model, optimizer,
                        step, step, _config_dict(args.arch, model), scaler=scaler,
                        extra=checkpoint_extra())
    elif final_state == "finished":
        final_state = "paused"  # incomplete: must be resumed, not treated as done
        log(f"WARNING: loop ended at step {step} of {args.max_steps} without a stop request; "
            f"NOT writing finetune_final.pt. Reporting 'paused' so the pipeline resumes it.")
    _save_curves(RESULTS_DIR / f"finetune_loss_{args.arch}", history, f"finetune {args.arch}")
    status.update(step, step * args.batch_size, force=True)
    status.set_state(final_state)
    log(f"done state={final_state} step={step} active_work={status.active_seconds/3600:.2f}h")
    return {"state": final_state, "step": step, "n_examples": len(examples),
           "active_seconds": status.active_seconds}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=["cracked", "cracked_full", "sliced"])
    # Every default comes from the frozen config. When they were hand-written
    # here they disagreed with it (batch 8 vs 16, max_steps 2000 vs 20000), and a
    # CLI resume past step 2000 then yielded ZERO batches, left final_state as
    # "finished", and wrote finetune_final.pt -- so the pipeline skipped
    # fine-tuning forever and the deliverable was silently under-trained.
    p.add_argument("--block-size", type=int, default=config.BLOCK_SIZE)
    p.add_argument("--batch-size", type=int, default=config.FINETUNE_BATCH_SIZE)
    p.add_argument("--max-steps", type=int, default=config.FINETUNE_MAX_STEPS)
    p.add_argument("--lr", type=float, default=config.FINETUNE["lr"])
    p.add_argument("--min-lr", type=float, default=config.FINETUNE["min_lr"])
    p.add_argument("--warmup-steps", type=int, default=config.FINETUNE["warmup_steps"])
    p.add_argument("--weight-decay", type=float, default=config.FINETUNE["weight_decay"])
    p.add_argument("--dtype", default=config.FINETUNE["dtype"],
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--log-interval", type=int, default=config.LOG_INTERVAL)
    p.add_argument("--eval-interval", type=int, default=config.FINETUNE_EVAL_INTERVAL)
    p.add_argument("--ckpt-interval-min", type=float, default=config.CKPT_INTERVAL_MIN)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--pretrained", default="", help="pretrained checkpoint to start from")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    p.add_argument("--finetune-dir", default=str(ARTIFACTS_DIR))
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
