"""Training loop for the Fase A mini-LLM.

Usage:
    python -m mini_llm.train.train --max-steps 2000 --batch-size 32
"""

import argparse
import math
import time
from pathlib import Path

import torch

from mini_llm.data.loader import BinaryTokenDataset
from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer import BPETokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float,
                     betas: tuple[float, float]) -> torch.optim.Optimizer:
    """Only weight-decay the 2D+ parameters (linear/embedding weights); leave
    RMSNorm gains undecayed, as is standard practice for Transformers."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def lr_at_step(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, dataset: BinaryTokenDataset, batch_size: int,
                   device: str, eval_iters: int = 20) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = dataset.get_batch(batch_size, device=device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-kv-head", type=int, default=2)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--out-dir", default=str(CHECKPOINT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)

    tokenizer = BPETokenizer.from_dir(DATA_DIR / "tokenizer")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
    )
    model = GPT(config).to(args.device)
    if args.compile:
        model = torch.compile(model)
    print(f"model: {model.num_parameters():,} parameters, config={config}")

    train_ds = BinaryTokenDataset(DATA_DIR / "train.bin", args.block_size)
    val_ds = BinaryTokenDataset(DATA_DIR / "val.bin", args.block_size)

    optimizer = build_optimizer(model, args.max_lr, args.weight_decay, betas=(0.9, 0.95))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    use_amp = args.device == "cuda" and dtype != torch.float32
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    t0 = time.time()
    for step in range(args.max_steps):
        lr = lr_at_step(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum_steps):
            x, y = train_ds.get_batch(args.batch_size, device=args.device)
            with torch.autocast(device_type=args.device, dtype=dtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0:
            dt = time.time() - t0
            toks_per_sec = args.batch_size * args.block_size * args.grad_accum_steps * args.log_interval / max(dt, 1e-9)
            print(f"step {step:6d} | loss {loss.item() * args.grad_accum_steps:.4f} | lr {lr:.2e} | {toks_per_sec:,.0f} tok/s")
            t0 = time.time()

        if step > 0 and step % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_ds, args.batch_size, args.device, args.eval_iters)
            print(f"  [eval] step {step} val_loss {val_loss:.4f}")
            torch.save(
                {"model": model.state_dict(), "config": config, "step": step},
                out_dir / "ckpt.pt",
            )

    torch.save({"model": model.state_dict(), "config": config, "step": args.max_steps},
               out_dir / "ckpt.pt")
    print(f"done. checkpoint saved to {out_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
