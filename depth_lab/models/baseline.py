"""Baseline decoder-only architecture for Fase B: a thin wrapper around
mini_llm.model.GPT (RoPE, RMSNorm, SwiGLU, GQA) with a tiny config, trained on
the same "<expr> => <value>" next-token-prediction framing used by every
architecture in this comparison. This is the "free" comparison arm -- zero new
model code, exactly the recipe the He 2026 paper shows fails to generalize to
depths beyond training.

Usage:
    python -m depth_lab.models.baseline --max-steps 2000
"""

import argparse
import dataclasses
from pathlib import Path

import torch
import torch.nn.functional as F

from depth_lab.data.build_dataset import ARTIFACTS_DIR, TEST_DEPTHS, load_jsonl
from depth_lab.data.loader import ExprDataset
from depth_lab.tokenizer import CharTokenizer
from mini_llm.model import GPT, GPTConfig

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
BLOCK_SIZE = 384  # comfortably covers the confirmed max expr length (335 chars) + " => False<eos>"


def build_config(vocab_size: int, block_size: int = BLOCK_SIZE, n_layer: int = 4,
                  n_embd: int = 256, n_head: int = 4, n_kv_head: int = 2) -> GPTConfig:
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
    )


@torch.no_grad()
def evaluate_exact_match(model: GPT, tokenizer: CharTokenizer, examples: list[dict],
                          device: str, max_new_tokens: int = 8) -> float:
    """Feeds "<bos><expr> => " and generates greedily until <eos> (or
    max_new_tokens), then compares the decoded completion against the true
    value as a string -- exact-match, the same metric the paper reports."""
    model.eval()
    dataset = ExprDataset(examples, tokenizer, block_size=model.config.block_size)
    correct = 0
    for i, ex in enumerate(examples):
        idx = torch.tensor([dataset.prompt_ids(i)], dtype=torch.long, device=device)
        out_ids: list[int] = []
        for grown in model.generate_stream(idx, max_new_tokens, temperature=1e-6):
            next_id = grown[0, -1].item()
            if next_id == tokenizer.eos_id:
                break
            out_ids.append(next_id)
        prediction = tokenizer.decode(out_ids)
        if prediction == str(ex["value"]):
            correct += 1
    model.train()
    return correct / len(examples)


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def train_steps(model: GPT, tokenizer: CharTokenizer, train_ds: ExprDataset, optimizer: torch.optim.Optimizer,
                 device: str, max_steps: int, batch_size: int, log_interval: int = 0) -> None:
    """Shared training loop used both by the full CLI run and by tests that
    just need to overfit a tiny batch."""
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    step = 0
    while step < max_steps:
        for x, y in loader:
            if step >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                    ignore_index=tokenizer.pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if log_interval and step % log_interval == 0:
                print(f"step {step:6d} | loss {loss.item():.4f}")
            step += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="bool", choices=["bool", "arith"])
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-kv-head", type=int, default=2)
    p.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=str(CHECKPOINT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)

    tokenizer = CharTokenizer()
    train_examples = load_jsonl(ARTIFACTS_DIR / f"{args.domain}_train.jsonl")
    val_examples = load_jsonl(ARTIFACTS_DIR / f"{args.domain}_val.jsonl")

    config = build_config(tokenizer.vocab_size, args.block_size, args.n_layer,
                           args.n_embd, args.n_head, args.n_kv_head)
    model = GPT(config).to(args.device)
    print(f"model: {model.num_parameters():,} parameters, config={config}")

    train_ds = ExprDataset(train_examples, tokenizer, args.block_size)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    while step < args.max_steps:
        for x, y in loader:
            if step >= args.max_steps:
                break
            x, y = x.to(args.device), y.to(args.device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                    ignore_index=tokenizer.pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.log_interval == 0:
                print(f"step {step:6d} | loss {loss.item():.4f}")

            if step > 0 and step % args.eval_interval == 0:
                val_acc = evaluate_exact_match(model, tokenizer, val_examples[:200], args.device)
                print(f"  [eval] step {step} val exact-match {val_acc:.3f}")
                torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config), "step": step},
                           out_dir / f"baseline_{args.domain}.pt")
            step += 1

    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config), "step": args.max_steps},
               out_dir / f"baseline_{args.domain}.pt")

    print("\nfinal OOD exact-match by depth:")
    for d in TEST_DEPTHS:
        test_examples = load_jsonl(ARTIFACTS_DIR / f"{args.domain}_test_depth{d}.jsonl")
        acc = evaluate_exact_match(model, tokenizer, test_examples, args.device)
        print(f"  depth {d:2d}: {acc:.3f}")


if __name__ == "__main__":
    main()
