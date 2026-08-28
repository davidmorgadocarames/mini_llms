"""Fase C.2: "Cracked" -- fine-tunes the existing Fase A Coconut checkpoint
on Alpaca instructions (coconut_lab.data.prepare_instructions) so it
responds to direct requests instead of only continuing WikiText-style
prose. Full fine-tuning, not LoRA: at 26.4M parameters the whole model
comfortably fits in an 8GB GPU, so there's no need for parameter-efficient
tricks aimed at much larger models.

Saves to a new checkpoint (coconut_lab/checkpoints/cracked.pt), never
overwrites the Fase A checkpoint -- Cracked is a derived artifact, not a
replacement.

Usage:
    python -m coconut_lab.models.cracked --max-steps 2000
"""

import argparse
import dataclasses
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from coconut_lab.data.loader import InstructionDataset
from coconut_lab.data.prepare_instructions import ARTIFACTS_DIR, load_jsonl
from mini_llm.model import GPT
from mini_llm.tokenizer import BPETokenizer
from mini_llm.tokenizer.bpe import EOT_TOKEN

HF_REPO = "davidmorgado/coconut-mini-llm"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

SAMPLE_PROMPTS = [
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\nName three colors.\n\n### Response:\n",
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
]


def load_base_model(device: str) -> tuple[GPT, BPETokenizer]:
    ckpt_path = hf_hub_download(HF_REPO, "ckpt.pt")
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")

    tokenizer = BPETokenizer(vocab_path, merges_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, tokenizer


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


def masked_loss(logits: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
    loss_per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    loss_per_token = loss_per_token.reshape(y.shape)
    return (loss_per_token * y_mask).sum() / y_mask.sum().clamp_min(1.0)


def train_steps(model: GPT, train_ds: InstructionDataset, optimizer: torch.optim.Optimizer,
                 device: str, max_steps: int, batch_size: int, log_interval: int = 0) -> None:
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    step = 0
    while step < max_steps:
        for x, y, y_mask in loader:
            if step >= max_steps:
                break
            x, y, y_mask = x.to(device), y.to(device), y_mask.to(device)
            logits, _ = model(x)
            loss = masked_loss(logits, y, y_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if log_interval and step % log_interval == 0:
                print(f"step {step:6d} | loss {loss.item():.4f}")
            step += 1


@torch.no_grad()
def generate_response(model: GPT, tokenizer: BPETokenizer, prompt: str, device: str,
                       max_new_tokens: int = 150, temperature: float = 0.8, top_k: int = 50) -> str:
    model.eval()
    eot_id = tokenizer.encode(EOT_TOKEN)[0]
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out_ids: list[int] = []
    for grown in model.generate_stream(idx, max_new_tokens, temperature=temperature, top_k=top_k):
        next_id = grown[0, -1].item()
        if next_id == eot_id:
            break
        out_ids.append(next_id)
    model.train()
    return tokenizer.decode(out_ids)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-5)  # small: fine-tuning, not pretraining from scratch
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=str(CHECKPOINT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)

    model, tokenizer = load_base_model(args.device)
    config = model.config
    print(f"base model: {model.num_parameters():,} parameters, config={config}")

    train_examples = load_jsonl(ARTIFACTS_DIR / "alpaca_train.jsonl")
    val_examples = load_jsonl(ARTIFACTS_DIR / "alpaca_val.jsonl")
    train_ds = InstructionDataset(train_examples, tokenizer, block_size=config.block_size)
    val_ds = InstructionDataset(val_examples, tokenizer, block_size=config.block_size)
    print(f"train: {len(train_ds)}/{len(train_examples)} examples fit block_size={config.block_size}")
    print(f"val:   {len(val_ds)}/{len(val_examples)} examples fit block_size={config.block_size}")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_steps(model, train_ds, optimizer, args.device, args.max_steps, args.batch_size, args.log_interval)
    elapsed = time.time() - t0

    print(f"\ntraining took {elapsed:.1f}s for {args.max_steps} steps ({elapsed / args.max_steps:.3f}s/step)")
    if args.device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak VRAM: {peak_gb:.2f} GB")

    val_loss = 0.0
    with torch.no_grad():
        model.eval()
        loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        n_batches = 0
        for x, y, y_mask in loader:
            x, y, y_mask = x.to(args.device), y.to(args.device), y_mask.to(args.device)
            logits, _ = model(x)
            val_loss += masked_loss(logits, y, y_mask).item()
            n_batches += 1
        model.train()
    print(f"val loss: {val_loss / max(n_batches, 1):.4f}")

    print("\nsample generations:")
    for prompt in SAMPLE_PROMPTS:
        response = generate_response(model, tokenizer, prompt, args.device)
        instruction = prompt.split("### Instruction:\n", 1)[1].split("\n\n### Response:")[0]
        print(f"  instruction: {instruction!r}")
        print(f"  response: {response!r}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config), "max_steps": args.max_steps},
               out_dir / "cracked.pt")
    print(f"done. checkpoint saved to {out_dir / 'cracked.pt'}")


if __name__ == "__main__":
    main()
