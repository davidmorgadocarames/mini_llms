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
import sys
import time
from pathlib import Path

# Generated text can contain arbitrary Unicode (including odd byte-level BPE
# artifacts); stdout defaults to the console's codepage when redirected to a
# file on Windows, which can't represent all of it and crashes the whole
# script -- reconfigure to UTF-8 with lossy fallback instead of failing.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from coconut_lab.data.loader import ConversationDataset, InstructionDataset
from coconut_lab.data.prepare_instructions import ARTIFACTS_DIR, load_jsonl
from mini_llm.model import GPT
from mini_llm.tokenizer import BPETokenizer
from mini_llm.tokenizer.bpe import EOT_TOKEN
from mini_llm.train.checkpoint import load_checkpoint, save_checkpoint

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
                 device: str, max_steps: int, batch_size: int, log_interval: int = 0,
                 grad_accum_steps: int = 1, dtype: str = "bfloat16",
                 checkpoint_path: Path | None = None, checkpoint_interval: int = 0) -> None:
    """grad_accum_steps splits each optimizer step into that many smaller
    mini-batches (gradients summed, not overwritten) -- the standard fix for
    a batch that would otherwise need more VRAM than fits, at the same
    effective batch size (mini_llm/train/train.py already does this for
    Fase A; Fase C's scripts hadn't picked it up). autocast is only enabled
    on CUDA, so CPU-run tests are completely unaffected -- same exact
    behavior and results as before this was added.

    checkpoint_path/checkpoint_interval (both no-ops by default, same
    backward-compatible pattern): if checkpoint_path already exists when
    training starts, resumes from it (model+optimizer+step) instead of
    starting at 0; every checkpoint_interval steps (and at the end) the same
    path is overwritten with the current state. Needed for Fase C's k-fold
    (25 unattended training runs) so a crash/power loss mid-run doesn't lose
    everything -- see mini_llm.train.checkpoint."""
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    use_amp = device == "cuda" and amp_dtype != torch.float32

    start_step = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        start_step = load_checkpoint(checkpoint_path, model, optimizer, device)
        print(f"resumed from {checkpoint_path} at step {start_step}")

    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    data_iter = iter(loader)
    step = start_step
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        last_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                x, y, y_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y, y_mask = next(data_iter)
            x, y, y_mask = x.to(device), y.to(device), y_mask.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(x)
                loss = masked_loss(logits, y, y_mask) / grad_accum_steps
            loss.backward()
            last_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if log_interval and step % log_interval == 0:
            print(f"step {step:6d} | loss {last_loss:.4f}")
        step += 1

        if checkpoint_path is not None and checkpoint_interval and step % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, model, optimizer, step, model.config)

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, optimizer, step, model.config)


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
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=str(CHECKPOINT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)

    model, tokenizer = load_base_model(args.device)
    config = model.config
    print(f"base model: {model.num_parameters():,} parameters, config={config}")

    alpaca_train = load_jsonl(ARTIFACTS_DIR / "alpaca_train.jsonl")
    alpaca_val = load_jsonl(ARTIFACTS_DIR / "alpaca_val.jsonl")
    instruction_train_ds = InstructionDataset(alpaca_train, tokenizer, block_size=config.block_size)
    instruction_val_ds = InstructionDataset(alpaca_val, tokenizer, block_size=config.block_size)
    print(f"alpaca train: {len(instruction_train_ds)}/{len(alpaca_train)} examples fit block_size={config.block_size}")
    print(f"alpaca val:   {len(instruction_val_ds)}/{len(alpaca_val)} examples fit block_size={config.block_size}")

    # prepare_conversations.py writes into the same coconut_lab/data/artifacts/ dir
    oasst1_train = load_jsonl(ARTIFACTS_DIR / "oasst1_train.jsonl")
    oasst1_val = load_jsonl(ARTIFACTS_DIR / "oasst1_val.jsonl")
    conversation_train_ds = ConversationDataset(oasst1_train, tokenizer, block_size=config.block_size)
    conversation_val_ds = ConversationDataset(oasst1_val, tokenizer, block_size=config.block_size)
    print(f"oasst1 train: {len(conversation_train_ds)}/{len(oasst1_train)} conversations fit block_size={config.block_size}")
    print(f"oasst1 val:   {len(conversation_val_ds)}/{len(oasst1_val)} conversations fit block_size={config.block_size}")

    train_ds = torch.utils.data.ConcatDataset([instruction_train_ds, conversation_train_ds])
    val_ds = torch.utils.data.ConcatDataset([instruction_val_ds, conversation_val_ds])
    print(f"combined train: {len(train_ds)} examples | combined val: {len(val_ds)} examples")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_steps(model, train_ds, optimizer, args.device, args.max_steps, args.batch_size, args.log_interval,
                args.grad_accum_steps, args.dtype)
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

    # Save before printing sample generations -- generated text is
    # unpredictable and must never be able to cost us the trained weights.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config), "max_steps": args.max_steps},
               out_dir / "cracked.pt")
    print(f"done. checkpoint saved to {out_dir / 'cracked.pt'}")

    print("\nsample generations:")
    for prompt in SAMPLE_PROMPTS:
        response = generate_response(model, tokenizer, prompt, args.device)
        instruction = prompt.split("### Instruction:\n", 1)[1].split("\n\n### Response:")[0]
        print(f"  instruction: {instruction!r}")
        print(f"  response: {response!r}\n")


if __name__ == "__main__":
    main()
