"""Fase C.4: "Sliced" -- an encoder-decoder Transformer trained from scratch
on the same instruction/conversation data as Cracked (C.1a Alpaca + C.1b
oasst1), using Fase A's BPE tokenizer.

Reuses the architecture classes from depth_lab.models.encoder_decoder
(EncoderDecoderTransformer, EncDecConfig, sinusoidal PE, standard MHA,
LayerNorm, ReLU MLP -- the classic "Attention Is All You Need" recipe
already hand-built in Fase B) as-is: those classes have zero coupling to
depth_lab's character-level tokenizer, they only ever operate on token id
tensors and config numbers. What's new here is the data/training glue for
Fase A's BPE vocabulary instead of depth_lab's CharTokenizer.

Unlike Cracked, Sliced trains from scratch -- there's no pretrained
encoder-decoder checkpoint to start from, so it does *not* benefit from
Fase A's WikiText pretraining the way Cracked does. That asymmetry is
real and reported honestly in the Fase C plan, not hidden.

Usage:
    python -m coconut_lab.models.sliced --max-steps 3000
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from coconut_lab.data.loader import Seq2SeqDataset
from coconut_lab.data.prepare_conversations import ARTIFACTS_DIR as CONV_ARTIFACTS_DIR
from coconut_lab.data.prepare_conversations import format_turns, load_jsonl as load_conv_jsonl
from coconut_lab.data.prepare_instructions import ARTIFACTS_DIR, load_jsonl
from depth_lab.models.encoder_decoder import EncDecConfig, EncoderDecoderTransformer
from mini_llm.tokenizer import BPETokenizer
from mini_llm.tokenizer.bpe import EOT_TOKEN

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HF_REPO = "davidmorgado/coconut-mini-llm"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

SRC_BLOCK_SIZE = 384
TGT_BLOCK_SIZE = 192

SAMPLE_PROMPTS = ["Name three colors.", "What is the capital of France?"]


def alpaca_to_seq2seq(examples: list[dict]) -> list[dict]:
    return [{"src": ex["prompt"], "tgt": ex["response"]} for ex in examples]


def conversations_to_seq2seq(examples: list[dict]) -> list[dict]:
    """One example per conversation: encoder reads every turn except the
    last, decoder generates the final assistant turn."""
    out = []
    for ex in examples:
        turns = ex["turns"]
        if len(turns) < 2 or turns[-1]["role"] != "assistant":
            continue
        out.append({"src": format_turns(turns[:-1]), "tgt": turns[-1]["text"]})
    return out


def load_tokenizer() -> BPETokenizer:
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    return BPETokenizer(vocab_path, merges_path)


def build_config(vocab_size: int, n_layer: int = 3, d_model: int = 256, n_head: int = 4,
                  d_ff: int = 1024) -> EncDecConfig:
    return EncDecConfig(vocab_size=vocab_size, d_model=d_model, n_head=n_head, n_layer=n_layer, d_ff=d_ff,
                         max_src_len=SRC_BLOCK_SIZE, max_tgt_len=TGT_BLOCK_SIZE + 1)


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


def train_steps(model: EncoderDecoderTransformer, train_ds: Seq2SeqDataset, optimizer: torch.optim.Optimizer,
                 device: str, max_steps: int, batch_size: int, pad_id: int, log_interval: int = 0,
                 grad_accum_steps: int = 1, dtype: str = "bfloat16") -> None:
    """See coconut_lab.models.cracked.train_steps's docstring for why
    grad_accum_steps/dtype exist -- same standard fix, same
    CUDA-only/backward-compatible defaults."""
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    use_amp = device == "cuda" and amp_dtype != torch.float32

    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    data_iter = iter(loader)
    step = 0
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        last_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                src, tgt_in, tgt_out = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                src, tgt_in, tgt_out = next(data_iter)
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_pad_mask = src == pad_id
            tgt_pad_mask = tgt_in == pad_id
            # Position 0 of tgt_in is always the real BOS token (Seq2SeqDataset
            # prepends it), never padding -- but BOS and pad share the same id
            # in this project's BPE vocab (no dedicated tokens for either), so
            # the equality check above misclassifies it. Left uncorrected, the
            # decoder's first self-attention query has zero valid keys under
            # causal+padding masking and can never learn to predict the first
            # output token from real context.
            tgt_pad_mask[:, 0] = False

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits = model(src, tgt_in, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                                        ignore_index=pad_id) / grad_accum_steps
            loss.backward()
            last_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if log_interval and step % log_interval == 0:
            print(f"step {step:6d} | loss {last_loss:.4f}")
        step += 1


@torch.no_grad()
def generate_response(model: EncoderDecoderTransformer, tokenizer: BPETokenizer, src_text: str, device: str,
                       max_new_tokens: int = 150, temperature: float = 0.0, top_k: int | None = None) -> str:
    model.eval()
    pad_id = tokenizer.encode(EOT_TOKEN)[0]
    src_ids = tokenizer.encode(src_text)[:SRC_BLOCK_SIZE]
    src_ids = src_ids + [pad_id] * (SRC_BLOCK_SIZE - len(src_ids))
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad_mask = src == pad_id

    out_ids: list[int] = []
    for grown in model.generate_stream(src, pad_id, max_new_tokens, src_key_padding_mask=src_pad_mask,
                                        temperature=temperature, top_k=top_k):
        next_id = grown[0, -1].item()
        if next_id == pad_id:
            break
        out_ids.append(next_id)
    model.train()
    return tokenizer.decode(out_ids)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
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

    tokenizer = load_tokenizer()
    config = build_config(tokenizer.vocab_size, args.n_layer, args.d_model, args.n_head, args.d_ff)
    model = EncoderDecoderTransformer(config).to(args.device)
    print(f"model: {model.num_parameters():,} parameters, config={config}")

    alpaca_train = alpaca_to_seq2seq(load_jsonl(ARTIFACTS_DIR / "alpaca_train.jsonl"))
    alpaca_val = alpaca_to_seq2seq(load_jsonl(ARTIFACTS_DIR / "alpaca_val.jsonl"))
    conv_train = conversations_to_seq2seq(load_conv_jsonl(CONV_ARTIFACTS_DIR / "oasst1_train.jsonl"))
    conv_val = conversations_to_seq2seq(load_conv_jsonl(CONV_ARTIFACTS_DIR / "oasst1_val.jsonl"))

    train_ds_a = Seq2SeqDataset(alpaca_train, tokenizer, SRC_BLOCK_SIZE, TGT_BLOCK_SIZE)
    train_ds_b = Seq2SeqDataset(conv_train, tokenizer, SRC_BLOCK_SIZE, TGT_BLOCK_SIZE)
    val_ds_a = Seq2SeqDataset(alpaca_val, tokenizer, SRC_BLOCK_SIZE, TGT_BLOCK_SIZE)
    val_ds_b = Seq2SeqDataset(conv_val, tokenizer, SRC_BLOCK_SIZE, TGT_BLOCK_SIZE)
    print(f"alpaca train: {len(train_ds_a)}/{len(alpaca_train)} fit | oasst1 train: {len(train_ds_b)}/{len(conv_train)} fit")
    pad_id = train_ds_a.pad_id
    train_ds = torch.utils.data.ConcatDataset([train_ds_a, train_ds_b])
    val_ds = torch.utils.data.ConcatDataset([val_ds_a, val_ds_b])
    print(f"combined train: {len(train_ds)} | combined val: {len(val_ds)}")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_steps(model, train_ds, optimizer, args.device, args.max_steps, args.batch_size, pad_id, args.log_interval,
                args.grad_accum_steps, args.dtype)
    elapsed = time.time() - t0
    print(f"\ntraining took {elapsed:.1f}s for {args.max_steps} steps ({elapsed / args.max_steps:.3f}s/step)")
    if args.device == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    pad_id = train_ds_a.pad_id
    val_loss, n_batches = 0.0, 0
    with torch.no_grad():
        model.eval()
        for src, tgt_in, tgt_out in torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size):
            src, tgt_in, tgt_out = src.to(args.device), tgt_in.to(args.device), tgt_out.to(args.device)
            logits = model(src, tgt_in, src_key_padding_mask=src == pad_id, tgt_key_padding_mask=tgt_in == pad_id)
            val_loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                                         ignore_index=pad_id).item()
            n_batches += 1
        model.train()
    print(f"val loss: {val_loss / max(n_batches, 1):.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config), "max_steps": args.max_steps},
               out_dir / "sliced.pt")
    print(f"done. checkpoint saved to {out_dir / 'sliced.pt'}")

    print("\nsample generations:")
    for prompt in SAMPLE_PROMPTS:
        response = generate_response(model, tokenizer, prompt, args.device)
        print(f"  instruction: {prompt!r}")
        print(f"  response: {response!r}\n")


if __name__ == "__main__":
    main()
