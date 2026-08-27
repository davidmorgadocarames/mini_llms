"""Classic encoder-decoder Transformer (Vaswani et al. 2017, "Attention Is
All You Need") for Fase B -- built from scratch, deliberately NOT reusing
Fase A's modern stack (RoPE/RMSNorm/SwiGLU/GQA), so this is a faithful
reproduction of the original recipe: sinusoidal positional encoding, standard
multi-head attention, post-norm LayerNorm, ReLU MLP.

Architectural bet vs. the decoder-only baseline: the encoder reads the whole
expression *bidirectionally* (no causal mask), so it can build a
representation of a deeply nested expression by looking in both directions at
once, instead of only ever attending backwards through a single growing
string. The decoder then generates "<value><eos>" autoregressively,
cross-attending to that representation. The paper being reproduced in Fase B
only tests decoder-only and its own Looped Locate-and-Replace pipeline --
this encoder-decoder arm is this project's own addition to the comparison.

Usage:
    python -m depth_lab.models.encoder_decoder --max-steps 2000
"""

import argparse
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_lab.data.build_dataset import ARTIFACTS_DIR, TEST_DEPTHS, load_jsonl
from depth_lab.data.loader import Seq2SeqDataset, render_value
from depth_lab.tokenizer import CharTokenizer

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
SRC_BLOCK_SIZE = 384   # matches the baseline: covers the confirmed max expr length (335 chars)
TGT_BLOCK_SIZE = 6     # "False" (5 chars) + <eos> fits with room to spare


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product multi-head attention -- no RoPE, no GQA
    (every head gets its own K/V, unlike Fase A). Used for encoder
    self-attention, decoder self-attention, and decoder cross-attention;
    which one it is depends only on what query/key/value and masks the
    caller passes in."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        q = self.q_proj(query).view(B, Tq, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, self.n_head, self.d_head).transpose(1, 2)

        # sdpa boolean mask convention: True = attend, False = block.
        mask = attn_mask
        if key_padding_mask is not None:
            keep = ~key_padding_mask[:, None, None, :]  # (B, 1, 1, Tk)
            mask = keep if mask is None else (mask & keep)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.n_head * self.d_head)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, causal_mask: torch.Tensor,
                tgt_key_padding_mask: torch.Tensor | None = None,
                memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(
            self.self_attn(x, x, x, attn_mask=causal_mask, key_padding_mask=tgt_key_padding_mask)
        ))
        x = self.norm2(x + self.dropout(
            self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        ))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


@dataclass
class EncDecConfig:
    vocab_size: int
    d_model: int = 256
    n_head: int = 4
    n_layer: int = 3
    d_ff: int = 1024
    max_src_len: int = SRC_BLOCK_SIZE
    max_tgt_len: int = TGT_BLOCK_SIZE + 1
    dropout: float = 0.0

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, config: EncDecConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.register_buffer(
            "src_pe", sinusoidal_positional_encoding(config.max_src_len, config.d_model), persistent=False
        )
        self.register_buffer(
            "tgt_pe", sinusoidal_positional_encoding(config.max_tgt_len, config.d_model), persistent=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config.d_model, config.n_head, config.d_ff, config.dropout) for _ in range(config.n_layer)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config.d_model, config.n_head, config.d_ff, config.dropout) for _ in range(config.n_layer)]
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, src_idx: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = src_idx.size(1)
        x = self.tok_emb(src_idx) * math.sqrt(self.config.d_model) + self.src_pe[:T]
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask)
        return x

    def decode(self, tgt_idx: torch.Tensor, memory: torch.Tensor,
               memory_key_padding_mask: torch.Tensor | None = None,
               tgt_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = tgt_idx.size(1)
        x = self.tok_emb(tgt_idx) * math.sqrt(self.config.d_model) + self.tgt_pe[:T]
        x = self.dropout(x)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt_idx.device))
        for layer in self.decoder_layers:
            x = layer(x, memory, causal_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.lm_head(x)

    def forward(self, src_idx: torch.Tensor, tgt_idx: torch.Tensor,
                src_key_padding_mask: torch.Tensor | None = None,
                tgt_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.encode(src_idx, src_key_padding_mask)
        return self.decode(tgt_idx, memory, src_key_padding_mask, tgt_key_padding_mask)

    def generate_stream(self, src_idx: torch.Tensor, bos_id: int, max_new_tokens: int,
                         src_key_padding_mask: torch.Tensor | None = None):
        """Encodes once, then decodes autoregressively -- yields the growing
        target sequence after each newly generated token, mirroring
        mini_llm.model.GPT.generate_stream's interface."""
        with torch.no_grad():
            memory = self.encode(src_idx, src_key_padding_mask)
            tgt = torch.full((src_idx.size(0), 1), bos_id, dtype=torch.long, device=src_idx.device)
            for _ in range(max_new_tokens):
                logits = self.decode(tgt, memory, src_key_padding_mask)
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_id], dim=1)
                yield tgt

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_config(vocab_size: int, n_layer: int = 3, d_model: int = 256, n_head: int = 4,
                  d_ff: int = 1024) -> EncDecConfig:
    return EncDecConfig(vocab_size=vocab_size, d_model=d_model, n_head=n_head, n_layer=n_layer, d_ff=d_ff)


@torch.no_grad()
def evaluate_exact_match(model: EncoderDecoderTransformer, tokenizer: CharTokenizer, examples: list[dict],
                          device: str, src_block_size: int = SRC_BLOCK_SIZE, max_new_tokens: int = 6) -> float:
    model.eval()
    pad_id = tokenizer.pad_id
    correct = 0
    for ex in examples:
        src_ids = tokenizer.encode(ex["expr"])
        if len(src_ids) > src_block_size:
            raise ValueError(f"expr needs {len(src_ids)} tokens, exceeds src_block_size={src_block_size}")
        src_ids = src_ids + [pad_id] * (src_block_size - len(src_ids))
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_pad_mask = src == pad_id

        out_ids: list[int] = []
        for grown in model.generate_stream(src, tokenizer.bos_id, max_new_tokens, src_key_padding_mask=src_pad_mask):
            next_id = grown[0, -1].item()
            if next_id == tokenizer.eos_id:
                break
            out_ids.append(next_id)
        prediction = tokenizer.decode(out_ids)
        if prediction == render_value(ex["value"]):
            correct += 1
    model.train()
    return correct / len(examples)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
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


def train_steps(model: EncoderDecoderTransformer, tokenizer: CharTokenizer, train_ds: Seq2SeqDataset,
                 optimizer: torch.optim.Optimizer, device: str, max_steps: int, batch_size: int,
                 log_interval: int = 0) -> None:
    """Shared training loop used both by the full CLI run and by the
    overfit-a-tiny-batch test."""
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    step = 0
    while step < max_steps:
        for src, tgt_in, tgt_out in loader:
            if step >= max_steps:
                break
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_pad_mask = src == tokenizer.pad_id
            tgt_pad_mask = tgt_in == tokenizer.pad_id

            logits = model(src, tgt_in, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
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
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
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

    config = build_config(tokenizer.vocab_size, args.n_layer, args.d_model, args.n_head, args.d_ff)
    model = EncoderDecoderTransformer(config).to(args.device)
    print(f"model: {model.num_parameters():,} parameters, config={config}")

    train_ds = Seq2SeqDataset(train_examples, tokenizer, SRC_BLOCK_SIZE, TGT_BLOCK_SIZE)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    while step < args.max_steps:
        for src, tgt_in, tgt_out in loader:
            if step >= args.max_steps:
                break
            src, tgt_in, tgt_out = src.to(args.device), tgt_in.to(args.device), tgt_out.to(args.device)
            src_pad_mask = src == tokenizer.pad_id
            tgt_pad_mask = tgt_in == tokenizer.pad_id

            logits = model(src, tgt_in, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
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
                           out_dir / f"encdec_{args.domain}.pt")
            step += 1

    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config), "step": args.max_steps},
               out_dir / f"encdec_{args.domain}.pt")

    print("\nfinal OOD exact-match by depth:")
    for d in TEST_DEPTHS:
        test_examples = load_jsonl(ARTIFACTS_DIR / f"{args.domain}_test_depth{d}.jsonl")
        acc = evaluate_exact_match(model, tokenizer, test_examples, args.device)
        print(f"  depth {d:2d}: {acc:.3f}")


if __name__ == "__main__":
    main()
