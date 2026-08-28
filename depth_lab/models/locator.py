"""Locator: the "locate" half of Fase B.5's Looped Locate-and-Replace
pipeline. Given the current (partially reduced) expression string, predicts
-- per character -- whether it belongs to the next innermost sub-expression
that should be reduced (see depth_lab/data/reduce.py for the ground-truth
definition of "next innermost").

Uses ALiBi (Attention with Linear Biases, Press et al. 2021) instead of any
learned/sinusoidal positional encoding: attention scores get a fixed,
non-learned penalty proportional to the distance between query and key
positions, with a different slope per head. No positional information is
added to the token embeddings at all -- position only ever enters through
this per-head distance bias. Self-attention here is bidirectional (no causal
mask): this is a per-token classification problem, not generation, so a
token's label can depend on context in both directions.

Usage:
    python -m depth_lab.models.locator --max-steps 2000
"""

import argparse
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_lab.data.build_dataset import ARTIFACTS_DIR, load_jsonl
from depth_lab.data.loader import LocatorDataset, build_locator_examples
from depth_lab.tokenizer import CharTokenizer
from mini_llm.train.checkpoint import load_checkpoint, save_checkpoint

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
BLOCK_SIZE = 384  # same bound as the other Fase B models: max confirmed expr length is 335 chars


def alibi_slopes(n_head: int) -> torch.Tensor:
    """The geometric-sequence slopes from the ALiBi paper. Only implements
    the power-of-2 case (our configs always use one) -- the general case
    needs an extra interpolation step the paper describes for other head
    counts, which we don't need here."""
    assert n_head > 0 and (n_head & (n_head - 1)) == 0, "alibi_slopes assumes n_head is a power of 2"
    exponents = torch.arange(1, n_head + 1, dtype=torch.float32)
    return 2.0 ** (-8.0 * exponents / n_head)


def alibi_bias(seq_len: int, n_head: int, device=None) -> torch.Tensor:
    """(n_head, T, T) additive attention bias: 0 on the diagonal, growing
    more negative (i.e. more suppressed) the farther apart two positions
    are, symmetrically in both directions since this is bidirectional."""
    slopes = alibi_slopes(n_head).to(device)
    positions = torch.arange(seq_len, device=device)
    distance = (positions[None, :] - positions[:, None]).abs().float()  # (T, T)
    return -slopes[:, None, None] * distance[None, :, :]


class ALiBiSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, n_head, T, T)
        scores = scores + alibi_bias(T, self.n_head, device=x.device)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (B, n_head, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_head)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocatorLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = ALiBiSelfAttention(d_model, n_head, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, key_padding_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


@dataclass
class LocatorConfig:
    vocab_size: int
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 3
    d_ff: int = 512
    dropout: float = 0.0


class Locator(nn.Module):
    def __init__(self, config: LocatorConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [LocatorLayer(config.d_model, config.n_head, config.d_ff, config.dropout) for _ in range(config.n_layer)]
        )
        self.head = nn.Linear(config.d_model, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Returns per-token logits (B, T) -- apply sigmoid for probabilities."""
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.head(x).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_config(vocab_size: int, n_layer: int = 3, d_model: int = 128, n_head: int = 4,
                  d_ff: int = 512) -> LocatorConfig:
    return LocatorConfig(vocab_size=vocab_size, d_model=d_model, n_head=n_head, n_layer=n_layer, d_ff=d_ff)


def extract_span_from_probs(probs: torch.Tensor, threshold: float = 0.5) -> tuple[int, int]:
    """Decodes per-token probabilities into a single contiguous span: the
    highest-confidence contiguous run of tokens above `threshold`. Falls back
    to a single-token span around the highest-probability position if
    nothing clears the threshold (should only happen for a badly undertrained
    model)."""
    mask = (probs > threshold).tolist()
    n = len(mask)
    best_span = None
    best_score = -1.0
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            score = probs[i:j].sum().item()
            if score > best_score:
                best_score = score
                best_span = (i, j)
            i = j
        else:
            i += 1
    if best_span is None:
        idx = int(probs.argmax().item())
        best_span = (idx, idx + 1)
    return best_span


@torch.no_grad()
def predict_span(model: Locator, tokenizer: CharTokenizer, expr: str, device: str) -> tuple[int, int]:
    model.eval()
    ids = torch.tensor([tokenizer.encode(expr)], dtype=torch.long, device=device)
    logits = model(ids)
    probs = torch.sigmoid(logits[0])
    model.train()
    return extract_span_from_probs(probs)


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


def train_steps(model: Locator, train_ds: LocatorDataset, optimizer: torch.optim.Optimizer,
                 device: str, max_steps: int, batch_size: int, log_interval: int = 0,
                 checkpoint_path: Path | None = None, checkpoint_interval: int = 0) -> None:
    """checkpoint_path/checkpoint_interval: see
    coconut_lab.models.cracked.train_steps's docstring -- same
    resume-if-exists, save-every-N-steps, backward-compatible-by-default
    pattern, needed for Fase C's k-fold (this function trains Pressed's
    locator once per fold)."""
    start_step = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        start_step = load_checkpoint(checkpoint_path, model, optimizer, device)
        print(f"resumed from {checkpoint_path} at step {start_step}")

    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    step = start_step
    while step < max_steps:
        for ids, labels, pad_mask in loader:
            if step >= max_steps:
                break
            ids, labels, pad_mask = ids.to(device), labels.to(device), pad_mask.to(device)
            logits = model(ids, key_padding_mask=pad_mask)
            loss_per_token = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            valid = (~pad_mask).float()
            loss = (loss_per_token * valid).sum() / valid.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if log_interval and step % log_interval == 0:
                print(f"step {step:6d} | loss {loss.item():.4f}")
            step += 1

            if checkpoint_path is not None and checkpoint_interval and step % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, model, optimizer, step, model.config)

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, optimizer, step, model.config)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="bool", choices=["bool"])
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=str(CHECKPOINT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)

    tokenizer = CharTokenizer()
    train_examples = load_jsonl(ARTIFACTS_DIR / f"{args.domain}_train.jsonl")
    locator_examples = build_locator_examples(train_examples)
    print(f"{len(locator_examples)} locator training instances from {len(train_examples)} expressions")

    config = build_config(tokenizer.vocab_size, args.n_layer, args.d_model, args.n_head, args.d_ff)
    model = Locator(config).to(args.device)
    print(f"model: {model.num_parameters():,} parameters, config={config}")

    train_ds = LocatorDataset(locator_examples, tokenizer, args.block_size)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_steps(model, train_ds, optimizer, args.device, args.max_steps, args.batch_size, args.log_interval)

    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config), "step": args.max_steps},
               out_dir / f"locator_{args.domain}.pt")
    print(f"done. checkpoint saved to {out_dir / f'locator_{args.domain}.pt'}")


if __name__ == "__main__":
    main()
