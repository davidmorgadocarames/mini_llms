"""Replacer: the "replace" half of Fase B.5's Looped Locate-and-Replace
pipeline. Given an isolated base sub-expression (a single span the locator
pointed at -- always exactly one literal, or one "left op right", or one
"not literal"), generates its value.

Uses NoPE (No Positional Encoding): the token embeddings get *no* positional
information added at all, and attention has no distance bias either (unlike
the locator's ALiBi) -- the causal mask itself is the only source of order
information, which recent work (e.g. Kazemnejad et al. 2023, "The Impact of
Positional Encoding on Length Generalization in Transformers") shows is
enough for a decoder-only model to learn sequential structure. This is the
paper's second architectural choice for the replacer.

The paper also reverses the input sequence before feeding it in; here that
means the isolated span text is character-reversed before framing as
"<reversed-span> => <value>" (see depth_lab.data.loader.build_replacer_examples).
Reversing puts the operator at a more consistent distance from the start of
generation across examples than the un-reversed text would (operand lengths
differ -- "True" is 4 characters, "False" is 5 -- which otherwise shifts
where the operator falls).

This model deliberately mirrors mini_llm.model.GPT's forward/generate_stream
interface (logits, loss = forward(idx); generate_stream(idx, max_new_tokens,
temperature, top_k)) so it's a drop-in replacement for the baseline's
training loop, optimizer builder, and exact-match evaluator -- see
depth_lab/models/baseline.py's train_steps/evaluate_exact_match, reused
as-is in this module's main().

Usage:
    python -m depth_lab.models.replacer --max-steps 1500
"""

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_lab.data.build_dataset import ARTIFACTS_DIR, load_jsonl
from depth_lab.data.loader import ExprDataset, build_replacer_examples
from depth_lab.models.baseline import build_optimizer, evaluate_exact_match, train_steps
from depth_lab.tokenizer import CharTokenizer

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
BLOCK_SIZE = 32  # spans are always short: e.g. "(False and False)" reversed + " => " + value


class CausalSelfAttention(nn.Module):
    """Plain scaled dot-product attention with a causal mask -- no RoPE, no
    ALiBi, no learned position embeddings anywhere in this model. Position
    only ever implicitly enters through the causal mask itself."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
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


class ReplacerBlock(nn.Module):
    """Pre-norm, unlike the encoder-decoder's post-norm -- this model has no
    positional signal to lean on, so the more stable pre-norm gradient flow
    matters more here than matching the classic paper's recipe."""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


@dataclass
class ReplacerConfig:
    vocab_size: int
    block_size: int = BLOCK_SIZE
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 3
    d_ff: int = 512
    dropout: float = 0.0


class Replacer(nn.Module):
    def __init__(self, config: ReplacerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)  # no positional embedding -- this is NoPE
        self.blocks = nn.ModuleList(
            [ReplacerBlock(config.d_model, config.n_head, config.d_ff, config.dropout) for _ in range(config.n_layer)]
        )
        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} exceeds block_size {self.config.block_size}"
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def generate_stream(self, idx: torch.Tensor, max_new_tokens: int,
                         temperature: float = 1.0, top_k: int | None = None):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                yield idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_config(vocab_size: int, block_size: int = BLOCK_SIZE, n_layer: int = 3,
                  d_model: int = 128, n_head: int = 4, d_ff: int = 512) -> ReplacerConfig:
    return ReplacerConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                           d_model=d_model, n_head=n_head, d_ff=d_ff)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="bool", choices=["bool"])
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=1500)
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
    replacer_examples = build_replacer_examples(train_examples)
    print(f"{len(replacer_examples)} replacer training instances from {len(train_examples)} expressions")

    config = build_config(tokenizer.vocab_size, args.block_size, args.n_layer, args.d_model, args.n_head, args.d_ff)
    model = Replacer(config).to(args.device)
    print(f"model: {model.num_parameters():,} parameters, config={config}")

    train_ds = ExprDataset(replacer_examples, tokenizer, args.block_size)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_steps(model, tokenizer, train_ds, optimizer, args.device, args.max_steps, args.batch_size,
                args.log_interval)

    acc = evaluate_exact_match(model, tokenizer, replacer_examples[:200], args.device)
    print(f"train-subset exact-match: {acc:.3f}")

    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(config), "step": args.max_steps},
               out_dir / f"replacer_{args.domain}.pt")
    print(f"done. checkpoint saved to {out_dir / f'replacer_{args.domain}.pt'}")


if __name__ == "__main__":
    main()
