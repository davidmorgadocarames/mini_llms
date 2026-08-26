"""Generate text from a trained checkpoint.

Usage:
    python -m mini_llm.inference.generate --prompt "Once upon a time" --max-new-tokens 200
"""

import argparse
from pathlib import Path

import torch

from mini_llm.model import GPT
from mini_llm.tokenizer import BPETokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def load_model(checkpoint_path: Path, device: str) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "ckpt.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = BPETokenizer.from_dir(DATA_DIR / "tokenizer")
    model = load_model(Path(args.checkpoint), args.device)

    ids = tokenizer.encode(args.prompt) if args.prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long, device=args.device)

    out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
