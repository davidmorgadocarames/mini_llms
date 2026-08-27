"""Coconut — a tiny terminal UI for chatting with the Fase A mini-LLM,
loosely styled after Claude Code's startup banner and prompt.

Usage:
    python coconut.py
    python coconut.py --temperature 1.0 --top-k 40
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from mini_llm.cli.banner import render_banner
from mini_llm.model import GPT
from mini_llm.tokenizer import BPETokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

HELP_TEXT = """Comandos:
  /help               muestra esta ayuda
  /temp <valor>       cambia la temperature (actual: {temp})
  /tokens <n>         cambia cuantos tokens generar por turno (actual: {n})
  /salir, /exit       termina la sesion
Cualquier otro texto se usa como prompt inicial para el modelo."""


def _load_model(checkpoint_path: Path, device: str) -> tuple[GPT, int | None]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("step")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "ckpt.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=200)
    return p.parse_args()


def main() -> None:
    # Windows consoles default to a legacy codepage that can't render the
    # block-element glyphs in the banner; force UTF-8 output.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if os.name == "nt":
        os.system("")  # negotiates ANSI escape support in legacy cmd.exe consoles

    args = _parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"No hay checkpoint en {checkpoint_path}. Entrena primero con "
              f"mini_llm.train.train, o pasa --checkpoint <ruta>.")
        return

    tokenizer = BPETokenizer.from_dir(DATA_DIR / "tokenizer")
    model, step = _load_model(checkpoint_path, args.device)

    n_params = model.num_parameters()
    step_info = f"step {step:,}" if step is not None else "sin entrenar"
    model_info = f"{n_params / 1e6:.1f}M params · {step_info}"

    temperature = args.temperature
    max_new_tokens = args.max_new_tokens

    print()
    print(render_banner(model_info))
    print()
    print(HELP_TEXT.format(temp=temperature, n=max_new_tokens))
    print()

    while True:
        try:
            user_input = input("\x1b[1m> \x1b[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("/salir", "/exit", "/quit"):
            break
        if user_input == "/help":
            print(HELP_TEXT.format(temp=temperature, n=max_new_tokens))
            continue
        if user_input.startswith("/temp"):
            try:
                temperature = float(user_input.split()[1])
                print(f"temperature = {temperature}")
            except (IndexError, ValueError):
                print("uso: /temp <numero>")
            continue
        if user_input.startswith("/tokens"):
            try:
                max_new_tokens = int(user_input.split()[1])
                print(f"max_new_tokens = {max_new_tokens}")
            except (IndexError, ValueError):
                print("uso: /tokens <entero>")
            continue

        ids = tokenizer.encode(user_input)
        idx = torch.tensor([ids], dtype=torch.long, device=args.device)

        prev_text = tokenizer.decode(ids)
        print()
        print(prev_text, end="", flush=True)
        for out_idx in model.generate_stream(idx, max_new_tokens=max_new_tokens,
                                              temperature=temperature, top_k=args.top_k):
            full_text = tokenizer.decode(out_idx[0].tolist())
            print(full_text[len(prev_text):], end="", flush=True)
            prev_text = full_text
        print("\n")


if __name__ == "__main__":
    main()
