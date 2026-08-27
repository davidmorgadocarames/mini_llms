"""Concrete LLMProvider backed by our own trained Fase A checkpoint
(mini_llm.model.GPT). Model inference is synchronous/blocking (PyTorch on
CPU or CUDA), so we run it in a background thread and bridge its output back
into an async generator via a queue — this keeps the Textual event loop free
to keep rendering (spinners, input) while a generation is in flight."""

import asyncio
import threading
from pathlib import Path
from typing import AsyncIterator

import torch

from mini_llm.model import GPT
from mini_llm.tokenizer import BPETokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TOKENIZER_DIR = _REPO_ROOT / "mini_llm" / "data" / "artifacts" / "tokenizer"
DEFAULT_CHECKPOINT_PATH = _REPO_ROOT / "mini_llm" / "checkpoints" / "ckpt.pt"


class CoconutProvider:
    """LLMProvider implementation wrapping our Fase A GPT checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BPETokenizer.from_dir(tokenizer_dir)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = GPT(checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.step: int | None = checkpoint.get("step")

    @property
    def n_params(self) -> int:
        return self.model.num_parameters()

    async def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 50,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[str | None]" = asyncio.Queue()

        def _worker() -> None:
            ids = self.tokenizer.encode(prompt)
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            prev_text = self.tokenizer.decode(ids)
            for out_idx in self.model.generate_stream(
                idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
            ):
                full_text = self.tokenizer.decode(out_idx[0].tolist())
                delta = full_text[len(prev_text):]
                prev_text = full_text
                if delta:
                    asyncio.run_coroutine_threadsafe(queue.put(delta), loop)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
