from pathlib import Path

import pytest

from coconut_tui.providers.coconut_provider import CoconutProvider

CHECKPOINT_PATH = Path("mini_llm/checkpoints/ckpt.pt")

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(), reason="requires a trained Fase A checkpoint"
)


@pytest.mark.asyncio
async def test_stream_yields_nonempty_text_chunks():
    provider = CoconutProvider(device="cpu")
    chunks = []
    async for chunk in provider.stream("The history of", max_new_tokens=10, top_k=20):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert "".join(chunks).strip() != ""


@pytest.mark.asyncio
async def test_stream_respects_max_new_tokens_roughly():
    provider = CoconutProvider(device="cpu")
    text = ""
    async for chunk in provider.stream("Once upon a time", max_new_tokens=5, top_k=10):
        text += chunk
    # 5 BPE tokens is a small, bounded amount of text
    assert len(text) < 200
