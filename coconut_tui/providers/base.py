"""The seam that decouples the Agent from any specific model/checkpoint/API.

Swapping providers — a different checkpoint, a Fase B model, eventually a
remote API — means implementing this one method. Nothing in agent.py or the
UI needs to change."""

from typing import AsyncIterator, Protocol


class LLMProvider(Protocol):
    async def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 50,
    ) -> AsyncIterator[str]:
        """Yields text chunks as they're produced."""
        ...
