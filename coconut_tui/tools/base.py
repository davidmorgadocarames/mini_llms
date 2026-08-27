"""Tools are triggered by explicit user slash-commands in this MVP (our 26M
base model has no function-calling ability — it can't decide on its own which
tool to run). The protocol is intentionally uniform and simple so a future,
instruction-tuned provider could decide to invoke these itself without any
change to this contract."""

from typing import Protocol


class Tool(Protocol):
    name: str
    description: str

    async def run(self, argument: str) -> str:
        """Executes the tool and returns its raw text output. Raises on failure
        (the Agent turns that into an AgentError event)."""
        ...
