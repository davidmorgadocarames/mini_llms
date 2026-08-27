"""Minimal asyncio pub/sub bus. This is the seam between the Agent (producer of
AgentEvent) and any UI (consumer) — the Agent only ever calls `publish`, and
never knows who (if anyone) is listening or how they render it."""

import asyncio

from coconut_tui.events import AgentEvent


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[AgentEvent]] = []

    def subscribe(self) -> "asyncio.Queue[AgentEvent]":
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: "asyncio.Queue[AgentEvent]") -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def publish(self, event: AgentEvent) -> None:
        for queue in list(self._subscribers):
            await queue.put(event)
