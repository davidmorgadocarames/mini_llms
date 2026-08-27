import asyncio

import pytest

from coconut_tui.bus import EventBus
from coconut_tui.events import AgentFinished, AgentStarted, AssistantTextDelta


def test_events_are_constructible_and_typed():
    started = AgentStarted(prompt="hello")
    assert started.kind == "agent_started"
    assert started.prompt == "hello"
    assert started.ts is not None


@pytest.mark.asyncio
async def test_bus_delivers_events_to_subscriber_in_order():
    bus = EventBus()
    queue = bus.subscribe()

    await bus.publish(AgentStarted(prompt="hi"))
    await bus.publish(AssistantTextDelta(text="he"))
    await bus.publish(AssistantTextDelta(text="llo"))
    await bus.publish(AgentFinished(full_text="hello"))

    kinds = [(await queue.get()).kind for _ in range(4)]
    assert kinds == ["agent_started", "assistant_text_delta", "assistant_text_delta", "agent_finished"]


@pytest.mark.asyncio
async def test_bus_supports_multiple_subscribers_independently():
    bus = EventBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()

    await bus.publish(AgentStarted(prompt="x"))

    e1 = await asyncio.wait_for(q1.get(), timeout=1)
    e2 = await asyncio.wait_for(q2.get(), timeout=1)
    assert e1.kind == e2.kind == "agent_started"


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    bus = EventBus()
    queue = bus.subscribe()
    bus.unsubscribe(queue)

    await bus.publish(AgentStarted(prompt="x"))
    assert queue.empty()
