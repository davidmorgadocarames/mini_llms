import pytest

from coconut_tui.agent import Agent
from coconut_tui.bus import EventBus


class FakeProvider:
    """Deterministic stand-in for CoconutProvider — proves the Agent only
    depends on the LLMProvider Protocol, not on any real model/GPU."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.last_call_kwargs: dict | None = None

    async def stream(self, prompt, **kwargs):
        self.last_call_kwargs = kwargs
        for chunk in self.chunks:
            yield chunk


class FailingProvider:
    async def stream(self, prompt, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this an async generator


class FakeTool:
    name = "fake"
    description = "a fake tool"

    def __init__(self, output="tool output", should_raise=False):
        self.output = output
        self.should_raise = should_raise
        self.last_argument = None

    async def run(self, argument: str) -> str:
        self.last_argument = argument
        if self.should_raise:
            raise ValueError("tool failed")
        return self.output


async def drain(bus: EventBus, queue, count: int):
    return [await queue.get() for _ in range(count)]


@pytest.mark.asyncio
async def test_plain_prompt_streams_text_deltas_and_finishes():
    provider = FakeProvider(["Hello", " ", "world"])
    bus = EventBus()
    queue = bus.subscribe()
    agent = Agent(provider, bus)

    await agent.handle_message("Once upon a time")

    events = await drain(bus, queue, 5)
    assert [e.kind for e in events] == [
        "agent_started", "assistant_text_delta", "assistant_text_delta",
        "assistant_text_delta", "agent_finished",
    ]
    assert events[0].prompt == "Once upon a time"
    assert events[-1].full_text == "Hello world"


@pytest.mark.asyncio
async def test_provider_error_emits_agent_error_not_a_crash():
    bus = EventBus()
    queue = bus.subscribe()
    agent = Agent(FailingProvider(), bus)

    await agent.handle_message("some prompt")

    events = await drain(bus, queue, 2)
    assert events[0].kind == "agent_started"
    assert events[1].kind == "agent_error"
    assert "boom" in events[1].message


@pytest.mark.asyncio
async def test_known_slash_command_dispatches_to_registered_tool():
    tool = FakeTool(output="file contents here")
    bus = EventBus()
    queue = bus.subscribe()
    agent = Agent(FakeProvider([]), bus, tools={"fake": tool})

    await agent.handle_message("/fake some/path.py")

    events = await drain(bus, queue, 3)
    assert [e.kind for e in events] == ["tool_started", "tool_output", "tool_completed"]
    assert events[1].output == "file contents here"
    assert events[2].success is True
    assert tool.last_argument == "some/path.py"


@pytest.mark.asyncio
async def test_diff_tool_dispatch_also_emits_file_diff_event():
    tool = FakeTool(output="diff --git a/x b/x\n+added line")
    bus = EventBus()
    queue = bus.subscribe()
    agent = Agent(FakeProvider([]), bus, tools={"diff": tool})

    await agent.handle_message("/diff some/path.py")

    events = await drain(bus, queue, 4)
    kinds = [e.kind for e in events]
    assert kinds == ["tool_started", "tool_output", "file_diff", "tool_completed"]
    assert events[2].path == "some/path.py"


@pytest.mark.asyncio
async def test_failing_tool_reports_failure_without_crashing():
    tool = FakeTool(should_raise=True)
    bus = EventBus()
    queue = bus.subscribe()
    agent = Agent(FakeProvider([]), bus, tools={"fake": tool})

    await agent.handle_message("/fake x")

    events = await drain(bus, queue, 3)
    assert [e.kind for e in events] == ["tool_started", "tool_output", "tool_completed"]
    assert "tool failed" in events[1].output
    assert events[2].success is False


@pytest.mark.asyncio
async def test_unknown_slash_command_emits_agent_error():
    bus = EventBus()
    queue = bus.subscribe()
    agent = Agent(FakeProvider([]), bus)

    await agent.handle_message("/nope")

    event = await queue.get()
    assert event.kind == "agent_error"
    assert "nope" in event.message


@pytest.mark.asyncio
async def test_temp_command_updates_agent_state_and_is_used_in_next_stream():
    provider = FakeProvider(["ok"])
    bus = EventBus()
    queue = bus.subscribe()
    agent = Agent(provider, bus)

    await agent.handle_message("/temp 1.5")
    info = await queue.get()
    assert info.kind == "agent_info"
    assert agent.temperature == 1.5

    await agent.handle_message("hello")
    await drain(bus, queue, 3)  # started, delta, finished
    assert provider.last_call_kwargs["temperature"] == 1.5


@pytest.mark.asyncio
async def test_bad_temp_value_emits_error_and_does_not_change_state():
    bus = EventBus()
    queue = bus.subscribe()
    agent = Agent(FakeProvider([]), bus)

    await agent.handle_message("/temp not-a-number")
    event = await queue.get()
    assert event.kind == "agent_error"
    assert agent.temperature == 0.8  # unchanged default


@pytest.mark.asyncio
async def test_empty_message_is_a_noop():
    bus = EventBus()
    queue = bus.subscribe()
    agent = Agent(FakeProvider([]), bus)

    await agent.handle_message("   ")
    assert queue.empty()
