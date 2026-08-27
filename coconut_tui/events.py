"""Typed events that flow from the Agent to the UI. This module has zero
dependency on Textual (or any UI toolkit) on purpose: it's the contract that
lets the Agent run headless (testable with plain pytest) and lets the UI layer
be swapped without touching agent logic.

We deliberately do NOT model the LLM's internal reasoning/logits/hidden
state as an event — only observable actions and outputs (what it said, what
tool ran, what a tool produced). See coconut_tui/agent.py.
"""

from datetime import datetime, timezone
from typing import Literal, Union

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BaseEvent(BaseModel):
    ts: datetime = Field(default_factory=_now)


class AgentStarted(BaseEvent):
    kind: Literal["agent_started"] = "agent_started"
    prompt: str


class AssistantTextDelta(BaseEvent):
    kind: Literal["assistant_text_delta"] = "assistant_text_delta"
    text: str


class AgentFinished(BaseEvent):
    kind: Literal["agent_finished"] = "agent_finished"
    full_text: str


class ToolStarted(BaseEvent):
    kind: Literal["tool_started"] = "tool_started"
    tool: str
    description: str


class ToolOutput(BaseEvent):
    kind: Literal["tool_output"] = "tool_output"
    tool: str
    output: str


class ToolCompleted(BaseEvent):
    kind: Literal["tool_completed"] = "tool_completed"
    tool: str
    success: bool


class FileDiff(BaseEvent):
    kind: Literal["file_diff"] = "file_diff"
    path: str
    diff: str


class AgentError(BaseEvent):
    kind: Literal["agent_error"] = "agent_error"
    message: str


class AgentInfo(BaseEvent):
    """Plain informational message (help text, config acknowledgements) — not
    an error, not model output."""
    kind: Literal["agent_info"] = "agent_info"
    message: str


AgentEvent = Union[
    AgentStarted,
    AssistantTextDelta,
    AgentFinished,
    ToolStarted,
    ToolOutput,
    ToolCompleted,
    FileDiff,
    AgentError,
    AgentInfo,
]
