"""The Textual layer. This is the ONLY module in coconut_tui that imports
textual — agent.py, events.py, bus.py, providers/ and tools/ are all plain
asyncio and can be tested (or reused by another UI) without it.

Layout: conversation panel + status line + input on the left, activity panel
(collapsible tool/turn entries) on the right.
"""

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input

from coconut_tui.agent import Agent
from coconut_tui.bus import EventBus
from coconut_tui.providers.coconut_provider import CoconutProvider
from coconut_tui.tools.builtin import DiffTool, ReadFileTool, RunPytestTool
from coconut_tui.widgets.activity import ActivityPanel
from coconut_tui.widgets.conversation import ConversationPanel
from coconut_tui.widgets.status import StatusBar

EXAMPLE_PROMPTS = [
    "The history of the",
    "In 1943, the",
    "The film received",
    "The album was praised by critics for its",
]


class CoconutApp(App):
    TITLE = "Coconut"
    CSS = """
    Screen {
        layout: horizontal;
    }
    #main-col {
        width: 2fr;
    }
    ActivityPanel {
        width: 1fr;
    }
    #input {
        dock: bottom;
    }
    """

    def __init__(self, provider: CoconutProvider | None = None):
        super().__init__()
        self.bus = EventBus()
        self._provider = provider
        self.agent: Agent | None = None
        self._consumer_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="main-col"):
                yield ConversationPanel(id="conversation")
                yield StatusBar(id="status")
                yield Input(placeholder="Escribe un prompt, o /help para ver comandos", id="input")
            yield ActivityPanel(id="activity")

    async def on_mount(self) -> None:
        provider = self._provider or CoconutProvider()
        self.agent = Agent(
            provider,
            self.bus,
            tools={"read": ReadFileTool(), "test": RunPytestTool(), "diff": DiffTool()},
        )

        conversation = self.query_one("#conversation", ConversationPanel)
        step_info = f"step {provider.step:,}" if provider.step is not None else "sin entrenar"
        conversation.show_banner(f"{provider.n_params / 1e6:.1f}M params · {step_info}")
        conversation.show_suggestions(EXAMPLE_PROMPTS)

        self._consumer_task = asyncio.create_task(self._consume_events())
        self.query_one("#input", Input).focus()

    async def on_unmount(self) -> None:
        if self._consumer_task is not None:
            self._consumer_task.cancel()

    async def _consume_events(self) -> None:
        queue = self.bus.subscribe()
        conversation = self.query_one("#conversation", ConversationPanel)
        activity = self.query_one("#activity", ActivityPanel)
        status = self.query_one("#status", StatusBar)

        while True:
            event = await queue.get()

            if event.kind == "agent_started":
                status.set_working(True, "Coconut esta escribiendo...")
                conversation.add_user_message(event.prompt)
                conversation.begin_assistant_message()
                activity.begin_entry("assistant", "Generando respuesta")
            elif event.kind == "assistant_text_delta":
                conversation.append_to_assistant_message(event.text)
            elif event.kind == "agent_finished":
                conversation.finish_assistant_message(event.full_text)
                activity.finish_entry("assistant", success=True)
                status.set_working(False)
            elif event.kind == "tool_started":
                status.set_working(True, f"{event.description}...")
                activity.begin_entry(event.tool, event.description)
            elif event.kind == "tool_output":
                activity.set_entry_output(event.tool, event.output)
            elif event.kind == "file_diff":
                activity.set_entry_diff(event.tool, event.diff)
            elif event.kind == "tool_completed":
                activity.finish_entry(event.tool, success=event.success)
                status.set_working(False)
            elif event.kind == "agent_error":
                conversation.add_error(event.message)
                status.set_working(False)
            elif event.kind == "agent_info":
                conversation.add_info(event.message)

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        text = message.value
        input_widget = self.query_one("#input", Input)
        input_widget.value = ""
        if not text.strip() or self.agent is None:
            return
        input_widget.disabled = True
        try:
            await self.agent.handle_message(text)
        finally:
            input_widget.disabled = False
            input_widget.focus()
