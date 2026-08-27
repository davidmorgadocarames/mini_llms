import pytest

from coconut_tui.app import CoconutApp
from coconut_tui.widgets.activity import ActivityPanel
from coconut_tui.widgets.conversation import ConversationPanel


class FakeProvider:
    """Avoids loading the real GPU checkpoint in a UI-composition test."""

    n_params = 123_000
    step = 42

    async def stream(self, prompt, **kwargs):
        for chunk in ["Hello", " ", "world"]:
            yield chunk


@pytest.mark.asyncio
async def test_app_mounts_and_shows_banner_and_suggestions():
    app = CoconutApp(provider=FakeProvider())
    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ConversationPanel)
        assert conversation.query_one("#banner") is not None


@pytest.mark.asyncio
async def test_submitting_a_prompt_streams_assistant_reply_into_conversation():
    app = CoconutApp(provider=FakeProvider())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#input")
        await pilot.press(*"Once upon a time")
        await pilot.press("enter")
        await pilot.pause()

        conversation = app.query_one("#conversation", ConversationPanel)
        widget = conversation.query(".assistant-message").first()
        content = widget.content
        text = content.markup if hasattr(content, "markup") else str(content)
        assert "Hello world" in text


@pytest.mark.asyncio
async def test_slash_command_populates_activity_panel():
    app = CoconutApp(provider=FakeProvider())
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#input")
        for ch in "/read README.md":
            await pilot.press(ch if ch != " " else "space")
        await pilot.press("enter")
        await pilot.pause()

        activity = app.query_one("#activity", ActivityPanel)
        assert len(activity.query("Collapsible")) >= 1
