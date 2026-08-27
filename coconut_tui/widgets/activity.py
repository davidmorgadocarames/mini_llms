"""The activity panel: one collapsible entry per tool run (or per assistant
turn), spinner-style marker while in flight, checkmark/cross when done. This
is the panel that shows *what the agent did*, never its private reasoning."""

from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Static

from coconut_tui.widgets.diff_view import render_diff


class ActivityPanel(VerticalScroll):
    DEFAULT_CSS = """
    ActivityPanel {
        border-left: solid $panel;
        padding: 0 1;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._collapsibles: dict[str, Collapsible] = {}
        self._descriptions: dict[str, str] = {}

    def begin_entry(self, key: str, description: str) -> None:
        self._descriptions[key] = description
        collapsible = Collapsible(Static(""), title=f"⏺ {description}", collapsed=True)
        self._collapsibles[key] = collapsible
        self.mount(collapsible)
        self.scroll_end(animate=False)

    def set_entry_output(self, key: str, output: str) -> None:
        collapsible = self._collapsibles.get(key)
        if collapsible is None:
            return
        body = collapsible.query(Static).first()
        body.update(output)

    def set_entry_diff(self, key: str, diff: str) -> None:
        collapsible = self._collapsibles.get(key)
        if collapsible is None:
            return
        body = collapsible.query(Static).first()
        body.update(render_diff(diff))

    def finish_entry(self, key: str, success: bool = True) -> None:
        collapsible = self._collapsibles.pop(key, None)
        description = self._descriptions.pop(key, key)
        if collapsible is None:
            return
        mark = "✓" if success else "✗"
        collapsible.title = f"{mark} {description}"
