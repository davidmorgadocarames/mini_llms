from textual.widgets import Static


class StatusBar(Static):
    """One-line indicator that the agent is doing something. Not a spinner
    widget of its own — just a reactive label the app toggles on/off."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        color: $text-muted;
    }
    """

    def set_working(self, working: bool, label: str = "Coconut esta trabajando...") -> None:
        self.update(f"⏺ {label}" if working else "")
