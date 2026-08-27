"""The main chat panel: banner + prompt suggestions at the top, then a
scrolling history of user/assistant turns. Assistant text streams in live via
append_to_assistant_message(), then gets re-rendered as Markdown once finished
(rich.markdown.Markdown) for nicer typography."""

from rich.markdown import Markdown
from textual.containers import VerticalScroll
from textual.widgets import Static

from coconut_tui.logo import logo_for_width


class ConversationPanel(VerticalScroll):
    DEFAULT_CSS = """
    ConversationPanel {
        padding: 0 1;
    }
    ConversationPanel .user-message {
        color: $accent;
        margin: 1 0 0 0;
    }
    ConversationPanel .assistant-message {
        margin: 0 0 1 0;
    }
    ConversationPanel .info-message {
        color: $text-muted;
        margin: 0 0 1 0;
    }
    ConversationPanel .error-message {
        color: $error;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_assistant: Static | None = None
        self._current_assistant_text = ""

    def show_banner(self, model_info: str) -> None:
        width = self.app.size.width or 100
        self.mount(Static(logo_for_width(width), id="banner"))
        self.mount(Static(model_info, classes="info-message"))

    def show_suggestions(self, prompts: list[str]) -> None:
        body = "Coconut es un modelo base (sin fine-tuning de instrucciones): " \
               "escribe el principio de una frase para que la continue.\n\n" \
               "Prueba, por ejemplo:\n" + "\n".join(f"  > {p}" for p in prompts) + \
               "\n\n/help para ver los comandos disponibles."
        self.mount(Static(body, classes="info-message"))

    def add_user_message(self, text: str) -> None:
        self.mount(Static(f"> {text}", classes="user-message"))
        self.scroll_end(animate=False)

    def begin_assistant_message(self) -> None:
        self._current_assistant_text = ""
        self._current_assistant = Static("", classes="assistant-message")
        self.mount(self._current_assistant)
        self.scroll_end(animate=False)

    def append_to_assistant_message(self, delta: str) -> None:
        if self._current_assistant is None:
            self.begin_assistant_message()
        self._current_assistant_text += delta
        self._current_assistant.update(self._current_assistant_text)
        self.scroll_end(animate=False)

    def finish_assistant_message(self, full_text: str) -> None:
        if self._current_assistant is not None:
            self._current_assistant.update(Markdown(full_text))
        self._current_assistant = None
        self.scroll_end(animate=False)

    def add_info(self, message: str) -> None:
        self.mount(Static(message, classes="info-message"))
        self.scroll_end(animate=False)

    def add_error(self, message: str) -> None:
        self.mount(Static(f"⚠ {message}", classes="error-message"))
        self.scroll_end(animate=False)
