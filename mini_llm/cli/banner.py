"""ASCII-art banner for the Coconut terminal UI: a small brown coconut with its
three "eyes", drawn from Unicode block-element glyphs, next to a version /
model-info / cwd block — the same layout shape as tools like Claude Code use
for their startup banner."""

import os
from pathlib import Path

from mini_llm._version import __version__

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
BROWN = "\x1b[38;5;94m"

# Rounded "shell" built from block-element corner glyphs, with three bullet
# "eyes" on the middle row — the classic coconut face.
_LOGO = [
    " ▛███▜ ",
    " █●●●█ ",
    " ▙███▟ ",
]


def _short_cwd() -> str:
    home = Path.home()
    cwd = Path.cwd()
    try:
        rel = cwd.relative_to(home)
    except ValueError:
        return str(cwd)
    return "~" if str(rel) == "." else f"~{os.sep}{rel}"


def render_banner(model_info: str) -> str:
    lines_right = [
        f"{BOLD}Coconut v{__version__}{RESET}",
        model_info,
        f"{DIM}{_short_cwd()}{RESET}",
    ]
    rows = [
        f"{BROWN}{logo_row}{RESET}  {text_row}"
        for logo_row, text_row in zip(_LOGO, lines_right)
    ]
    return "\n".join(rows)
