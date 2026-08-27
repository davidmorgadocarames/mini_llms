"""Colors a unified diff the way `git diff --color` would, using plain Rich
markup: additions green, deletions red, hunk headers cyan."""

from rich.text import Text


def render_diff(diff_text: str) -> Text:
    result = Text()
    lines = diff_text.splitlines() or [""]
    for i, line in enumerate(lines):
        if line.startswith("+") and not line.startswith("+++"):
            style = "green"
        elif line.startswith("-") and not line.startswith("---"):
            style = "red"
        elif line.startswith("@@"):
            style = "cyan"
        elif line.startswith(("diff --git", "index ")):
            style = "dim"
        else:
            style = ""
        result.append(line, style=style)
        if i < len(lines) - 1:
            result.append("\n")
    return result
