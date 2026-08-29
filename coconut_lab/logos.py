"""Fase C.7: ASCII-art logos for the three architectures (Cracked/Sliced/
Pressed), same block-drawing style and pattern as coconut_tui.logo.LARGE_LOGO
-- a raw triple-quoted string per model, stripped of the leading/trailing
blank line the triple-quote introduces. Used by the Fase C demo page
(pages/2_Fase_C_Coconut_Interactivo.py) to swap the banner shown based on
the currently selected model.

Each logo is one piece of art (a coconut illustration) plus its wordmark
(the block-letter model name) side by side. The demo shows an animated GIF
for the illustration and the live ASCII wordmark next to it, so the
wordmark is sliced back out here rather than duplicated as a second
constant -- one source of truth for the art.
"""


def _wordmark(logo: str, split_col: int) -> str:
    """Slices the block-letter model name out of the right-hand side of a
    logo, dropping the illustration to its left and re-trimming the leading
    indent the slice leaves behind."""
    lines = [line[split_col:].rstrip() for line in logo.splitlines()]
    lines = [line for line in lines if line.strip()]
    indent = min(len(line) - len(line.lstrip()) for line in lines)
    return "\n".join(line[indent:] for line in lines)

CRACKED_LOGO = r"""
           ▓▓▓▓▓▓▒▒▒▒▒▒▓▓▓▓▓▓
        ▓█▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓██▓
      ▓█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▒▓▒▒▒▒▓▓██▓
    ▓█▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▓▓▓▒▓▓▓▓▓██▓
   ▓█▒▒▒▒▒▒▒▒▒▒▒▒▒▓█▒░░░░▒█▓▓▓▓▓████▓
  ▓█▒▒▒▒▒▒▒▒▓▓▓▓▓█▓▒░░░░░░░▒▒▓█▓█████▓
  █▒▒▒▒▒▒▒▒▒█▒░░░░░░░░░░░░░░░░▒█▓█████     ████ ████   ██    ████ ██  █ █████ ████
 ▓▓▒▒▒▒▒▒▒▓▓▒▒░░░░░░░░░░░░░░░░▓▓▓█████▓   ██▒▒▒▒██  █ █  █  ██▒▒▒▒██▒█ ▒██▒▒▒▒██  █
 ▓▓▒▒▒▒▒▓▓▓▒░░░░░░░░░░░░░░░░░▒▒▓██████▓   ██▒   ████ ▒█████ ██▒   ███ ▒ ████  ██  █▒
 ▓▓▒▒▒▒▒▓▓░░░░░░░░░░░░░░░░░▒▓▓████████▓   ██▒   ██▒█▒ █▒▒█▒▒██▒   ██▒█  ██▒▒▒ ██  █▒
 ▓█▓▓▓▓▓█▓░░░░░░░░░░░░░░░▒▓▓▓█████████▓    ████ ██▒ █ █▒ █▒  ████ ██▒ █ █████ ████ ▒
  █▓▓▓▓▓▓▓▒░░░░░░░░░░░░░░░▒███████████      ▒▒▒▒ ▒▒  ▒ ▒  ▒   ▒▒▒▒ ▒▒  ▒ ▒▒▒▒▒ ▒▒▒▒
  ▓█▓▓▓▓▓██▒░░░░░░░░░░░░░░░▒█████████▓
   ▓███▓██▓▓▓▒░░░░░░░░░░░░░▒▓███████▓
    ▓███████▓█▓▓▒░░▒▒▒▒▓▓▓▓▓▓██████▓
      ▓████████▓█▓█▓█████████████▓
        ▓██████████████████████▓
           ▓▓██████████████▓▓
""".strip("\n")

SLICED_LOGO = r"""
                                         ░░▒
                                       ░░░░▒
                                     ░░░░░░▒
                                    ░░░░░░░▓
                                  ░░░░░░░░▒▓
                                ░░░░░░░░░▒█▓    ████ ██    ███  ████ █████ ████
░░░                          ░░░░░░░░░░░░█▓▓   ██▒▒▒▒██▒    █▒▒██▒▒▒▒██▒▒▒▒██  █
░░░░░░░░                 ░░░░░░░░░░░░░░▒█▓██    ███  ██▒    █▒ ██▒   ████  ██  █▒
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓▓▓▓██     ▒██ ██▒    █▒ ██▒   ██▒▒▒ ██  █▒
▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓▓▓▓██▓   ████▒▒█████ ███  ████ █████ ████ ▒
▓█▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓█▓▓▓███▓     ▒▒▒▒  ▒▒▒▒▒ ▒▒▒  ▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒
██▓██▓▒░░░░░░░░░░░░░░░░░░░░░░░▒▓▓▓▓▓▓▓██▓
████▓▓▓▓▓▓▒▒░░░░░░░░░░░░░▒▒▓▓█▓▓▓█████▓
  ▓█████▓▓▓▓████▓▓▓▓▓████▓▓▓▓▓█▓███▓▓
     ▓████████▓█▓▓▓▓▓▓███▓██████▓▓
         ▓▓▓██████████████▓▓▓▓
                 ▓▓▓▓
""".strip("\n")

PRESSED_LOGO = r"""
        ▓███▓▒▒▒▓▓▓▒▓▓
       ▓▓▓██░░░░▒▒▒▒▒▒▒▓▓▓▓
      ▓█▓▓▓██▓░░▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓
      █▓▓▓▒▓▒▓█▓▒▒▒▒▒▒▒▒▒▒▒▒▒░▒█▓
      █▓▓▓▒▓▓▓▓▓▓█▓▓▒▒▒▒▒▒▒▒▒▒░░▒██
      █▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓█▓▓▓▒▒▒░░▒▒███
      ███▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█████████
      ▓████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓███████▓   ████  ████  █████  ████  ████ █████ ████
      ░░░▒███▓▓▓▓▓▓▓▓█▓█▓█████████▓    ██  █ ██  █ ██▒▒▒▒██▒▒▒▒██▒▒▒▒██▒▒▒▒██  █
      ░░░▓███████████████████████▓     ████ ▒████ ▒████   ███   ███  ████  ██  █▒
      ░░░ ▓████████████████████▓       ██▒▒▒ ██▒█▒ ██▒▒▒   ▒██   ▒██ ██▒▒▒ ██  █▒
      ░░░   ▓▓██████████████▓▓         ██▒   ██▒ █ █████ ████▒▒████▒▒█████ ████ ▒
     ▒░░░       ▓▓▓▓▓▓▓▓▓▓              ▒▒    ▒▒  ▒ ▒▒▒▒▒ ▒▒▒▒  ▒▒▒▒  ▒▒▒▒▒ ▒▒▒▒
     ▒░░
      ░▒
      ░░
      ░░
      ░░
      ░░      ░
   ░░░▒▒      ░
  ░▒░░▒
""".strip("\n")

# Column where each logo's illustration ends and its wordmark begins
# (verified by slicing and eyeballing both halves).
CRACKED_WORDMARK = _wordmark(CRACKED_LOGO, 39)
SLICED_WORDMARK = _wordmark(SLICED_LOGO, 44)
PRESSED_WORDMARK = _wordmark(PRESSED_LOGO, 37)
