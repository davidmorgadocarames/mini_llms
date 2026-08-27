"""ASCII-art banner for Coconut. The large logo needs a wide terminal (~90
columns); narrower terminals fall back to a compact one-liner so the layout
never breaks."""

LARGE_LOGO = r"""
                        ▒▒▓
                       ▒▓▓█▓
                      ▒▓▓ ▓█▓          ████    ████    ████    ████
                     ▒▓▓   ▓▓▒        ██▒▒▒▒  ██  ██  ██▒▒▒▒  ██  ██
                    ▒▓▓               ██▒     ██  ██▒ ██▒     ██  ██▒
          ▓▓▓▓▓▒▓▓▓▓▓█▓▓▒▓▓▓▓▓        ██▒     ██  ██▒ ██▒     ██  ██▒
      ▓▓██▒▒░░░░░░▒▓▓░░░░░░░▒▒██▓▓     ████    ████▒▒  ████    ████▒▒
     ▓██▓░░░░▒▒▒▒▒▓▓▒▒▒▒▒▒▒░░░░▓██▓     ▒▒▒▒    ▒▒▒▒    ▒▒▒▒    ▒▒▒▒
     ████▓▒░░░░▒▒▒▓▒▒▒▒▒▒░░░░▒▓████
     █▒▒▓▓▓██▓▓▓▓▒▒▒▒▒▒▓▓▓▓██▓█████   ██  ██  ██  ██  ██████
     █▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▒▓▒▓▒▓▓▓██████   ███ ██▒ ██▒ ██▒  ▒██▒▒▒
     █▓▒▒▒▒▒▒▒▒▒▒▓▒▒▒▓▒▓▓▓▓▓███████   ██████▒ ██▒ ██▒   ██▒
      █▓▓▓▓▓▓▓▓▓▒▓▒▓▓▓▓▓▓▓█▓██████    ██▒███▒ ██▒ ██▒   ██▒
      ▓█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██████████▓    ██▒ ██▒  ████▒▒   ██▒
        ██▓▓██▓▓▓▓▓█████████████       ▒▒  ▒▒   ▒▒▒▒     ▒▒
          ████████████████████
            ▓▓████████████▓▓
                 ▓▓▓▓▓▓
""".strip("\n")

LARGE_LOGO_WIDTH = max(len(line) for line in LARGE_LOGO.splitlines())

COMPACT_LOGO = "🥥 Coconut"


def logo_for_width(width: int) -> str:
    return LARGE_LOGO if width >= LARGE_LOGO_WIDTH else COMPACT_LOGO
