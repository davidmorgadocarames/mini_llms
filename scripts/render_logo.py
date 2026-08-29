"""Renders block-art ASCII logos (Coconut, and Fase C's Cracked/Sliced/
Pressed) to transparent PNGs.

Why: the ASCII art relies on Unicode block-element characters that many
mobile browsers render as generic "tofu" glyphs (confirmed on a real device),
so we ship it as a pre-rendered image instead of live text for the Streamlit
demo, where it always looks the same regardless of the visitor's font
support.

Usage: python scripts/render_logo.py
"""

import os
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

from coconut_lab.logos import (
    CRACKED_LOGO,
    CRACKED_WORDMARK,
    PRESSED_LOGO,
    PRESSED_WORDMARK,
    SLICED_LOGO,
    SLICED_WORDMARK,
)
from coconut_tui.logo import LARGE_LOGO

FONT_PATH = os.path.join(
    os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSansMono.ttf"
)
FONT_SIZE = 40
COLOR = (201, 138, 75, 255)  # #c98a4b, matches the app's brown accent

ASSETS_DIR = Path(__file__).resolve().parent.parent / "coconut_tui" / "assets"
LOGOS = {
    "logo.png": LARGE_LOGO,
    "cracked_logo.png": CRACKED_LOGO,
    "sliced_logo.png": SLICED_LOGO,
    "pressed_logo.png": PRESSED_LOGO,
    # Wordmark-only renders: the Fase C demo pairs an animated GIF of the
    # illustration with the live ASCII wordmark beside it, and falls back to
    # these on mobile for the same tofu-glyph reason as the full logos.
    "cracked_wordmark.png": CRACKED_WORDMARK,
    "sliced_wordmark.png": SLICED_WORDMARK,
    "pressed_wordmark.png": PRESSED_WORDMARK,
}


def render_one(text: str, out_path: Path) -> None:
    lines = text.splitlines()
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    cell_w = font.getlength("█")
    ascent, descent = font.getmetrics()
    cell_h = ascent + descent

    max_cols = max(len(line) for line in lines)
    img_w = int(cell_w * max_cols) + 4
    img_h = int(cell_h * len(lines)) + 4

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for row, line in enumerate(lines):
        y = row * cell_h
        for col, ch in enumerate(line):
            if ch == " ":
                continue
            draw.text((col * cell_w, y), ch, font=font, fill=COLOR)

    bbox = img.getbbox()
    if bbox:
        pad = 6
        left, top, right, bottom = bbox
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(img_w, right + pad)
        bottom = min(img_h, bottom + pad)
        img = img.crop((left, top, right, bottom))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"saved {out_path} ({img.width}x{img.height})")


def render() -> None:
    for filename, text in LOGOS.items():
        render_one(text, ASSETS_DIR / filename)


if __name__ == "__main__":
    render()
