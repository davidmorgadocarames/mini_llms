"""Renders coconut_tui.logo.LARGE_LOGO to a transparent PNG.

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

from coconut_tui.logo import LARGE_LOGO

FONT_PATH = os.path.join(
    os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSansMono.ttf"
)
FONT_SIZE = 40
COLOR = (201, 138, 75, 255)  # #c98a4b, matches the app's brown accent
OUT_PATH = Path(__file__).resolve().parent.parent / "coconut_tui" / "assets" / "logo.png"


def render() -> None:
    lines = LARGE_LOGO.splitlines()
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

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH)
    print(f"saved {OUT_PATH} ({img.width}x{img.height})")


if __name__ == "__main__":
    render()
