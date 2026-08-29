"""Makes the flat background of the Fase C model GIFs transparent.

The source GIFs ship with an opaque near-black background ((13, 10, 7)),
which shows up as a visible rectangle behind the illustration on the demo
page's own (slightly different) dark background. CSS blend modes don't fix
it cleanly because neither color is pure black, so the background is
knocked out in the asset itself instead.

Run after dropping in new source GIFs:
    python scripts/make_gifs_transparent.py
"""

from pathlib import Path

from PIL import Image, ImageSequence

ASSETS_DIR = Path(__file__).resolve().parent.parent / "coconut_tui" / "assets"
GIFS = ["cracked.gif", "sliced.gif", "pressed.gif"]

# Anything this dark is background, not artwork: the illustrations are drawn
# in bright oranges/whites, so there's a wide gap to threshold on.
BACKGROUND_MAX_CHANNEL = 32
TRANSPARENT_INDEX = 255


def make_transparent(path: Path) -> None:
    source = Image.open(path)
    duration = source.info.get("duration", 70)
    loop = source.info.get("loop", 0)

    frames = []
    for frame in ImageSequence.Iterator(source):
        rgb = frame.convert("RGB")
        # Reserve index 255 for transparency, so quantize to 255 colors.
        quantized = rgb.quantize(colors=TRANSPARENT_INDEX, method=Image.MEDIANCUT)

        pixels = rgb.load()
        out = quantized.load()
        width, height = rgb.size
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                if r <= BACKGROUND_MAX_CHANNEL and g <= BACKGROUND_MAX_CHANNEL and b <= BACKGROUND_MAX_CHANNEL:
                    out[x, y] = TRANSPARENT_INDEX
        quantized.info["transparency"] = TRANSPARENT_INDEX
        frames.append(quantized)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        transparency=TRANSPARENT_INDEX,
        disposal=2,  # restore to background between frames, so frames don't stack
        optimize=False,
    )
    print(f"{path.name}: {len(frames)} frames, background made transparent")


def main() -> None:
    for name in GIFS:
        make_transparent(ASSETS_DIR / name)


if __name__ == "__main__":
    main()
