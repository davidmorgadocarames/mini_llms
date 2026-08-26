"""Convert YouTube auto-generated .vtt subtitles into clean plain-text transcripts.

YouTube's auto-captions use "rolling" cues: each cue repeats the previous line
and appends new words one at a time, with per-word inline timestamps
(e.g. ``<00:00:01.860><c> word</c>``). This strips the VTT markup and collapses
the rolling repeats down to a single flowing line of text.
"""

import re
from pathlib import Path

VTT_DIR = Path(__file__).resolve().parent.parent / "docs" / "karpathy" / "vtt"
TXT_DIR = Path(__file__).resolve().parent.parent / "docs" / "karpathy" / "transcripts"

INLINE_TIMESTAMP_RE = re.compile(r"<\d\d:\d\d:\d\d\.\d\d\d>")
CUE_TAG_RE = re.compile(r"</?c>")


def clean_vtt(raw: str) -> str:
    raw = INLINE_TIMESTAMP_RE.sub("", raw)
    raw = CUE_TAG_RE.sub("", raw)

    lines: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line:
            continue
        lines.append(line)

    deduped: list[str] = []
    for line in lines:
        if deduped and deduped[-1] == line:
            continue
        if deduped and line.startswith(deduped[-1]):
            deduped[-1] = line
            continue
        if deduped and deduped[-1].startswith(line):
            continue
        deduped.append(line)

    text = " ".join(deduped)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    vtt_files = sorted(VTT_DIR.glob("*.vtt"))
    if not vtt_files:
        print(f"No .vtt files found in {VTT_DIR}")
        return

    for vtt_path in vtt_files:
        raw = vtt_path.read_text(encoding="utf-8")
        cleaned = clean_vtt(raw)
        stem = vtt_path.stem
        if stem.endswith(".en"):
            stem = stem[: -len(".en")]
        out_path = TXT_DIR / f"{stem}.txt"
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"{vtt_path.name} -> {out_path.name} ({len(cleaned)} chars)")


if __name__ == "__main__":
    main()
