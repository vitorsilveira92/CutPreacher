from __future__ import annotations

import re

from .models import Word

MAX_WORDS = 3
MAX_CHARS = 18
MAX_GAP = 0.6
HIGHLIGHT = r"{\c&H00E5FF&}"
NORMAL = r"{\c&HFFFFFF&}"
BREAK_AFTER = re.compile(r"[.!?…,;:]['\"”)]*$")


def format_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").upper()


def clip_words(words: list[Word], start: float, end: float) -> list[Word]:
    """Words inside [start, end], with times relative to `start`."""
    out = []
    for w in words:
        if w.end <= start or w.start >= end:
            continue
        out.append(Word(start=max(w.start, start) - start, end=min(w.end, end) - start, text=_clean(w.text)))
    return out


def group_words(words: list[Word]) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if current:
            chars = sum(len(x.text) + 1 for x in current) + len(w.text)
            if len(current) >= MAX_WORDS or chars > MAX_CHARS or w.start - current[-1].end > MAX_GAP:
                groups.append(current)
                current = []
        current.append(w)
        if BREAK_AFTER.search(w.text):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def build_ass(words: list[Word], width: int, height: int, font_name: str) -> str:
    font_size = round(width * 0.08)
    margin_v = round(height * 0.28)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,2,70,70,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    groups = group_words(words)
    for gi, group in enumerate(groups):
        next_start = groups[gi + 1][0].start if gi + 1 < len(groups) else None
        group_end = group[-1].end + 0.25
        if next_start is not None:
            group_end = min(group_end, next_start) if next_start - group[-1].end > 0.3 else next_start
        for wi, word in enumerate(group):
            ev_start = group[0].start if wi == 0 else word.start
            ev_end = group[wi + 1].start if wi + 1 < len(group) else group_end
            if ev_end <= ev_start:
                continue
            text = " ".join(
                f"{HIGHLIGHT}{w.text}{NORMAL}" if i == wi else w.text for i, w in enumerate(group)
            )
            lines.append(
                f"Dialogue: 0,{format_time(ev_start)},{format_time(ev_end)},Default,,0,0,0,,{text}"
            )
    return header + "\n".join(lines) + "\n"
