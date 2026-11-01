"""SRT parsing and rebuild.

The pipeline must keep cue timing intact across translation, so the file is parsed
into structured cues rather than filtered as text: stripping index lines with
`grep -v '^[0-9]'` also deletes dialogue that starts with a digit and leaves the
`-->` lines behind, which produces an unusable subtitle file.
"""

import re

CUE_RE = re.compile(
    r"(?ms)^\s*(\d+)\s*\n"             # cue index
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})[^\n]*\n"
    r"(.*?)(?=\n\s*\n\s*\d+\s*\n|\Z)"
)


def parse(text):
    cues = []
    for match in CUE_RE.finditer(text or ""):
        index, start, end, body = match.groups()
        cues.append(
            {
                "index": int(index),
                "start": start,
                "end": end,
                "text": " ".join(line.strip() for line in body.splitlines() if line.strip()),
            }
        )
    return cues


def read(path):
    return parse(path.read_text(encoding="utf-8", errors="ignore"))


def format(cues):
    blocks = []
    for position, cue in enumerate(cues, start=1):
        blocks.append(
            f"{position}\n{cue['start']} --> {cue['end']}\n{cue['text'].strip()}\n"
        )
    return "\n".join(blocks)


def write(path, cues):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format(cues), encoding="utf-8")


def plain_text(cues):
    return "\n".join(cue["text"].strip() for cue in cues if cue["text"].strip())


def with_translations(cues, translations):
    if len(cues) != len(translations):
        raise ValueError(f"expected {len(cues)} translations, got {len(translations)}")
    out = []
    for cue, text in zip(cues, translations):
        merged = dict(cue)
        merged["text"] = (text or "").strip() or cue["text"]
        out.append(merged)
    return out
