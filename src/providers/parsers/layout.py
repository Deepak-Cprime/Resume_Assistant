"""PDF page layout analysis and column-aware Markdown rendering, using PyMuPDF only.

Resumes with a sidebar are the one layout where plain top-to-bottom extraction mixes
sections. We find the gutter between two columns and read the left column, then the right,
with any full-width header above them and any full-width footer below them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

NARROW_BLOCK = 0.65  # a text block narrower than this share of the page is a column candidate
MIN_BLOCK_CHARS = 3  # ignore stray glyphs; short labels ("SQL", "CONTACT") count
MIN_BLOCKS_PER_COLUMN = 2
# Each side must carry this share of the narrow text. A sidebar of skill labels does; the
# right-aligned dates of a one-column resume (a few percent) do not, and must not become a column.
MIN_COLUMN_TEXT_SHARE = 0.10
MAX_STRADDLE_SHARE = 0.10
MIN_VERTICAL_OVERLAP = 0.5
SPLIT_TOLERANCE = 2.0

HEADING_RATIO = 1.15  # a line this much larger than body text is a heading
MAX_HEADING_LEVEL = 3
MAX_LABEL_CHARS = 60  # a short all-bold line is rendered as a bold label
WRAP_FILL = 0.75  # a line this full (vs the widest in its block) may continue on the next

_BULLET = re.compile(r"^[•▪◦●■□➢➤►‣⁃·–]\s*")
_BULLET_ONLY = set("•▪◦●■□➢➤►‣⁃·–")
_BOLD_FLAG = 16


def find_column_split(page: Any) -> float | None:
    """x position of the gutter if narrow text blocks form a left and a right column.

    A single-column page has full-width blocks, which are ignored. A false positive is NOT
    harmless: reading 'all left blocks, then all right blocks' would pull every right-aligned
    date away from its job title, so both sides must hold a real share of the text.

    Blocks are not assigned by their centre (a short heading of the main column can sit left
    of the page middle). Instead each block's left edge is tried as the start of the right
    column; it is a gutter if no block straddles it and both sides are substantial.
    """
    width = page.rect.width
    blocks = []  # (x0, y0, x1, y1, chars)
    for x0, y0, x1, y1, text, _no, kind in page.get_text("blocks"):
        n = len(text.strip())
        if kind == 0 and n >= MIN_BLOCK_CHARS and (x1 - x0) < NARROW_BLOCK * width:
            blocks.append((x0, y0, x1, y1, n))
    best: tuple[float, float] | None = None  # (smaller side's text share, gutter x)
    for start in sorted({round(b[0]) for b in blocks}):
        left = [b for b in blocks if b[2] <= start + SPLIT_TOLERANCE]
        right = [b for b in blocks if b[0] >= start - SPLIT_TOLERANCE]
        if not left or not right:
            continue
        # A centred name/title band may straddle the gutter; it is read as a header (see
        # order_blocks) as long as it is a small part of the page.
        straddle = sum(b[4] for b in blocks if b not in left and b not in right)
        if straddle > MAX_STRADDLE_SHARE * sum(b[4] for b in blocks):
            continue
        if len(left) < MIN_BLOCKS_PER_COLUMN or len(right) < MIN_BLOCKS_PER_COLUMN:
            continue
        l_chars, r_chars = sum(b[4] for b in left), sum(b[4] for b in right)
        share = min(l_chars, r_chars) / (l_chars + r_chars)
        if share < MIN_COLUMN_TEXT_SHARE:
            continue
        l_top, l_bot = min(b[1] for b in left), max(b[3] for b in left)
        r_top, r_bot = min(b[1] for b in right), max(b[3] for b in right)
        overlap = max(0.0, min(l_bot, r_bot) - max(l_top, r_top))
        shorter = min(l_bot - l_top, r_bot - r_top)
        if shorter <= 0 or overlap / shorter < MIN_VERTICAL_OVERLAP:
            continue
        gutter = (max(b[2] for b in left) + min(b[0] for b in right)) / 2
        if best is None or share > best[0]:
            best = (share, gutter)
    return best[1] if best else None


def detect_multi_column(page: Any) -> bool:
    return find_column_split(page) is not None


# --- reading order ----------------------------------------------------------------------


def order_blocks(blocks: list[dict], split: float) -> list[dict]:
    """Header band, left column, right column, footer band; each top to bottom."""
    left, right, full = [], [], []
    for b in blocks:
        x0, _y0, x1, _y1 = b["bbox"]
        if x1 <= split + SPLIT_TOLERANCE:
            left.append(b)
        elif x0 >= split - SPLIT_TOLERANCE:
            right.append(b)
        else:
            full.append(b)
    columns = left + right
    top = min((b["bbox"][1] for b in columns), default=0.0)
    bottom = max((b["bbox"][3] for b in columns), default=0.0)
    middle = (top + bottom) / 2
    head = [b for b in full if (b["bbox"][1] + b["bbox"][3]) / 2 < middle]
    tail = [b for b in full if b not in head]

    def key(b: dict) -> tuple[float, float]:
        return (b["bbox"][1], b["bbox"][0])

    return sorted(head, key=key) + sorted(left, key=key) + sorted(right, key=key) + sorted(
        tail, key=key
    )


# --- Markdown rendering -----------------------------------------------------------------


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    width: float


def _half_point(size: float) -> float:
    return round(size * 2) / 2


def _line_info(line: dict) -> _Line | None:
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return None
    text = "".join(s["text"] for s in line["spans"]).strip()
    words = [s for s in spans if s["text"].strip() not in _BULLET_ONLY]
    bold = bool(words) and all(
        (s["flags"] & _BOLD_FLAG) or "bold" in s["font"].lower() for s in words
    )
    x0, _, x1, _ = line["bbox"]
    return _Line(text, max(s["size"] for s in spans), bold, x1 - x0)


def _font_profile(blocks: list[dict]) -> tuple[float, dict[float, int]]:
    """Body font size (most characters) and a size -> heading level map."""
    weight: Counter[float] = Counter()
    for b in blocks:
        for line in b["lines"]:
            for s in line["spans"]:
                weight[_half_point(s["size"])] += len(s["text"].strip())
    if not weight:
        return 0.0, {}
    body = weight.most_common(1)[0][0]
    bigger = sorted({s for s in weight if s >= body * HEADING_RATIO}, reverse=True)
    return body, {s: min(i + 1, MAX_HEADING_LEVEL) for i, s in enumerate(bigger)}


def _continues(prev: _Line, prev_kind: str, cur: _Line, widest: float) -> bool:
    """True if `cur` is the wrapped remainder of `prev` rather than a new list item.

    Only a nearly full previous line can wrap. After a bullet, any non-bullet line is its
    continuation; in plain text we also need a lowercase/digit start (or a trailing comma),
    so a sidebar list such as "Apache Airflow" / "Spark" is not glued together.
    """
    if prev.width < WRAP_FILL * widest:
        return False
    if prev_kind == "bullet":
        return True
    first = cur.text[:1]
    return first.islower() or first.isdigit() or prev.text.endswith((",", "-", "/", "&"))


def render_block(block: dict, levels: dict[float, int]) -> list[str]:
    lines = [i for i in map(_line_info, block["lines"]) if i]
    if not lines:
        return []
    widest = max(i.width for i in lines)
    out: list[str] = []
    prev: _Line | None = None
    prev_kind, prev_level = "", 0
    for info in lines:
        if out and out[-1] == "- " and not _BULLET.match(info.text):
            out[-1] += info.text  # the bullet glyph was its own line; this is its text
            prev, prev_kind = info, "bullet"
            continue
        bullet = _BULLET.match(info.text)
        level = levels.get(_half_point(info.size), 0)
        if level and not bullet:
            if prev_kind == "heading" and prev_level == level:
                out[-1] += " " + info.text  # heading wrapped onto two lines
            else:
                out.append("#" * level + " " + info.text)
            prev_kind, prev_level = "heading", level
        elif bullet:
            out.append("- " + info.text[bullet.end():])
            prev_kind = "bullet"
        elif info.bold and len(info.text) <= MAX_LABEL_CHARS and not info.text.endswith((".", ",")):
            out.append(f"**{info.text}**")
            prev_kind = "label"
        elif prev is not None and prev_kind in ("bullet", "text") and _continues(
            prev, prev_kind, info, widest
        ):
            out[-1] += " " + info.text
        else:
            out.append(info.text)
            prev_kind = "text"
        prev = info
    return out


def column_markdown(page: Any, split: float) -> str:
    """Markdown for a two-column page, read column by column."""
    blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0 and b.get("lines")]
    _, levels = _font_profile(blocks)
    parts = []
    for block in order_blocks(blocks, split):
        rendered = render_block(block, levels)
        if rendered:
            parts.append("\n".join(rendered))
    text = "\n\n".join(parts)
    # A bullet glyph and its text can land in different blocks; join them.
    return re.sub(r"(?m)^([ \t]*)-[ \t]*\n+(?=\S)", r"\1- ", text)
