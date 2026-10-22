"""Normalise parser output before section detection.

Fixes ligatures and invisible characters, drops headers/footers repeated on several pages,
rejoins words hyphenated across lines, unifies bullet markers and tidies whitespace.
"""

from __future__ import annotations

import math
import re
from collections import Counter

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
_CHAR_MAP = str.maketrans(
    {
        **LIGATURES,
        " ": " ",
        " ": " ",
        " ": " ",
        "​": None,
        "‌": None,
        "‍": None,
        "﻿": None,
        "­": None,  # soft hyphen
    }
)

# lowercase fragment + hyphen + line break + lowercase continuation: "exper-\nience"
_HYPHEN_BREAK = re.compile(r"(?<![\w-])([a-z]{2,})-\n([a-z]{2,})")
# ':' and '.' are what OCR makes of a bullet glyph; a line never starts with them otherwise.
_BULLET = re.compile(r"(?m)^([ \t]*)(?:[•▪◦●■□➢➤►▸‣⁃·–:.]|[-*])[ \t]+")
_MAX_FOOTER_LEN = 80


def normalize_chars(text: str) -> str:
    return text.translate(_CHAR_MAP)


def _is_footer_candidate(line: str) -> bool:
    s = line.strip()
    return 0 < len(s) <= _MAX_FOOTER_LEN and not s.startswith(("- ", "* ", "• "))


def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())  # "Page 2 of 3" == "Page 1 of 3"


def drop_repeated_lines(pages: list[str]) -> list[str]:
    """Remove short lines that appear on most pages (running headers, footers, page numbers)."""
    if len(pages) < 2:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_line_key(ln) for ln in page.splitlines() if _is_footer_candidate(ln)})
    threshold = max(2, math.ceil(0.6 * len(pages)))
    repeated = {key for key, n in counts.items() if n >= threshold}
    return [
        "\n".join(
            ln for ln in page.splitlines()
            if not (_is_footer_candidate(ln) and _line_key(ln) in repeated)
        )
        for page in pages
    ]


def join_hyphenated(text: str) -> str:
    """Rejoin words split across lines. Only lowercase-start fragments are joined, so
    'Full-\\nstack' is left alone; 'cross-\\nfunctional' is joined wrongly (accepted trade-off)."""
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def normalize_bullets(text: str) -> str:
    return _BULLET.sub(r"\1- ", text)


def tidy_whitespace(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_pages(pages: list[str]) -> str:
    pages = [normalize_chars(p) for p in pages]
    pages = drop_repeated_lines(pages)
    text = "\n\n".join(p.strip("\n") for p in pages)
    return tidy_whitespace(normalize_bullets(join_hyphenated(text)))


def clean_markdown(text: str) -> str:
    return clean_pages([text])
