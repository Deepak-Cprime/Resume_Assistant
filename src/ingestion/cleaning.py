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

# word + hyphen + line break + word: "exper-\nience", "customer-\nvisible", "ananya-\nkrishnan"
_HYPHEN_BREAK = re.compile(r"(?<![\w-])([A-Za-z]{2,})-\n([a-z]{2,})")
# What follows the hyphen when a long word was split by the typesetter ("informa-tion"). In a
# resume a trailing hyphen is far more often a real one (compounds, URLs), so only these
# word endings are glued back; everything else keeps its hyphen.
_WORD_ENDING = re.compile(
    r"(?:tion|sion|ment|ing|ity|ness|able|ible|ence|ience|ance|ally|ous|ive|ize|ise|ical|ular"
    r"|ated|ation|ure)s?"
)
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


def _rejoin(match: re.Match[str]) -> str:
    left, right = match.groups()
    if _WORD_ENDING.fullmatch(right):
        return left + right  # typesetter hyphenation: "informa-\ntion" -> "information"
    return f"{left}-{right}"  # a real hyphen that happened to end the line: keep it


def join_hyphenated(text: str) -> str:
    """Remove the line break after a trailing hyphen. 'informa-\\ntion' becomes 'information';
    'customer-\\nvisible' and a URL such as 'in/ananya-\\nkrishnan' keep their hyphen."""
    return _HYPHEN_BREAK.sub(_rejoin, text)


# A converter that merges a whole block into one paragraph leaves bullet glyphs mid-line, and a
# date-first entry ("2018 - 2019 **Web Developer** ...") then looks like a list item.
_MIDLINE_BULLET = re.compile(r"(?<=\S)[ \t]+[•▪◦●■□➢➤►▸‣⁃][ \t]+")
_DATE_FIRST_ITEM = re.compile(
    r"(?mi)^[ \t]*[-*][ \t]+"
    r"(?=(?:19|20)\d{2}(?:[ \t]*[-–][ \t]*(?:(?:19|20)\d{2}|present|now))?[ \t]+\*\*)"
)


def normalize_bullets(text: str) -> str:
    text = _MIDLINE_BULLET.sub("\n- ", text)
    text = _BULLET.sub(r"\1- ", text)
    return _DATE_FIRST_ITEM.sub("", text)


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
