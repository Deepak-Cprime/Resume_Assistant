"""Line-level helpers and patterns shared by section detection and entry parsing."""

from __future__ import annotations

import re
import unicodedata

_BULLET_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def strip_md(line: str) -> str:
    """Plain text of one Markdown line: no heading marks, bullet marker, bold or italics."""
    s = re.sub(r"\\([_*#\[\]])", r"\1", line.strip())
    s = re.sub(r"^#{1,6}\s*", "", s)
    s = re.sub(r"^[-*•]\s+", "", s)
    s = re.sub(r"(\*\*|__)(.+?)\1", r"\2", s)
    s = re.sub(r"(?<![\w*])([*_])(.+?)\1(?![\w*])", r"\2", s)
    return s.strip()


def is_bullet(line: str) -> bool:
    return bool(_BULLET_LINE.match(line))


def bullet_text(line: str) -> str:
    return _BULLET_LINE.sub("", line, count=1).strip()


def normalize_key(text: str) -> str:
    """Lowercase, accent-free, punctuation-free form used for matching headings and names."""
    text = unicodedata.normalize("NFKD", text.replace("&", " and "))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


_CREDENTIALS = re.compile(
    r"\s*[,(]\s*(?:ph\.?\s?d|m\.?\s?b\.?\s?a|m\.?\s?sc?|b\.?\s?sc?|m\.?\s?d|p\.?\s?e|pmp|cpa|"
    r"cissp|cfa|esq|phd|msc|mba)\b\.?\)?(?:\s*[,/]\s*[A-Za-z.]+)*\s*$",
    re.IGNORECASE,
)


def strip_credentials(text: str) -> str:
    """'Mei Tanaka, PhD' -> 'Mei Tanaka' (degrees and certifications after a name)."""
    return _CREDENTIALS.sub("", text).strip()


def looks_like_name(text: str) -> bool:
    """2-4 capitalised alphabetic words, e.g. 'Priya Raman' or 'Lucía Fernández'."""
    words = text.split()
    if not 2 <= len(words) <= 4:
        return False
    for w in words:
        core = w.replace("'", "").replace("’", "").replace("-", "").replace(".", "")
        if not core or not core.isalpha() or not w[0].isupper():
            return False
    return True


# A degree starts a line ("B.Tech in Computer Science") or is spelled out; abbreviations are
# matched only at the line start so "Boston, MA" in a location is not read as an M.A.
DEGREE_ABBR = re.compile(
    r"^(?:B\.?\s?Tech|B\.?\s?E|B\.?\s?Sc?|B\.?\s?A|M\.?\s?Tech|M\.?\s?Sc?|M\.?\s?A|M\.?\s?E"
    r"|MBA|Ph\.?\s?D)\.?(?![A-Za-z])"
)
DEGREE_WORDS = re.compile(
    r"\b(?:bachelors?|masters?|doctorate|diploma|associate degree|grado|licenciatura"
    r"|m[aá]ster|ingenier[ií]a|ph\.?d)\b",
    re.IGNORECASE,
)
SCHOOL_RE = re.compile(
    r"\b(?:university|college|institute|school|academy|polytechnic|universidad|universit[eé]"
    r"|instituto|escuela)\b",
    re.IGNORECASE,
)
CERT_RE = re.compile(
    r"\b(?:certified|certificate|certification|licen[sc]e|specialist|associate|professional"
    r"|certificado|certificaci[oó]n)\b",
    re.IGNORECASE,
)
TITLE_RE = re.compile(
    r"\b(?:engineers?|developers?|programmers?|architects?|managers?|directors?|leads?|head"
    r"|analysts?|scientists?|consultants?|designers?|administrators?|specialists?|officers?"
    r"|interns?|associates?|founders?|ctos?|ceos?|vp|presidents?|sre|devops|technicians?"
    r"|coordinators?|supervisors?|researchers?|professors?|teachers?|trainers?|owners?"
    r"|ingenier[oa]s?|analistas?|desarrolladores?|desarrolladora|gerentes?|directoras?"
    r"|consultor(?:a|es)?|arquitect[oa]s?|t[eé]cnic[oa]s?|especialistas?|jef[ea]s?"
    r"|l[ií]deres?)\b",
    re.IGNORECASE,
)


def is_degree_line(text: str) -> bool:
    return bool(DEGREE_ABBR.match(text) or DEGREE_WORDS.search(text))
