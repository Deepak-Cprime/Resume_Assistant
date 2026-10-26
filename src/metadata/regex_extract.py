"""Contact fields and the candidate's name, found with regular expressions."""

from __future__ import annotations

import re
from collections.abc import Iterator

from src.structure.dates import find_range
from src.structure.sections import SectionMatcher, load_matcher
from src.structure.text import looks_like_name, strip_credentials, strip_md

# Starts with a letter/digit and ends the domain on one, so Markdown italics around an address
# ("_a@b.com_") are not read as part of it.
EMAIL_RE = re.compile(r"[^\W_][\w.+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/in/[\w\-%]+/?", re.I)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+(?![\w/])", re.I)
URL_RE = re.compile(r"https?://\S+")
PHONE_CANDIDATE = re.compile(r"(?<![\w@/])\+?\(?\d[\d\s().\-]{6,}\d")
_LOCATION = re.compile(r"[^\W\d_][\w .'’-]*,\s*[^\W\d_][\w .'’-]*")
_NOT_A_NAME = {"resume", "curriculum", "vitae", "cv", "profile"}
MIN_PHONE_DIGITS, MAX_PHONE_DIGITS, LONG_PHONE_DIGITS = 8, 15, 10
NAME_SEARCH_LINES = 8
MERGED_NAME_LINES = 3


def find_email(text: str) -> str:
    m = EMAIL_RE.search(text)
    return m.group(0).rstrip(".") if m else ""


def iter_phones(text: str) -> Iterator[re.Match[str]]:
    """Phone-like matches. A date range such as '2018 - 2021' has the same digits, so those are
    skipped, and a bare number needs a '+' or '(' or at least ten digits."""
    for m in PHONE_CANDIDATE.finditer(text):
        raw = m.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if not MIN_PHONE_DIGITS <= len(digits) <= MAX_PHONE_DIGITS or find_range(raw):
            continue
        if raw[0] in "+(" or len(digits) >= LONG_PHONE_DIGITS:
            yield m


def find_phone(text: str) -> str:
    m = next(iter_phones(text), None)
    return m.group(0).strip() if m else ""


def find_linkedin(text: str) -> str:
    m = LINKEDIN_RE.search(text)
    return m.group(0).rstrip("/") if m else ""


def find_github(text: str) -> str:
    m = GITHUB_RE.search(text)
    return m.group(0) if m else ""


def find_location(lines: list[str]) -> str:
    """'City, ST' or 'City, Country' from the contact area, ignoring email/phone/URLs."""
    for line in lines:
        text = EMAIL_RE.sub("|", strip_md(line))
        text = LINKEDIN_RE.sub("|", GITHUB_RE.sub("|", text))
        for segment in re.split(r"\s*[|·•]\s*", text):
            segment = segment.strip(" ,")
            if segment and not find_phone(segment) and _LOCATION.fullmatch(segment):
                return segment
    return ""


def find_name(lines: list[str], matcher: SectionMatcher | None = None) -> str:
    """Prefer a level-1 heading that reads like a name (it may follow a sidebar), then the
    first name-like line near the top."""
    matcher = matcher or load_matcher()

    def usable(text: str) -> bool:
        words = {w.lower().strip(".,") for w in text.split()}
        return (
            looks_like_name(text)
            and matcher.lookup(text) is None
            and not words & _NOT_A_NAME
        )

    candidates = [strip_credentials(strip_md(ln)) for ln in lines if re.match(r"^#\s+\S", ln)]
    near_top = [
        strip_credentials(strip_md(ln)) for ln in lines[: NAME_SEARCH_LINES * 3] if ln.strip()
    ]
    for text in candidates + near_top[:NAME_SEARCH_LINES]:
        if usable(text):
            return text.title() if text.isupper() else text
    # OCR can run the name into the contact line: "Rahul Verma rahul@x.com | +1 555 ..."
    for text in near_top[:MERGED_NAME_LINES]:
        cut = re.search(r"\S+@|[|+(]|\d", text)
        head = text[: cut.start()].strip(" ,|-") if cut else ""
        if head and usable(head):
            return head.title() if head.isupper() else head
    return ""
