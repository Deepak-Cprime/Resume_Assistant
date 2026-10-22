"""Split a resume's Markdown into canonical sections.

1. A line is a heading when it matches a known heading (exact synonym, then fuzzy for
   Markdown/bold/UPPERCASE lines) from configs/sections_resume.yaml.
2. Unrecognised Markdown headings become `unknown` sections so nothing is silently merged.
3. A resume with fewer than two known headings (no headings, or only designer labels) falls
   back to `infer_sections`, which locates sections from their content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT
from src.metadata.skills import default_vocabulary
from src.structure.dates import find_range
from src.structure.text import (
    CERT_RE,
    SCHOOL_RE,
    is_bullet,
    is_degree_line,
    looks_like_name,
    normalize_key,
    strip_md,
)

DEFAULT_SECTIONS_PATH = PROJECT_ROOT / "configs" / "sections_resume.yaml"
FUZZY_THRESHOLD = 0.86
MIN_FUZZY_CHARS = 5
MAX_HEADING_CHARS = 60
MAX_HEADING_WORDS = 6
MAX_UNKNOWN_WORDS = 5
MAX_UNKNOWN_LEVEL = 2  # '#' and '##' can start an unknown section; deeper levels never do
MIN_KNOWN_SECTIONS = 2
MIN_SUMMARY_WORDS = 8
MAX_TITLE_WORDS = 15
NON_CONTENT_LABELS = ("header", "contact", "unknown")


@dataclass
class Section:
    label: str  # canonical label, "header" (text before the first heading) or "unknown"
    heading: str  # heading as written; "" for header and inferred sections
    lines: list[str]
    method: str  # synonym | fuzzy | inline | unknown | preamble | heuristic
    start: int  # index of the heading line in the document

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass(frozen=True)
class Heading:
    label: str
    text: str
    method: str
    rest: str = ""  # content that followed an inline heading ("Skills: Python, Go")


class SectionMatcher:
    def __init__(self, sections: dict[str, list[str]], inline_ok: list[str]) -> None:
        self.exact = {
            normalize_key(syn): label for label, syns in sections.items() for syn in syns
        }
        self.inline_ok = set(inline_ok)

    def lookup(self, text: str) -> str | None:
        return self.exact.get(normalize_key(text))

    def fuzzy(self, text: str) -> str | None:
        key = normalize_key(text)
        if len(key) < MIN_FUZZY_CHARS:
            return None
        best, label = 0.0, None
        for syn, lab in self.exact.items():
            if abs(len(syn) - len(key)) > 3:
                continue
            ratio = SequenceMatcher(None, key, syn).ratio()
            if ratio > best:
                best, label = ratio, lab
        return label if best >= FUZZY_THRESHOLD else None

    def classify(self, raw: str, *, allow_unknown: bool) -> Heading | None:
        s = raw.strip()
        if not s or is_bullet(s):
            return None
        is_md = bool(re.match(r"^#{1,6}\s", s))
        is_bold = bool(re.fullmatch(r"(\*\*|__).+\1:?", s))
        text = strip_md(s)

        if ":" in text:  # inline form: "Skills: Python, Go"
            head, _, rest = text.partition(":")
            label = self.lookup(head) if len(head.split()) <= 4 else None
            if label in self.inline_ok and rest.strip():
                return Heading(label, head.strip(), "inline", rest.strip())

        text = text.rstrip(":").strip()
        if (
            not text
            or len(text) > MAX_HEADING_CHARS
            or len(text.split()) > MAX_HEADING_WORDS
            or text.endswith(".")
        ):
            return None
        if label := self.lookup(text):
            return Heading(label, text, "synonym")
        emphasised = is_md or is_bold or (text.isupper() and sum(c.isalpha() for c in text) >= 3)
        if emphasised and (label := self.fuzzy(text)):
            return Heading(label, text, "fuzzy")
        level = len(s) - len(s.lstrip("#")) if is_md else 0
        if is_md and allow_unknown and self._plausible_unknown(text, level):
            return Heading("unknown", text, "unknown")
        return None

    @staticmethod
    def _plausible_unknown(text: str, level: int) -> bool:
        """An unrecognised Markdown heading that is probably a section title.

        PDF converters mark any slightly larger line as a heading, so deeper levels
        ('### M.S. in Statistics', '#### Anand Institute') and lines that read like degrees,
        schools or dates are entry details, not sections."""
        if level > MAX_UNKNOWN_LEVEL or len(text.split()) > MAX_UNKNOWN_WORDS:
            return False
        if re.search(r"[\d@|,]", text) or is_degree_line(text) or SCHOOL_RE.search(text):
            return False
        # '# Priya Raman' (level 1) is the name; '## Volunteer Work' is an unknown section.
        return not (level == 1 and looks_like_name(text))


@lru_cache(maxsize=4)
def load_matcher(path: Path | str | None = None) -> SectionMatcher:
    raw = yaml.safe_load(Path(path or DEFAULT_SECTIONS_PATH).read_text(encoding="utf-8"))
    return SectionMatcher(raw["sections"], raw.get("inline_ok", []))


def split_sections(markdown: str, matcher: SectionMatcher | None = None) -> list[Section]:
    matcher = matcher or load_matcher()
    lines = markdown.splitlines()
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    marks: list[tuple[int, Heading]] = []
    for i, line in enumerate(lines):
        heading = matcher.classify(line, allow_unknown=first is not None and i > first)
        if heading:
            marks.append((i, heading))

    sections: list[Section] = []
    preamble = lines[: marks[0][0] if marks else len(lines)]
    if any(ln.strip() for ln in preamble):
        sections.append(Section("header", "", preamble, "preamble", 0))
    for k, (i, h) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        body = ([h.rest] if h.rest else []) + lines[i + 1 : end]
        sections.append(Section(h.label, h.text, body, h.method, i))

    if known_labels(sections) < MIN_KNOWN_SECTIONS:
        inferred = infer_sections(lines)
        if inferred and known_labels(inferred) > known_labels(sections):
            return inferred
    return sections


def known_labels(sections: list[Section]) -> int:
    return len({s.label for s in sections if s.label not in NON_CONTENT_LABELS})


def by_label(sections: list[Section]) -> dict[str, list[Section]]:
    out: dict[str, list[Section]] = {}
    for s in sections:
        out.setdefault(s.label, []).append(s)
    return out


def section_lines(sections: list[Section], label: str) -> list[str]:
    return [ln for s in sections if s.label == label for ln in s.lines]


def known_token_share(sections: list[Section]) -> float:
    """Share of words that sit in a recognised section (header and contact count as known)."""
    total = known = 0
    for s in sections:
        n = len(s.text.split())
        total += n
        if s.label != "unknown":
            known += n
    return known / total if total else 0.0


# --- inference for resumes without usable headings -------------------------------------


def _is_title_candidate(text: str) -> bool:
    return bool(text) and len(text.split()) <= MAX_TITLE_WORDS and not text.endswith(".")


_SCHOOL_THEN_REST = re.compile(r"^(?P<edu>.*?\b(?:19|20)\d{2})\s+(?P<rest>\S.*)$")


def _split_school_and_skills(lines: list[str], vocab) -> list[str]:
    """PDF converters join consecutive short lines of one text block, so without headings
    'School, 2019' and a skills line can arrive as one: split them apart again."""
    out: list[str] = []
    for line in lines:
        m = _SCHOOL_THEN_REST.match(strip_md(line))
        if m and SCHOOL_RE.search(m["edu"]) and vocab.looks_like_skill_list(m["rest"]):
            out += [m["edu"], m["rest"]]
        else:
            out.append(line)
    return out


def infer_sections(lines: list[str]) -> list[Section] | None:
    """Locate sections from content: the first dated entry starts experience, a degree line
    next to a school starts education, a line of known skills starts skills, and so on.
    Assumes the usual order summary -> experience -> education -> skills -> certifications."""
    vocab = default_vocabulary()
    lines = _split_school_and_skills(lines, vocab)
    plain = [strip_md(ln) for ln in lines]
    nonblank = [i for i, p in enumerate(plain) if p]

    def near_education(i: int) -> bool:
        return any(is_degree_line(plain[j]) for j in nonblank if abs(j - i) <= 2)

    exp_date = next(
        (
            i for i in nonblank
            if not is_bullet(lines[i]) and find_range(plain[i]) and not near_education(i)
        ),
        None,
    )
    if exp_date is None:
        return None
    prev = [j for j in nonblank if j < exp_date]
    exp_start = prev[-1] if prev and _is_title_candidate(plain[prev[-1]]) else exp_date

    def first_after(start: int, test) -> int | None:
        return next((i for i in nonblank if i > start and test(i)), None)

    edu_start = first_after(
        exp_date,
        lambda i: not is_bullet(lines[i]) and is_degree_line(plain[i])
        and any(SCHOOL_RE.search(plain[j]) for j in nonblank if i <= j <= i + 2),
    )
    skills_start = first_after(
        edu_start or exp_date,
        lambda i: not is_bullet(lines[i]) and vocab.looks_like_skill_list(plain[i]),
    )
    cert_from = skills_start or edu_start or exp_date
    cert_start = first_after(
        cert_from, lambda i: bool(CERT_RE.search(plain[i])) and not is_degree_line(plain[i])
    )
    summary_start = next(
        (
            i for i in nonblank
            if i < exp_start and len(plain[i].split()) >= MIN_SUMMARY_WORDS
            and "@" not in plain[i] and "|" not in plain[i]
        ),
        None,
    )

    marks = {"summary": summary_start, "experience": exp_start, "education": edu_start,
             "skills": skills_start, "certifications": cert_start}
    ordered = sorted((idx, label) for label, idx in marks.items() if idx is not None)
    sections: list[Section] = []
    header_end = ordered[0][0]
    if any(ln.strip() for ln in lines[:header_end]):
        sections.append(Section("header", "", lines[:header_end], "preamble", 0))
    for k, (idx, label) in enumerate(ordered):
        end = ordered[k + 1][0] if k + 1 < len(ordered) else len(lines)
        sections.append(Section(label, "", lines[idx:end], "heuristic", idx))
    return sections
