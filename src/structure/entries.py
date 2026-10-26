"""Split a section into entries and parse their fields: jobs, education, certifications.

A job entry is anchored on its date range. The title/company header sits on the line above
the dates (`**Title, Company**` / `Mar 2021 - Dec 2024 | City`), on the same line
(`Title, Company  Mar 2021 - Dec 2024`), or just below (`2019 - 2021` / `Title - Company`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.metadata.schema import Certification, Education, Experience
from src.structure.dates import DateRange, find_range, strip_range
from src.structure.text import (
    SCHOOL_RE,
    TITLE_RE,
    bullet_text,
    is_bullet,
    is_degree_line,
    strip_md,
)

MAX_HEADER_WORDS = 14  # a dated line with more words around the range is prose, not a header
MAX_ROLE_WORDS = 8  # a job title line under a company header is short
MAX_PROSE_WORDS = 8  # a sentence (ends with '.') longer than this is body text, not a header
MAX_TITLE_WORDS = 15
MAX_CERT_WORDS = 14
_SEPARATORS = (" | ", " @ ", " at ", " — ", " – ", " - ", ", ", " · ")
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_INLINE_BULLET = re.compile(r"(?:(?<=\s)|^)[:•·.](?=\s+[A-Z])")  # a lone glyph before a Capital
# Words that end a degree's field of study ("... Computer Science | Anand Institute"), so they
# are never pulled into a school name when degree and school share a line.
_FIELD_WORDS = {
    "science", "sciences", "engineering", "technology", "arts", "studies", "mathematics",
    "statistics", "design", "systems", "administration", "business", "management", "physics",
}
MAX_SCHOOL_PREFIX_WORDS = 2
_STATE_CODE = re.compile(r"[A-Z]{2}$")  # a bare "MA" after "Boston," is a state, not a degree
_EDU_SEGMENTS = re.compile(r"\s*[|;]\s*|,\s+|\s+[–—-]\s+")


@dataclass
class RawEntry:
    header: list[str]  # plain-text header lines, date range removed
    date_range: DateRange | None
    location: str
    body: list[str]  # Markdown lines after the header
    raw_text: str


def _is_title_candidate(plain: str) -> bool:
    return bool(plain) and len(plain.split()) <= MAX_TITLE_WORDS and not plain.endswith(".")


# PDF converters sometimes write a bold company line as a list item: "- **Nimbus Retail** , Pune".
_BOLD_BULLET = re.compile(r"^\s*[-*•]\s+\*\*")


def _is_content_bullet(line: str) -> bool:
    """A real bullet point, as opposed to a header line that a converter formatted as one:
    a bold company name, or nothing but a date range ('- January 2015 - March 2019')."""
    if not is_bullet(line) or _BOLD_BULLET.match(line):
        return False
    text = strip_md(line)
    rng = find_range(text)
    if rng is not None:
        before, after = strip_range(text, rng)
        if not (before or after):
            return False
    return True


def _explode_inline_entry(line: str) -> list[str]:
    """OCR and some converters flatten a whole job into one paragraph:
    'Title, Company Jul 2019 - Sep 2022 | City : did this . did that'. Split it back into a
    header line plus '- ' bullets, using the stray ':' / '.' / '•' that stand for bullet glyphs."""
    if is_bullet(line):
        return [line]
    text = strip_md(line)
    rng = find_range(text)
    if rng is None:
        return [line]
    before, after = strip_range(text, rng)
    if len(before.split()) < 2 or not _INLINE_BULLET.search(after):
        return [line]
    location, *bullets = (p.strip(" |,;") for p in _INLINE_BULLET.split(after))
    dates = text[rng.span[0] : rng.span[1]]
    header = f"{before}  {dates}" + (f" | {location}" if location else "")
    return [header] + [f"- {b}" for b in bullets if b]


def split_entries(lines: list[str]) -> list[RawEntry]:
    lines = [part for ln in lines for part in _explode_inline_entry(ln)]
    idx = [i for i, ln in enumerate(lines) if ln.strip()]
    plain = {i: strip_md(lines[i]) for i in idx}
    dated: dict[int, DateRange] = {}
    for i in idx:
        if _is_content_bullet(lines[i]):
            continue
        rng = find_range(plain[i])
        if rng is None:
            continue
        before, after = strip_range(plain[i], rng)
        words = len(f"{before} {after}".split())
        is_prose = plain[i].endswith(".") and words > MAX_PROSE_WORDS
        if words <= MAX_HEADER_WORDS and not is_prose:
            dated[i] = rng
    if not dated:
        body = [ln for ln in lines if ln.strip()]
        return [RawEntry([], None, "", body, "\n".join(body))] if body else []

    # per entry: header line indexes, date line index, header texts, location
    metas: list[tuple[list[int], int, list[str], str]] = []
    prev_end = -1
    for d in sorted(dated):
        before, after = strip_range(plain[d], dated[d])
        header = [d]
        texts: list[str] = []
        location = ""
        if len(before.split()) >= 2:  # "Title, Company  Mar 2021 - Dec 2024"
            texts, location = [before], after
        else:
            above = [i for i in idx if prev_end < i < d]
            below = [i for i in idx if i > d]
            if above and _usable_header(lines, plain, dated, above[-1]):
                header = [above[-1], d]
                if _stacked_header(lines, plain, dated, above):
                    header = [above[-2], above[-1], d]
                texts, location = [plain[i] for i in header if i != d], after
            elif below and _usable_header(lines, plain, dated, below[0]):
                header = [d, below[0]]
                texts, location = [plain[below[0]]], after
            elif before:  # one word next to the dates: "Consultant  2020 - 2021"
                texts, location = [before], after
            elif after:  # only a date line: "2019 - 2021 | Acme Corp"
                texts = [after]
        metas.append((header, d, texts, location))
        prev_end = max(header)

    entries: list[RawEntry] = []
    for k, (header, d, texts, location) in enumerate(metas):
        start = min(header)
        end = max(header)
        next_start = min(metas[k + 1][0]) if k + 1 < len(metas) else len(lines)
        body = [ln for ln in lines[end + 1 : next_start] if ln.strip()]
        raw = "\n".join(lines[start : end + 1] + body)
        entries.append(RawEntry(texts, dated[d], location, body, raw))
    return entries


def _usable_header(lines: list[str], plain: dict[int, str], dated: dict, i: int) -> bool:
    return i not in dated and not _is_content_bullet(lines[i]) and _is_title_candidate(plain[i])


def _stacked_header(lines: list[str], plain: dict[int, str], dated: dict, above: list[int]) -> bool:
    """'Company' / 'Title' / dates: two short lines above the dates, exactly one of them a job
    title, and nothing but a bullet (or the section start) before them, so a previous entry's
    prose line is never pulled into this header."""
    if len(above) < 2 or not _usable_header(lines, plain, dated, above[-2]):
        return False
    first, second = plain[above[-2]], plain[above[-1]]
    if bool(TITLE_RE.search(first)) == bool(TITLE_RE.search(second)):
        return False
    return len(above) == 2 or is_bullet(lines[above[-3]])


def _assign(a: str, b: str) -> tuple[str, str, str]:
    a_title, b_title = bool(TITLE_RE.search(a)), bool(TITLE_RE.search(b))
    if a_title and not b_title:
        return a, b, "high"
    if b_title and not a_title:
        return b, a, "high"
    return a, b, "low"  # both or neither look like a title: assume "Title, Company"


def classify_header(texts: list[str]) -> tuple[str, str, str]:
    """(title, company, confidence) from header text such as 'Staff SRE, Orchard Streaming'."""
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        return "", "", "none"
    if len(texts) >= 2:
        return _assign(texts[0], texts[1])
    text = texts[0]
    for sep in _SEPARATORS:
        if sep in text:
            left, right = (p.strip() for p in text.split(sep, 1))
            if left and right:
                return _assign(left, right)
    return (text, "", "low") if TITLE_RE.search(text) else ("", text, "low")


def _title_below_company(
    raw: RawEntry, title: str, company: str
) -> tuple[str, str, str, list[str]] | None:
    """A header that is only 'Company, City' (no job title in it) followed by a line holding the
    title: returns (title, company, city, remaining body), else None."""
    if not raw.body or any(TITLE_RE.search(t) for t in raw.header):
        return None
    first_line = raw.body[0]
    first = strip_md(first_line)
    if is_bullet(first_line) or not TITLE_RE.search(first) or first.endswith("."):
        return None
    if len(first.split()) > MAX_ROLE_WORDS:
        return None
    name, city = (title, company) if title else (company, "")
    return first, name, city, raw.body[1:]


def parse_experience(raw: RawEntry) -> Experience:
    title, company, confidence = classify_header(raw.header)
    location = raw.location
    body = raw.body
    below = _title_below_company(raw, title, company)
    if below is not None:
        title, company, city, body = below
        location = location or city
        confidence = "high"
    bullets: list[str] = []
    description: list[str] = []
    for line in body:
        if is_bullet(line):
            bullets.append(bullet_text(line))
        elif bullets:
            bullets[-1] += " " + strip_md(line)  # wrapped continuation of the last bullet
        else:
            description.append(strip_md(line))
    rng = raw.date_range
    return Experience(
        title=title,
        company=company,
        location=location,
        start=rng.start.iso() if rng else None,
        end=rng.end.iso() if rng and rng.end else None,
        is_current=bool(rng and rng.is_current),
        bullets=bullets,
        description=" ".join(description),
        confidence=confidence,  # type: ignore[arg-type]
        raw_text=raw.raw_text,
    )


def _is_role_only(exp: Experience) -> bool:
    return bool(exp.title) and not exp.company


def parse_experiences(lines: list[str]) -> list[Experience]:
    """Jobs in reading order. Resumes often list a company once with its dates and then several
    roles under it ('Zenith Cloud, Bengaluru, 2021 - now' / 'Staff Engineer (2023 - now)' / ...).
    Such a company line has no text of its own; each role below it takes its company and, when
    it has none, its dates."""
    raws = split_entries(lines)
    parsed = [parse_experience(r) for r in raws]
    out: list[Experience] = []
    carry: Experience | None = None  # the company line currently in force
    for i, (raw, exp) in enumerate(zip(raws, parsed, strict=True)):
        following = parsed[i + 1] if i + 1 < len(parsed) else None
        if (
            not raw.body
            and following is not None
            and _is_role_only(following)
            and not any(TITLE_RE.search(t) for t in raw.header)
        ):
            # classify_header read 'Company, City' as (title, company); it is (company, city)
            name, city = (exp.title, exp.company) if exp.title else (exp.company, "")
            exp.company, exp.location, exp.title = name, city or exp.location, ""
            carry = exp
            continue
        if carry is not None and _is_role_only(exp):
            exp.company = carry.company
            exp.location = exp.location or carry.location
            if exp.start is None:
                exp.start, exp.end, exp.is_current = carry.start, carry.end, carry.is_current
            exp.confidence = "high"
        elif exp.company:
            carry = None
        out.append(exp)
    return out


# --- education --------------------------------------------------------------------------


def parse_education(lines: list[str]) -> list[Education]:
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        text = strip_md(line)
        if not text:
            continue
        if is_degree_line(text) and any(is_degree_line(t) for t in current):
            groups.append(current)
            current = []
        current.append(text)
    if current:
        groups.append(current)

    out: list[Education] = []
    for group in groups:
        segments = [s.strip() for t in group for s in _EDU_SEGMENTS.split(t) if s.strip()]
        degree = next((s for s in segments if is_degree_line(s) and not _STATE_CODE.match(s)), "")
        school = next((s for s in segments if SCHOOL_RE.search(s)), "")
        if degree and degree == school:  # "B.E. in Information Technology Georgia Valley College"
            degree, school = _split_degree_school(degree)
        if not (degree or school):
            continue
        out.append(Education(degree=degree, school=school, year=_education_year(group)))
    return out


def _split_degree_school(segment: str) -> tuple[str, str]:
    """Split 'DEGREE in FIELD School Name' at the school: the school keyword plus up to two
    capitalised words before it (stopping at field words) and an 'of X' tail after it."""
    words = segment.split()
    idx = next((i for i, w in enumerate(words) if SCHOOL_RE.fullmatch(w.strip(".,"))), None)
    if idx is None:
        return segment, segment
    start = idx
    while (
        start > 0
        and idx - start < MAX_SCHOOL_PREFIX_WORDS
        and words[start - 1][:1].isupper()
        and words[start - 1].lower().strip(".,") not in _FIELD_WORDS
        and not is_degree_line(words[start - 1])
    ):
        start -= 1
    end = idx + 1
    if end < len(words) and words[end].lower() in ("of", "de", "del", "for"):
        end = min(len(words), end + 3)
    degree = " ".join(words[:start]).strip(" ,")
    return (degree or segment), " ".join(words[start:end])


def _education_year(group: list[str]) -> int | None:
    text = " ".join(group)
    rng = find_range(text)
    if rng:
        return (rng.end or rng.start).year
    years = _YEAR.findall(text)
    return int(years[-1]) if years else None


# --- certifications ---------------------------------------------------------------------


def parse_certifications(lines: list[str]) -> list[Certification]:
    out: list[Certification] = []
    for line in lines:
        for part in re.split(r"\s*[|;]\s*", strip_md(line)):
            years = _YEAR.findall(part)
            name = re.sub(r"[\s(,\-–—]*\(?\b(?:19|20)\d{2}\b\)?[\s),]*$", "", part)
            name = name.strip(" ,-–—:•·")
            if len(name) < 3 or len(name.split()) > MAX_CERT_WORDS:
                continue
            out.append(Certification(name=name, year=int(years[-1]) if years else None))
    return list({c.name: c for c in out}.values())
