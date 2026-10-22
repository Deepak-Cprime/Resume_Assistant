"""Context added to a chunk before it is embedded: structural prefix, skill tags, PII masking.

`text` is what the interviewer sees and what citations quote. `embed_text` is what the embedder
and BM25 see: the prefix makes a bullet that says "Cut infra cost by 32%" searchable by company,
title and dates. The candidate is referred to by ID, never by name.
"""

from __future__ import annotations

from collections.abc import Callable

from src.metadata.pii import mask_pii
from src.metadata.schema import Experience

SECTION_TITLES = {
    "profile": "Profile",
    "experience": "Experience",
    "skills": "Skills",
    "education": "Education",
    "certifications": "Certifications",
    "projects": "Projects",
    "awards": "Awards",
    "publications": "Publications",
    "languages": "Languages",
    "volunteering": "Volunteering",
    "interests": "Interests",
    "other": "Other",
}
DASH = "–"


def candidate_label(candidate_id: int) -> str:
    return f"C{candidate_id}"


def section_title(label: str) -> str:
    return " + ".join(SECTION_TITLES.get(part, part.title()) for part in label.split("+"))


def dates_label(start: str | None, end: str | None, is_current: bool) -> str:
    """'2021-03 – 2024-12', '2021-03 – present', '2018' or '' when the resume gave no dates."""
    if not start:
        return ""
    if is_current or not end:
        return f"{start} {DASH} present" if is_current else start
    return f"{start} {DASH} {end}"


def entry_title(exp: Experience) -> str:
    """'Senior Backend Engineer @ Northwind Payments' (either part may be missing)."""
    if exp.title and exp.company:
        return f"{exp.title} @ {exp.company}"
    return exp.title or exp.company


def entry_header(exp: Experience) -> str:
    """The title line shown at the top of an entry chunk: title @ company (dates)."""
    dates = dates_label(exp.start, exp.end, exp.is_current)
    title = entry_title(exp)
    return f"{title} ({dates})" if title and dates else title or dates


def section_prefix(candidate_id: int, label: str) -> str:
    return f"Candidate {candidate_label(candidate_id)} | {section_title(label)}"


def entry_prefix(candidate_id: int, exp: Experience) -> str:
    return f"{section_prefix(candidate_id, 'experience')} > {entry_header(exp)}".rstrip(" >")


def tags_line(skills: list[str]) -> str:
    return f"Skills: {', '.join(skills)}" if skills else ""


def assemble_embed_text(
    prefix: str,
    body: str,
    tags: str,
    *,
    names: list[str],
    limit: int,
    count: Callable[[str], int],
) -> str:
    """prefix + body + skill tags, PII-masked, and guaranteed to fit `limit` tokens.

    If it does not fit, the skill tags go first, then the end of the body is cut word by word.
    """
    def build(b: str, t: str) -> str:
        return mask_pii("\n".join(p for p in (prefix, b, t) if p), names)

    text = build(body, tags)
    if count(text) <= limit:
        return text
    text = build(body, "")
    words = body.split()
    while count(text) > limit and words:
        words.pop()
        text = build(" ".join(words), "")
    while count(text) > limit and len(text) > 1:
        text = text[: int(len(text) * 0.9)]  # last resort for a single enormous token
    return text
