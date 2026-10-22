"""Optional LLM pass for job entries whose title/company the rules could not tell apart.

The model only sees one entry at a time and is asked for values copied from that text. Every
returned value is then checked against the source (grounding): a company or title that does
not appear in the entry is rejected, so the model cannot invent employers.
"""

from __future__ import annotations

from typing import Any

from src.logging_setup import get_logger
from src.metadata.schema import Experience
from src.providers.base import LLMProvider
from src.structure.text import normalize_key

log = get_logger(__name__)

MAX_ENTRY_CHARS = 800
ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "company": {"type": ["string", "null"]},
    },
    "required": ["title", "company"],
}
SYSTEM = (
    "You extract fields from one job entry of a resume. Copy the job title and the employer "
    "name exactly as written in the text. Use null when a field is not present. "
    "Never guess and never add words that are not in the text."
)


def needs_llm(exp: Experience) -> bool:
    return exp.confidence != "high" or not exp.title or not exp.company


def is_grounded(value: str, source: str) -> bool:
    """True if `value` appears in `source`, ignoring case, accents and punctuation."""
    key = normalize_key(value)
    return bool(key) and key in normalize_key(source)


def extract_entry(llm: LLMProvider, exp: Experience) -> dict[str, str]:
    """Grounded {'title': ..., 'company': ...} for one entry (missing keys = rejected)."""
    source = exp.raw_text[:MAX_ENTRY_CHARS]
    prompt = f"Resume entry:\n{source}\n\nReturn the job title and the company."
    try:
        data = llm.generate_json(prompt, ENTRY_SCHEMA, system=SYSTEM)
    except Exception as e:  # noqa: BLE001 - a failing model must not break indexing
        log.warning("LLM extraction failed: %s", e)
        return {}
    out: dict[str, str] = {}
    for field in ("title", "company"):
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            if is_grounded(value, source):
                out[field] = value.strip()
            else:
                log.warning("rejected ungrounded %s %r", field, value)
    return out


def enrich(llm: LLMProvider, experiences: list[Experience]) -> int:
    """Fill title/company of uncertain entries in place; returns how many were changed."""
    changed = 0
    for exp in experiences:
        if not needs_llm(exp):
            continue
        found = extract_entry(llm, exp)
        if "title" in found and "company" in found:
            exp.title, exp.company = found["title"], found["company"]
            exp.confidence, exp.source = "high", "llm"
            changed += 1
    return changed
