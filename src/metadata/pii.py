"""Mask personal data before text is embedded or indexed, so a query cannot match a
candidate by email, phone or name and the vectors carry no contact details."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from src.metadata.regex_extract import (
    EMAIL_RE,
    GITHUB_RE,
    LINKEDIN_RE,
    URL_RE,
    iter_phones,
)

MIN_NAME_TOKEN = 3


def email_hash(email: str) -> str:
    """Stable identity for a candidate without storing the address."""
    email = email.strip().lower()
    return hashlib.sha256(email.encode("utf-8")).hexdigest() if email else ""


def mask_pii(text: str, names: Iterable[str] = ()) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    for rx in (LINKEDIN_RE, GITHUB_RE, URL_RE):
        text = rx.sub("[URL]", text)
    for m in reversed(list(iter_phones(text))):
        text = text[: m.start()] + "[PHONE]" + text[m.end() :]
    terms: set[str] = set()
    for name in names:
        name = name.strip()
        if len(name) >= MIN_NAME_TOKEN:
            terms.add(name)
        terms.update(t for t in re.split(r"\s+", name) if len(t.strip(".,")) >= MIN_NAME_TOKEN)
    for term in sorted(terms, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", "[NAME]", text, flags=re.IGNORECASE)
    return text
