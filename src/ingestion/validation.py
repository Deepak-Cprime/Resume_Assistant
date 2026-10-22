"""Decide whether parsed text is usable. Fatal issues send a document to `needs_review`
(or to the fallback parser first); warnings are only logged and reported."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from src.config import Settings

MIN_WORDS = 30
GARBAGE_FATAL = 0.05
GARBAGE_WARN = 0.01
MIN_ALPHA_RATIO = 0.4


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    fatal: bool


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def fatal(self) -> bool:
        return any(i.fatal for i in self.issues)

    @property
    def codes(self) -> list[str]:
        return [i.code for i in self.issues]

    def summary(self) -> str:
        return "; ".join(i.message for i in self.issues)


def garbage_ratio(text: str) -> float:
    """Share of characters that are replacement/control characters or '(cid:N)' glyph ids."""
    if not text:
        return 0.0
    bad = text.count("�") + text.count("(cid:") * 8
    bad += sum(1 for c in text if unicodedata.category(c) == "Cc" and c not in "\n\t\r")
    return min(1.0, bad / len(text))


def validate_markdown(
    markdown: str, n_pages: int, settings: Settings
) -> ValidationReport:
    report = ValidationReport()
    text = markdown.strip()
    if not text:
        report.issues.append(Issue("empty", "no text extracted", True))
        return report

    per_page = len(text) / max(n_pages, 1)
    if per_page < settings.parser.min_chars_per_page:
        report.issues.append(
            Issue("no_text", f"only {per_page:.0f} characters per page extracted", True)
        )
    if len(text.split()) < MIN_WORDS:
        report.issues.append(Issue("too_short", f"only {len(text.split())} words extracted", True))

    garbage = garbage_ratio(text)
    if garbage > GARBAGE_FATAL:
        report.issues.append(Issue("garbage", f"{garbage:.0%} garbage characters", True))
    elif garbage > GARBAGE_WARN:
        report.issues.append(Issue("some_garbage", f"{garbage:.1%} garbage characters", False))

    non_space = [c for c in text if not c.isspace()]
    alpha = sum(c.isalpha() for c in non_space) / max(len(non_space), 1)
    if alpha < MIN_ALPHA_RATIO:
        report.issues.append(Issue("low_alpha", f"only {alpha:.0%} letters", True))

    if "@" not in text:
        report.issues.append(Issue("no_email", "no email address found", False))
    return report
