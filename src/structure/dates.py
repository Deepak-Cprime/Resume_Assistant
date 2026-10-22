"""Parse the many ways resumes write dates and date ranges.

Month precision is kept when the resume gives it ("Mar 2021", "07/2018"); a bare year stays a
year ("2018") and counts as mid-year when measuring durations, so that "2018 - 2021" measures
3 years on average instead of the 4 that Jan-to-Dec would give.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

MONTHS = {
    "jan": 1, "january": 1, "ene": 1, "enero": 1,
    "feb": 2, "february": 2, "febrero": 2,
    "mar": 3, "march": 3, "marzo": 3,
    "apr": 4, "april": 4, "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "june": 6, "junio": 6,
    "jul": 7, "july": 7, "julio": 7,
    "aug": 8, "august": 8, "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "september": 9, "set": 9, "septiembre": 9,
    "oct": 10, "october": 10, "octubre": 10,
    "nov": 11, "november": 11, "noviembre": 11,
    "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
}
MID_YEAR_MONTH = 7
MIN_YEAR, MAX_YEAR = 1950, 2100

_MON = "|".join(sorted(MONTHS, key=len, reverse=True))
_MONTH_NUM = r"(?:0?[1-9]|1[0-2])"
PRESENT = (
    r"(?:present|currently|current|now|today|ongoing|till\s+date|to\s+date|till\s+now"
    r"|actualidad|presente|hoy|actual)"
)
DATE_PATTERN = (
    rf"(?:\b(?:{_MON})\.?,?(?:\s+de)?\s+\d{{4}}"  # Jan 2021, Sept. 2021, enero de 2020
    rf"|(?<!\d){_MONTH_NUM}\s*[/.\-]\s*\d{{4}}"  # 07/2018, 7.2018
    rf"|\d{{4}}\s*[/\-]\s*{_MONTH_NUM}(?!\d)"  # 2021-03
    r"|(?<!\d)\d{4}(?!\d))"  # 2018
)
_SEPARATOR = r"(?:\s*[-–—‑]\s*|\s+(?:to|until|through|a|hasta)\s+)"
RANGE_RE = re.compile(
    rf"(?P<start>{DATE_PATTERN}){_SEPARATOR}(?P<end>{DATE_PATTERN}|\b{PRESENT}\b)", re.IGNORECASE
)
PRESENT_RE = re.compile(rf"^\s*{PRESENT}\s*$", re.IGNORECASE)
_NAMED_RE = re.compile(  # "Mar 2021", "Sept. 2021", "enero de 2020" or just "2018"
    rf"(?:(?P<month>{_MON})\.?,?(?:\s+de)?\s+)?(?P<year>\d{{4}})", re.IGNORECASE
)
_NUMERIC_RE = re.compile(rf"(?<!\d)({_MONTH_NUM})\s*[/.\-]\s*(\d{{4}})")
_ISO_RE = re.compile(rf"(\d{{4}})\s*[/\-]\s*({_MONTH_NUM})(?!\d)")


@dataclass(frozen=True)
class DateValue:
    year: int
    month: int | None = None  # None = the resume gave only a year

    def iso(self) -> str:
        return f"{self.year:04d}-{self.month:02d}" if self.month else f"{self.year:04d}"

    def month_index(self) -> int:
        """Months since year 0; a bare year counts as July."""
        return self.year * 12 + (self.month or MID_YEAR_MONTH) - 1


@dataclass(frozen=True)
class DateRange:
    start: DateValue
    end: DateValue | None  # None while is_current
    is_current: bool
    span: tuple[int, int]  # character offsets of the range inside the searched text

    def end_or(self, as_of: date) -> DateValue:
        return self.end or DateValue(as_of.year, as_of.month)


def parse_date(token: str) -> DateValue | None:
    """'Mar 2021', 'March 2021', '07/2018', '2021-03', '2018' -> DateValue; else None."""
    token = token.strip().strip(",.")
    if m := _ISO_RE.fullmatch(token):
        year, month = int(m.group(1)), int(m.group(2))
    elif m := _NUMERIC_RE.fullmatch(token):
        year, month = int(m.group(2)), int(m.group(1))
    elif m := _NAMED_RE.fullmatch(token):
        year = int(m.group("year"))
        month = MONTHS[m.group("month").lower()] if m.group("month") else None
    else:
        return None
    if not MIN_YEAR <= year <= MAX_YEAR:
        return None
    return DateValue(year, month)


def is_present(token: str) -> bool:
    return bool(PRESENT_RE.match(token))


def find_range(text: str) -> DateRange | None:
    """First date range in `text`, e.g. 'Mar 2021 - Present' or '2018 – 2021'."""
    for m in RANGE_RE.finditer(text):
        start = parse_date(m.group("start"))
        if start is None:
            continue
        end_text = m.group("end")
        if is_present(end_text):
            return DateRange(start, None, True, m.span())
        end = parse_date(end_text)
        if end is not None and end.month_index() >= start.month_index():
            return DateRange(start, end, False, m.span())
    return None


def strip_range(text: str, rng: DateRange) -> tuple[str, str]:
    """(text before the range, text after it), each trimmed of separators."""
    sep = " \t|,;:·•–—-()[]"
    return text[: rng.span[0]].strip(sep), text[rng.span[1] :].strip(sep)
