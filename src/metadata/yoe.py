"""Years of experience as the union of job intervals, plus career gaps.

Summing job durations double-counts overlapping roles (a full-time job plus freelance work);
the union counts each calendar month once.
"""

from __future__ import annotations

from datetime import date

from src.metadata.schema import Experience, Gap
from src.structure.dates import DateValue, parse_date

DEFAULT_GAP_MONTHS = 3


def month_index_to_iso(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def intervals(experiences: list[Experience], as_of: date) -> list[tuple[int, int]]:
    """(start, end) month indexes for every job with a usable start date."""
    out: list[tuple[int, int]] = []
    for exp in experiences:
        start = parse_date(exp.start) if exp.start else None
        if start is None:
            continue
        if exp.is_current or not exp.end:
            end = DateValue(as_of.year, as_of.month)
        else:
            end = parse_date(exp.end) or DateValue(as_of.year, as_of.month)
        if end.month_index() >= start.month_index():
            out.append((start.month_index(), end.month_index()))
    return out


def merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def months_covered(spans: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge(spans))


def compute_yoe(
    experiences: list[Experience], as_of: date, gap_months: int = DEFAULT_GAP_MONTHS
) -> tuple[float, list[Gap]]:
    """(years, gaps longer than `gap_months` between consecutive employment periods)."""
    merged = merge(intervals(experiences, as_of))
    gaps = [
        Gap(
            start=month_index_to_iso(a_end),
            end=month_index_to_iso(b_start),
            months=b_start - a_end,
        )
        for (_, a_end), (b_start, _) in zip(merged, merged[1:], strict=False)
        if b_start - a_end > gap_months
    ]
    return round(sum(e - s for s, e in merged) / 12, 2), gaps
