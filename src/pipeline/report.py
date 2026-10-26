"""Parse report: how every resume fared in the pipeline, and the Phase 1 quality gates
(plan section 7.5) computed from the stored results.

Read-only over the database and data folder. `scripts/parse_report.py` runs the pipeline on a
folder of resumes and writes this report as Markdown.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from src.config import Settings
from src.pipeline import overview
from src.stores import db

NON_CONTENT = ("header", "unknown")  # not counted as a recognised section
MIN_KNOWN_SECTIONS = 4
GATES = {
    "parsed_ok": 0.90,
    "known_tokens": 0.90,
    "four_sections": 0.95,
    "yoe_within": 0.90,
    "under_min": 0.05,
}
YOE_TOLERANCE = 0.5


@dataclass
class DocReport:
    doc_id: str
    filename: str
    status: str
    parser: str = ""
    seconds: float | None = None
    name: str = ""
    known_sections: int = 0
    known_share: float = 0.0
    jobs: int = 0
    yoe: float = 0.0
    education: int = 0
    skills: int = 0
    certifications: int = 0
    chunks: int = 0
    embedded: int = 0
    max_tokens: int = 0
    over_limit: int = 0
    entry_chunks: int = 0
    entry_chunks_complete: int = 0  # entry chunks that carry both a company and a start date
    expected_yoe: float | None = None
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def yoe_ok(self) -> bool | None:
        if self.expected_yoe is None:
            return None
        return abs(self.yoe - self.expected_yoe) <= YOE_TOLERANCE


@dataclass
class Gate:
    name: str
    value: float | None  # None = not measurable on this run
    threshold: str
    ok: bool | None
    note: str = ""
    advisory: bool = False  # reported, but never fails a run

    @property
    def failed(self) -> bool:
        return self.ok is False and not self.advisory


def document_reports(
    conn: sqlite3.Connection,
    settings: Settings,
    seconds: dict[str, float] | None = None,
    expected_yoe: dict[str, float] | None = None,
) -> list[DocReport]:
    """One row per upload. `expected_yoe` maps a file name stem to its ground-truth YOE."""
    seconds = seconds or {}
    expected_yoe = expected_yoe or {}
    out: list[DocReport] = []
    limit = settings.chunking.model_limit
    for row in reversed(db.list_document_rows(conn)):
        view = overview.document_view(conn, settings, row["doc_id"])
        rep = DocReport(
            doc_id=row["doc_id"],
            filename=row["filename"],
            status=row["status"],
            parser=row["parser"] or "",
            seconds=seconds.get(row["doc_id"]),
            error=row["error"] or "",
            expected_yoe=expected_yoe.get(row["filename"].rsplit(".", 1)[0]),
        )
        total = sum(s["words"] for s in view.sections)
        known = [s for s in view.sections if s["label"] not in NON_CONTENT]
        rep.known_sections = len({s["label"] for s in known})
        # header and contact text is recognised too, so only `unknown` counts against the share
        rep.known_share = (
            sum(s["words"] for s in view.sections if s["label"] != "unknown") / total
            if total else 0.0
        )
        if view.profile is not None:
            p = view.profile
            rep.name, rep.jobs, rep.yoe = p.name, len(p.experiences), p.total_yoe
            rep.education, rep.skills = len(p.education), len(p.skills)
            rep.certifications, rep.warnings = len(p.certifications), list(p.warnings)
        embedded = [c for c in view.chunks if c["embedded"]]
        rep.chunks, rep.embedded = len(view.chunks), len(embedded)
        rep.max_tokens = max((c["tokens"] or 0 for c in embedded), default=0)
        rep.over_limit = sum((c["tokens"] or 0) > limit for c in embedded)
        entries = [c for c in view.chunks if c["level"] == 2]
        rep.entry_chunks = len(entries)
        rep.entry_chunks_complete = sum(_has_company_and_date(c, view) for c in entries)
        out.append(rep)
    return out


def _has_company_and_date(chunk: dict, view: overview.DocumentView) -> bool:
    """An experience entry chunk must state its company and its start date in the embedded text."""
    if chunk["section"] != "experience":
        return True
    text = chunk["embed_text"]
    if view.profile is None:
        return False
    for exp in view.profile.experiences:
        if exp.company and chunk["entry"].endswith(exp.company):
            return exp.company in text and bool(exp.start) and exp.start in text
    return False


def _share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_gates(
    reports: list[DocReport], conn: sqlite3.Connection, settings: Settings
) -> list[Gate]:
    n = len(reports)
    gates: list[Gate] = []

    def add(name: str, value: float | None, threshold: str, ok: bool | None, note: str = "",
            advisory: bool = False):
        gates.append(Gate(name, value, threshold, ok, note, advisory))

    good = sum(r.status not in ("failed", "needs_review") for r in reports)
    v = _share(good, n)
    add("Parsed without needs_review / failed", v, f">= {GATES['parsed_ok']:.0%}",
        None if v is None else v >= GATES["parsed_ok"])

    parsed = [r for r in reports if r.known_share or r.known_sections]
    words = [r.known_share for r in parsed]
    v = sum(words) / len(words) if words else None
    add("Words in recognised sections (mean)", v, f">= {GATES['known_tokens']:.0%}",
        None if v is None else v >= GATES["known_tokens"])

    v = _share(sum(r.known_sections >= MIN_KNOWN_SECTIONS for r in parsed), len(parsed))
    add(f"Resumes with >= {MIN_KNOWN_SECTIONS} known sections", v,
        f">= {GATES['four_sections']:.0%}", None if v is None else v >= GATES["four_sections"])

    truth = [r for r in reports if r.expected_yoe is not None]
    v = _share(sum(bool(r.yoe_ok) for r in truth), len(truth))
    add(f"YOE within +/-{YOE_TOLERANCE} years of ground truth", v,
        f">= {GATES['yoe_within']:.0%}", None if v is None else v >= GATES["yoe_within"],
        "" if truth else "no ground truth for these files")

    over = sum(r.over_limit for r in reports)
    add("Chunks over the model limit", float(over), "0", over == 0)

    entries = sum(r.entry_chunks for r in reports)
    complete = sum(r.entry_chunks_complete for r in reports)
    v = _share(complete, entries)
    add("Experience chunks with company and dates", v, "100%", None if v is None else v == 1.0)

    sizes = _under_min_share(conn)
    add("Chunks under min_tok", sizes, f"< {GATES['under_min']:.0%}",
        None if sizes is None else sizes < GATES["under_min"],
        "advisory: meaningful on realistic 5-10 year resumes; synthetic ones have 2 short "
        "bullets per job", advisory=True)

    info = overview.index_overview(conn, settings)
    if info.embeddable_chunks == 0 or not any(m.manifest for m in info.models):
        add("Store count mismatch (DB vs vectors vs BM25)", None, "0", None,
            "no index built in this run")
    else:
        mismatch = sum(
            len(m.report.missing) + len(m.report.stale) + len(m.report.changed)
            for m in info.models if m.manifest
        ) + (0 if info.keyword_report.ok else 1)
        add("Store count mismatch (DB vs vectors vs BM25)", float(mismatch), "0", mismatch == 0)
    return gates


def _under_min_share(conn: sqlite3.Connection) -> float | None:
    from src.chunking.budgets import Budgets

    raw = db.get_chunk_budgets(conn)
    rows = db.list_chunks(conn, embedded_only=True)
    if not raw or not rows:
        return None
    budgets = Budgets.from_dict(raw)
    under = sum((r["n_tokens"] or 0) < budgets.for_section(r["section"]).min_tok for r in rows)
    return under / len(rows)


# --- Markdown ----------------------------------------------------------------------------


def _fmt(value: float | None, threshold: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}" if threshold == "0" else f"{value:.1%}"


def render_markdown(
    reports: list[DocReport], gates: list[Gate], title: str = "Parse report"
) -> str:
    lines = [f"# {title}", "", f"{len(reports)} resume(s).", "", "## Phase 1 gates", "",
             "| Gate | Value | Threshold | Result |", "|---|---|---|---|"]
    for g in gates:
        if g.ok is None:
            result = "not measured"
        elif g.ok:
            result = "PASS"
        else:
            result = "advisory miss" if g.advisory else "**FAIL**"
        note = f" ({g.note})" if g.note else ""
        value = _fmt(g.value, g.threshold)
        lines.append(f"| {g.name} | {value} | {g.threshold} | {result}{note} |")

    lines += ["", "## Per resume", "",
              "| File | Status | Parser | Sec | Known words | Jobs | YOE | Edu | Skills | Certs "
              "| Chunks | Max tok | Seconds |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        secs = "" if r.seconds is None else f"{r.seconds:.1f}"
        lines.append(
            f"| {r.filename} | {r.status} | {r.parser} | {r.known_sections} | "
            f"{r.known_share:.0%} | {r.jobs} | {r.yoe:.1f} | {r.education} | {r.skills} | "
            f"{r.certifications} | {r.embedded}/{r.chunks} | {r.max_tokens} | {secs} |"
        )

    flagged = [r for r in reports if r.warnings or r.error or r.yoe_ok is False]
    if flagged:
        lines += ["", "## Needs a look", ""]
        for r in flagged:
            lines.append(f"### {r.filename}")
            if r.error:
                lines.append(f"- error: {r.error}")
            if r.yoe_ok is False:
                lines.append(f"- YOE {r.yoe:.1f}, ground truth {r.expected_yoe}")
            lines += [f"- {w}" for w in r.warnings]
    times = [r.seconds for r in reports if r.seconds is not None]
    if times:
        lines += ["", f"Processing time: mean {sum(times) / len(times):.1f} s, "
                      f"max {max(times):.1f} s per resume (parse to chunk, no embedding)."]
    return "\n".join(lines) + "\n"
