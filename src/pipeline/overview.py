"""Read-only views of the corpus, assembled for the UI: counts, tables, the per-resume
inspection (raw file -> Markdown -> sections -> metadata -> chunks) and index health.

Nothing here writes. Pages call these functions and only format what they return."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from src.chunking.budgets import Budgets
from src.config import Settings
from src.metadata.schema import CandidateProfile
from src.pipeline import chunk_stage, index_stage, keyword_stage
from src.pipeline import manifest as mf
from src.stores import db, files

HISTOGRAM_BIN = 50  # tokens per bar


@dataclass
class CorpusStats:
    resumes: int
    candidates: int
    chunks: int
    embedded_chunks: int
    by_status: dict[str, int]

    @property
    def failed(self) -> int:
        return self.by_status.get("failed", 0)

    @property
    def needs_review(self) -> int:
        return self.by_status.get("needs_review", 0)

    @property
    def pending(self) -> int:
        return sum(self.by_status.get(s, 0) for s in db.PENDING_STATUSES)


def corpus_stats(conn: sqlite3.Connection) -> CorpusStats:
    by_status = db.status_counts(conn)
    return CorpusStats(
        resumes=sum(by_status.values()),
        candidates=db.count_candidates(conn),
        chunks=db.count_chunks(conn),
        embedded_chunks=db.count_chunks(conn, embedded_only=True),
        by_status=by_status,
    )


def document_rows(conn: sqlite3.Connection) -> list[dict]:
    return [
        {
            "file": r["filename"],
            "status": r["status"],
            "type": r["doc_type"] or "",
            "candidate": r["display_name"] or "",
            "parser": r["parser"] or "",
            "note": r["error"] or "",
            "uploaded": r["created_at"],
            "doc_id": r["doc_id"],
        }
        for r in db.list_document_rows(conn)
    ]


def candidate_rows(conn: sqlite3.Connection) -> list[dict]:
    return [
        {
            "candidate_id": r["candidate_id"],
            "name": r["display_name"] or "(name not found)",
            "headline": r["headline"] or "",
            "years": r["total_yoe"],
            "seniority": r["seniority"] or "",
            "chunks": r["n_chunks"],
            "uploads": r["n_documents"],
        }
        for r in db.list_candidate_rows(conn)
    ]


# --- one resume, stage by stage --------------------------------------------------------


@dataclass
class DocumentView:
    row: dict
    raw_path: Path | None
    markdown: str
    sections: list[dict]
    profile: CandidateProfile | None
    chunks: list[dict]
    notes: list[str] = field(default_factory=list)  # why a tab is empty

    @property
    def facts(self) -> dict[str, str]:
        """Headline numbers and contact details for the Metadata tab."""
        p = self.profile
        if p is None:
            return {}
        return {
            "Name": p.name or "(not found)",
            "Email": p.contact.email or "(not found)",
            "Phone": p.contact.phone or "(not found)",
            "Location": p.contact.location or "(not found)",
            "Headline": p.headline or "(none)",
            "Seniority": p.seniority or "(none)",
            "Years of experience": f"{p.total_yoe:.1f}",
        }

    @property
    def gaps_table(self) -> list[dict]:
        return [] if self.profile is None else [
            {"from": g.start, "to": g.end, "months": g.months} for g in self.profile.gaps
        ]

    @property
    def education_table(self) -> list[dict]:
        return [] if self.profile is None else [
            {"degree": e.degree, "school": e.school, "year": e.year or ""}
            for e in self.profile.education
        ]

    @property
    def skills_table(self) -> list[dict]:
        return [] if self.profile is None else [
            {"skill": s.skill, "evidence": s.evidence, "claimed": s.claimed, "used": s.used,
             "years": s.years or ""}
            for s in self.profile.skills
        ]

    @property
    def certifications_table(self) -> list[dict]:
        return [] if self.profile is None else [
            {"certification": c.name, "year": c.year or ""} for c in self.profile.certifications
        ]

    @property
    def experience_table(self) -> list[dict]:
        if self.profile is None:
            return []
        return [
            {
                "title": e.title,
                "company": e.company,
                "start": e.start or "",
                "end": "present" if e.is_current else (e.end or ""),
                "location": e.location,
                "split": e.confidence + (" (llm)" if e.source == "llm" else ""),
            }
            for e in self.profile.experiences
        ]


def _read_text(path_str: str | None) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def document_view(conn: sqlite3.Connection, settings: Settings, doc_id: str) -> DocumentView:
    """Everything known about one upload, whatever stage it reached."""
    row = db.get_document(conn, doc_id)
    if row is None:
        raise KeyError(doc_id)
    notes: list[str] = []

    sections_file = files.sections_path(settings, doc_id)
    sections = (
        json.loads(sections_file.read_text(encoding="utf-8")) if sections_file.exists() else []
    )
    for s in sections:
        s["words"] = len(s["text"].split())

    chunk_rows = db.list_chunks_for_document(conn, doc_id)
    key_by_id = {c["chunk_id"]: c["chunk_key"] for c in chunk_rows}
    chunks = [
        {
            "key": c["chunk_key"],
            "level": c["level"],
            "section": c["section"],
            "entry": c["entry_title"] or "",
            "tokens": c["n_tokens"],
            "embedded": bool(c["embed"]),
            "parent": key_by_id.get(c["parent_id"], ""),
            "skills": ", ".join(json.loads(c["skills"] or "[]")),
            "text": c["text"],
            "embed_text": c["embed_text"],
        }
        for c in chunk_rows
    ]

    status = row["status"]
    if not row["markdown_path"]:
        notes.append(f"Not parsed yet (status: {status}).")
    if row["error"]:
        notes.append(row["error"])
    if row["markdown_path"] and not sections and status not in ("parsed",):
        notes.append("No sections were saved for this document.")
    if row["candidate_id"] is not None and not chunks:
        notes.append("Another upload of this candidate owns the current chunks.")

    return DocumentView(
        row=dict(row),
        raw_path=Path(row["raw_path"]) if row["raw_path"] else None,
        markdown=_read_text(row["markdown_path"]),
        sections=sections,
        profile=chunk_stage.load_profile(settings, doc_id),
        chunks=chunks,
        notes=notes,
    )


# --- index health ----------------------------------------------------------------------


def size_histogram(sizes: list[int], bin_width: int = HISTOGRAM_BIN) -> list[tuple[str, int]]:
    """(label, count) per bin, including empty bins between the smallest and largest."""
    if not sizes:
        return []
    top = max(sizes) // bin_width
    counts = [0] * (top + 1)
    for n in sizes:
        counts[n // bin_width] += 1
    return [(f"{i * bin_width}-{(i + 1) * bin_width - 1}", c) for i, c in enumerate(counts)]


@dataclass
class ModelIndexInfo:
    model: str
    configured: bool
    manifest: mf.IndexManifest | None
    report: index_stage.IndexReport


@dataclass
class IndexOverview:
    models: list[ModelIndexInfo]
    keyword_manifest: keyword_stage.KeywordManifest | None
    keyword_report: keyword_stage.KeywordReport
    budgets: Budgets | None
    sizes: list[tuple[str, int]]
    embeddable_chunks: int

    @property
    def budget_rows(self) -> list[dict]:
        if self.budgets is None:
            return []
        return [
            {"section": name, "min tokens": b.min_tok, "max tokens": b.max_tok,
             "samples": b.n_units, "source": b.source}
            for name, b in sorted(self.budgets.types.items())
        ]


def _safe_report(
    conn: sqlite3.Connection, settings: Settings, model: str
) -> index_stage.IndexReport:
    """A damaged or half-deleted index must show up as 'out of date', not crash the page."""
    try:
        return index_stage.verify_model_index(conn, settings, model)
    except Exception:  # noqa: BLE001 - unreadable vectors file, unknown store, ...
        wanted = [r["chunk_id"] for r in db.list_chunks(conn, embedded_only=True)]
        return index_stage.IndexReport(model, len(wanted), 0, missing=sorted(wanted))


def index_overview(conn: sqlite3.Connection, settings: Settings) -> IndexOverview:
    """Manifests and consistency reports for every configured model, plus any model that has an
    index on disk, the keyword index and the chunk budgets. Loads no embedding model."""
    configured = [m.name for m in settings.embedding.models]
    on_disk = [r["emb_model"] for r in db.current_index_manifests(conn)]
    names = list(dict.fromkeys([*configured, *on_disk]))
    models = [
        ModelIndexInfo(
            model=name,
            configured=name in configured,
            manifest=mf.read_manifest(settings, name),
            report=_safe_report(conn, settings, name),
        )
        for name in names
    ]
    raw = db.get_chunk_budgets(conn)
    return IndexOverview(
        models=models,
        keyword_manifest=keyword_stage.read_keyword_manifest(settings),
        keyword_report=keyword_stage.verify_keyword_index(conn, settings),
        budgets=Budgets.from_dict(raw) if raw else None,
        sizes=size_histogram(db.embedded_chunk_sizes(conn)),
        embeddable_chunks=db.count_chunks(conn, embedded_only=True),
    )
