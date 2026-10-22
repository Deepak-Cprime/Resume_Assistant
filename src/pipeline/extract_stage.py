"""Extract stage for one document: structured -> extracted (profile saved, DB rows written)."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from src.config import Settings
from src.logging_setup import get_logger
from src.metadata.profile import build_profile
from src.providers import factory
from src.providers.base import LLMProvider
from src.stores import db, files

log = get_logger(__name__)


def _llm_for(settings: Settings) -> LLMProvider | None:
    if not settings.metadata.use_llm:
        return None
    try:
        return factory.create_llm(settings)
    except Exception as e:  # noqa: BLE001 - extraction still works from rules alone
        log.warning("LLM extraction requested but unavailable (%s); using rules only", e)
        return None


def extract_document(
    conn: sqlite3.Connection,
    settings: Settings,
    doc_id: str,
    *,
    llm: LLMProvider | None = None,
) -> str:
    """Build the candidate profile, save it as JSON and write candidate rows.

    `llm` overrides the configured one (tests pass a fake). A resume with no job entries is
    still saved but marked `needs_review`. Documents not in `structured` are skipped.
    """
    row = db.get_document(conn, doc_id)
    if row is None:
        raise KeyError(doc_id)
    if row["status"] != "structured":
        return row["status"]

    try:
        markdown = Path(row["markdown_path"]).read_text(encoding="utf-8")
        profile = build_profile(
            markdown,
            as_of=settings.metadata.as_of or date.today(),
            gap_months=settings.metadata.gap_months,
            llm=llm or _llm_for(settings),
        )
        files.write_text_atomic(
            files.profile_path(settings, doc_id), profile.model_dump_json(indent=2)
        )
        candidate_id = db.upsert_candidate_profile(
            conn,
            display_name=profile.name,
            email_hash=profile.email_hash,
            headline=profile.headline,
            total_yoe=profile.total_yoe,
            seniority=profile.seniority,
            experiences=[
                {"title": e.title, "company": e.company, "start": e.start, "end": e.end}
                for e in profile.experiences
            ],
            skills=[
                {"skill": s.skill, "evidence": s.evidence, "years": s.years}
                for s in profile.skills
            ],
            certifications=[{"name": c.name, "year": c.year} for c in profile.certifications],
        )
    except Exception as e:  # noqa: BLE001 - one bad document must not stop the batch
        log.exception("extracting %s failed", doc_id[:12])
        db.set_document_status(conn, doc_id, "failed", error=f"extract failed: {e}")
        return "failed"

    if not profile.experiences:
        db.set_document_status(
            conn, doc_id, "needs_review",
            error="no job entries found", candidate_id=candidate_id,
        )
        return "needs_review"
    db.set_document_status(conn, doc_id, "extracted", candidate_id=candidate_id)
    return "extracted"
