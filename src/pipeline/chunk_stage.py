"""Chunk stage for one document: extracted -> chunked (budgets ensured, chunks written)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.chunking.budgets import Budgets, compute_budgets, needs_recompute
from src.chunking.chunker import Counter, build_chunks, unit_lengths
from src.chunking.tokens import estimate_tokens
from src.config import Settings
from src.logging_setup import get_logger
from src.metadata.schema import CandidateProfile
from src.stores import db, files

log = get_logger(__name__)
CHUNKED_OR_LATER = ("extracted", "chunked", "indexed")


def load_profile(settings: Settings, doc_id: str) -> CandidateProfile | None:
    path: Path = files.profile_path(settings, doc_id)
    if not path.exists():
        return None
    return CandidateProfile.model_validate_json(path.read_text(encoding="utf-8"))


def corpus_lengths(
    conn: sqlite3.Connection, settings: Settings, count: Counter
) -> tuple[dict[str, list[int]], int]:
    """Unit lengths of every extracted resume (one per candidate) and the candidate count."""
    rows = conn.execute(
        "SELECT doc_id, candidate_id FROM documents WHERE candidate_id IS NOT NULL "
        f"AND status IN ({', '.join('?' for _ in CHUNKED_OR_LATER)}) ORDER BY created_at, doc_id",
        CHUNKED_OR_LATER,
    ).fetchall()
    lengths: dict[str, list[int]] = {}
    seen: set[int] = set()
    for row in rows:
        if row["candidate_id"] in seen:
            continue
        profile = load_profile(settings, row["doc_id"])
        if profile is None:
            continue
        seen.add(row["candidate_id"])
        for label, values in unit_lengths(profile, count).items():
            lengths.setdefault(label, []).extend(values)
    return lengths, len(seen)


def ensure_budgets(
    conn: sqlite3.Connection, settings: Settings, count: Counter = estimate_tokens
) -> tuple[Budgets, bool]:
    """The role's stored budgets, recomputed only when needs_recompute says so.

    Returns (budgets, recomputed). When `recomputed` is True, resumes chunked earlier were built
    with different budgets and should be re-chunked (the pipeline orchestrator does that)."""
    lengths, n_candidates = corpus_lengths(conn, settings, count)
    raw = db.get_chunk_budgets(conn)
    stored = Budgets.from_dict(raw) if raw else None
    if not needs_recompute(stored, lengths, n_candidates, settings.chunking):
        return stored, False  # type: ignore[return-value]  # needs_recompute is True when None
    fresh = compute_budgets(lengths, n_candidates=n_candidates, cfg=settings.chunking)
    db.set_chunk_budgets(conn, fresh.to_dict())
    log.info("chunk budgets recomputed from %d candidates (%s)", n_candidates, fresh.fingerprint())
    return fresh, True


def chunk_document(
    conn: sqlite3.Connection,
    settings: Settings,
    doc_id: str,
    *,
    count: Counter = estimate_tokens,
) -> str:
    """Chunk one extracted resume and replace its candidate's chunks in the database.

    Returns the new status. Documents not in `extracted` are skipped; to re-chunk one after a
    budget change, move it back to `extracted` first."""
    row = db.get_document(conn, doc_id)
    if row is None:
        raise KeyError(doc_id)
    if row["status"] != "extracted":
        return row["status"]

    try:
        profile = load_profile(settings, doc_id)
        if profile is None:
            raise FileNotFoundError(f"profile for {doc_id[:12]} is missing")
        budgets, _ = ensure_budgets(conn, settings, count)
        chunks = build_chunks(
            profile,
            candidate_id=row["candidate_id"],
            doc_id=doc_id,
            budgets=budgets,
            model_limit=settings.chunking.model_limit,
            count=count,
        )
        db.replace_chunks(conn, row["candidate_id"], [c.to_row() for c in chunks])
    except Exception as e:  # noqa: BLE001 - one bad document must not stop the batch
        log.exception("chunking %s failed", doc_id[:12])
        db.set_document_status(conn, doc_id, "failed", error=f"chunk failed: {e}")
        return "failed"
    db.set_document_status(conn, doc_id, "chunked")
    return "chunked"
