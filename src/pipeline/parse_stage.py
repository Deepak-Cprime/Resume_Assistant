"""Parse stage for one document: uploaded -> parsed (or needs_review / failed)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import Settings
from src.logging_setup import get_logger
from src.providers import factory
from src.providers.parsers import router
from src.stores import db, files

log = get_logger(__name__)


def parse_document(
    conn: sqlite3.Connection,
    settings: Settings,
    doc_id: str,
    *,
    parser_factory: router.ParserFactory = factory.create_parser,
) -> str:
    """Parse one uploaded document, save its Markdown and update its status.

    Returns the new status. Documents past `uploaded` are skipped, so re-running a batch
    never re-parses finished work (retry a failed one by resetting it to `uploaded`).
    """
    row = db.get_document(conn, doc_id)
    if row is None:
        raise KeyError(doc_id)
    if row["status"] != "uploaded":
        return row["status"]

    try:
        outcome = router.parse_resume(
            Path(row["raw_path"]), row["doc_type"], settings, parser_factory=parser_factory
        )
    except Exception as e:  # noqa: BLE001 - one bad document must not stop the batch
        log.exception("parsing %s failed", doc_id[:12])
        db.set_document_status(conn, doc_id, "failed", error=f"parse failed: {e}")
        return "failed"

    md_path = files.markdown_path(settings, doc_id)
    files.write_text_atomic(md_path, outcome.markdown)
    status = "needs_review" if outcome.needs_review else "parsed"
    error = outcome.report.summary() if outcome.needs_review else None
    db.set_document_status(
        conn, doc_id, status, error=error,
        parser=outcome.parsed.parser,
        parser_version=outcome.parsed.parser_version,
        markdown_path=str(md_path),
    )
    if outcome.report.issues and not outcome.needs_review:
        log.info("%s parsed with warnings: %s", doc_id[:12], outcome.report.summary())
    return status
