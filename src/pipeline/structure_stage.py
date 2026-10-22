"""Structure stage for one document: parsed -> structured (sections found and saved)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.config import Settings
from src.logging_setup import get_logger
from src.stores import db, files
from src.structure.sections import MIN_KNOWN_SECTIONS, known_labels, split_sections

log = get_logger(__name__)


def structure_document(conn: sqlite3.Connection, settings: Settings, doc_id: str) -> str:
    """Detect sections in the saved Markdown and store them for inspection.

    Returns the new status. A resume with fewer than two recognised sections cannot be
    structured reliably and is sent to `needs_review`. Documents not in `parsed` are skipped.
    """
    row = db.get_document(conn, doc_id)
    if row is None:
        raise KeyError(doc_id)
    if row["status"] != "parsed":
        return row["status"]

    try:
        markdown = Path(row["markdown_path"]).read_text(encoding="utf-8")
        sections = split_sections(markdown)
        payload = [
            {"label": s.label, "heading": s.heading, "method": s.method, "text": s.text}
            for s in sections
        ]
        files.write_text_atomic(
            files.sections_path(settings, doc_id),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
    except Exception as e:  # noqa: BLE001 - one bad document must not stop the batch
        log.exception("structuring %s failed", doc_id[:12])
        db.set_document_status(conn, doc_id, "failed", error=f"structure failed: {e}")
        return "failed"

    found = known_labels(sections)
    if found < MIN_KNOWN_SECTIONS:
        db.set_document_status(
            conn, doc_id, "needs_review",
            error=f"only {found} recognised section(s); headings missing or unusual",
        )
        return "needs_review"
    db.set_document_status(conn, doc_id, "structured")
    return "structured"
