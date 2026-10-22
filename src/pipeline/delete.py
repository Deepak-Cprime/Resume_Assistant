"""Delete a candidate (or a stuck upload) from every place it lives.

    database rows -> vector indexes (every model) -> keyword index -> files on disk

The database goes first: it is the source of truth, and if a later step fails the leftovers
are harmless (a vector or keyword entry with no chunk behind it) and the next pipeline run
removes them as stale. No embedding model is loaded, so deleting is instant.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Settings
from src.logging_setup import get_logger
from src.pipeline import index_stage, keyword_stage
from src.stores import db, files

log = get_logger(__name__)


@dataclass
class DeleteResult:
    candidate_id: int | None
    documents: int  # document rows removed
    chunks: int  # chunk rows removed
    vectors: dict[str, int] = field(default_factory=dict)  # vectors removed per embedding model
    keyword_rebuilt: bool = False
    files: int = 0  # files removed from data/
    errors: list[str] = field(default_factory=list)  # index/file clean-up problems (non-fatal)


def _document_files(settings: Settings, rows: list[sqlite3.Row]) -> list[Path]:
    paths: list[Path] = []
    for row in rows:
        for stored in (row["raw_path"], row["markdown_path"]):
            if stored:
                paths.append(Path(stored))
        paths.append(files.profile_path(settings, row["doc_id"]))
        paths.append(files.sections_path(settings, row["doc_id"]))
    return paths


def _clean_up(
    conn: sqlite3.Connection, settings: Settings, chunk_ids: list[int], paths: list[Path],
    result: DeleteResult,
) -> None:
    try:
        if chunk_ids:
            result.vectors = index_stage.remove_from_indexes(conn, settings, chunk_ids)
    except Exception as e:  # noqa: BLE001 - the next pipeline run removes stale vectors anyway
        log.exception("removing vectors failed")
        result.errors.append(f"vector indexes: {e}")
    try:
        if keyword_stage.read_keyword_manifest(settings) is not None:
            result.keyword_rebuilt = keyword_stage.build_keyword_index(conn, settings).rebuilt
    except Exception as e:  # noqa: BLE001
        log.exception("rebuilding the keyword index failed")
        result.errors.append(f"keyword index: {e}")
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                result.files += 1
        except OSError as e:
            result.errors.append(f"{path.name}: {e}")


def delete_candidate(
    conn: sqlite3.Connection, settings: Settings, candidate_id: int
) -> DeleteResult:
    """Remove a candidate: every upload of their resume, all extracted rows, every chunk, their
    vectors in every embedding model's index, their keyword entries, and their files."""
    if db.get_candidate(conn, candidate_id) is None:
        raise KeyError(f"no candidate {candidate_id}")
    documents = db.documents_of_candidate(conn, candidate_id)
    chunk_ids = db.candidate_chunk_ids(conn, candidate_id)
    paths = _document_files(settings, documents)

    db.delete_candidate_rows(conn, candidate_id)
    result = DeleteResult(candidate_id, len(documents), len(chunk_ids))
    _clean_up(conn, settings, chunk_ids, paths, result)
    log.info("deleted candidate %d (%d documents, %d chunks)", candidate_id, len(documents),
             len(chunk_ids))
    return result


def delete_document(conn: sqlite3.Connection, settings: Settings, doc_id: str) -> DeleteResult:
    """Remove an upload. If it belongs to a candidate the whole candidate goes (their chunks
    belong to the person, not to one file); a failed or unprocessed upload just disappears,
    and the same file can then be uploaded again."""
    row = db.get_document(conn, doc_id)
    if row is None:
        raise KeyError(doc_id)
    if row["candidate_id"] is not None:
        return delete_candidate(conn, settings, row["candidate_id"])
    paths = _document_files(settings, [row])
    db.delete_document_row(conn, doc_id)
    result = DeleteResult(None, 1, 0)
    _clean_up(conn, settings, [], paths, result)
    return result
