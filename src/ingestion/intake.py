"""Validate, hash, dedupe and store uploaded resumes."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path

from src.config import Settings
from src.logging_setup import get_logger
from src.stores import db, files

log = get_logger(__name__)

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"


class IntakeError(ValueError):
    """The file is not an acceptable resume; the message is shown to the user."""


@dataclass(frozen=True)
class IntakeResult:
    doc_id: str
    filename: str
    status: str
    is_new: bool
    error: str | None = None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_type(data: bytes) -> str:
    """'pdf' or 'docx' judged by content (magic bytes), never by file extension."""
    if data.startswith(PDF_MAGIC):
        return "pdf"
    if data.startswith(ZIP_MAGIC):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if "word/document.xml" in z.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
        raise IntakeError("ZIP file is not a DOCX document")
    raise IntakeError("Unsupported or corrupt file (expected PDF or DOCX)")


def _check_pdf(data: bytes, max_pages: int) -> None:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise IntakeError(f"Corrupt PDF: {e}") from e
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise IntakeError("PDF is password-protected or encrypted")
        if doc.page_count == 0:
            raise IntakeError("PDF has no pages")
        if doc.page_count > max_pages:
            raise IntakeError(f"PDF has {doc.page_count} pages (limit {max_pages})")
    finally:
        doc.close()


def validate(filename: str, data: bytes, settings: Settings) -> str:
    """Return the detected type ('pdf' | 'docx') or raise IntakeError."""
    if not data:
        raise IntakeError("File is empty")
    if len(data) > settings.ingestion.max_file_mb * 1024 * 1024:
        raise IntakeError(f"File is larger than {settings.ingestion.max_file_mb} MB")
    kind = detect_type(data)
    if kind == "pdf":
        _check_pdf(data, settings.ingestion.max_pages)
    return kind


def ingest(
    conn: sqlite3.Connection, settings: Settings, filename: str, data: bytes
) -> IntakeResult:
    """Accept one upload. Idempotent: identical bytes never create a second row or file.

    Rejected files are recorded as `failed` (without storing the bytes) so the UI can list
    them; re-uploading the same bad bytes returns that same record.
    """
    doc_id = sha256_hex(data)
    existing = db.get_document(conn, doc_id)
    if existing is not None:
        log.info("duplicate upload %s (%s)", filename, doc_id[:12])
        return IntakeResult(
            doc_id, existing["filename"], existing["status"], False, existing["error"]
        )

    try:
        kind = validate(filename, data, settings)
    except IntakeError as e:
        db.insert_document(
            conn, doc_id=doc_id, filename=Path(filename).name, doc_type=None, raw_path=None,
            status="failed", error=str(e),
        )
        log.warning("rejected %s: %s", filename, e)
        return IntakeResult(doc_id, Path(filename).name, "failed", True, str(e))

    path = files.raw_path(settings, doc_id, f".{kind}")
    files.write_bytes_atomic(path, data)
    db.insert_document(
        conn, doc_id=doc_id, filename=Path(filename).name, doc_type=kind, raw_path=str(path)
    )
    return IntakeResult(doc_id, Path(filename).name, "uploaded", True)
