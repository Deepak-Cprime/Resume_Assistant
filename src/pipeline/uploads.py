"""Accept several uploaded files at once and say what happened to each, in words."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from src.config import Settings
from src.ingestion import intake
from src.ingestion.intake import IntakeResult
from src.logging_setup import get_logger

log = get_logger(__name__)


def ingest_files(
    conn: sqlite3.Connection, settings: Settings, items: Iterable[tuple[str, bytes]]
) -> list[IntakeResult]:
    """Validate, hash, dedupe and store each (filename, bytes). One bad file never stops the
    others; an unexpected error becomes a `failed` result for that file."""
    results: list[IntakeResult] = []
    for name, data in items:
        try:
            results.append(intake.ingest(conn, settings, name, data))
        except Exception as e:  # noqa: BLE001 - e.g. disk full while storing the raw file
            log.exception("could not store %s", name)
            results.append(IntakeResult("", name, "failed", False, f"could not store file: {e}"))
    return results


def outcome(result: IntakeResult) -> str:
    """One line for the Upload page: new, already uploaded, or rejected and why."""
    if result.status == "failed":
        return f"rejected: {result.error}"
    if not result.is_new:
        return "already uploaded (same content)"
    return "added"
