"""Build and check the BM25 keyword index over the same chunks the vector indexes hold.

BM25 scores depend on corpus-wide statistics, so the index is rebuilt whole (seconds, even for
tens of thousands of chunks) rather than patched; a rebuild happens only when the chunk set,
the tokenizer or the BM25 parameters changed. Document statuses are not touched here: the
orchestrator (M8) marks a resume `indexed` once vectors and keywords are both current.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.config import Settings
from src.logging_setup import get_logger
from src.pipeline.manifest import IndexMismatch, chunk_set_hash
from src.providers import factory
from src.providers.base import KeywordIndex
from src.providers.keyword.tokenizer import tokenizer_fingerprint
from src.stores import db, files

log = get_logger(__name__)
INDEX_DIR = "index"
MANIFEST_FILE = "manifest.json"


@dataclass
class KeywordManifest:
    tokenizer: str
    tokenizer_fingerprint: str
    k1: float
    b: float
    n_docs: int
    chunk_set_hash: str
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> KeywordManifest:
        return cls(**json.loads(text))


@dataclass
class KeywordBuildResult:
    n_docs: int
    rebuilt: bool
    seconds: float


@dataclass
class KeywordReport:
    n_db: int  # embeddable chunks in the database
    n_index: int  # documents in the keyword index
    missing: list[int] = field(default_factory=list)  # in the database, not in the index
    stale: list[int] = field(default_factory=list)  # in the index, not in the database
    up_to_date: bool = False  # indexed text is what the database holds now

    @property
    def ok(self) -> bool:
        return self.up_to_date and not (self.missing or self.stale)


def _paths(settings: Settings) -> tuple[Path, Path]:
    directory = files.keyword_dir(settings)
    return directory / INDEX_DIR, directory / MANIFEST_FILE


def read_keyword_manifest(settings: Settings) -> KeywordManifest | None:
    _, manifest_path = _paths(settings)
    if not manifest_path.exists():
        return None
    return KeywordManifest.from_json(manifest_path.read_text(encoding="utf-8"))


def _is_current(manifest: KeywordManifest | None, settings: Settings, chunks_hash: str) -> bool:
    """True if the saved index was built from these chunks with the current settings."""
    cfg = settings.keyword
    return (
        manifest is not None
        and manifest.chunk_set_hash == chunks_hash
        and manifest.tokenizer == cfg.tokenizer
        and manifest.tokenizer_fingerprint == tokenizer_fingerprint(cfg.tokenizer)
        and (manifest.k1, manifest.b) == (cfg.k1, cfg.b)
    )


def build_keyword_index(
    conn: sqlite3.Connection, settings: Settings, *, rebuild: bool = False
) -> KeywordBuildResult:
    """Make the keyword index match the embeddable chunks in the database."""
    started = time.perf_counter()
    rows = db.list_chunks(conn, embedded_only=True)
    chunks_hash = chunk_set_hash((r["chunk_id"], r["content_hash"]) for r in rows)
    index_path, manifest_path = _paths(settings)
    current = read_keyword_manifest(settings)
    if not rebuild and index_path.exists() and _is_current(current, settings, chunks_hash):
        return KeywordBuildResult(len(rows), False, time.perf_counter() - started)

    index = factory.create_keyword_index(settings)
    index.build([r["chunk_id"] for r in rows], [r["embed_text"] for r in rows])
    index.save(index_path)
    cfg = settings.keyword
    manifest = KeywordManifest(
        tokenizer=cfg.tokenizer,
        tokenizer_fingerprint=tokenizer_fingerprint(cfg.tokenizer),
        k1=cfg.k1,
        b=cfg.b,
        n_docs=index.count(),
        chunk_set_hash=chunks_hash,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    files.write_text_atomic(manifest_path, manifest.to_json())
    log.info("keyword index rebuilt: %d documents (%s)", manifest.n_docs, cfg.tokenizer)
    return KeywordBuildResult(manifest.n_docs, True, time.perf_counter() - started)


def load_keyword_index(settings: Settings) -> tuple[KeywordIndex, KeywordManifest]:
    """Open the saved index for searching.

    Raises FileNotFoundError if none was built, and IndexMismatch if it was built with a
    different tokenizer or alias file than the current one (queries would be tokenized
    differently from the documents, so results would be silently wrong)."""
    manifest = read_keyword_manifest(settings)
    index_path, _ = _paths(settings)
    if manifest is None or not index_path.exists():
        raise FileNotFoundError("no keyword index has been built")
    cfg = settings.keyword
    if (manifest.tokenizer, manifest.tokenizer_fingerprint) != (
        cfg.tokenizer, tokenizer_fingerprint(cfg.tokenizer),
    ):
        raise IndexMismatch(
            f"keyword index was built with tokenizer '{manifest.tokenizer_fingerprint}', "
            f"current is '{tokenizer_fingerprint(cfg.tokenizer)}'; rebuild the keyword index"
        )
    index = factory.create_keyword_index(settings)
    index.load(index_path)
    return index, manifest


def verify_keyword_index(conn: sqlite3.Connection, settings: Settings) -> KeywordReport:
    """Compare the saved keyword index with the database without changing either."""
    rows = db.list_chunks(conn, embedded_only=True)
    wanted = {r["chunk_id"] for r in rows}
    try:
        index, manifest = load_keyword_index(settings)
    except (FileNotFoundError, IndexMismatch):
        return KeywordReport(len(wanted), 0, missing=sorted(wanted))
    chunks_hash = chunk_set_hash((r["chunk_id"], r["content_hash"]) for r in rows)
    in_index = set(index.ids())
    return KeywordReport(
        n_db=len(wanted),
        n_index=index.count(),
        missing=sorted(wanted - in_index),
        stale=sorted(in_index - wanted),
        up_to_date=_is_current(manifest, settings, chunks_hash),
    )
