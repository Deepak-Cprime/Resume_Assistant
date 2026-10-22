"""Embed the database's chunks into one persisted vector index per embedding model.

Incremental: a hash of each chunk's embedded text is stored beside the vectors, so a rebuild
embeds only new or changed chunks and drops vectors whose chunk no longer exists. Another model
gets its own folder, so several indexes live side by side and each can be queried on its own.

Document statuses are not touched here: `indexed` means vectors AND keyword index are current,
which the orchestrator (M8) decides once both stages have run.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from src.chunking.budgets import Budgets
from src.config import Settings
from src.logging_setup import get_logger
from src.pipeline import manifest as mf
from src.providers import factory
from src.providers.base import EmbeddingProvider, VectorStore
from src.stores import db, files

log = get_logger(__name__)
Progress = Callable[[int, int], None]  # (done, total) chunks embedded so far
MAX_LISTED = 5


class IndexBuildError(RuntimeError):
    pass


@dataclass
class BuildResult:
    model: str
    embedded: int  # chunks embedded in this run (new or changed)
    removed: int  # vectors dropped because their chunk is gone or was replaced
    total: int  # vectors in the index afterwards
    index_version: str
    seconds: float
    rebuilt_from_scratch: bool = False


@dataclass
class IndexReport:
    model: str
    n_db: int  # embeddable chunks in the database
    n_index: int  # vectors in the index
    missing: list[int] = field(default_factory=list)  # in the database, not in the index
    stale: list[int] = field(default_factory=list)  # in the index, not in the database
    changed: list[int] = field(default_factory=list)  # text changed since it was embedded

    @property
    def ok(self) -> bool:
        return not (self.missing or self.stale or self.changed) and self.n_db == self.n_index


def _budgets_fingerprint(conn: sqlite3.Connection) -> str:
    raw = db.get_chunk_budgets(conn)
    return Budgets.from_dict(raw).fingerprint() if raw else ""


def check_token_limits(embedder: EmbeddingProvider, rows: list[sqlite3.Row]) -> None:
    """Every chunk must fit the model's context as the model's own tokenizer counts it."""
    limit = embedder.max_tokens
    over = [
        (r["chunk_key"], n)
        for r in rows
        if (n := embedder.count_tokens(r["embed_text"])) > limit
    ]
    if over:
        shown = ", ".join(f"{key} ({n})" for key, n in over[:MAX_LISTED])
        raise IndexBuildError(
            f"{len(over)} chunk(s) exceed {embedder.name}'s limit of {limit} tokens: {shown}. "
            "Lower chunking.model_limit or the budget caps and re-chunk."
        )


def _open_store(
    settings: Settings, embedder: EmbeddingProvider, rebuild: bool
) -> tuple[VectorStore, dict[int, str], bool]:
    """(store, chunk-hash map, rebuilt_from_scratch): reuse the saved index when it is still
    compatible with this embedder, otherwise start empty."""
    store = factory.create_vector_store(embedder.dim, settings)
    existing = mf.read_manifest(settings, embedder.name)
    path = mf.vectors_path(settings, embedder.name, store)
    compatible = (
        existing is not None
        and path.exists()
        and existing.dim == embedder.dim
        and existing.doc_prefix == getattr(embedder, "doc_prefix", "")
        and existing.query_prefix == getattr(embedder, "query_prefix", "")
        and existing.vector_store == settings.vector_store.provider
    )
    if rebuild or not compatible:
        if existing is not None and not rebuild:
            log.warning("index for %s is incompatible with the embedder; rebuilding", embedder.name)
        return store, {}, True
    store.load(path)
    return store, mf.read_hashes(settings, embedder.name), False


def build_index(
    conn: sqlite3.Connection,
    settings: Settings,
    embedder: EmbeddingProvider,
    *,
    rebuild: bool = False,
    progress: Progress | None = None,
) -> BuildResult:
    """Bring `embedder`'s index in line with the embeddable chunks in the database."""
    started = time.perf_counter()
    wanted = {r["chunk_id"]: r for r in db.list_chunks(conn, embedded_only=True)}
    store, hashes, from_scratch = _open_store(settings, embedder, rebuild)

    stale = [cid for cid in hashes if cid not in wanted]
    todo = [cid for cid, r in wanted.items() if hashes.get(cid) != r["content_hash"]]
    check_token_limits(embedder, [wanted[cid] for cid in todo])

    current = mf.read_manifest(settings, embedder.name)
    if not (todo or stale or from_scratch) and current is not None:
        return BuildResult(  # nothing changed: leave the files and the manifest history alone
            embedder.name, 0, 0, store.count(), current.index_version,
            time.perf_counter() - started,
        )

    store.remove(stale)
    for cid in stale:
        del hashes[cid]
    batch_size = settings.embedding.batch_size
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        vectors = embedder.embed_documents([wanted[cid]["embed_text"] for cid in batch])
        store.add(batch, vectors)
        hashes.update({cid: wanted[cid]["content_hash"] for cid in batch})
        if progress:
            progress(min(start + batch_size, len(todo)), len(todo))

    manifest = _save(conn, settings, embedder, store, hashes)
    log.info("index %s: +%d -%d = %d vectors", embedder.name, len(todo), len(stale), store.count())
    return BuildResult(
        embedder.name, len(todo), len(stale), store.count(), manifest.index_version,
        time.perf_counter() - started, from_scratch,
    )


def _save(
    conn: sqlite3.Connection,
    settings: Settings,
    embedder: EmbeddingProvider,
    store: VectorStore,
    hashes: dict[int, str],
) -> mf.IndexManifest:
    """Write vectors, hashes and manifest (vectors first, manifest last) and log the build."""
    directory = files.index_dir(settings, embedder.name)
    doc_prefix = getattr(embedder, "doc_prefix", "")
    query_prefix = getattr(embedder, "query_prefix", "")
    budgets_fp = _budgets_fingerprint(conn)
    chunks_hash = mf.chunk_set_hash(hashes.items())
    manifest = mf.IndexManifest(
        emb_model=embedder.name,
        dim=embedder.dim,
        n_vectors=store.count(),
        index_version=mf.index_version(
            embedder.name, embedder.dim, doc_prefix, query_prefix, budgets_fp, chunks_hash
        ),
        doc_prefix=doc_prefix,
        query_prefix=query_prefix,
        budgets_fingerprint=budgets_fp,
        chunk_set_hash=chunks_hash,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        vector_store=settings.vector_store.provider,
    )
    store.save(mf.vectors_path(settings, embedder.name, store))
    files.write_text_atomic(
        directory / mf.HASHES_FILE, json.dumps({str(k): v for k, v in sorted(hashes.items())})
    )
    files.write_text_atomic(directory / mf.MANIFEST_FILE, manifest.to_json())
    _log_manifest(conn, settings, manifest)
    return manifest


def _log_manifest(
    conn: sqlite3.Connection, settings: Settings, manifest: mf.IndexManifest
) -> None:
    db.add_index_manifest(
        conn,
        emb_model=manifest.emb_model,
        dim=manifest.dim,
        n_vectors=manifest.n_vectors,
        chunker_cfg={
            "strategy": settings.chunking.strategy,
            "model_limit": settings.chunking.model_limit,
            "budgets_fingerprint": manifest.budgets_fingerprint,
            "index_version": manifest.index_version,
            "doc_prefix": manifest.doc_prefix,
            "query_prefix": manifest.query_prefix,
        },
    )


def remove_from_indexes(
    conn: sqlite3.Connection, settings: Settings, chunk_ids: list[int]
) -> dict[str, int]:
    """Drop these chunks from every saved vector index; returns {model: vectors removed}.

    Works from the files alone (no embedding model is loaded), so deleting a candidate stays
    instant. Each touched index gets a fresh manifest and hashes file."""
    removed: dict[str, int] = {}
    root = settings.indexes_dir
    if not root.exists():
        return removed
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest_file = directory / mf.MANIFEST_FILE
        if directory.name.startswith("_") or not manifest_file.exists():
            continue  # the keyword index (or anything else that is not a vector index)
        manifest = mf.IndexManifest.from_json(manifest_file.read_text(encoding="utf-8"))
        store = factory.create_vector_store(manifest.dim, settings, provider=manifest.vector_store)
        vectors_file = directory / f"{mf.VECTORS_STEM}{store.suffix}"
        store.load(vectors_file)
        present = sorted(set(store.ids()) & set(chunk_ids))
        removed[manifest.emb_model] = len(present)
        if not present:
            continue
        store.remove(present)
        store.save(vectors_file)
        hashes = {k: v for k, v in mf.read_hashes(settings, manifest.emb_model).items()
                  if k not in set(present)}
        files.write_text_atomic(
            directory / mf.HASHES_FILE, json.dumps({str(k): v for k, v in sorted(hashes.items())})
        )
        chunks_hash = mf.chunk_set_hash(hashes.items())
        updated = replace(
            manifest,
            n_vectors=store.count(),
            chunk_set_hash=chunks_hash,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            index_version=mf.index_version(
                manifest.emb_model, manifest.dim, manifest.doc_prefix, manifest.query_prefix,
                manifest.budgets_fingerprint, chunks_hash,
            ),
        )
        files.write_text_atomic(manifest_file, updated.to_json())
        _log_manifest(conn, settings, updated)
    return removed


def verify_index(
    conn: sqlite3.Connection, settings: Settings, embedder: EmbeddingProvider
) -> IndexReport:
    """Compare the saved index with the database without changing either."""
    return verify_model_index(conn, settings, embedder.name)


def verify_model_index(conn: sqlite3.Connection, settings: Settings, model: str) -> IndexReport:
    """Same check by model name, so the UI can show index health without loading the model."""
    wanted = {r["chunk_id"]: r["content_hash"] for r in db.list_chunks(conn, embedded_only=True)}
    manifest = mf.read_manifest(settings, model)
    if manifest is None:
        return IndexReport(model, len(wanted), 0, missing=sorted(wanted))
    store = factory.create_vector_store(manifest.dim, settings, provider=manifest.vector_store)
    store.load(mf.vectors_path(settings, model, store))
    in_index = set(store.ids())
    hashes = mf.read_hashes(settings, model)
    return IndexReport(
        model=model,
        n_db=len(wanted),
        n_index=store.count(),
        missing=sorted(set(wanted) - in_index),
        stale=sorted(in_index - set(wanted)),
        changed=sorted(c for c in in_index & set(wanted) if hashes.get(c) != wanted[c]),
    )
