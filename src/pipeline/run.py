"""Run the whole indexing pipeline over every resume that still has work to do.

    uploaded -> parse -> parsed -> structure -> structured -> extract -> extracted
             -> chunk -> chunked -> [vector index + keyword index] -> indexed

Every stage stores its own status, so a crash or a closed browser tab leaves a resumable state:
running again continues each document from where it stopped and never repeats finished work.
One failing resume is marked `failed` (or `needs_review`) and the batch carries on. If the
chunk budgets change, only the resumes chunked under the old budgets are re-chunked, and the
indexes then re-embed only the chunks whose text actually changed.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from src.chunking.tokens import estimate_tokens
from src.config import Settings
from src.logging_setup import get_logger
from src.pipeline import (
    chunk_stage,
    extract_stage,
    index_stage,
    keyword_stage,
    parse_stage,
    structure_stage,
)
from src.pipeline.keyword_stage import KeywordBuildResult
from src.providers import factory
from src.providers.base import EmbeddingProvider, LLMProvider
from src.providers.parsers.router import ParserFactory
from src.stores import db

log = get_logger(__name__)

# Called as progress(phase, done, total): phase is "documents", "embedding:<model>" or "keyword".
Progress = Callable[[str, int, int], None]
FLAGGED = ("failed", "needs_review")


@dataclass
class DocResult:
    doc_id: str
    filename: str
    status: str  # where the document ended up
    stages: list[str] = field(default_factory=list)  # stages that ran in this call
    error: str | None = None


@dataclass
class RunResult:
    documents: list[DocResult] = field(default_factory=list)
    rechunked: list[str] = field(default_factory=list)  # doc ids re-chunked after a budget change
    vectors: dict[str, index_stage.BuildResult] = field(default_factory=dict)
    keyword: KeywordBuildResult | None = None
    indexed: list[str] = field(default_factory=list)  # doc ids that became `indexed`
    errors: list[str] = field(default_factory=list)  # index build problems (documents stay chunked)

    @property
    def flagged(self) -> list[DocResult]:
        return [d for d in self.documents if d.status in FLAGGED]

    def summary(self) -> str:
        parts = [f"{len(self.documents)} document(s) processed"]
        if self.flagged:
            parts.append(f"{len(self.flagged)} need attention")
        if self.rechunked:
            parts.append(f"{len(self.rechunked)} re-chunked")
        for model, built in self.vectors.items():
            parts.append(f"{model}: +{built.embedded} -{built.removed} = {built.total} vectors")
        if self.keyword is not None:
            parts.append(f"keyword index {'rebuilt' if self.keyword.rebuilt else 'unchanged'}")
        if self.errors:
            parts.append(f"{len(self.errors)} index error(s)")
        return "; ".join(parts)


def process_document(
    conn: sqlite3.Connection,
    settings: Settings,
    doc_id: str,
    *,
    parser_factory: ParserFactory = factory.create_parser,
    llm: LLMProvider | None = None,
) -> DocResult:
    """Advance one document through parse, structure, extract and chunk, from its current
    status; stops early at `failed` / `needs_review`. Safe to call on any document: stages
    already done are skipped."""
    stages: list[str] = []
    steps = {
        "uploaded": ("parse", lambda: parse_stage.parse_document(
            conn, settings, doc_id, parser_factory=parser_factory)),
        "parsed": ("structure", lambda: structure_stage.structure_document(
            conn, settings, doc_id)),
        "structured": ("extract", lambda: extract_stage.extract_document(
            conn, settings, doc_id, llm=llm)),
        "extracted": ("chunk", lambda: chunk_stage.chunk_document(conn, settings, doc_id)),
    }
    status = db.get_document(conn, doc_id)["status"]
    try:
        while status in steps:
            name, step = steps[status]
            new_status = step()
            stages.append(name)
            if new_status == status:  # a stage that makes no progress must not loop forever
                break
            status = new_status
    except Exception as e:  # noqa: BLE001 - a bug in one document must not stop the batch
        log.exception("pipeline failed on %s", doc_id[:12])
        row = db.get_document(conn, doc_id)
        if row["status"] not in FLAGGED:
            db.set_document_status(conn, doc_id, "failed", error=f"pipeline error: {e}")
    row = db.get_document(conn, doc_id)
    return DocResult(doc_id, row["filename"], row["status"], stages, row["error"])


def rechunk_stale(
    conn: sqlite3.Connection, settings: Settings, count: Callable[[str], int] = estimate_tokens
) -> list[str]:
    """Re-chunk resumes whose chunks were built under budgets other than the current ones.

    Only the document that owns a candidate's live chunks is touched, so an older upload of
    the same person is never allowed to overwrite a newer one."""
    budgets, _ = chunk_stage.ensure_budgets(conn, settings, count)
    current = budgets.fingerprint()
    stale = [r["doc_id"] for r in db.chunk_owner_documents(conn) if r["chunk_fp"] != current]
    for doc_id in stale:
        db.set_document_status(conn, doc_id, "extracted")  # step back one stage
        chunk_stage.chunk_document(conn, settings, doc_id, count=count)
    if stale:
        log.info("re-chunked %d resume(s) after a budget change", len(stale))
    return stale


def mark_indexed(conn: sqlite3.Connection) -> list[str]:
    """`chunked` -> `indexed` for every document; call only once the indexes are current."""
    done = []
    for row in db.list_documents(conn, status="chunked"):
        db.set_document_status(conn, row["doc_id"], "indexed")
        done.append(row["doc_id"])
    return done


def _build_indexes(
    conn: sqlite3.Connection,
    settings: Settings,
    embedders: list[EmbeddingProvider],
    result: RunResult,
    progress: Progress | None,
) -> None:
    for embedder in embedders:
        phase = f"embedding:{embedder.name}"

        def report(done: int, total: int, phase: str = phase) -> None:
            if progress:
                progress(phase, done, total)

        try:
            result.vectors[embedder.name] = index_stage.build_index(
                conn, settings, embedder, progress=report
            )
        except Exception as e:  # noqa: BLE001 - e.g. a chunk over the model limit, Ollama down
            log.exception("vector index for %s failed", embedder.name)
            result.errors.append(f"{embedder.name}: {e}")
    try:
        result.keyword = keyword_stage.build_keyword_index(conn, settings)
        if progress:
            progress("keyword", result.keyword.n_docs, result.keyword.n_docs)
    except Exception as e:  # noqa: BLE001
        log.exception("keyword index failed")
        result.errors.append(f"keyword index: {e}")


def run_pipeline(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    embedders: list[EmbeddingProvider] | None = None,
    doc_ids: set[str] | None = None,
    build_indexes: bool = True,
    parser_factory: ParserFactory = factory.create_parser,
    llm: LLMProvider | None = None,
    progress: Progress | None = None,
) -> RunResult:
    """Process pending resumes, re-chunk stale ones, sync the indexes, mark `indexed`.

    `embedders` defaults to the configured default model only; pass more to build several
    indexes side by side. `doc_ids` limits which pending resumes are processed (the indexes
    are always synced with the whole database). A resume becomes `indexed` only when the
    default model's vector index AND the keyword index were both built without error."""
    result = RunResult()
    pending = [r["doc_id"] for r in db.list_pending_documents(conn)
               if doc_ids is None or r["doc_id"] in doc_ids]
    for i, doc_id in enumerate(pending):
        if progress:
            progress("documents", i, len(pending))
        result.documents.append(
            process_document(conn, settings, doc_id, parser_factory=parser_factory, llm=llm)
        )
    if progress and pending:
        progress("documents", len(pending), len(pending))

    result.rechunked = rechunk_stale(conn, settings)
    if not build_indexes:
        return result

    try:
        models = embedders if embedders is not None else [
            factory.create_embedder(settings.embedding.default, settings)
        ]
    except Exception as e:  # noqa: BLE001 - model files missing, no network on first use ...
        result.errors.append(f"embedding model: {e}")
        models = []
    _build_indexes(conn, settings, models, result, progress)

    default_built = settings.embedding.default in result.vectors
    if default_built and result.keyword is not None and not result.errors:
        result.indexed = mark_indexed(conn)
    return result


def retry_failed(
    conn: sqlite3.Connection, doc_ids: list[str] | None = None, *, include_review: bool = True
) -> list[str]:
    """Send failed (and by default needs_review) documents back to `uploaded` so the next run
    parses them again, e.g. after fixing a setting or installing a missing component."""
    wanted = set(doc_ids) if doc_ids is not None else None
    statuses = FLAGGED if include_review else ("failed",)
    reset = []
    for status in statuses:
        for row in db.list_documents(conn, status=status):
            if wanted is None or row["doc_id"] in wanted:
                db.set_document_status(conn, row["doc_id"], "uploaded")
                reset.append(row["doc_id"])
    return reset
