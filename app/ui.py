"""Streamlit wiring shared by the pages: cached resources, a progress display and small
formatters. Business logic lives in `src/`; nothing here decides what the pipeline does."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `streamlit run app/Home.py` leaves the project root off the path
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import Settings, load_settings
from src.logging_setup import setup_logging
from src.pipeline import run as pipeline
from src.providers import factory
from src.providers.base import EmbeddingProvider
from src.stores import db

STATUS_ICONS = {
    "uploaded": "⬆️",
    "parsed": "\U0001f4c4",
    "structured": "\U0001f9e9",
    "extracted": "\U0001f3f7️",
    "chunked": "✂️",
    "indexed": "✅",
    "failed": "❌",
    "needs_review": "⚠️",
}


def get_settings() -> Settings:
    settings = load_settings()
    setup_logging(settings.logging.level)
    return settings


@st.cache_resource(show_spinner=False)
def _connection(db_path: str) -> sqlite3.Connection:
    return db.connect(db_path)


def get_conn(settings: Settings) -> sqlite3.Connection:
    """One database connection per database file, kept across reruns."""
    return _connection(str(settings.paths.db))


@st.cache_resource(show_spinner=False)
def _embedder(model_name: str, data_dir: str, _settings: Settings) -> EmbeddingProvider:
    # data_dir is part of the cache key only; the model itself loads on first use
    return factory.create_embedder(model_name, _settings)


def get_embedder(settings: Settings, model_name: str | None = None) -> EmbeddingProvider:
    """The embedder object is cached; its weights load lazily on the first embedding."""
    name = model_name or settings.embedding.default
    return _embedder(name, str(settings.paths.data_dir), settings)


def status_text(status: str) -> str:
    return f"{STATUS_ICONS.get(status, '')} {status}".strip()


def phase_label(phase: str) -> str:
    if phase == "documents":
        return "Processing resumes"
    if phase.startswith("embedding:"):
        return f"Embedding with {phase.split(':', 1)[1]}"
    if phase == "keyword":
        return "Building the keyword index"
    return phase


def run_with_progress(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    doc_ids: set[str] | None = None,
    models: list[str] | None = None,
    rebuild: bool = False,
) -> pipeline.RunResult:
    """Run the pipeline inside an `st.status` box with a progress bar."""
    names = models or [settings.embedding.default]
    embedders = [get_embedder(settings, n) for n in names]
    with st.status("Running the pipeline", expanded=True) as status:
        bar = st.progress(0.0, text="Starting")
        st.caption("The first run loads the embedding model, which can take a few seconds.")

        def on_progress(phase: str, done: int, total: int) -> None:
            fraction = 1.0 if total == 0 else min(done / total, 1.0)
            bar.progress(fraction, text=f"{phase_label(phase)}: {done}/{total}")

        result = pipeline.run_pipeline(
            conn, settings, embedders=embedders, doc_ids=doc_ids, rebuild_indexes=rebuild,
            progress=on_progress,
        )
        status.update(
            label=result.summary(), state="error" if result.errors else "complete", expanded=False
        )
    return result


def flash(message: str) -> None:
    """Keep a message across the next rerun (used after an action that calls st.rerun)."""
    st.session_state["_flash"] = message


def show_flash() -> None:
    message = st.session_state.pop("_flash", None)
    if message:
        st.success(message)


def render_result(result: pipeline.RunResult) -> None:
    """Show what a run did: problems first, then the summary."""
    for message in result.errors:
        st.error(message)
    for doc in result.flagged:
        st.warning(f"{doc.filename}: {doc.status}. {doc.error or ''}")
    if not result.errors:
        st.success(result.summary())


# --- user / developer view ---------------------------------------------------------------

USER_MODE, DEVELOPER_MODE = "User", "Developer"
MODE_KEY = "view_mode"


def mode_switch() -> None:
    """The User / Developer switch in the sidebar, shared by every page (same session key).

    User mode shows what an interviewer needs: candidates, their facts, a one-click index
    update. Developer mode adds the pipeline internals: raw file, Markdown, sections, chunks,
    embedding models, rebuilds and sizes."""
    st.sidebar.radio(
        "View",
        [USER_MODE, DEVELOPER_MODE],
        key=MODE_KEY,
        horizontal=True,
        help="Developer view adds the pipeline internals (parsing, chunks, index details).",
    )


def is_developer() -> bool:
    return st.session_state.get(MODE_KEY, USER_MODE) == DEVELOPER_MODE
