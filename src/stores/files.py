"""Where files live under data/, and atomic writes."""

from __future__ import annotations

import re
from pathlib import Path

from src import fsutil
from src.config import Settings


def raw_path(settings: Settings, doc_id: str, ext: str) -> Path:
    return settings.raw_dir / f"{doc_id}{ext}"


def markdown_path(settings: Settings, doc_id: str) -> Path:
    return settings.markdown_dir / f"{doc_id}.md"


def model_slug(model_name: str) -> str:
    """Folder-safe model name: 'BAAI/bge-small-en-v1.5' -> 'baai_bge-small-en-v1.5'."""
    return re.sub(r"[^\w.-]+", "_", model_name.strip().lower()).strip("_")


def index_dir(settings: Settings, model_name: str) -> Path:
    """One folder per embedding model: vectors, per-chunk hashes and the manifest side by side."""
    return settings.indexes_dir / model_slug(model_name)


def keyword_dir(settings: Settings) -> Path:
    """The BM25 index; independent of any embedding model. The leading underscore keeps it
    from ever colliding with a model folder (model_slug never starts with one)."""
    return settings.indexes_dir / "_bm25"


def sections_path(settings: Settings, doc_id: str) -> Path:
    return settings.profiles_dir / f"{doc_id}.sections.json"


def profile_path(settings: Settings, doc_id: str) -> Path:
    return settings.profiles_dir / f"{doc_id}.json"


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file then rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    fsutil.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))
