"""Where files live under data/, and atomic writes."""

from __future__ import annotations

import os
from pathlib import Path

from src.config import Settings


def raw_path(settings: Settings, doc_id: str, ext: str) -> Path:
    return settings.raw_dir / f"{doc_id}{ext}"


def markdown_path(settings: Settings, doc_id: str) -> Path:
    return settings.markdown_dir / f"{doc_id}.md"


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file then rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))
