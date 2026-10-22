"""One place to configure logging for the app, scripts and tests."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Idempotent: safe to call on every Streamlit rerun."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not any(getattr(h, "_rag_handler", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._rag_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
