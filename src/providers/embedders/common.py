"""Pieces shared by every embedder: normalisation, the error type, per-model prefixes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class EmbeddingError(RuntimeError):
    """The embedding backend failed (model missing, server down ...). Says what to do."""


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """float32 copy with every row scaled to unit length (an all-zero row stays zero)."""
    out = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


@dataclass(frozen=True)
class Prefixes:
    """Instruction text a model was trained to expect before queries and before documents."""

    query: str = ""
    document: str = ""


_BGE_QUERY = "Represent this sentence for searching relevant passages: "
_PREFIXES = {
    "baai/bge-small-en-v1.5": Prefixes(query=_BGE_QUERY),
    "baai/bge-base-en-v1.5": Prefixes(query=_BGE_QUERY),
    "baai/bge-large-en-v1.5": Prefixes(query=_BGE_QUERY),
    "nomic-embed-text": Prefixes("search_query: ", "search_document: "),
    "nomic-ai/nomic-embed-text-v1.5": Prefixes("search_query: ", "search_document: "),
    "intfloat/e5-small-v2": Prefixes("query: ", "passage: "),
    "intfloat/e5-base-v2": Prefixes("query: ", "passage: "),
}
# Context length (tokens) for models whose backend does not report it.
_CONTEXT = {"nomic-embed-text": 8192, "nomic-ai/nomic-embed-text-v1.5": 8192}
DEFAULT_CONTEXT = 2048


def _key(model_name: str) -> str:
    return model_name.strip().lower().removesuffix(":latest")


def prefixes_for(model_name: str) -> Prefixes:
    """Documented prefixes for a model; no prefixes for models that do not use them."""
    return _PREFIXES.get(_key(model_name), Prefixes())


def context_for(model_name: str) -> int:
    return _CONTEXT.get(_key(model_name), DEFAULT_CONTEXT)
