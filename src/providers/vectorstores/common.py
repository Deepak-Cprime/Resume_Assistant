"""Input checks every vector store applies, so a bad batch fails loudly instead of silently
corrupting similarity scores."""

from __future__ import annotations

import numpy as np

NORM_TOLERANCE = 1e-3


def check_batch(ids: list[int], vectors: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """(int64 ids, float32 vectors) after validating shape, uniqueness and unit length.

    Inner product equals cosine similarity only for unit vectors, so an un-normalised batch
    (an embedder that forgot to normalise) is rejected rather than ranked wrongly."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"expected vectors of shape (n, {dim}), got {arr.shape}")
    if len(ids) != arr.shape[0]:
        raise ValueError(f"{len(ids)} ids for {arr.shape[0]} vectors")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids in one batch")
    if not np.isfinite(arr).all():
        raise ValueError("vectors contain NaN or infinity")
    if arr.size and not np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=NORM_TOLERANCE):
        raise ValueError("vectors must be L2-normalised")
    return np.asarray(ids, dtype=np.int64), np.ascontiguousarray(arr)


def check_query(query: np.ndarray, dim: int) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    if q.shape[0] != dim:
        raise ValueError(f"query has {q.shape[0]} dimensions, the index has {dim}")
    return q
