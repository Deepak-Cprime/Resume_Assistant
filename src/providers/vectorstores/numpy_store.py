"""Exact cosine search in plain NumPy: no extra dependency, same behaviour as the FAISS store.

Useful where faiss-cpu has no wheel for the Python version, and as the reference
implementation the FAISS store is tested against. Fine up to roughly 100k chunks.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.providers.base import VectorHit
from src.providers.vectorstores.common import check_batch, check_query


class NumpyStore:
    suffix = ".npz"

    def __init__(self, dim: int, index: str = "flat_ip") -> None:
        if index != "flat_ip":
            raise ValueError(f"unsupported index '{index}' (only 'flat_ip')")
        self.dim = dim
        self._ids = np.zeros(0, dtype=np.int64)
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        id_array, batch = check_batch(ids, vectors, self.dim)
        if not len(id_array):
            return
        self.remove(list(id_array))
        self._ids = np.concatenate([self._ids, id_array])
        self._vectors = np.concatenate([self._vectors, batch])

    def remove(self, ids: list[int]) -> None:
        if not len(ids) or not len(self._ids):
            return
        keep = ~np.isin(self._ids, np.asarray(ids, dtype=np.int64))
        self._ids, self._vectors = self._ids[keep], self._vectors[keep]

    def search(self, query: np.ndarray, k: int) -> list[VectorHit]:
        q = check_query(query, self.dim)
        if not len(self._ids) or k <= 0:
            return []
        scores = self._vectors @ q
        # highest score first; equal scores ordered by id so results are reproducible
        order = np.lexsort((self._ids, -scores))[:k]
        return [VectorHit(int(self._ids[i]), float(scores[i])) for i in order]

    def ids(self) -> list[int]:
        return [int(i) for i in self._ids]

    def count(self) -> int:
        return int(len(self._ids))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez(f, ids=self._ids, vectors=self._vectors)
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        with np.load(path) as data:
            ids, vectors = data["ids"], data["vectors"]
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"{path.name} has dimension {vectors.shape[-1]}, expected {self.dim}")
        self._ids, self._vectors = ids.astype(np.int64), vectors.astype(np.float32)
