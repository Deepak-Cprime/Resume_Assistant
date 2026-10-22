"""Exact (brute-force) cosine search with FAISS: IndexIDMap(IndexFlatIP) on unit vectors.

Flat search is exact and fast enough for tens of thousands of chunks; the config key
`vector_store.index` is where an approximate index (HNSW, IVF) would be added later.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.providers.base import VectorHit
from src.providers.vectorstores.common import check_batch, check_query


class FaissStore:
    suffix = ".faiss"

    def __init__(self, dim: int, index: str = "flat_ip") -> None:
        import faiss

        if index != "flat_ip":
            raise ValueError(f"unsupported faiss index '{index}' (only 'flat_ip')")
        self._faiss = faiss
        self.dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        id_array, batch = check_batch(ids, vectors, self.dim)
        if len(id_array):
            self._index.remove_ids(id_array)  # IDMap would otherwise keep both copies
            self._index.add_with_ids(batch, id_array)

    def remove(self, ids: list[int]) -> None:
        if ids:
            self._index.remove_ids(np.asarray(ids, dtype=np.int64))

    def search(self, query: np.ndarray, k: int) -> list[VectorHit]:
        q = check_query(query, self.dim)
        if self._index.ntotal == 0 or k <= 0:
            return []
        scores, found = self._index.search(q.reshape(1, -1), min(k, self._index.ntotal))
        return [
            VectorHit(int(i), float(s)) for i, s in zip(found[0], scores[0], strict=True) if i != -1
        ]

    def ids(self) -> list[int]:
        return [int(i) for i in self._faiss.vector_to_array(self._index.id_map)]

    def count(self) -> int:
        return int(self._index.ntotal)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        self._faiss.write_index(self._index, str(tmp))
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        index = self._faiss.read_index(str(path))
        if index.d != self.dim:
            raise ValueError(f"{path.name} has dimension {index.d}, expected {self.dim}")
        self._index = index
