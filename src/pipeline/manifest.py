"""What an index was built from, and the check that keeps queries and vectors compatible.

Each embedding model has its own folder under data/indexes/ holding `vectors.<ext>`,
`hashes.json` (content hash of every embedded chunk, so rebuilds re-embed only changes) and
`manifest.json`. A query must be embedded by the same model, with the same prefix, that built
the index; otherwise cosine scores are meaningless, so `load_index` refuses a mismatch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import Settings
from src.providers import factory
from src.providers.base import EmbeddingProvider, VectorStore
from src.stores import files

VECTORS_STEM = "vectors"
HASHES_FILE = "hashes.json"
MANIFEST_FILE = "manifest.json"


class IndexMismatch(RuntimeError):
    """A query embedder does not match the index it is asked to search."""


@dataclass
class IndexManifest:
    emb_model: str
    dim: int
    n_vectors: int
    index_version: str
    doc_prefix: str
    query_prefix: str
    budgets_fingerprint: str
    chunk_set_hash: str
    created_at: str
    vector_store: str = "faiss"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> IndexManifest:
        return cls(**json.loads(text))


def chunk_set_hash(pairs: Iterable[tuple[int, str]]) -> str:
    """Fingerprint of exactly which chunks (id + content) an index holds."""
    digest = hashlib.sha256()
    for chunk_id, content_hash in sorted(pairs):
        digest.update(f"{chunk_id}:{content_hash};".encode())
    return digest.hexdigest()[:16]


def index_version(
    emb_model: str, dim: int, doc_prefix: str, query_prefix: str, budgets_fp: str, chunks_hash: str
) -> str:
    """Changes whenever the model, its prefixes, the chunk boundaries or the chunk text change."""
    raw = json.dumps([emb_model, dim, doc_prefix, query_prefix, budgets_fp, chunks_hash])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def vectors_path(settings: Settings, model_name: str, store: VectorStore) -> Path:
    return files.index_dir(settings, model_name) / f"{VECTORS_STEM}{store.suffix}"


def read_manifest(settings: Settings, model_name: str) -> IndexManifest | None:
    path = files.index_dir(settings, model_name) / MANIFEST_FILE
    return IndexManifest.from_json(path.read_text(encoding="utf-8")) if path.exists() else None


def read_hashes(settings: Settings, model_name: str) -> dict[int, str]:
    path = files.index_dir(settings, model_name) / HASHES_FILE
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def check_query_embedder(manifest: IndexManifest, embedder: EmbeddingProvider) -> None:
    """Raise IndexMismatch unless `embedder` is the one that built the index."""
    problems = []
    if embedder.name != manifest.emb_model:
        problems.append(f"model is '{embedder.name}', index was built with '{manifest.emb_model}'")
    if embedder.dim != manifest.dim:
        problems.append(f"dimension is {embedder.dim}, index has {manifest.dim}")
    if getattr(embedder, "query_prefix", "") != manifest.query_prefix:
        problems.append("query prefix differs from the one the index was built for")
    if problems:
        raise IndexMismatch("; ".join(problems))


def load_index(
    settings: Settings, embedder: EmbeddingProvider
) -> tuple[VectorStore, IndexManifest]:
    """Open the persisted index for `embedder`'s model, ready to search.

    Raises FileNotFoundError if that model has no index yet, and IndexMismatch if the embedder
    differs from the one that built it (different model, dimension or query prefix)."""
    manifest = read_manifest(settings, embedder.name)
    if manifest is None:
        raise FileNotFoundError(f"no index has been built for model '{embedder.name}'")
    check_query_embedder(manifest, embedder)
    store = factory.create_vector_store(manifest.dim, settings, provider=manifest.vector_store)
    store.load(vectors_path(settings, embedder.name, store))
    return store, manifest
