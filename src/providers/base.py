"""Provider interfaces. Every external dependency is used only through these Protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class ParsedDocument:
    markdown: str
    n_pages: int
    parser: str
    parser_version: str = ""
    warnings: list[str] = field(default_factory=list)
    # Per-page character counts of the PDF text layer, used by the router and validation.
    chars_per_page: list[int] = field(default_factory=list)
    # Markdown per page when the parser can split it (lets cleaning drop repeated footers).
    pages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VectorHit:
    chunk_id: int
    score: float


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: int
    score: float


@runtime_checkable
class DocumentParser(Protocol):
    name: str

    def parse(self, path: Path) -> ParsedDocument: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int
    max_tokens: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return float32 array (n, dim), L2-normalised; applies the model's document prefix."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return float32 array (dim,), L2-normalised; applies the model's query prefix."""
        ...

    def count_tokens(self, text: str) -> int:
        """Tokens `text` costs when embedded as a document (prefix and special tokens included),
        so it can be compared with `max_tokens`."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def generate(self, prompt: str, *, system: str | None = None) -> str: ...

    def generate_json(
        self, prompt: str, schema: dict[str, Any], *, system: str | None = None
    ) -> dict[str, Any]:
        """Return a dict conforming to the JSON schema, or raise ValueError."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    dim: int
    suffix: str  # file extension of the persisted index, e.g. ".faiss"

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        """Insert or replace: adding an id that already exists replaces its vector.
        Vectors must be float32, shape (n, dim) and L2-normalised (inner product = cosine)."""
        ...

    def remove(self, ids: list[int]) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[VectorHit]:
        """Best `k` hits, highest score first; fewer if the store holds fewer."""
        ...

    def ids(self) -> list[int]: ...

    def count(self) -> int: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


@runtime_checkable
class KeywordIndex(Protocol):
    def build(self, ids: list[int], texts: list[str]) -> None:
        """Replace the whole index with these documents."""
        ...

    def search(self, query: str, k: int) -> list[KeywordHit]:
        """Up to `k` documents sharing a term with the query, best first (zero scores dropped)."""
        ...

    def ids(self) -> list[int]: ...

    def count(self) -> int: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...
