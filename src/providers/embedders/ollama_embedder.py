"""Embeddings from a local Ollama server (default: nomic-embed-text, ~270 MB, runs offline).

Needs Ollama running and the model pulled once: `ollama pull nomic-embed-text`. Ollama has no
public tokenizer endpoint, so `count_tokens` uses the same conservative estimate as the chunker.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.chunking.tokens import estimate_tokens
from src.providers.embedders.common import (
    EmbeddingError,
    context_for,
    l2_normalize,
    prefixes_for,
)

DIM_PROBE = "dimension probe"


class OllamaEmbedder:
    def __init__(
        self,
        name: str = "nomic-embed-text",
        batch_size: int = 32,
        device: str | None = None,  # accepted for interface parity; Ollama decides placement
        host: str | None = None,
        client: Any = None,
    ) -> None:
        self.name = name
        self.batch_size = batch_size
        self.host = host or os.environ.get("OLLAMA_HOST")
        self._client = client  # tests inject a stub
        self._dim: int | None = None
        prefixes = prefixes_for(name)
        self.query_prefix, self.doc_prefix = prefixes.query, prefixes.document

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host) if self.host else ollama.Client()
        return self._client

    @property
    def max_tokens(self) -> int:
        return context_for(self.name)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._embed([DIM_PROBE]).shape[1])
        return self._dim

    def _embed(self, inputs: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        try:
            for start in range(0, len(inputs), self.batch_size):
                batch = inputs[start : start + self.batch_size]
                response = self._get_client().embed(model=self.name, input=batch)
                rows.extend(response["embeddings"])
        except Exception as e:  # noqa: BLE001 - connection, missing model, bad response ...
            raise EmbeddingError(
                f"Ollama embedding with '{self.name}' failed: {e}. Is Ollama running, and has "
                f"the model been pulled (`ollama pull {self.name}`)?"
            ) from e
        if len(rows) != len(inputs):
            raise EmbeddingError(f"Ollama returned {len(rows)} vectors for {len(inputs)} inputs")
        return l2_normalize(np.asarray(rows, dtype=np.float32).reshape(len(inputs), -1))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._embed([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(self.doc_prefix + text)
