"""Local embeddings with sentence-transformers (BAAI/bge-small-en-v1.5 by default).

The model is loaded on first use, not at construction, so building the object is free; cache the
object (st.cache_resource in the UI) so the model loads once. The first load downloads the
weights from the Hugging Face hub (about 130 MB for bge-small); afterwards it works offline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.providers.embedders.common import EmbeddingError, l2_normalize, prefixes_for


class SentenceTransformerEmbedder:
    def __init__(
        self,
        name: str,
        batch_size: int = 32,
        device: str | None = None,
        model: Any = None,
    ) -> None:
        self.name = name
        self.batch_size = batch_size
        self.device = device
        self._model = model  # tests inject a stub; normally loaded lazily
        prefixes = prefixes_for(name)
        self.query_prefix, self.doc_prefix = prefixes.query, prefixes.document

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name, device=self.device)
            except Exception as e:  # noqa: BLE001 - turn any load failure into one clear error
                raise EmbeddingError(
                    f"could not load embedding model '{self.name}': {e}. Check the model name "
                    "and your internet connection (the first load downloads the weights)."
                ) from e
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        # renamed in recent sentence-transformers; keep the old name for older installs
        getter = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        return int(getter())

    @property
    def max_tokens(self) -> int:
        return int(self._load().max_seq_length)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(vectors)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

    def count_tokens(self, text: str) -> int:
        tokenizer = self._load().tokenizer
        return len(tokenizer.encode(self.doc_prefix + text, add_special_tokens=True))
