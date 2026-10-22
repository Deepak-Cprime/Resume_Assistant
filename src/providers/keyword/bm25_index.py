"""BM25 keyword search over chunk text with bm25s (pure NumPy/SciPy, no model).

The index returns chunk ids, the same ids the vector index uses, so Phase 2 can fuse the two
rankings. A document is whatever text the caller passes (the chunks' `embed_text`: structural
prefix, text and skill tags), tokenized by the configured tokenizer; queries go through the
same tokenizer with `is_query=True`.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from src.providers.base import KeywordHit
from src.providers.keyword.tokenizer import Tokenizer, get_tokenizer

IDS_FILE = "ids.json"
MODEL_DIR = "bm25"
EMPTY_DOCUMENT = "<empty>"  # bm25s cannot index a document with no tokens


class Bm25Index:
    def __init__(self, tokenizer: str = "tech_aware", k1: float = 1.5, b: float = 0.75) -> None:
        self.tokenizer_name = tokenizer
        self._tokenize: Tokenizer = get_tokenizer(tokenizer)
        self.k1, self.b = k1, b
        self._ids: list[int] = []
        self._retriever: Any = None  # None while the index is empty

    def _tokens(self, text: str, *, is_query: bool = False) -> list[str]:
        return self._tokenize(text, is_query=is_query)

    def build(self, ids: list[int], texts: list[str]) -> None:
        """Replace the whole index with these documents (BM25 statistics are corpus-wide, so an
        index is rebuilt rather than patched)."""
        if len(ids) != len(texts):
            raise ValueError(f"{len(ids)} ids for {len(texts)} texts")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids")
        self._ids = list(ids)
        self._retriever = None
        if not ids:
            return
        import bm25s

        corpus = [self._tokens(t) or [EMPTY_DOCUMENT] for t in texts]
        retriever = bm25s.BM25(k1=self.k1, b=self.b)
        retriever.index(corpus, show_progress=False)
        self._retriever = retriever

    def search(self, query: str, k: int) -> list[KeywordHit]:
        """Up to `k` chunks that share at least one query term, best first."""
        if self._retriever is None or k <= 0:
            return []
        known = self._retriever.vocab_dict
        terms = [t for t in self._tokens(query, is_query=True) if t in known]
        if not terms:
            return []
        documents, scores = self._retriever.retrieve(
            [terms], k=min(k, len(self._ids)), show_progress=False
        )
        return [
            KeywordHit(self._ids[int(i)], float(s))
            for i, s in zip(documents[0], scores[0], strict=True)
            if s > 0
        ]

    def ids(self) -> list[int]:
        return list(self._ids)

    def count(self) -> int:
        return len(self._ids)

    def save(self, path: Path) -> None:
        """Write the index into directory `path`, replacing what was there."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / IDS_FILE).write_text(json.dumps(self._ids), encoding="utf-8")
        if self._retriever is not None:
            self._retriever.save(str(tmp / MODEL_DIR))
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        import bm25s

        path = Path(path)
        self._ids = [int(i) for i in json.loads((path / IDS_FILE).read_text(encoding="utf-8"))]
        model_dir = path / MODEL_DIR
        self._retriever = bm25s.BM25.load(str(model_dir)) if model_dir.exists() else None
        if self._ids and self._retriever is None:
            raise ValueError(f"{path} lists {len(self._ids)} documents but has no BM25 data")
