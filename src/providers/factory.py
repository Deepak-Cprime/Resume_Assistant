"""Create providers from config. Implementations are imported lazily, so heavy libraries
(torch, docling, faiss) load only when that provider is actually used.

To add a paid/alternative component: write a class satisfying the Protocol in base.py and
register it here (or call `register`) under a new provider name; then change config.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from src.config import Settings

# kind -> provider name -> "module:attribute" (a class or factory callable) or a callable.
_REGISTRY: dict[str, dict[str, str | Callable[..., Any]]] = {
    "parser": {
        "pymupdf4llm": "src.providers.parsers.pymupdf4llm_parser:PyMuPDF4LLMParser",
        "python_docx": "src.providers.parsers.docx_parser:PythonDocxParser",
        "docling": "src.providers.parsers.docling_parser:DoclingParser",  # optional, heavy
    },
    "embedder": {
        "sentence_transformers": "src.providers.embedders.st_embedder:SentenceTransformerEmbedder",
        "ollama": "src.providers.embedders.ollama_embedder:OllamaEmbedder",
    },
    "llm": {
        "ollama": "src.providers.llm.ollama_llm:OllamaLLM",
    },
    "vector_store": {
        "faiss": "src.providers.vectorstores.faiss_store:FaissStore",
    },
    "keyword": {
        "bm25s": "src.providers.keyword.bm25_index:Bm25Index",
    },
}


class ProviderError(RuntimeError):
    pass


def register(kind: str, name: str, target: str | Callable[..., Any]) -> None:
    """Register (or override) a provider; used for plugins and for test fakes."""
    _REGISTRY.setdefault(kind, {})[name] = target


def _resolve(kind: str, name: str) -> Callable[..., Any]:
    try:
        target = _REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY.get(kind, {}))) or "none"
        raise ProviderError(f"unknown {kind} provider '{name}' (known: {known})") from None
    if callable(target):
        return target
    module_name, _, attr = target.partition(":")
    try:
        return getattr(import_module(module_name), attr)
    except ModuleNotFoundError as e:
        raise ProviderError(f"{kind} provider '{name}' is not available yet: {e}") from e


def create_parser(name: str, settings: Settings) -> Any:
    return _resolve("parser", name)(settings=settings)


def create_embedder(model_name: str, settings: Settings) -> Any:
    cfg = settings.embedding.get(model_name)
    return _resolve("embedder", cfg.provider)(name=cfg.name)


def create_llm(settings: Settings) -> Any:
    return _resolve("llm", settings.llm.provider)(
        model=settings.llm.model, temperature=settings.llm.temperature
    )


def create_vector_store(dim: int, settings: Settings) -> Any:
    return _resolve("vector_store", settings.vector_store.provider)(
        dim=dim, index=settings.vector_store.index
    )


def create_keyword_index(settings: Settings) -> Any:
    return _resolve("keyword", settings.keyword.provider)(tokenizer=settings.keyword.tokenizer)
