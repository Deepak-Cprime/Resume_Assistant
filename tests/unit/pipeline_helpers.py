"""Shared setup for whole-pipeline tests: a parser that returns canned Markdown per upload, a
settings tweak that makes a fake embedder the default model, and distinct resumes."""

from datetime import date
from pathlib import Path

from src.config import EmbeddingModelConfig, Settings
from src.ingestion import intake
from src.providers.base import ParsedDocument
from tests.unit.fakes import FULL_MD


class MarkdownParser:
    """Stands in for pymupdf4llm. Raw files are stored as <doc_id>.pdf, so canned text is keyed
    by doc id. Records every call and can be told to fail for chosen documents."""

    name = "pymupdf4llm"

    def __init__(self) -> None:
        self.texts: dict[str, str] = {}
        self.calls: list[str] = []
        self.fail: set[str] = set()

    def parse(self, path: Path) -> ParsedDocument:
        doc_id = path.stem
        self.calls.append(doc_id)
        if doc_id in self.fail:
            raise RuntimeError("cannot parse this file")
        text = self.texts[doc_id]
        return ParsedDocument(markdown=text, n_pages=1, parser=self.name, pages=[text])

    @property
    def factory(self):
        return lambda name, settings: self


def resume(i: int, *, bullet: str = "Led migration of 40 services") -> str:
    """FULL_MD for candidate i: own email (so own candidate) and a unique company token."""
    text = FULL_MD.replace("priya.raman@example.com", f"p{i}@example.com")
    text = text.replace("Northwind Payments", f"Marker{i}Corp")
    return text.replace("Led migration of 40 services", bullet)


def configure(settings: Settings, *models: str) -> None:
    """Fixed 'Present' date, the NumPy vector store, and these fake models as the configured
    embedding models (the first is the default)."""
    settings.metadata.as_of = date(2025, 6, 1)
    settings.vector_store.provider = "numpy"
    settings.embedding.models.clear()  # drop the real model from configs/config.yaml
    for name in models or ("fake-embed",):
        settings.embedding.models.append(
            EmbeddingModelConfig(provider="sentence_transformers", name=name)
        )
    settings.embedding.default = (models or ("fake-embed",))[0]


def upload(conn, settings, fixture_files, spec_id: str, parser: MarkdownParser, text: str) -> str:
    """Store a fixture file as an upload and register the Markdown the parser will return."""
    path = fixture_files[spec_id]
    doc_id = intake.ingest(conn, settings, path.name, path.read_bytes()).doc_id
    parser.texts[doc_id] = text
    return doc_id
