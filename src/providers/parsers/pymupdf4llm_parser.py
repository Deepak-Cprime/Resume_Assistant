"""Default PDF parser: pymupdf4llm for ordinary pages, column-aware rendering for pages with
a sidebar. CPU only, no ML models, milliseconds per page."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

from src.config import Settings
from src.providers.base import ParsedDocument
from src.providers.parsers import layout


class PyMuPDF4LLMParser:
    name = "pymupdf4llm"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    def parse(self, path: Path) -> ParsedDocument:
        import pymupdf

        warnings: list[str] = []
        with pymupdf.open(str(path)) as doc:
            chars = [len(page.get_text().strip()) for page in doc]
            splits = {i: layout.find_column_split(page) for i, page in enumerate(doc)}
            splits = {i: s for i, s in splits.items() if s is not None}
            ordinary = [i for i in range(doc.page_count) if i not in splits]
            texts = dict(zip(ordinary, self._markdown_pages(doc, ordinary), strict=True))
            for i, split in splits.items():
                texts[i] = layout.column_markdown(doc[i], split)
                warnings.append(f"page {i + 1}: two-column layout, read column by column")
            n_pages = doc.page_count
        pages = [texts[i] for i in range(n_pages)]
        return ParsedDocument(
            markdown="\n\n".join(pages),
            n_pages=n_pages,
            parser=self.name,
            parser_version=_version("pymupdf4llm"),
            warnings=warnings,
            chars_per_page=chars,
            pages=pages,
        )

    @staticmethod
    def _markdown_pages(doc: Any, page_numbers: list[int]) -> list[str]:
        if not page_numbers:
            return []
        import pymupdf4llm

        kwargs: dict[str, Any] = {"pages": page_numbers, "page_chunks": True}
        try:
            chunks = pymupdf4llm.to_markdown(doc, show_progress=False, **kwargs)
        except TypeError:  # older releases have no show_progress argument
            chunks = pymupdf4llm.to_markdown(doc, **kwargs)
        return [c["text"] for c in chunks]


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return ""
