"""Layout-model path for two-column, scanned and DOCX files (Docling, OCR built in)."""

from __future__ import annotations

import shutil
from importlib import metadata
from pathlib import Path
from typing import Any

from src.config import Settings
from src.logging_setup import get_logger
from src.providers.base import ParsedDocument

log = get_logger(__name__)


class DoclingParser:
    name = "docling"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self._converter: Any = None  # built on first use; loading models is slow

    def _ocr_options(self) -> Any:
        """Tesseract if configured and installed; otherwise Docling's default OCR engine."""
        wanted = self.settings.parser.ocr if self.settings else ""
        if wanted != "tesseract":
            return None
        if shutil.which("tesseract") is None:
            log.warning("parser.ocr is 'tesseract' but the binary is not on PATH; using default")
            return None
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions

        return TesseractCliOcrOptions(lang=["eng"])

    def _get_converter(self) -> Any:
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions()
            options.do_ocr = True
            ocr = self._ocr_options()
            if ocr is not None:
                options.ocr_options = ocr
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    def parse(self, path: Path) -> ParsedDocument:
        result = self._get_converter().convert(str(path))
        doc = result.document
        try:
            n_pages = int(doc.num_pages())
        except Exception:
            n_pages = 0
        is_pdf = path.suffix.lower() == ".pdf"
        pages = self._page_markdown(doc, n_pages) if is_pdf else [doc.export_to_markdown()]
        return ParsedDocument(
            markdown="\n\n".join(pages),
            n_pages=max(n_pages, 1),
            parser=self.name,
            parser_version=_version("docling"),
            pages=pages,
        )

    @staticmethod
    def _page_markdown(doc: Any, n_pages: int) -> list[str]:
        if n_pages > 1:
            try:
                pages = [doc.export_to_markdown(page_no=i) for i in range(1, n_pages + 1)]
                if any(p.strip() for p in pages):
                    return pages
            except Exception:  # per-page export not supported by this Docling version
                pass
        return [doc.export_to_markdown()]


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return ""
