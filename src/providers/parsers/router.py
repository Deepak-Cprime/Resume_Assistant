"""Pick a parser per document from cheap layout features, then parse, clean and validate,
trying the optional fallback parser when the first result is unusable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Settings
from src.ingestion.cleaning import clean_pages
from src.ingestion.validation import ValidationReport, validate_markdown
from src.logging_setup import get_logger
from src.providers import factory
from src.providers.base import DocumentParser, ParsedDocument
from src.providers.parsers.layout import detect_multi_column

log = get_logger(__name__)

ParserFactory = Callable[[str, Settings], DocumentParser]


class ParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class LayoutFeatures:
    doc_type: str
    n_pages: int = 1
    chars_per_page: list[int] = field(default_factory=list)
    scanned: bool = False
    multi_column: bool = False


@dataclass(frozen=True)
class RouteDecision:
    parser: str
    fallback: str | None
    reasons: list[str]


@dataclass
class ParseOutcome:
    parsed: ParsedDocument
    markdown: str
    route: RouteDecision
    features: LayoutFeatures
    report: ValidationReport
    attempts: list[str]

    @property
    def needs_review(self) -> bool:
        return self.report.fatal


def analyze(path: Path, doc_type: str, settings: Settings) -> LayoutFeatures:
    if doc_type != "pdf":
        return LayoutFeatures(doc_type=doc_type)
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        chars = [len(page.get_text().strip()) for page in doc]
        multi = any(detect_multi_column(page) for page in doc)  # informational: parser handles it
        n_pages = doc.page_count
    empty_pages = sum(c < settings.parser.min_chars_per_page for c in chars)
    scanned = n_pages > 0 and empty_pages / n_pages >= 0.5
    return LayoutFeatures(
        doc_type=doc_type, n_pages=n_pages, chars_per_page=chars,
        scanned=scanned, multi_column=multi,
    )


def choose(features: LayoutFeatures, settings: Settings) -> RouteDecision:
    """DOCX -> python-docx; every PDF -> the default parser, which handles multi-column pages
    itself and runs its built-in light OCR (RapidOCR, no torch) on pages without a text
    layer. The optional heavy parser (parser.fallback, e.g. Docling) is only tried when the
    default parser's result is unusable."""
    p = settings.parser
    if features.doc_type == "docx":
        return RouteDecision(p.docx, None, ["DOCX"])
    if features.scanned:
        kind = "scanned (built-in OCR)"
    else:
        kind = "multi-column" if features.multi_column else "single-column"
    return RouteDecision(p.default, p.fallback, [f"{kind} PDF"])


def parse_resume(
    path: Path,
    doc_type: str,
    settings: Settings,
    *,
    parser_factory: ParserFactory = factory.create_parser,
) -> ParseOutcome:
    """Route, parse, clean and validate. Raises ParseError only if every parser crashed;
    an unusable result is returned with `needs_review` set so the UI can show it."""
    features = analyze(path, doc_type, settings)
    route = choose(features, settings)
    names = [route.parser] + ([route.fallback] if route.fallback else [])
    attempts: list[str] = []
    last: ParseOutcome | None = None
    for name in names:
        try:
            parsed = parser_factory(name, settings).parse(path)
        except Exception as e:  # noqa: BLE001 - any parser crash should try the fallback
            log.warning("%s failed on %s: %s", name, path.name, e)
            attempts.append(f"{name}: crashed ({e})")
            continue
        markdown = clean_pages(parsed.pages or [parsed.markdown])
        report = validate_markdown(markdown, parsed.n_pages, settings)
        attempts.append(f"{name}: {'unusable - ' + report.summary() if report.fatal else 'ok'}")
        last = ParseOutcome(parsed, markdown, route, features, report, list(attempts))
        if not report.fatal:
            return last
    if last is None:
        raise ParseError("; ".join(attempts))
    return last
