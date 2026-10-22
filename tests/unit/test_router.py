from pathlib import Path

import pymupdf
import pytest

from src.providers.base import ParsedDocument
from src.providers.parsers import router
from src.providers.parsers.router import LayoutFeatures

GOOD = " ".join(["engineer"] * 60) + " someone@example.com"


class FakeParser:
    def __init__(self, name: str, text: str | None = GOOD, error: Exception | None = None):
        self.name, self.text, self.error, self.calls = name, text, error, 0

    def parse(self, path: Path) -> ParsedDocument:
        self.calls += 1
        if self.error:
            raise self.error
        return ParsedDocument(markdown=self.text, n_pages=1, parser=self.name, pages=[self.text])


def factory_for(*parsers: FakeParser):
    by_name = {p.name: p for p in parsers}
    return lambda name, settings: by_name[name]


# --- layout features on the generated fixtures ------------------------------------------


@pytest.mark.parametrize(
    "spec_id",
    ["single_col_backend", "single_col_overlap", "single_col_gap",
     "single_col_no_headings", "single_col_spanish"],
)
def test_single_column_pdfs_use_the_fast_parser(fixture_files, settings, spec_id):
    f = router.analyze(fixture_files[spec_id], "pdf", settings)
    assert (f.scanned, f.multi_column) == (False, False)
    assert router.choose(f, settings).parser == "pymupdf4llm"


@pytest.mark.parametrize("spec_id", ["two_col_sidebar", "two_col_sidebar_right"])
def test_two_column_pdfs_are_detected_but_use_the_default_parser(fixture_files, settings, spec_id):
    f = router.analyze(fixture_files[spec_id], "pdf", settings)
    assert f.multi_column and not f.scanned
    assert router.choose(f, settings).parser == "pymupdf4llm"


def test_scanned_pdf_has_no_text_layer(fixture_files, settings):
    f = router.analyze(fixture_files["scanned_devops"], "pdf", settings)
    assert f.scanned
    assert max(f.chars_per_page) < settings.parser.min_chars_per_page


def test_scanned_pdf_stays_on_default_parser(settings):
    scanned = LayoutFeatures(doc_type="pdf", scanned=True)
    decision = router.choose(scanned, settings)
    assert decision.parser == "pymupdf4llm" and "OCR" in decision.reasons[0]
    settings.parser.fallback = "docling"
    decision = router.choose(scanned, settings)
    assert (decision.parser, decision.fallback) == ("pymupdf4llm", "docling")


def test_digital_fixtures_have_a_text_layer(fixture_files):
    with pymupdf.open(str(fixture_files["single_col_backend"])) as doc:
        assert "Northwind Payments" in doc[0].get_text()


def test_docx_goes_to_python_docx(settings):
    decision = router.choose(LayoutFeatures(doc_type="docx"), settings)
    assert decision.parser == "python_docx" and decision.fallback is None


def test_no_heavy_fallback_by_default(settings):
    decision = router.choose(LayoutFeatures(doc_type="pdf"), settings)
    assert (decision.parser, decision.fallback) == ("pymupdf4llm", None)


@pytest.fixture
def with_fallback(settings):
    settings.parser.fallback = "docling"
    return settings


# --- parse_resume: parse, clean, validate, fall back ------------------------------------


def test_good_first_parser_is_used(fixture_files, with_fallback):
    fast, heavy = FakeParser("pymupdf4llm"), FakeParser("docling")
    out = router.parse_resume(fixture_files["single_col_backend"], "pdf", with_fallback,
                              parser_factory=factory_for(fast, heavy))
    assert out.parsed.parser == "pymupdf4llm" and not out.needs_review
    assert (fast.calls, heavy.calls) == (1, 0)


def test_unusable_result_falls_back(fixture_files, with_fallback):
    fast, heavy = FakeParser("pymupdf4llm", text=""), FakeParser("docling")
    out = router.parse_resume(fixture_files["single_col_backend"], "pdf", with_fallback,
                              parser_factory=factory_for(fast, heavy))
    assert out.parsed.parser == "docling" and not out.needs_review
    assert len(out.attempts) == 2 and "unusable" in out.attempts[0]


def test_crash_falls_back(fixture_files, with_fallback):
    fast = FakeParser("pymupdf4llm", error=RuntimeError("boom"))
    out = router.parse_resume(fixture_files["single_col_backend"], "pdf", with_fallback,
                              parser_factory=factory_for(fast, FakeParser("docling")))
    assert out.parsed.parser == "docling" and "crashed" in out.attempts[0]


def test_both_unusable_means_needs_review(fixture_files, with_fallback):
    fast, heavy = FakeParser("pymupdf4llm", text=""), FakeParser("docling", text="x")
    out = router.parse_resume(fixture_files["single_col_backend"], "pdf", with_fallback,
                              parser_factory=factory_for(fast, heavy))
    assert out.needs_review


def test_unusable_without_fallback_means_needs_review(fixture_files, settings):
    fast = FakeParser("pymupdf4llm", text="")
    out = router.parse_resume(fixture_files["single_col_backend"], "pdf", settings,
                              parser_factory=factory_for(fast))
    assert out.needs_review and len(out.attempts) == 1


def test_all_parsers_crashing_raises(fixture_files, with_fallback):
    boom = RuntimeError("boom")
    with pytest.raises(router.ParseError, match="crashed"):
        router.parse_resume(
            fixture_files["single_col_backend"], "pdf", with_fallback,
            parser_factory=factory_for(
                FakeParser("pymupdf4llm", error=boom), FakeParser("docling", error=boom)
            ),
        )


def test_output_is_cleaned(fixture_files, settings):
    text = "ﬁnance " + GOOD
    out = router.parse_resume(fixture_files["single_col_backend"], "pdf", settings,
                              parser_factory=factory_for(FakeParser("pymupdf4llm", text=text),
                                                         FakeParser("docling")))
    assert out.markdown.startswith("finance")
