"""Real parsers on the generated fixtures. The PDF and DOCX parsers are light and run by
default; the optional Docling parser is marked `slow` (model download, seconds per page)."""

import re

import pytest

from src.ingestion.cleaning import clean_pages
from src.providers.parsers import router
from src.providers.parsers.docx_parser import PythonDocxParser
from src.providers.parsers.pymupdf4llm_parser import PyMuPDF4LLMParser
from tests.unit.helpers import check_reading_order, norm


def test_single_column_is_read_in_order(fixture_files, specs, settings):
    pytest.importorskip("pymupdf4llm")
    spec = specs["single_col_backend"]
    parsed = PyMuPDF4LLMParser(settings).parse(fixture_files["single_col_backend"])
    assert parsed.parser == "pymupdf4llm" and parsed.n_pages == 1 and not parsed.warnings
    md = clean_pages(parsed.pages)

    for job in spec["experience"]:
        assert job["company"] in md
    text = norm(md)
    sections = [text.find(h) for h in ("Experience", "Education", "Skills", "Certifications")]
    assert -1 not in sections and sections == sorted(sections)
    check_reading_order(md, spec)


def test_single_column_headings_become_markdown_headings(fixture_files, settings):
    pytest.importorskip("pymupdf4llm")
    md = PyMuPDF4LLMParser(settings).parse(fixture_files["single_col_backend"]).markdown
    heading = re.search(r"(?m)^#+\s*\**Experience", md)
    assert heading, "section headings should become Markdown headings"


@pytest.mark.parametrize("spec_id", ["two_col_sidebar", "two_col_sidebar_right"])
def test_sidebar_stays_out_of_experience(fixture_files, specs, settings, spec_id):
    parsed = PyMuPDF4LLMParser(settings).parse(fixture_files[spec_id])
    assert any("two-column" in w for w in parsed.warnings)
    md = clean_pages(parsed.pages)
    check_reading_order(md, specs[spec_id])
    assert re.search(r"(?m)^#+ .*Experience", md), "headings should survive column rendering"


def test_scanned_pdf_is_read_with_builtin_ocr(fixture_files, specs, settings):
    """pymupdf4llm OCRs pages without a text layer (RapidOCR, light, no torch)."""
    pytest.importorskip("pymupdf4llm")
    spec = specs["scanned_devops"]
    out = router.parse_resume(fixture_files["scanned_devops"], "pdf", settings)
    assert out.features.scanned and not out.needs_review
    text = norm(out.markdown)
    assert spec["name"] in text
    found = [j["company"].split()[0] in text for j in spec["experience"]]
    assert all(found), "OCR should recover the company names"


def test_docx_parser(fixture_files, specs, settings):
    spec = specs["docx_frontend"]
    parsed = PythonDocxParser(settings).parse(fixture_files["docx_frontend"])
    md = parsed.markdown
    assert md.startswith(f"# {spec['name']}")
    assert "## Experience" in md and "## Skills" in md
    for job in spec["experience"]:
        assert job["company"] in md
        for bullet in job["bullets"]:
            assert f"- {bullet}" in md
    check_reading_order(md, spec)


# --- optional Docling parser (needs `pip install -r requirements-docling.txt`) ------------


@pytest.mark.slow
@pytest.mark.parametrize("spec_id", ["two_col_sidebar", "two_col_sidebar_right"])
def test_docling_keeps_sidebar_out_of_experience(fixture_files, specs, settings, spec_id):
    pytest.importorskip("docling")
    from src.providers.parsers.docling_parser import DoclingParser

    parsed = DoclingParser(settings).parse(fixture_files[spec_id])
    check_reading_order(clean_pages(parsed.pages), specs[spec_id])


@pytest.mark.slow
def test_docling_reads_scanned_pdf(fixture_files, specs, settings):
    pytest.importorskip("docling")
    from src.providers.parsers.docling_parser import DoclingParser

    parsed = DoclingParser(settings).parse(fixture_files["scanned_devops"])
    for job in specs["scanned_devops"]["experience"]:
        assert job["company"].split()[0] in parsed.markdown
