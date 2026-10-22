from pathlib import Path

import pytest

from src.ingestion import intake
from src.pipeline import parse_stage
from src.providers.base import ParsedDocument
from src.stores import db

GOOD = " ".join(["engineer"] * 60) + " someone@example.com"


class FakeParser:
    name = "pymupdf4llm"

    def __init__(self, text=GOOD, error=None):
        self.text, self.error, self.calls = text, error, 0

    def parse(self, path: Path) -> ParsedDocument:
        self.calls += 1
        if self.error:
            raise self.error
        return ParsedDocument(
            markdown=self.text, n_pages=1, parser=self.name, parser_version="1.0", pages=[self.text]
        )


@pytest.fixture
def conn(settings):
    c = db.connect(settings.paths.db)
    yield c
    c.close()


@pytest.fixture
def doc_id(conn, settings, fixture_files):
    path = fixture_files["single_col_backend"]
    return intake.ingest(conn, settings, path.name, path.read_bytes()).doc_id


def run(conn, settings, doc_id, parser):
    return parse_stage.parse_document(
        conn, settings, doc_id, parser_factory=lambda name, s: parser
    )


def test_success_saves_markdown_and_status(conn, settings, doc_id):
    assert run(conn, settings, doc_id, FakeParser()) == "parsed"
    row = db.get_document(conn, doc_id)
    assert (row["parser"], row["parser_version"], row["error"]) == ("pymupdf4llm", "1.0", None)
    assert Path(row["markdown_path"]).read_text(encoding="utf-8").startswith("engineer")


def test_unusable_output_is_needs_review_but_markdown_kept(conn, settings, doc_id):
    assert run(conn, settings, doc_id, FakeParser(text="tiny")) == "needs_review"
    row = db.get_document(conn, doc_id)
    assert row["error"] and Path(row["markdown_path"]).exists()


def test_crash_marks_failed_with_message(conn, settings, doc_id):
    status = run(conn, settings, doc_id, FakeParser(error=RuntimeError("boom")))
    assert status == "failed"
    assert "boom" in db.get_document(conn, doc_id)["error"]


def test_already_parsed_is_skipped(conn, settings, doc_id):
    parser = FakeParser()
    run(conn, settings, doc_id, parser)
    assert run(conn, settings, doc_id, parser) == "parsed"
    assert parser.calls == 1


def test_failed_document_can_be_retried(conn, settings, doc_id):
    run(conn, settings, doc_id, FakeParser(error=RuntimeError("boom")))
    db.set_document_status(conn, doc_id, "uploaded")
    assert run(conn, settings, doc_id, FakeParser()) == "parsed"


def test_unknown_document(conn, settings):
    with pytest.raises(KeyError):
        run(conn, settings, "nope", FakeParser())
