import pymupdf
import pytest

from src.ingestion import intake
from src.pipeline import uploads
from src.stores import db


def pdf(text: str = "Jane Doe") -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


@pytest.fixture
def conn(settings):
    c = db.connect(settings.paths.db)
    yield c
    c.close()


def test_each_file_gets_its_own_outcome(conn, settings):
    good = pdf("one")
    results = uploads.ingest_files(conn, settings, [
        ("a.pdf", good), ("b.pdf", good), ("virus.pdf", b"MZ junk"), ("c.pdf", pdf("two")),
    ])
    assert [uploads.outcome(r) for r in results] == [
        "added",
        "already uploaded (same content)",
        "rejected: Unsupported or corrupt file (expected PDF or DOCX)",
        "added",
    ]
    assert len(db.list_documents(conn)) == 3  # a, virus (recorded as failed), c


def test_an_unexpected_error_affects_only_that_file(conn, settings, monkeypatch):
    real = intake.ingest

    def flaky(conn, settings, filename, data):
        if filename == "broken.pdf":
            raise OSError("disk full")
        return real(conn, settings, filename, data)

    monkeypatch.setattr(intake, "ingest", flaky)
    results = uploads.ingest_files(conn, settings, [
        ("broken.pdf", pdf("x")), ("fine.pdf", pdf("y")),
    ])
    assert results[0].status == "failed" and "disk full" in uploads.outcome(results[0])
    assert uploads.outcome(results[1]) == "added"


def test_no_files(conn, settings):
    assert uploads.ingest_files(conn, settings, []) == []


def test_accepts_any_iterable_of_pairs(conn, settings):
    pairs = ((name, pdf(name)) for name in ("x", "y"))
    assert len(uploads.ingest_files(conn, settings, pairs)) == 2
