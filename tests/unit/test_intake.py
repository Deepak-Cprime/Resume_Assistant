import io

import pymupdf
import pytest
from docx import Document

from src.ingestion import intake
from src.stores import db


def make_pdf(pages: int = 1, password: str | None = None) -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"Jane Doe resume page {i}")
    if password:
        return doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password + "o"
        )
    return doc.tobytes()


def make_docx() -> bytes:
    d = Document()
    d.add_paragraph("Jane Doe")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


@pytest.fixture
def conn(settings):
    c = db.connect(settings.paths.db)
    yield c
    c.close()


def test_valid_pdf_and_docx(conn, settings):
    r1 = intake.ingest(conn, settings, "a.pdf", make_pdf())
    r2 = intake.ingest(conn, settings, "b.docx", make_docx())
    assert (r1.status, r1.is_new) == ("uploaded", True)
    assert (r2.status, r2.is_new) == ("uploaded", True)
    assert db.get_document(conn, r1.doc_id)["doc_type"] == "pdf"
    assert db.get_document(conn, r2.doc_id)["doc_type"] == "docx"
    assert (settings.raw_dir / f"{r1.doc_id}.pdf").exists()


def test_same_bytes_same_doc_id():
    data = make_pdf()
    assert intake.sha256_hex(data) == intake.sha256_hex(bytes(data))


def test_reupload_creates_nothing_new(conn, settings):
    data = make_pdf()
    first = intake.ingest(conn, settings, "a.pdf", data)
    files_before = sorted(p.name for p in settings.raw_dir.iterdir())
    second = intake.ingest(conn, settings, "renamed.pdf", data)
    assert second.doc_id == first.doc_id
    assert second.is_new is False
    assert len(db.list_documents(conn)) == 1
    assert sorted(p.name for p in settings.raw_dir.iterdir()) == files_before


@pytest.mark.parametrize(
    "name,data,msg",
    [
        ("empty.pdf", b"", "empty"),
        ("fake.pdf", b"MZ\x90\x00 this is an exe", "Unsupported"),
        ("corrupt.pdf", b"%PDF-1.4 not really a pdf", "Corrupt"),
        ("zip.docx", b"PK\x03\x04garbage", "not a DOCX"),
    ],
)
def test_rejects_bad_files(conn, settings, name, data, msg):
    r = intake.ingest(conn, settings, name, data)
    assert r.status == "failed"
    assert msg.lower() in r.error.lower()
    assert not settings.raw_dir.exists() or not list(settings.raw_dir.iterdir())


def test_rejects_encrypted_pdf(conn, settings):
    r = intake.ingest(conn, settings, "locked.pdf", make_pdf(password="secret"))
    assert r.status == "failed" and "encrypted" in r.error


def test_rejects_too_many_pages(conn, settings):
    r = intake.ingest(conn, settings, "long.pdf", make_pdf(pages=settings.ingestion.max_pages + 1))
    assert r.status == "failed" and "pages" in r.error


def test_rejects_oversize(conn, settings):
    settings.ingestion.max_file_mb = 1
    r = intake.ingest(conn, settings, "big.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 1))
    assert r.status == "failed" and "larger" in r.error


def test_rejected_reupload_returns_same_record(conn, settings):
    first = intake.ingest(conn, settings, "x.pdf", b"MZ junk")
    second = intake.ingest(conn, settings, "x.pdf", b"MZ junk")
    assert first.doc_id == second.doc_id and second.is_new is False
    assert len(db.list_documents(conn)) == 1


def test_extension_is_ignored(conn, settings):
    r = intake.ingest(conn, settings, "resume.docx", make_pdf())
    assert db.get_document(conn, r.doc_id)["doc_type"] == "pdf"
