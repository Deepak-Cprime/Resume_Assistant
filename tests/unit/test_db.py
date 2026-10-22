import sqlite3

import pytest

from src.stores import db


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()


def _doc(conn, doc_id="d1"):
    db.insert_document(conn, doc_id=doc_id, filename="a.pdf", doc_type="pdf", raw_path="x")


def test_schema_and_single_role(conn):
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"roles", "candidates", "documents", "experiences", "skills", "certifications",
            "chunks", "index_manifest"} <= tables
    assert conn.execute("SELECT COUNT(*) FROM roles").fetchone()[0] == 1


def test_connect_is_idempotent(tmp_path):
    p = tmp_path / "sub" / "app.db"
    db.connect(p).close()
    c = db.connect(p)
    assert c.execute("SELECT COUNT(*) FROM roles").fetchone()[0] == 1
    c.close()


def test_insert_document_once(conn):
    assert db.insert_document(conn, doc_id="d1", filename="a", doc_type="pdf", raw_path="x")
    assert not db.insert_document(conn, doc_id="d1", filename="b", doc_type="pdf", raw_path="y")
    assert db.get_document(conn, "d1")["filename"] == "a"


def test_forward_transitions(conn):
    _doc(conn)
    for s in db.PIPELINE_STATUSES[1:]:
        db.set_document_status(conn, "d1", s)
    assert db.get_document(conn, "d1")["status"] == "indexed"


def test_cannot_skip_stage(conn):
    _doc(conn)
    with pytest.raises(db.InvalidTransition):
        db.set_document_status(conn, "d1", "chunked")
    assert db.get_document(conn, "d1")["status"] == "uploaded"


def test_failed_then_retry(conn):
    _doc(conn)
    db.set_document_status(conn, "d1", "failed", error="boom")
    assert db.get_document(conn, "d1")["error"] == "boom"
    with pytest.raises(db.InvalidTransition):
        db.set_document_status(conn, "d1", "parsed")
    db.set_document_status(conn, "d1", "uploaded")
    assert db.get_document(conn, "d1")["error"] is None


def test_backward_rerun_allowed(conn):
    _doc(conn)
    for s in ("parsed", "structured", "extracted", "chunked"):
        db.set_document_status(conn, "d1", s)
    db.set_document_status(conn, "d1", "structured")


def test_rejects_unknown_status_and_fields(conn):
    _doc(conn)
    with pytest.raises(db.InvalidTransition):
        db.set_document_status(conn, "d1", "bogus")
    with pytest.raises(ValueError):
        db.set_document_status(conn, "d1", "parsed", filename="evil")


def test_extra_fields_saved(conn):
    _doc(conn)
    db.set_document_status(conn, "d1", "parsed", parser="pymupdf4llm", markdown_path="m.md")
    row = db.get_document(conn, "d1")
    assert (row["parser"], row["markdown_path"]) == ("pymupdf4llm", "m.md")


def test_status_counts(conn):
    _doc(conn, "a")
    _doc(conn, "b")
    db.set_document_status(conn, "b", "failed", error="x")
    assert db.status_counts(conn) == {"uploaded": 1, "failed": 1}


def test_update_jd_touches_nothing_else(conn):
    _doc(conn)
    db.update_role(conn, title="Backend Engineer", jd_text="Python, K8s")
    role = db.get_role(conn)
    assert (role["title"], role["jd_text"]) == ("Backend Engineer", "Python, K8s")
    assert db.get_document(conn, "d1")["status"] == "uploaded"


def test_foreign_keys_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO experiences (candidate_id, title) VALUES (999, 'x')")
