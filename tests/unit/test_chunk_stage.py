"""parse -> structure -> extract -> chunk for whole documents, with a fake parser."""

import json
from datetime import date
from pathlib import Path

import pytest

from src.chunking.budgets import Budgets
from src.ingestion import intake
from src.pipeline import chunk_stage, extract_stage, parse_stage, structure_stage
from src.providers.base import ParsedDocument
from src.stores import db
from tests.unit.fakes import FULL_MD

FIXTURES = ["single_col_backend", "single_col_overlap", "single_col_gap", "single_col_spanish"]


class FakeParser:
    name = "pymupdf4llm"

    def __init__(self, text):
        self.text = text

    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(markdown=self.text, n_pages=1, parser=self.name, pages=[self.text])


@pytest.fixture
def conn(settings):
    settings.metadata.as_of = date(2025, 6, 1)
    c = db.connect(settings.paths.db)
    yield c
    c.close()


def extracted(conn, settings, fixture_files, spec_id, email="priya.raman@example.com"):
    """Run one fixture file through parse -> structure -> extract with FULL_MD as its text."""
    path = fixture_files[spec_id]
    doc_id = intake.ingest(conn, settings, path.name, path.read_bytes()).doc_id
    parser = FakeParser(FULL_MD.replace("priya.raman@example.com", email))
    parse_stage.parse_document(conn, settings, doc_id, parser_factory=lambda n, s: parser)
    structure_stage.structure_document(conn, settings, doc_id)
    assert extract_stage.extract_document(conn, settings, doc_id) == "extracted"
    return doc_id


def test_chunk_document_writes_chunks_and_budgets(conn, settings, fixture_files):
    doc_id = extracted(conn, settings, fixture_files, "single_col_backend")
    assert chunk_stage.chunk_document(conn, settings, doc_id) == "chunked"

    row = db.get_document(conn, doc_id)
    chunks = db.list_chunks(conn, candidate_id=row["candidate_id"])
    cid = row["candidate_id"]
    assert sorted(c["chunk_key"] for c in chunks) == sorted([
        f"c{cid}-profile", f"c{cid}-experience", f"c{cid}-exp-1-p1", f"c{cid}-exp-2-p1",
        f"c{cid}-education-skills-certifications",
    ])
    assert db.count_chunks(conn) == 5 and db.count_chunks(conn, embedded_only=True) == 4
    assert all(c["doc_id"] == doc_id and c["role_id"] == 1 for c in chunks)
    entry = next(c for c in chunks if c["chunk_key"].endswith("exp-1-p1"))
    assert entry["level"] == 2 and entry["parent_id"] is not None
    assert json.loads(entry["skills"]) == ["Kubernetes", "Go"]
    assert entry["start"] == "2021-03" and entry["end"] == "2024-12"
    assert entry["n_tokens"] > 0 and len(entry["content_hash"]) == 64

    stored = Budgets.from_dict(db.get_chunk_budgets(conn))
    assert stored.n_candidates == 1 and "default" in stored.types


def test_rerun_does_not_duplicate(conn, settings, fixture_files):
    doc_id = extracted(conn, settings, fixture_files, "single_col_backend")
    chunk_stage.chunk_document(conn, settings, doc_id)
    before = [(c["chunk_id"], c["content_hash"]) for c in db.list_chunks(conn)]
    assert chunk_stage.chunk_document(conn, settings, doc_id) == "chunked"  # skipped
    db.set_document_status(conn, doc_id, "extracted")  # force a real re-chunk
    assert chunk_stage.chunk_document(conn, settings, doc_id) == "chunked"
    assert [(c["chunk_id"], c["content_hash"]) for c in db.list_chunks(conn)] == before


def test_refreshed_candidate_replaces_chunks(conn, settings, fixture_files):
    first = extracted(conn, settings, fixture_files, "single_col_backend")
    chunk_stage.chunk_document(conn, settings, first)
    second = extracted(conn, settings, fixture_files, "single_col_overlap")  # same email
    chunk_stage.chunk_document(conn, settings, second)
    chunks = db.list_chunks(conn)
    assert len(chunks) == 5 and {c["doc_id"] for c in chunks} == {second}


def test_budgets_stay_stable_as_the_corpus_grows_a_little(conn, settings, fixture_files):
    docs = [
        extracted(conn, settings, fixture_files, spec, email=f"p{i}@example.com")
        for i, spec in enumerate(FIXTURES[:3])
    ]
    first, recomputed = chunk_stage.ensure_budgets(conn, settings)
    assert recomputed  # nothing stored yet
    assert not chunk_stage.ensure_budgets(conn, settings)[1]  # unchanged corpus: stable

    extracted(conn, settings, fixture_files, FIXTURES[3], email="p9@example.com")
    again, recomputed = chunk_stage.ensure_budgets(conn, settings)
    assert not recomputed and again == first  # 3 -> 4 candidates is under +50%
    assert chunk_stage.chunk_document(conn, settings, docs[0]) == "chunked"


def test_budgets_recompute_when_the_corpus_has_grown_by_half(conn, settings, fixture_files):
    extracted(conn, settings, fixture_files, FIXTURES[0], email="p0@example.com")
    stored, _ = chunk_stage.ensure_budgets(conn, settings)
    assert stored.n_candidates == 1
    extracted(conn, settings, fixture_files, FIXTURES[1], email="p1@example.com")
    grown, recomputed = chunk_stage.ensure_budgets(conn, settings)
    assert recomputed and grown.n_candidates == 2  # +100%


def test_skips_documents_in_the_wrong_status(conn, settings, fixture_files):
    path = fixture_files["single_col_backend"]
    doc_id = intake.ingest(conn, settings, path.name, path.read_bytes()).doc_id
    assert chunk_stage.chunk_document(conn, settings, doc_id) == "uploaded"
    assert db.count_chunks(conn) == 0


def test_missing_profile_marks_the_document_failed(conn, settings, fixture_files):
    doc_id = extracted(conn, settings, fixture_files, "single_col_backend")
    (settings.profiles_dir / f"{doc_id}.json").unlink()
    assert chunk_stage.chunk_document(conn, settings, doc_id) == "failed"
    assert "missing" in db.get_document(conn, doc_id)["error"]


def test_unknown_document(conn, settings):
    with pytest.raises(KeyError):
        chunk_stage.chunk_document(conn, settings, "nope")


def test_deleting_a_candidate_removes_its_chunks(conn, settings, fixture_files):
    doc_id = extracted(conn, settings, fixture_files, "single_col_backend")
    chunk_stage.chunk_document(conn, settings, doc_id)
    cid = db.get_document(conn, doc_id)["candidate_id"]
    with db.transaction(conn):
        conn.execute("DELETE FROM candidates WHERE candidate_id = ?", (cid,))
    assert db.count_chunks(conn) == 0  # ON DELETE CASCADE
