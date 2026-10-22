"""delete_candidate / delete_document: nothing of the person may remain anywhere."""

from pathlib import Path

import pytest

from src.ingestion import intake
from src.pipeline import delete, index_stage, keyword_stage
from src.pipeline import manifest as mf
from src.pipeline import run as pipeline
from src.stores import db, files
from tests.unit.fakes import FakeEmbedder
from tests.unit.pipeline_helpers import MarkdownParser, configure, resume, upload

SPECS = ["single_col_backend", "single_col_overlap", "single_col_gap"]
TABLES_BY_CANDIDATE = ("experiences", "skills", "certifications", "chunks", "documents")


@pytest.fixture
def conn(settings):
    configure(settings, "fake-a", "fake-b")
    c = db.connect(settings.paths.db)
    yield c
    c.close()


@pytest.fixture
def models():
    return [FakeEmbedder("fake-a", dim=16), FakeEmbedder("fake-b", dim=24)]


@pytest.fixture
def parser():
    return MarkdownParser()


@pytest.fixture
def world(conn, settings, fixture_files, parser, models):
    """Three indexed candidates (2 vector indexes + keyword index); returns their doc ids."""
    docs = [
        upload(conn, settings, fixture_files, spec, parser, resume(i))
        for i, spec in enumerate(SPECS, 1)
    ]
    result = pipeline.run_pipeline(
        conn, settings, embedders=models, parser_factory=parser.factory
    )
    assert not result.errors and db.status_counts(conn) == {"indexed": 3}
    return docs


def rows_for(conn, candidate_id):
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t} WHERE candidate_id = ?",
                              (candidate_id,)).fetchone()[0] for t in TABLES_BY_CANDIDATE}
    counts["candidates"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()[0]
    return counts


def keyword_hits(settings, term):
    index, _ = keyword_stage.load_keyword_index(settings)
    return index.search(term, 10)


def test_delete_candidate_removes_every_trace(conn, settings, world, models):
    cid = db.get_document(conn, world[0])["candidate_id"]
    chunk_ids = set(db.candidate_chunk_ids(conn, cid))
    doc_row = db.get_document(conn, world[0])
    paths = [Path(doc_row["raw_path"]), Path(doc_row["markdown_path"]),
             files.profile_path(settings, world[0]), files.sections_path(settings, world[0])]
    assert all(p.exists() for p in paths)
    assert len(chunk_ids) == 5  # 4 embedded + the stored full-experience parent
    assert keyword_hits(settings, "marker1corp")  # findable before

    result = delete.delete_candidate(conn, settings, cid)

    assert (result.documents, result.chunks, result.files) == (1, 5, 4) and not result.errors
    assert result.vectors == {"fake-a": 4, "fake-b": 4} and result.keyword_rebuilt
    assert set(rows_for(conn, cid).values()) == {0}  # zero rows in every table
    assert not any(p.exists() for p in paths)

    for embedder in models:  # gone from every vector index
        store, manifest = mf.load_index(settings, embedder)
        assert not chunk_ids & set(store.ids()) and store.count() == 8 == manifest.n_vectors
        assert index_stage.verify_index(conn, settings, embedder).ok
    assert keyword_hits(settings, "marker1corp") == []  # and from keyword search
    assert keyword_stage.verify_keyword_index(conn, settings).ok


def test_other_candidates_are_untouched(conn, settings, world, models):
    other = db.get_document(conn, world[1])["candidate_id"]
    before = (rows_for(conn, other), db.candidate_chunk_ids(conn, other))
    delete.delete_candidate(conn, settings, db.get_document(conn, world[0])["candidate_id"])
    assert (rows_for(conn, other), db.candidate_chunk_ids(conn, other)) == before
    assert keyword_hits(settings, "marker2corp")
    assert db.status_counts(conn) == {"indexed": 2}


def test_the_pipeline_finds_nothing_to_fix_afterwards(conn, settings, world, models, parser):
    delete.delete_candidate(conn, settings, db.get_document(conn, world[0])["candidate_id"])
    result = pipeline.run_pipeline(
        conn, settings, embedders=models, parser_factory=parser.factory
    )
    assert result.documents == [] and result.rechunked == []
    assert all(r.embedded == 0 and r.removed == 0 for r in result.vectors.values())
    assert not result.keyword.rebuilt


def test_the_same_file_can_be_uploaded_again_after_deleting(
    conn, settings, fixture_files, world, models, parser
):
    delete.delete_candidate(conn, settings, db.get_document(conn, world[0])["candidate_id"])
    path = fixture_files[SPECS[0]]
    again = intake.ingest(conn, settings, path.name, path.read_bytes())
    assert again.is_new and again.doc_id == world[0]
    parser.texts[again.doc_id] = resume(1)
    pipeline.run_pipeline(conn, settings, embedders=models, parser_factory=parser.factory)
    assert db.status_counts(conn) == {"indexed": 3} and db.count_chunks(conn) == 15
    assert keyword_hits(settings, "marker1corp")


def test_every_upload_of_a_candidate_goes_with_them(
    conn, settings, fixture_files, world, models, parser
):
    newer = upload(conn, settings, fixture_files, "single_col_spanish", parser,
                   resume(1, bullet="Led migration of 41 services"))
    pipeline.run_pipeline(conn, settings, embedders=models, parser_factory=parser.factory)
    cid = db.get_document(conn, world[0])["candidate_id"]
    raws = [Path(db.get_document(conn, d)["raw_path"]) for d in (world[0], newer)]

    result = delete.delete_candidate(conn, settings, cid)

    assert result.documents == 2 and not any(p.exists() for p in raws)
    assert db.get_document(conn, world[0]) is None and db.get_document(conn, newer) is None


def test_delete_document_of_a_failed_upload(conn, settings, fixture_files, parser, models):
    doc = upload(conn, settings, fixture_files, SPECS[0], parser, resume(1))
    parser.fail.add(doc)
    pipeline.run_pipeline(conn, settings, embedders=models, parser_factory=parser.factory)
    assert db.get_document(conn, doc)["status"] == "failed"
    raw = Path(db.get_document(conn, doc)["raw_path"])

    result = delete.delete_document(conn, settings, doc)

    assert (result.candidate_id, result.documents, result.chunks) == (None, 1, 0)
    assert db.get_document(conn, doc) is None and not raw.exists()


def test_delete_document_of_a_processed_upload_removes_the_candidate(conn, settings, world):
    cid = db.get_document(conn, world[0])["candidate_id"]
    result = delete.delete_document(conn, settings, world[0])
    assert result.candidate_id == cid and db.get_candidate(conn, cid) is None


def test_deleting_before_anything_was_indexed(conn, settings, fixture_files, parser):
    doc = upload(conn, settings, fixture_files, SPECS[0], parser, resume(1))
    pipeline.run_pipeline(conn, settings, build_indexes=False, parser_factory=parser.factory)
    cid = db.get_document(conn, doc)["candidate_id"]

    result = delete.delete_candidate(conn, settings, cid)

    assert result.vectors == {} and not result.keyword_rebuilt and not result.errors
    assert result.chunks == 5 and db.count_chunks(conn) == 0


def test_missing_files_are_not_an_error(conn, settings, world):
    row = db.get_document(conn, world[0])
    Path(row["raw_path"]).unlink()
    files.profile_path(settings, world[0]).unlink()
    result = delete.delete_candidate(conn, settings, row["candidate_id"])
    assert not result.errors and result.files == 2


def test_a_failed_index_cleanup_is_reported_and_repaired_by_the_next_run(
    conn, settings, world, models, parser, monkeypatch
):
    def broken(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(index_stage, "remove_from_indexes", broken)
    cid = db.get_document(conn, world[0])["candidate_id"]
    result = delete.delete_candidate(conn, settings, cid)
    assert any("disk unavailable" in e for e in result.errors)
    assert db.count_chunks(conn, embedded_only=True) == 8  # the database is already clean

    monkeypatch.undo()
    fixed = pipeline.run_pipeline(conn, settings, embedders=models, parser_factory=parser.factory)
    assert all(r.removed == 4 for r in fixed.vectors.values())  # stale vectors dropped
    for embedder in models:
        assert index_stage.verify_index(conn, settings, embedder).ok


def test_unknown_targets(conn, settings, world):
    with pytest.raises(KeyError):
        delete.delete_candidate(conn, settings, 999)
    with pytest.raises(KeyError):
        delete.delete_document(conn, settings, "nope")
    cid = db.get_document(conn, world[0])["candidate_id"]
    delete.delete_candidate(conn, settings, cid)
    with pytest.raises(KeyError):
        delete.delete_candidate(conn, settings, cid)  # already gone
