"""Building, updating and checking vector indexes from the chunks in the database."""

import importlib.util
from datetime import date

import pytest

from src.pipeline import chunk_stage, index_stage
from src.pipeline import manifest as mf
from src.stores import db, files
from tests.unit.fakes import FakeEmbedder
from tests.unit.test_chunk_stage import extracted

HAS_FAISS = importlib.util.find_spec("faiss") is not None
EMBEDDED_PER_RESUME = 4  # profile, two entries, merged sections


@pytest.fixture(
    params=[
        "numpy",
        pytest.param("faiss", marks=pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu missing")),
    ]
)
def conn(request, settings):
    settings.metadata.as_of = date(2025, 6, 1)
    settings.vector_store.provider = request.param
    c = db.connect(settings.paths.db)
    yield c
    c.close()


@pytest.fixture
def corpus(conn, settings, fixture_files):
    """Two candidates, extracted and chunked: 8 embeddable chunks."""
    docs = []
    for i, spec in enumerate(["single_col_backend", "single_col_overlap"]):
        doc_id = extracted(conn, settings, fixture_files, spec, email=f"p{i}@example.com")
        assert chunk_stage.chunk_document(conn, settings, doc_id) == "chunked"
        docs.append(doc_id)
    return docs


def total_embeddable(conn):
    return db.count_chunks(conn, embedded_only=True)


def test_first_build_embeds_every_embeddable_chunk(conn, settings, corpus):
    embedder = FakeEmbedder()
    result = index_stage.build_index(conn, settings, embedder)
    assert total_embeddable(conn) == 2 * EMBEDDED_PER_RESUME
    assert (result.embedded, result.removed, result.total) == (8, 0, 8)
    assert result.rebuilt_from_scratch and len(result.index_version) == 12

    report = index_stage.verify_index(conn, settings, embedder)
    assert report.ok and report.n_db == report.n_index == 8

    suffix = ".faiss" if settings.vector_store.provider == "faiss" else ".npz"
    directory = files.index_dir(settings, embedder.name)
    assert sorted(p.name for p in directory.iterdir()) == [
        "hashes.json", "manifest.json", f"vectors{suffix}"]
    (row,) = db.current_index_manifests(conn)
    assert (row["emb_model"], row["dim"], row["n_vectors"]) == ("fake-embed", 32, 8)


def test_stored_parents_are_not_embedded(conn, settings, corpus):
    index_stage.build_index(conn, settings, FakeEmbedder())
    store, _ = mf.load_index(settings, FakeEmbedder())
    parent_ids = {r["chunk_id"] for r in db.list_chunks(conn) if not r["embed"]}
    assert parent_ids and not parent_ids & set(store.ids())


def test_second_build_changes_nothing(conn, settings, corpus):
    embedder = FakeEmbedder()
    first = index_stage.build_index(conn, settings, embedder)
    calls = embedder.document_calls
    second = index_stage.build_index(conn, settings, embedder)
    assert (second.embedded, second.removed) == (0, 0) and not second.rebuilt_from_scratch
    assert second.index_version == first.index_version
    assert embedder.document_calls == calls  # nothing was sent to the model
    assert len(conn.execute("SELECT * FROM index_manifest").fetchall()) == 1


def test_a_changed_chunk_is_the_only_one_re_embedded(conn, settings, corpus):
    embedder = FakeEmbedder()
    first = index_stage.build_index(conn, settings, embedder)
    key = db.list_chunks(conn, embedded_only=True)[0]["chunk_key"]
    with db.transaction(conn):
        conn.execute(
            "UPDATE chunks SET embed_text = embed_text || ' updated', content_hash = 'new' "
            "WHERE chunk_key = ?", (key,),
        )
    assert not index_stage.verify_index(conn, settings, embedder).ok

    second = index_stage.build_index(conn, settings, embedder)
    assert (second.embedded, second.removed, second.total) == (1, 0, 8)
    assert second.index_version != first.index_version
    assert index_stage.verify_index(conn, settings, embedder).ok


def test_deleting_a_candidate_removes_its_vectors(conn, settings, corpus):
    embedder = FakeEmbedder()
    index_stage.build_index(conn, settings, embedder)
    cid = db.get_document(conn, corpus[0])["candidate_id"]
    with db.transaction(conn):
        conn.execute("DELETE FROM candidates WHERE candidate_id = ?", (cid,))

    report = index_stage.verify_index(conn, settings, embedder)
    assert not report.ok and len(report.stale) == EMBEDDED_PER_RESUME

    result = index_stage.build_index(conn, settings, embedder)
    assert (result.embedded, result.removed, result.total) == (0, EMBEDDED_PER_RESUME, 4)
    assert index_stage.verify_index(conn, settings, embedder).ok


def test_two_model_indexes_live_side_by_side(conn, settings, corpus):
    small, wide = FakeEmbedder("fake-small", dim=16), FakeEmbedder("fake-wide", dim=24)
    for embedder in (small, wide):
        index_stage.build_index(conn, settings, embedder)

    assert files.index_dir(settings, "fake-small") != files.index_dir(settings, "fake-wide")
    for embedder in (small, wide):
        store, manifest = mf.load_index(settings, embedder)
        assert store.count() == total_embeddable(conn) == manifest.n_vectors
        assert store.dim == embedder.dim == manifest.dim
        assert index_stage.verify_index(conn, settings, embedder).ok
    assert {r["emb_model"]: r["dim"] for r in db.current_index_manifests(conn)} == {
        "fake-small": 16, "fake-wide": 24}


def test_a_query_model_that_did_not_build_the_index_is_refused(conn, settings, corpus):
    index_stage.build_index(conn, settings, FakeEmbedder("fake-small", dim=16))
    with pytest.raises(mf.IndexMismatch):
        mf.load_index(settings, FakeEmbedder("fake-small", dim=24))
    with pytest.raises(mf.IndexMismatch, match="query prefix"):
        mf.load_index(settings, FakeEmbedder("fake-small", dim=16, query_prefix="other: "))
    with pytest.raises(FileNotFoundError):
        mf.load_index(settings, FakeEmbedder("fake-wide", dim=24))


def test_search_finds_the_relevant_entry(conn, settings, corpus):
    embedder = FakeEmbedder(dim=256)
    index_stage.build_index(conn, settings, embedder)
    store, _ = mf.load_index(settings, embedder)
    hits = store.search(embedder.embed_query("Kubernetes migration services EKS"), 2)
    by_id = {r["chunk_id"]: r["chunk_key"] for r in db.list_chunks(conn)}
    assert any(by_id[h.chunk_id].endswith("exp-1-p1") for h in hits)


def test_chunks_over_the_models_limit_stop_the_build(conn, settings, corpus):
    with pytest.raises(index_stage.IndexBuildError, match=r"exceed fake-embed's limit of 5"):
        index_stage.build_index(conn, settings, FakeEmbedder(max_tokens=5))
    assert mf.read_manifest(settings, "fake-embed") is None  # nothing half-written
    assert len(conn.execute("SELECT * FROM index_manifest").fetchall()) == 0


def test_progress_is_reported_per_batch(conn, settings, corpus):
    settings.embedding.batch_size = 3
    seen = []
    index_stage.build_index(
        conn, settings, FakeEmbedder(), progress=lambda done, total: seen.append((done, total))
    )
    assert seen == [(3, 8), (6, 8), (8, 8)]


def test_rebuild_flag_re_embeds_everything(conn, settings, corpus):
    embedder = FakeEmbedder()
    index_stage.build_index(conn, settings, embedder)
    result = index_stage.build_index(conn, settings, embedder, rebuild=True)
    assert (result.embedded, result.total, result.rebuilt_from_scratch) == (8, 8, True)


def test_a_prefix_change_forces_a_rebuild(conn, settings, corpus):
    index_stage.build_index(conn, settings, FakeEmbedder(doc_prefix="a: "))
    result = index_stage.build_index(conn, settings, FakeEmbedder(doc_prefix="b: "))
    assert result.rebuilt_from_scratch and result.embedded == 8


def test_verify_reports_a_missing_index(conn, settings, corpus):
    report = index_stage.verify_index(conn, settings, FakeEmbedder())
    assert not report.ok and report.n_index == 0 and len(report.missing) == 8


def test_empty_database(conn, settings):
    embedder = FakeEmbedder()
    result = index_stage.build_index(conn, settings, embedder)
    assert result.total == 0
    assert index_stage.verify_index(conn, settings, embedder).ok
    store, _ = mf.load_index(settings, embedder)
    assert store.search(embedder.embed_query("anything"), 5) == []
