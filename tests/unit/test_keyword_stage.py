"""The keyword index built from the database, and kept in step with the vector index."""

import json
from datetime import date

import pytest

from src.pipeline import chunk_stage, index_stage, keyword_stage
from src.pipeline import manifest as mf
from src.stores import db, files
from tests.unit.fakes import FakeEmbedder
from tests.unit.test_chunk_stage import extracted

pytest.importorskip("bm25s")


@pytest.fixture
def conn(settings):
    settings.metadata.as_of = date(2025, 6, 1)
    settings.vector_store.provider = "numpy"
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


def keys(conn, hits):
    by_id = {r["chunk_id"]: r["chunk_key"] for r in db.list_chunks(conn)}
    return [by_id[h.chunk_id] for h in hits]


def test_build_indexes_every_embeddable_chunk(conn, settings, corpus):
    result = keyword_stage.build_keyword_index(conn, settings)
    assert (result.n_docs, result.rebuilt) == (8, True)

    report = keyword_stage.verify_keyword_index(conn, settings)
    assert report.ok and report.n_db == report.n_index == 8

    directory = files.keyword_dir(settings)
    assert (directory / "manifest.json").exists() and (directory / "index" / "ids.json").exists()
    manifest = keyword_stage.read_keyword_manifest(settings)
    assert manifest.tokenizer == "tech_aware" and manifest.n_docs == 8


def test_second_build_changes_nothing(conn, settings, corpus):
    keyword_stage.build_keyword_index(conn, settings)
    stamp = keyword_stage.read_keyword_manifest(settings).created_at
    again = keyword_stage.build_keyword_index(conn, settings)
    assert not again.rebuilt
    assert keyword_stage.read_keyword_manifest(settings).created_at == stamp


def test_same_chunk_ids_as_the_vector_index(conn, settings, corpus):
    embedder = FakeEmbedder()
    index_stage.build_index(conn, settings, embedder)
    keyword_stage.build_keyword_index(conn, settings)
    vectors, _ = mf.load_index(settings, embedder)
    words, _ = keyword_stage.load_keyword_index(settings)
    assert set(words.ids()) == set(vectors.ids())  # the store-count-mismatch gate is 0
    assert words.count() == vectors.count() == db.count_chunks(conn, embedded_only=True)


def test_search_finds_the_relevant_entry(conn, settings, corpus):
    keyword_stage.build_keyword_index(conn, settings)
    index, _ = keyword_stage.load_keyword_index(settings)
    hits = index.search("kubernetes migration", 3)
    assert keys(conn, hits)[0].endswith("exp-1-p1")


def test_tech_terms_are_searchable_and_aliases_work(conn, settings, corpus):
    keyword_stage.build_keyword_index(conn, settings)
    index, _ = keyword_stage.load_keyword_index(settings)
    assert index.search("k8s", 5) and index.search("golang", 5)
    # "GitHub Actions" is only in the second job (and in the profile chunk's top skills)
    found = keys(conn, index.search("GitHub Actions", 5))
    assert found and all("exp-2" in k or "profile" in k for k in found)


def test_personal_data_is_not_searchable(conn, settings, corpus):
    keyword_stage.build_keyword_index(conn, settings)
    index, _ = keyword_stage.load_keyword_index(settings)
    for term in ("priya", "raman", "p0@example.com", "example.com", "555 0101"):
        assert index.search(term, 5) == [], term


def test_a_changed_chunk_triggers_a_rebuild(conn, settings, corpus):
    keyword_stage.build_keyword_index(conn, settings)
    key = db.list_chunks(conn, embedded_only=True)[0]["chunk_key"]
    with db.transaction(conn):
        conn.execute(
            "UPDATE chunks SET embed_text = embed_text || ' zebraquartz', content_hash = 'new' "
            "WHERE chunk_key = ?", (key,),
        )
    assert not keyword_stage.verify_keyword_index(conn, settings).ok
    assert keyword_stage.build_keyword_index(conn, settings).rebuilt

    index, _ = keyword_stage.load_keyword_index(settings)
    assert keys(conn, index.search("zebraquartz", 5)) == [key]
    assert keyword_stage.verify_keyword_index(conn, settings).ok


def test_deleting_a_candidate_removes_its_documents(conn, settings, corpus):
    keyword_stage.build_keyword_index(conn, settings)
    cid = db.get_document(conn, corpus[0])["candidate_id"]
    with db.transaction(conn):
        conn.execute("DELETE FROM candidates WHERE candidate_id = ?", (cid,))
    report = keyword_stage.verify_keyword_index(conn, settings)
    assert not report.ok and len(report.stale) == 4

    assert keyword_stage.build_keyword_index(conn, settings).n_docs == 4
    assert keyword_stage.verify_keyword_index(conn, settings).ok


def test_changing_the_tokenizer_makes_the_old_index_unusable(conn, settings, corpus):
    keyword_stage.build_keyword_index(conn, settings)
    settings.keyword.tokenizer = "simple"
    with pytest.raises(mf.IndexMismatch, match="rebuild the keyword index"):
        keyword_stage.load_keyword_index(settings)
    assert not keyword_stage.verify_keyword_index(conn, settings).ok

    assert keyword_stage.build_keyword_index(conn, settings).rebuilt
    index, manifest = keyword_stage.load_keyword_index(settings)
    assert manifest.tokenizer == "simple" and index.count() == 8


def test_a_changed_alias_file_is_detected(conn, settings, corpus):
    keyword_stage.build_keyword_index(conn, settings)
    path = files.keyword_dir(settings) / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tokenizer_fingerprint"] = "tech_aware:1:0123456789ab"  # as if skills_aliases.yaml changed
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(mf.IndexMismatch):
        keyword_stage.load_keyword_index(settings)
    assert keyword_stage.build_keyword_index(conn, settings).rebuilt  # a rebuild repairs it
    keyword_stage.load_keyword_index(settings)


def test_changing_bm25_parameters_rebuilds(conn, settings, corpus):
    keyword_stage.build_keyword_index(conn, settings)
    settings.keyword.k1 = 1.2
    assert keyword_stage.build_keyword_index(conn, settings).rebuilt
    assert not keyword_stage.build_keyword_index(conn, settings).rebuilt


def test_force_rebuild(conn, settings, corpus):
    keyword_stage.build_keyword_index(conn, settings)
    assert keyword_stage.build_keyword_index(conn, settings, rebuild=True).rebuilt


def test_no_index_yet(conn, settings, corpus):
    with pytest.raises(FileNotFoundError):
        keyword_stage.load_keyword_index(settings)
    report = keyword_stage.verify_keyword_index(conn, settings)
    assert not report.ok and len(report.missing) == 8 and report.n_index == 0


def test_empty_database(conn, settings):
    assert keyword_stage.build_keyword_index(conn, settings).n_docs == 0
    assert keyword_stage.verify_keyword_index(conn, settings).ok
    index, _ = keyword_stage.load_keyword_index(settings)
    assert index.search("kubernetes", 5) == []
