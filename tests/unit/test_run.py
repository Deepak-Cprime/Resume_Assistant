"""run_pipeline over whole batches: resumable, idempotent, failure-tolerant, budget-aware."""

import pytest

from src.config import EmbeddingModelConfig
from src.pipeline import extract_stage, index_stage, keyword_stage, parse_stage, structure_stage
from src.pipeline import manifest as mf
from src.pipeline import run as pipeline
from src.providers import factory
from src.providers.embedders.common import EmbeddingError
from src.stores import db
from tests.unit.fakes import FakeEmbedder
from tests.unit.pipeline_helpers import MarkdownParser, configure, resume, upload

SPECS = ["single_col_backend", "single_col_overlap", "single_col_gap"]
EMBEDDABLE_PER_RESUME = 4


@pytest.fixture
def conn(settings):
    configure(settings)
    c = db.connect(settings.paths.db)
    yield c
    c.close()


@pytest.fixture
def parser():
    return MarkdownParser()


@pytest.fixture
def embedder():
    return FakeEmbedder("fake-embed")


@pytest.fixture
def docs(conn, settings, fixture_files, parser):
    """Three uploaded resumes (candidates 1-3), nothing processed yet."""
    return [
        upload(conn, settings, fixture_files, spec, parser, resume(i))
        for i, spec in enumerate(SPECS, 1)
    ]


def run(conn, settings, parser, embedder, **kwargs):
    return pipeline.run_pipeline(
        conn, settings, embedders=[embedder], parser_factory=parser.factory, **kwargs
    )


def snapshot(conn):
    return [(c["chunk_id"], c["content_hash"], c["doc_id"]) for c in db.list_chunks(conn)]


def test_a_full_run_indexes_every_resume(conn, settings, parser, embedder, docs):
    result = run(conn, settings, parser, embedder)

    assert [d.stages for d in result.documents] == [["parse", "structure", "extract", "chunk"]] * 3
    assert db.status_counts(conn) == {"indexed": 3}
    assert sorted(result.indexed) == sorted(docs) and not result.errors

    embeddable = db.count_chunks(conn, embedded_only=True)
    assert embeddable == 3 * EMBEDDABLE_PER_RESUME
    # the store-count-mismatch gate: database, vector index and keyword index all agree
    store, _ = mf.load_index(settings, embedder)
    words, _ = keyword_stage.load_keyword_index(settings)
    assert store.count() == words.count() == embeddable
    assert set(store.ids()) == set(words.ids())
    assert index_stage.verify_index(conn, settings, embedder).ok
    assert keyword_stage.verify_keyword_index(conn, settings).ok
    assert "3 document(s) processed" in result.summary()


def test_rerunning_changes_nothing(conn, settings, parser, embedder, docs):
    run(conn, settings, parser, embedder)
    before = (snapshot(conn), len(parser.calls), embedder.document_calls,
              len(conn.execute("SELECT * FROM index_manifest").fetchall()))

    again = run(conn, settings, parser, embedder)

    assert again.documents == [] and again.rechunked == []
    assert again.vectors["fake-embed"].embedded == 0 and not again.keyword.rebuilt
    assert (snapshot(conn), len(parser.calls), embedder.document_calls,
            len(conn.execute("SELECT * FROM index_manifest").fetchall())) == before
    assert db.status_counts(conn) == {"indexed": 3}


def test_a_failing_resume_does_not_stop_the_batch(conn, settings, parser, embedder, docs):
    parser.fail.add(docs[1])
    result = run(conn, settings, parser, embedder)

    assert [d.doc_id for d in result.flagged] == [docs[1]]
    bad = db.get_document(conn, docs[1])
    assert bad["status"] == "failed" and "cannot parse" in bad["error"]
    assert db.status_counts(conn) == {"indexed": 2, "failed": 1}
    assert db.count_chunks(conn, embedded_only=True) == 2 * EMBEDDABLE_PER_RESUME

    parser.fail.clear()
    assert pipeline.retry_failed(conn) == [docs[1]]
    run(conn, settings, parser, embedder)
    assert db.status_counts(conn) == {"indexed": 3}
    assert db.count_chunks(conn, embedded_only=True) == 3 * EMBEDDABLE_PER_RESUME
    assert index_stage.verify_index(conn, settings, embedder).ok


def test_an_unexpected_error_in_a_stage_is_contained(
    conn, settings, parser, embedder, docs, monkeypatch
):
    real = extract_stage.extract_document

    def boom(conn, settings, doc_id, **kwargs):
        if doc_id == docs[0]:
            raise RuntimeError("boom")
        return real(conn, settings, doc_id, **kwargs)

    monkeypatch.setattr(extract_stage, "extract_document", boom)
    run(conn, settings, parser, embedder)
    row = db.get_document(conn, docs[0])
    assert row["status"] == "failed" and "pipeline error: boom" in row["error"]
    assert db.status_counts(conn) == {"indexed": 2, "failed": 1}


def test_a_document_resumes_from_its_last_finished_stage(conn, settings, parser, embedder, docs):
    parse_stage.parse_document(conn, settings, docs[0], parser_factory=parser.factory)
    structure_stage.structure_document(conn, settings, docs[0])
    assert db.get_document(conn, docs[0])["status"] == "structured"
    parser.calls.clear()

    result = run(conn, settings, parser, embedder)

    assert docs[0] not in parser.calls  # already parsed: not parsed again
    first = next(d for d in result.documents if d.doc_id == docs[0])
    assert first.stages == ["extract", "chunk"]
    assert db.status_counts(conn) == {"indexed": 3}


def test_a_changed_file_re_embeds_only_the_chunks_that_changed(
    conn, settings, fixture_files, parser, embedder, docs
):
    run(conn, settings, parser, embedder)
    embedded_before = embedder.document_calls

    # a new upload of candidate 1 (same email) with one bullet edited
    newer = upload(conn, settings, fixture_files, "single_col_spanish", parser,
                   resume(1, bullet="Led migration of 41 services"))
    result = run(conn, settings, parser, embedder)

    assert [d.doc_id for d in result.documents] == [newer]
    assert embedder.document_calls - embedded_before == 1  # one entry chunk, nothing else
    assert result.vectors["fake-embed"].total == 3 * EMBEDDABLE_PER_RESUME
    assert result.keyword.rebuilt
    owners = {r["doc_id"] for r in db.chunk_owner_documents(conn)}
    assert newer in owners and docs[0] not in owners  # the older upload no longer owns chunks
    assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 3
    assert index_stage.verify_index(conn, settings, embedder).ok


def test_a_budget_change_re_chunks_only_what_was_chunked_under_the_old_budgets(
    conn, settings, parser, embedder, docs
):
    run(conn, settings, parser, embedder)
    old = {r["chunk_fp"] for r in db.list_documents(conn)}
    assert len(old) == 1
    embedded = embedder.document_calls

    settings.chunking.model_limit = 256  # changes the budgets' fingerprint
    result = run(conn, settings, parser, embedder)

    assert sorted(result.rechunked) == sorted(docs)
    assert {r["chunk_fp"] for r in db.list_documents(conn)} not in (old, {None})
    assert db.status_counts(conn) == {"indexed": 3}
    assert embedder.document_calls == embedded  # chunk text did not change: nothing re-embedded

    assert pipeline.run_pipeline(
        conn, settings, embedders=[embedder], parser_factory=parser.factory
    ).rechunked == []  # and it is stable afterwards


def test_a_failed_index_build_leaves_resumes_chunked_until_it_succeeds(
    conn, settings, parser, docs
):
    tiny = FakeEmbedder("fake-embed", max_tokens=5)  # every chunk is over its limit
    result = run(conn, settings, parser, tiny)

    assert result.errors and "exceed" in result.errors[0] and result.indexed == []
    assert db.status_counts(conn) == {"chunked": 3}

    good = FakeEmbedder("fake-embed")
    result = run(conn, settings, parser, good)
    assert not result.errors and len(result.indexed) == 3
    assert db.status_counts(conn) == {"indexed": 3}


def test_resumes_needing_review_are_reported_not_fatal(conn, settings, fixture_files, parser,
                                                       embedder, docs):
    parser.texts[docs[2]] = " ".join(["plain words without any resume structure"] * 10) + " a@b.co"
    result = run(conn, settings, parser, embedder)

    assert [d.doc_id for d in result.flagged] == [docs[2]]
    assert db.get_document(conn, docs[2])["status"] == "needs_review"
    assert db.status_counts(conn) == {"indexed": 2, "needs_review": 1}
    assert "1 need attention" in result.summary()

    assert pipeline.retry_failed(conn, include_review=False) == []  # only `failed` this time
    assert pipeline.retry_failed(conn) == [docs[2]]


def test_only_the_requested_documents_are_processed(conn, settings, parser, embedder, docs):
    result = run(conn, settings, parser, embedder, doc_ids={docs[0]})
    assert [d.doc_id for d in result.documents] == [docs[0]]
    statuses = {r["doc_id"]: r["status"] for r in db.list_documents(conn)}
    assert statuses == {docs[0]: "indexed", docs[1]: "uploaded", docs[2]: "uploaded"}
    assert result.vectors["fake-embed"].total == EMBEDDABLE_PER_RESUME


def test_documents_can_be_processed_without_building_indexes(conn, settings, parser, docs):
    result = pipeline.run_pipeline(
        conn, settings, build_indexes=False, parser_factory=parser.factory
    )
    assert db.status_counts(conn) == {"chunked": 3} and result.vectors == {}
    assert result.keyword is None and result.indexed == []


def test_progress_is_reported(conn, settings, parser, embedder, docs):
    events = []
    run(conn, settings, parser, embedder, progress=lambda *e: events.append(e))
    assert events[0] == ("documents", 0, 3) and ("documents", 3, 3) in events
    assert any(phase == "embedding:fake-embed" for phase, _, _ in events)
    assert any(phase == "keyword" for phase, _, _ in events)


def test_the_configured_default_model_is_used_when_none_is_given(
    conn, settings, parser, docs, monkeypatch
):
    asked = []

    def fake_create(name, settings):
        asked.append(name)
        return FakeEmbedder(name)

    monkeypatch.setattr(factory, "create_embedder", fake_create)
    pipeline.run_pipeline(conn, settings, parser_factory=parser.factory)
    assert asked == ["fake-embed"] and db.status_counts(conn) == {"indexed": 3}


def test_an_unavailable_embedding_model_is_reported_not_raised(
    conn, settings, parser, docs, monkeypatch
):
    def unavailable(name, settings):
        raise EmbeddingError("model files missing")

    monkeypatch.setattr(factory, "create_embedder", unavailable)
    result = pipeline.run_pipeline(conn, settings, parser_factory=parser.factory)
    assert any("model files missing" in e for e in result.errors)
    assert db.status_counts(conn) == {"chunked": 3}  # processed, waiting for an index


def test_a_second_model_is_indexed_alongside_without_blocking_the_default(
    conn, settings, parser, docs
):
    settings.embedding.models.append(
        EmbeddingModelConfig(provider="sentence_transformers", name="fake-second")
    )
    first, second = FakeEmbedder("fake-embed", dim=16), FakeEmbedder("fake-second", dim=24)
    result = pipeline.run_pipeline(
        conn, settings, embedders=[first, second], parser_factory=parser.factory
    )
    assert set(result.vectors) == {"fake-embed", "fake-second"}
    assert db.status_counts(conn) == {"indexed": 3}
    for embedder in (first, second):
        assert index_stage.verify_index(conn, settings, embedder).ok


def test_empty_database(conn, settings, embedder):
    result = pipeline.run_pipeline(conn, settings, embedders=[embedder])
    assert result.documents == [] and not result.errors and result.indexed == []
    assert result.vectors["fake-embed"].total == 0


@pytest.mark.integration
def test_every_fixture_resume_end_to_end(conn, settings, fixture_files, parsed, parser, embedder):
    """The plan's corpus run: real parsed Markdown of all 9 fixtures -> sections -> profiles ->
    chunks -> vector and keyword indexes, with the three stores agreeing on every chunk."""
    for spec_id in fixture_files:
        upload(conn, settings, fixture_files, spec_id, parser, parsed[spec_id])

    result = run(conn, settings, parser, embedder)

    assert not result.errors and not result.flagged
    assert db.status_counts(conn) == {"indexed": len(fixture_files)}  # none needs review
    assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == len(fixture_files)

    embeddable = db.count_chunks(conn, embedded_only=True)
    store, _ = mf.load_index(settings, embedder)
    words, _ = keyword_stage.load_keyword_index(settings)
    assert store.count() == words.count() == embeddable > len(fixture_files)  # 0 count mismatch
    assert set(store.ids()) == set(words.ids())

    again = run(conn, settings, parser, embedder)  # and the whole corpus re-runs as a no-op
    assert again.documents == [] and again.vectors["fake-embed"].embedded == 0
    assert not again.keyword.rebuilt
