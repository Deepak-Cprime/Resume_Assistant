"""The whole pipeline over every synthetic resume with the real parsers (only the embedding model
is faked): the Phase 1 gates of plan section 7.5, end to end."""

import pytest

from src.pipeline import overview, report
from src.pipeline import run as pipeline
from src.stores import db
from tests.unit.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


def test_every_fixture_reaches_indexed(corpus):
    _, _, result, reports = corpus
    assert not result.errors
    assert {r.status for r in reports} == {"indexed"}, [(r.filename, r.status) for r in reports]


def test_the_phase_one_gates_pass(corpus):
    conn, settings, _, reports = corpus
    gates = report.compute_gates(reports, conn, settings)
    failed = [f"{g.name}: {g.value} (needs {g.threshold})" for g in gates if g.failed]
    assert not failed, "\n".join(failed)
    measured = {g.name for g in gates if g.ok is not None}
    assert len(measured) >= 6, "most gates should be measurable on the fixtures"


def test_stores_agree_after_the_run(corpus):
    conn, settings, _, _ = corpus
    info = overview.index_overview(conn, settings)
    assert info.keyword_report.ok and all(m.report.ok for m in info.models if m.manifest)
    assert db.count_chunks(conn, embedded_only=True) == info.models[0].manifest.n_vectors


def test_a_second_run_changes_nothing(corpus):
    conn, settings, _, _ = corpus
    before = [(c["chunk_id"], c["content_hash"]) for c in db.list_chunks(conn)]
    again = pipeline.run_pipeline(conn, settings, embedders=[FakeEmbedder("fake-embed")])
    assert not again.documents and not again.rechunked
    assert [(c["chunk_id"], c["content_hash"]) for c in db.list_chunks(conn)] == before


def test_the_report_renders_every_resume(corpus):
    conn, settings, _, reports = corpus
    text = report.render_markdown(reports, report.compute_gates(reports, conn, settings))
    assert "## Phase 1 gates" in text
    for r in reports:
        assert r.filename in text
