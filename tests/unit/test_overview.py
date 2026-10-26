"""The read-only views the UI pages render."""

import pytest

from src.pipeline import index_stage, overview
from src.pipeline import run as pipeline
from src.stores import db, files
from tests.unit.fakes import FakeEmbedder
from tests.unit.pipeline_helpers import MarkdownParser, configure, resume, upload


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
def world(conn, settings, fixture_files, parser, embedder):
    """Two processed resumes (candidates 1 and 2) and one upload that failed to parse."""
    docs = [
        upload(conn, settings, fixture_files, spec, parser, resume(i))
        for i, spec in enumerate(["single_col_backend", "single_col_overlap"], 1)
    ]
    bad = upload(conn, settings, fixture_files, "single_col_gap", parser, resume(3))
    parser.fail.add(bad)
    pipeline.run_pipeline(conn, settings, embedders=[embedder], parser_factory=parser.factory)
    return docs, bad


def test_corpus_stats(conn, world):
    stats = overview.corpus_stats(conn)
    assert (stats.resumes, stats.candidates, stats.chunks, stats.embedded_chunks) == (3, 2, 10, 8)
    assert stats.by_status == {"indexed": 2, "failed": 1}
    assert (stats.failed, stats.needs_review, stats.pending) == (1, 0, 0)


def test_corpus_stats_of_an_empty_database(conn):
    stats = overview.corpus_stats(conn)
    assert (stats.resumes, stats.candidates, stats.chunks, stats.pending) == (0, 0, 0, 0)


def test_document_rows(conn, world):
    docs, bad = world
    rows = overview.document_rows(conn)
    assert [r["doc_id"] for r in rows][0] == bad  # newest upload first
    by_id = {r["doc_id"]: r for r in rows}
    assert by_id[docs[0]]["status"] == "indexed" and by_id[docs[0]]["candidate"] == "Priya Raman"
    assert by_id[bad]["status"] == "failed" and "cannot parse" in by_id[bad]["note"]
    assert by_id[docs[0]]["parser"] == "pymupdf4llm" and by_id[docs[0]]["type"] == "pdf"


def test_candidate_rows(conn, world):
    rows = overview.candidate_rows(conn)
    assert len(rows) == 2
    first = rows[0]
    assert first["name"] == "Priya Raman" and first["chunks"] == 5 and first["uploads"] == 1
    assert first["years"] == pytest.approx(6.4, abs=0.1) and first["seniority"] == "senior"


def test_document_view_of_a_processed_resume(conn, settings, world):
    docs, _ = world
    view = overview.document_view(conn, settings, docs[0])

    assert "Marker1Corp" in view.markdown and view.raw_path.exists() and view.notes == []
    labels = [s["label"] for s in view.sections]
    assert labels[:3] == ["header", "summary", "experience"]
    assert all(s["words"] for s in view.sections)

    assert view.profile.name == "Priya Raman"
    first_job = view.experience_table[0]
    assert (first_job["company"], first_job["start"], first_job["end"], first_job["split"]) == (
        "Marker1Corp", "2021-03", "2024-12", "high")
    assert view.facts["Years of experience"] == "6.4" and view.facts["Name"] == "Priya Raman"
    skills = {s["skill"]: s for s in view.skills_table}
    assert skills["Kubernetes"]["evidence"] == "used" and skills["AWS"]["evidence"] == "claimed"
    assert view.education_table[0]["degree"] == "B.Tech in Computer Science"
    assert view.certifications_table[0]["certification"] == "AWS Certified Solutions Architect"
    assert view.gaps_table == []


def test_document_view_chunks(conn, settings, world):
    docs, _ = world
    chunks = {c["key"]: c for c in overview.document_view(conn, settings, docs[0]).chunks}
    assert len(chunks) == 5
    entry, parent = chunks["c1-exp-1-p1"], chunks["c1-experience"]
    assert entry["parent"] == "c1-experience" and entry["level"] == 2 and entry["embedded"]
    assert not parent["embedded"] and parent["parent"] == ""
    assert entry["skills"] == "Kubernetes, Go" and entry["tokens"] > 0
    assert entry["embed_text"].startswith("Candidate C1 | Experience")
    assert "Candidate" not in entry["text"]


def test_document_view_of_a_failed_upload(conn, settings, world):
    _, bad = world
    view = overview.document_view(conn, settings, bad)
    assert view.markdown == "" and view.sections == [] and view.profile is None
    assert view.chunks == [] and view.facts == {} and view.experience_table == []
    assert any("Not parsed yet" in n for n in view.notes)
    assert any("cannot parse" in n for n in view.notes)


def test_document_view_of_an_unprocessed_upload(conn, settings, fixture_files, parser):
    doc = upload(conn, settings, fixture_files, "single_col_backend", parser, resume(1))
    view = overview.document_view(conn, settings, doc)
    assert view.notes == ["Not parsed yet (status: uploaded)."] and view.raw_path.exists()


def test_a_replaced_upload_explains_why_it_has_no_chunks(
    conn, settings, fixture_files, parser, embedder, world
):
    docs, _ = world
    upload(conn, settings, fixture_files, "single_col_spanish", parser, resume(1))
    pipeline.run_pipeline(conn, settings, embedders=[embedder], parser_factory=parser.factory)
    view = overview.document_view(conn, settings, docs[0])  # the older upload of candidate 1
    assert view.chunks == []
    assert "Another upload of this candidate owns the current chunks." in view.notes


def test_unknown_document(conn, settings):
    with pytest.raises(KeyError):
        overview.document_view(conn, settings, "nope")


@pytest.mark.parametrize(
    "sizes,expected",
    [
        ([], []),
        ([10], [("0-49", 1)]),
        ([10, 60, 61, 160], [("0-49", 1), ("50-99", 2), ("100-149", 0), ("150-199", 1)]),
        ([49, 50], [("0-49", 1), ("50-99", 1)]),
    ],
)
def test_size_histogram(sizes, expected):
    assert overview.size_histogram(sizes) == expected


def test_index_overview_before_and_after_building(conn, settings, parser, embedder, fixture_files):
    upload(conn, settings, fixture_files, "single_col_backend", parser, resume(1))
    pipeline.run_pipeline(conn, settings, embedders=[embedder], parser_factory=parser.factory,
                          build_indexes=False)

    before = overview.index_overview(conn, settings)
    (model,) = before.models
    assert model.model == "fake-embed" and model.configured and model.manifest is None
    assert not model.report.ok and not before.keyword_report.ok
    assert before.embeddable_chunks == 4 and before.sizes and before.budget_rows

    pipeline.run_pipeline(conn, settings, embedders=[embedder], parser_factory=parser.factory)
    after = overview.index_overview(conn, settings)
    assert after.models[0].manifest.n_vectors == 4 and after.models[0].report.ok
    assert after.keyword_manifest.n_docs == 4 and after.keyword_report.ok
    assert {"section", "min tokens", "max tokens", "samples", "source"} <= set(after.budget_rows[0])


def test_index_overview_lists_models_that_are_built_but_no_longer_configured(
    conn, settings, parser, fixture_files
):
    upload(conn, settings, fixture_files, "single_col_backend", parser, resume(1))
    old = FakeEmbedder("retired-model")
    pipeline.run_pipeline(conn, settings, embedders=[old], parser_factory=parser.factory,
                          build_indexes=False)
    index_stage.build_index(conn, settings, old)
    view = overview.index_overview(conn, settings)
    flags = {m.model: m.configured for m in view.models}
    assert flags == {"fake-embed": True, "retired-model": False}


def test_a_damaged_index_shows_as_out_of_date_instead_of_crashing(
    conn, settings, parser, embedder, fixture_files
):
    upload(conn, settings, fixture_files, "single_col_backend", parser, resume(1))
    pipeline.run_pipeline(conn, settings, embedders=[embedder], parser_factory=parser.factory)
    (files.index_dir(settings, "fake-embed") / "vectors.npz").unlink()
    (model,) = overview.index_overview(conn, settings).models
    assert not model.report.ok and len(model.report.missing) == 4


def test_empty_database_index_overview(conn, settings):
    view = overview.index_overview(conn, settings)
    assert view.embeddable_chunks == 0 and view.sizes == [] and view.budget_rows == []
    assert view.keyword_report.n_db == 0 and view.models[0].manifest is None


def test_every_table_is_arrow_compatible(conn, settings, world):
    """Streamlit converts tables through Arrow; a column mixing numbers and text ('' for a
    missing value) makes it log a conversion error, so every column must keep one type."""
    pandas = pytest.importorskip("pandas")
    pyarrow = pytest.importorskip("pyarrow")
    docs, bad = world
    tables = [overview.document_rows(conn), overview.candidate_rows(conn)]
    for doc_id in (*docs, bad):
        view = overview.document_view(conn, settings, doc_id)
        tables += [view.experience_table, view.skills_table, view.education_table,
                   view.certifications_table, view.gaps_table, view.sections, view.chunks]
    for rows in tables:
        if rows:
            pyarrow.Table.from_pandas(pandas.DataFrame(rows))  # raises ArrowInvalid on a mix


def test_a_skill_without_years_has_an_empty_text_cell(conn, settings, world):
    docs, _ = world
    skills = {s["skill"]: s for s in overview.document_view(conn, settings, docs[0]).skills_table}
    assert skills["AWS"]["years"] == ""  # claimed, never used in a job
    assert skills["Kubernetes"]["years"] == "3.8"  # 45 months, shown as text
