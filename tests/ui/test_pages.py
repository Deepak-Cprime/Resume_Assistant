"""Each page loads and its main actions work, driven by Streamlit's AppTest.

AppTest runs the real page scripts without a browser, so it checks wiring and behaviour, not
how the page looks: that part is the manual QA walkthrough in the plan."""

import pytest

from src.pipeline import index_stage, keyword_stage, uploads
from src.stores import db

pytestmark = pytest.mark.ui

SPECS = ("single_col_backend", "two_col_sidebar")
PAGES = ["Home.py", "pages/1_Upload.py", "pages/2_Inspect.py", "pages/3_Index.py"]


def texts(elements):
    return [e.value for e in elements]


def frame_with(at, column):
    """The first dataframe on the page that has this column."""
    return next(d.value for d in at.dataframe if column in d.value.columns)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_on_an_empty_database(open_page, page):
    at = open_page(page)
    assert not at.exception


# --- Home -----------------------------------------------------------------------------


def test_home_shows_the_corpus_counts(open_page, add_resumes):
    add_resumes(*SPECS)
    at = open_page("Home.py")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Resumes"] == "2" and metrics["Candidates"] == "2"
    assert metrics["Waiting to process"] == "0" and metrics["Need attention"] == "0"
    names = list(at.dataframe[0].value["name"])
    assert "Priya Raman" in names and len(names) == 2


def test_the_job_description_saves_and_persists(open_page, env):
    at = open_page("Home.py")
    at.text_input(key="role_title").set_value("Backend Engineer")
    at.text_area(key="jd_text").set_value("Go, Kubernetes, 5+ years")
    at.button(key="save_role").click().run()
    assert not at.exception and at.success

    role = db.get_role(env.conn)
    assert (role["title"], role["jd_text"]) == ("Backend Engineer", "Go, Kubernetes, 5+ years")
    fresh = open_page("Home.py")  # a brand-new session sees it
    assert fresh.text_area(key="jd_text").value == "Go, Kubernetes, 5+ years"


def test_editing_the_job_description_never_re_indexes(open_page, env, add_resumes):
    add_resumes(*SPECS)
    before = (
        [(c["chunk_id"], c["content_hash"]) for c in db.list_chunks(env.conn)],
        db.status_counts(env.conn),
        [r["created_at"] for r in db.current_index_manifests(env.conn)],
    )
    at = open_page("Home.py")
    at.text_area(key="jd_text").set_value("A completely different job description")
    at.button(key="save_role").click().run()
    assert not at.exception
    assert before == (
        [(c["chunk_id"], c["content_hash"]) for c in db.list_chunks(env.conn)],
        db.status_counts(env.conn),
        [r["created_at"] for r in db.current_index_manifests(env.conn)],
    )


# --- Upload ---------------------------------------------------------------------------


def test_upload_page_processes_pending_resumes_to_indexed(open_page, env, add_resumes):
    add_resumes(*SPECS, process=False)
    at = open_page("pages/1_Upload.py")
    assert not at.exception
    assert all("uploaded" in s for s in at.dataframe[0].value["status"])

    at.button(key="process_btn").click().run()

    assert not at.exception
    assert db.status_counts(env.conn) == {"indexed": 2}
    assert all("indexed" in s for s in at.dataframe[0].value["status"])
    assert any("2 document(s) processed" in m for m in texts(at.success))


def test_a_rejected_file_is_listed_and_can_be_removed(open_page, env):
    uploads.ingest_files(env.conn, env.settings, [("virus.pdf", b"MZ not a pdf")])
    at = open_page("pages/1_Upload.py")
    assert not at.exception
    assert any("virus.pdf" in w and "failed" in w for w in texts(at.warning))
    assert not at.button(key="retry_btn").disabled  # something to retry

    at.checkbox(key="remove_confirm").check().run()
    at.button(key="remove_btn").click().run()
    assert not at.exception
    assert db.list_documents(env.conn) == []
    assert any("Removed 1 upload" in m for m in texts(at.success))


def test_remove_needs_confirmation(open_page, env):
    uploads.ingest_files(env.conn, env.settings, [("virus.pdf", b"MZ not a pdf")])
    at = open_page("pages/1_Upload.py")
    assert at.button(key="remove_btn").disabled
    assert len(db.list_documents(env.conn)) == 1


# --- Inspect --------------------------------------------------------------------------


def test_inspect_with_nothing_uploaded(open_page):
    at = open_page("pages/2_Inspect.py")
    assert not at.exception and any("Nothing to inspect" in m for m in texts(at.info))


def test_inspect_shows_every_stage_of_a_resume(open_page, add_resumes):
    add_resumes(*SPECS)
    at = open_page("pages/2_Inspect.py")
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Raw file", "Markdown", "Sections", "Metadata", "Chunks"]

    box = at.selectbox(key="inspect_doc")
    assert len(box.options) == 2
    box.select(next(o for o in box.options if o.startswith("single_col_backend"))).run()
    assert not at.exception

    assert any("Northwind Payments" in code for code in texts(at.code))  # the Markdown tab
    sections = frame_with(at, "heading")
    assert {"experience", "education", "skills"} <= set(sections["label"])
    jobs = frame_with(at, "company")
    assert list(jobs["company"]) == ["Northwind Payments", "Brightline Logistics"]
    assert list(jobs["start"]) == ["2021-03", "2018-06"]
    chunks = frame_with(at, "tokens")
    assert len(chunks) == 5 and set(chunks["embedded"]) == {True, False}
    assert any(k.endswith("exp-1-p1") for k in chunks["key"])


def test_inspect_can_show_one_chunk(open_page, add_resumes):
    add_resumes("single_col_backend")
    at = open_page("pages/2_Inspect.py")
    choice = at.selectbox(key="chunk_choice")
    choice.select(next(o for o in choice.options if o.endswith("exp-1-p1"))).run()
    codes = texts(at.code)
    assert any(c.startswith("Candidate C") and "Experience > " in c for c in codes)  # embed_text
    assert any(c.startswith("Senior Backend Engineer @ Northwind Payments") for c in codes)


def test_inspect_explains_a_document_that_never_parsed(open_page, env):
    uploads.ingest_files(env.conn, env.settings, [("virus.pdf", b"MZ not a pdf")])
    at = open_page("pages/2_Inspect.py")
    assert not at.exception
    assert any("Not parsed yet" in m for m in texts(at.info))


# --- Index ----------------------------------------------------------------------------


def test_index_page_with_no_chunks(open_page):
    at = open_page("pages/3_Index.py")
    assert not at.exception and any("No chunks yet" in m for m in texts(at.info))


def test_build_button_brings_the_indexes_up_to_date(open_page, env, add_resumes):
    add_resumes(*SPECS, build_indexes=False)
    at = open_page("pages/3_Index.py")
    assert any("missing or out of date" in m for m in texts(at.warning))

    at.button(key="build_btn").click().run()

    assert not at.exception
    assert any("Every index matches the database" in m for m in texts(at.success))
    assert db.status_counts(env.conn) == {"indexed": 2}
    vectors = frame_with(at, "vectors")
    assert int(vectors["vectors"][0]) == db.count_chunks(env.conn, embedded_only=True)
    assert list(vectors["matches database"]) == ["yes"]


def test_rebuild_from_scratch_runs(open_page, env, add_resumes):
    add_resumes(*SPECS)
    at = open_page("pages/3_Index.py")
    at.button(key="rebuild_btn").click().run()
    assert not at.exception
    assert any("Every index matches the database" in m for m in texts(at.success))


def test_deleting_a_candidate_removes_them_everywhere(open_page, env, add_resumes):
    add_resumes(*SPECS)
    total = db.count_chunks(env.conn, embedded_only=True)
    at = open_page("pages/3_Index.py")
    at.checkbox(key="delete_confirm").check().run()
    at.button(key="delete_btn").click().run()

    assert not at.exception
    assert db.count_candidates(env.conn) == 1
    assert any("Deleted 1 upload" in m for m in texts(at.success))
    assert db.count_chunks(env.conn, embedded_only=True) < total
    assert index_stage.verify_model_index(env.conn, env.settings, "fake-embed").ok
    assert keyword_stage.verify_keyword_index(env.conn, env.settings).ok


def test_delete_needs_confirmation(open_page, env, add_resumes):
    add_resumes(*SPECS)
    at = open_page("pages/3_Index.py")
    assert at.button(key="delete_btn").disabled
    assert db.count_candidates(env.conn) == 2
