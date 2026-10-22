"""parse -> structure -> extract for whole documents, with a fake parser so no PDF library runs."""

import json
from datetime import date
from pathlib import Path

import pytest

from src.ingestion import intake
from src.pipeline import extract_stage, parse_stage, structure_stage
from src.providers.base import ParsedDocument
from src.stores import db
from tests.unit.fakes import FULL_MD, FakeLLM

NO_STRUCTURE = " ".join(["plain words without any resume structure at all"] * 8) + " a@b.com"


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


def upload(conn, settings, fixture_files, spec_id):
    path = fixture_files[spec_id]
    return intake.ingest(conn, settings, path.name, path.read_bytes()).doc_id


def run_parse(conn, settings, doc_id, text):
    parser = FakeParser(text)
    return parse_stage.parse_document(conn, settings, doc_id, parser_factory=lambda n, s: parser)


def test_full_flow_to_extracted(conn, settings, fixture_files):
    doc_id = upload(conn, settings, fixture_files, "single_col_backend")
    assert run_parse(conn, settings, doc_id, FULL_MD) == "parsed"
    assert structure_stage.structure_document(conn, settings, doc_id) == "structured"
    assert extract_stage.extract_document(conn, settings, doc_id) == "extracted"

    row = db.get_document(conn, doc_id)
    cand = db.get_candidate(conn, row["candidate_id"])
    assert (cand["display_name"], cand["seniority"]) == ("Priya Raman", "senior")
    assert cand["total_yoe"] == pytest.approx(77 / 12, abs=0.01)
    assert [e["company"] for e in db.list_experiences(conn, cand["candidate_id"])] == [
        "Northwind Payments", "Brightline Logistics"]
    skills = {s["skill"]: s["evidence"] for s in db.list_skills(conn, cand["candidate_id"])}
    assert skills["Kubernetes"] == "used" and skills["AWS"] == "claimed"
    assert [c["name"] for c in db.list_certifications(conn, cand["candidate_id"])] == [
        "AWS Certified Solutions Architect"]


def test_artifacts_are_saved(conn, settings, fixture_files):
    doc_id = upload(conn, settings, fixture_files, "single_col_backend")
    run_parse(conn, settings, doc_id, FULL_MD)
    structure_stage.structure_document(conn, settings, doc_id)
    extract_stage.extract_document(conn, settings, doc_id)

    sections = json.loads((settings.profiles_dir / f"{doc_id}.sections.json").read_text("utf-8"))
    assert [s["label"] for s in sections][:3] == ["header", "summary", "experience"]
    profile = json.loads((settings.profiles_dir / f"{doc_id}.json").read_text("utf-8"))
    assert profile["name"] == "Priya Raman" and len(profile["experiences"]) == 2
    assert len(profile["email_hash"]) == 64


def test_stages_skip_documents_in_the_wrong_status(conn, settings, fixture_files):
    doc_id = upload(conn, settings, fixture_files, "single_col_backend")
    assert structure_stage.structure_document(conn, settings, doc_id) == "uploaded"
    assert extract_stage.extract_document(conn, settings, doc_id) == "uploaded"
    run_parse(conn, settings, doc_id, FULL_MD)
    assert extract_stage.extract_document(conn, settings, doc_id) == "parsed"


def test_rerun_is_a_no_op(conn, settings, fixture_files):
    doc_id = upload(conn, settings, fixture_files, "single_col_backend")
    run_parse(conn, settings, doc_id, FULL_MD)
    structure_stage.structure_document(conn, settings, doc_id)
    extract_stage.extract_document(conn, settings, doc_id)
    assert extract_stage.extract_document(conn, settings, doc_id) == "extracted"
    assert conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0] == 2


def test_resume_without_structure_needs_review(conn, settings, fixture_files):
    doc_id = upload(conn, settings, fixture_files, "single_col_backend")
    run_parse(conn, settings, doc_id, NO_STRUCTURE)
    assert structure_stage.structure_document(conn, settings, doc_id) == "needs_review"
    assert "recognised section" in db.get_document(conn, doc_id)["error"]


def test_resume_without_jobs_needs_review_but_is_saved(conn, settings, fixture_files):
    md = "# Jane Doe\njane@example.com\n\n## Skills\nPython, Go, SQL\n\n## Summary\nEngineer."
    doc_id = upload(conn, settings, fixture_files, "single_col_backend")
    run_parse(conn, settings, doc_id, md + " " + NO_STRUCTURE)
    assert structure_stage.structure_document(conn, settings, doc_id) == "structured"
    assert extract_stage.extract_document(conn, settings, doc_id) == "needs_review"
    row = db.get_document(conn, doc_id)
    assert row["error"] == "no job entries found" and row["candidate_id"] is not None


def test_same_email_reuses_the_candidate(conn, settings, fixture_files):
    ids = []
    for spec_id in ("single_col_backend", "single_col_overlap"):
        doc_id = upload(conn, settings, fixture_files, spec_id)
        run_parse(conn, settings, doc_id, FULL_MD)
        structure_stage.structure_document(conn, settings, doc_id)
        extract_stage.extract_document(conn, settings, doc_id)
        ids.append(db.get_document(conn, doc_id)["candidate_id"])
    assert ids[0] == ids[1]
    assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 1
    # the second resume replaces the candidate's experiences instead of adding to them
    assert conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0] == 2


def test_unknown_document(conn, settings):
    with pytest.raises(KeyError):
        structure_stage.structure_document(conn, settings, "nope")
    with pytest.raises(KeyError):
        extract_stage.extract_document(conn, settings, "nope")


def test_llm_is_used_when_passed(conn, settings, fixture_files):
    ambiguous = "**Northwind, Payments**"
    md = FULL_MD.replace("**Senior Backend Engineer, Northwind Payments**", ambiguous)
    doc_id = upload(conn, settings, fixture_files, "single_col_backend")
    run_parse(conn, settings, doc_id, md)
    structure_stage.structure_document(conn, settings, doc_id)
    llm = FakeLLM({"title": "Payments", "company": "Northwind"})
    extract_stage.extract_document(conn, settings, doc_id, llm=llm)
    assert llm.prompts, "the ambiguous entry should have been sent to the model"
