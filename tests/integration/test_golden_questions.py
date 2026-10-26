"""eval/golden_questions.yaml is well-formed and every evidence quote is really in the candidate's
chunks, so the labels stay valid as the chunker changes (they are never tied to chunk ids)."""

import re
from collections import Counter

import pytest
import yaml

from src.config import PROJECT_ROOT
from src.stores import db

pytestmark = pytest.mark.integration

GOLDEN = PROJECT_ROOT / "eval" / "golden_questions.yaml"
INTENTS = {"skill_lookup", "experience_depth", "structured", "compare", "jd_match", "negative",
           "adversarial"}


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@pytest.fixture(scope="module")
def questions():
    return yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))


def test_there_are_at_least_forty_questions(questions):
    assert len(questions) >= 40
    assert len({q["id"] for q in questions}) == len(questions)


def test_every_question_has_the_required_fields(questions, specs):
    for q in questions:
        assert q["intent"] in INTENTS, q["id"]
        assert q["split"] in ("dev", "test"), q["id"]
        assert q["question"].strip() and isinstance(q["answerable"], bool), q["id"]
        for fixture in q["expected_candidates"]:
            assert fixture in specs, (q["id"], fixture)
        if q["answerable"]:
            assert q["evidence"] and q["expected_candidates"], q["id"]
        else:
            assert not q["evidence"] and not q["expected_candidates"], q["id"]


def test_every_intent_is_covered_and_both_splits_are_used(questions):
    counts = Counter(q["intent"] for q in questions)
    assert all(counts[i] >= 2 for i in INTENTS), counts
    assert {q["split"] for q in questions} == {"dev", "test"}


def test_every_evidence_quote_is_in_that_candidates_chunks(questions, corpus):
    conn, _, _, reports = corpus
    candidate_of = {r.filename.rsplit(".", 1)[0]: db.get_document(conn, r.doc_id)["candidate_id"]
                    for r in reports}
    text_of = {
        fixture: squash(" ".join(c["text"] for c in db.list_chunks(conn, candidate_id=cid)))
        for fixture, cid in candidate_of.items()
    }
    missing = [
        f"{q['id']}: {e['candidate']}: {e['quote']!r}"
        for q in questions for e in q["evidence"]
        if squash(e["quote"]) not in text_of[e["candidate"]]
    ]
    assert not missing, "\n".join(missing)
    for q in questions:  # evidence candidates are the expected ones
        assert {e["candidate"] for e in q["evidence"]} <= set(q["expected_candidates"]), q["id"]
