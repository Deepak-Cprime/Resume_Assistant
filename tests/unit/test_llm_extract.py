from src.metadata import llm_extract as L
from src.metadata.schema import Experience
from tests.unit.fakes import FakeLLM


def uncertain(raw="Acme Corp\nSenior Engineer\nJan 2020 - Dec 2021") -> Experience:
    return Experience(title="Acme Corp", company="Senior Engineer", confidence="low", raw_text=raw)


def test_needs_llm_only_for_uncertain_entries():
    assert L.needs_llm(uncertain())
    assert L.needs_llm(Experience(title="Engineer", company="", confidence="high"))
    assert not L.needs_llm(Experience(title="Engineer", company="Acme", confidence="high"))


def test_grounded_value_is_accepted_ignoring_case_and_punctuation():
    llm = FakeLLM({"title": "senior engineer", "company": "ACME CORP."})
    found = L.extract_entry(llm, uncertain())
    assert found == {"title": "senior engineer", "company": "ACME CORP."}


def test_company_not_in_the_text_is_rejected():
    llm = FakeLLM({"title": "Senior Engineer", "company": "Globex Corporation"})
    assert L.extract_entry(llm, uncertain()) == {"title": "Senior Engineer"}


def test_null_and_wrong_types_are_ignored():
    assert L.extract_entry(FakeLLM({"title": None, "company": 42}), uncertain()) == {}


def test_model_errors_do_not_propagate():
    assert L.extract_entry(FakeLLM(error=ValueError("bad json")), uncertain()) == {}


def test_prompt_contains_only_this_entry():
    llm = FakeLLM({"title": "Senior Engineer", "company": "Acme Corp"})
    L.extract_entry(llm, uncertain())
    assert "Acme Corp\nSenior Engineer" in llm.prompts[0]


def test_enrich_fixes_swapped_fields_and_marks_source():
    exp = uncertain()
    llm = FakeLLM({"title": "Senior Engineer", "company": "Acme Corp"})
    assert L.enrich(llm, [exp]) == 1
    assert (exp.title, exp.company, exp.confidence, exp.source) == (
        "Senior Engineer", "Acme Corp", "high", "llm")


def test_enrich_needs_both_grounded_fields_and_skips_confident_entries():
    partial = uncertain()
    confident = Experience(title="Engineer", company="Acme", confidence="high", raw_text="x")
    llm = FakeLLM({"title": "Senior Engineer", "company": "Invented Inc"})
    assert L.enrich(llm, [partial, confident]) == 0
    assert partial.title == "Acme Corp" and partial.source == "rules"  # unchanged
    assert len(llm.prompts) == 1  # the confident entry was never sent


def test_long_entries_are_truncated_before_prompting():
    llm = FakeLLM({})
    L.extract_entry(llm, uncertain("x" * 5000))
    assert len(llm.prompts[0]) < 1200
