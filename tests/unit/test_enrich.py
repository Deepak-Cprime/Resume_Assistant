import pytest

from src.chunking import enrich
from src.chunking.tokens import estimate_tokens
from src.metadata.schema import Experience

JOB = Experience(
    title="Senior Backend Engineer", company="Northwind Payments", start="2021-03", end="2024-12"
)


@pytest.mark.parametrize(
    "start,end,current,label",
    [
        ("2021-03", "2024-12", False, "2021-03 – 2024-12"),
        ("2021-03", None, True, "2021-03 – present"),
        ("2018", None, False, "2018"),
        (None, None, False, ""),
        (None, "2020-01", False, ""),
    ],
)
def test_dates_label(start, end, current, label):
    assert enrich.dates_label(start, end, current) == label


def test_entry_header_and_prefix():
    assert enrich.entry_header(JOB) == (
        "Senior Backend Engineer @ Northwind Payments (2021-03 – 2024-12)")
    assert enrich.entry_prefix(12, JOB) == (
        "Candidate C12 | Experience > Senior Backend Engineer @ Northwind Payments "
        "(2021-03 – 2024-12)")


def test_prefix_with_missing_parts():
    assert enrich.entry_header(Experience(company="Acme")) == "Acme"
    assert enrich.entry_header(Experience(title="Engineer", company="Acme")) == "Engineer @ Acme"
    assert enrich.entry_header(Experience(start="2020-01")) == "2020-01"
    assert enrich.entry_prefix(3, Experience()) == "Candidate C3 | Experience"


def test_section_titles():
    assert enrich.section_prefix(7, "skills") == "Candidate C7 | Skills"
    assert enrich.section_title("education+certifications") == "Education + Certifications"
    assert enrich.section_title("other") == "Other"


def test_tags_line():
    assert enrich.tags_line([]) == ""
    assert enrich.tags_line(["Go", "SQL"]) == "Skills: Go, SQL"


def assemble(prefix, body, tags="", limit=512, names=()):
    return enrich.assemble_embed_text(
        prefix, body, tags, names=list(names), limit=limit, count=estimate_tokens
    )


def test_assemble_layout_and_pii_masking():
    out = assemble(
        "Candidate C1 | Experience", "Priya Raman (priya@example.com) built APIs", "Skills: Go",
        names=["Priya Raman"],
    )
    assert out == "Candidate C1 | Experience\n[NAME] ([EMAIL]) built APIs\nSkills: Go"


def test_assemble_drops_tags_before_cutting_the_body():
    body = " ".join(["word"] * 10)
    limit = estimate_tokens("P\n" + body)
    out = assemble("P", body, "Skills: alpha beta gamma", limit=limit)
    assert out == "P\n" + body and "Skills" not in out


def test_assemble_cuts_the_body_when_still_too_long():
    out = assemble("P", " ".join(f"w{i}" for i in range(200)), "", limit=40)
    assert estimate_tokens(out) <= 40 and out.startswith("P\nw0 w1")


def test_assemble_always_fits_even_for_one_huge_token():
    out = assemble("P", "x" * 5000, "", limit=20)
    assert estimate_tokens(out) <= 20
