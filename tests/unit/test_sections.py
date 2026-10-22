import pytest
import yaml

from src.structure import sections as S
from src.structure.sections import load_matcher, split_sections
from tests.unit.fakes import FULL_MD, NO_HEADINGS_MD, SIDEBAR_FIRST_MD


def labels(sections):
    return [s.label for s in sections]


def classify(line, allow_unknown=False):
    return load_matcher().classify(line, allow_unknown=allow_unknown)


def test_every_synonym_maps_to_its_label():
    raw = yaml.safe_load(S.DEFAULT_SECTIONS_PATH.read_text(encoding="utf-8"))
    for label, synonyms in raw["sections"].items():
        for syn in synonyms:
            heading = classify(f"## {syn}")
            assert heading is not None and heading.label == label, (label, syn)


@pytest.mark.parametrize(
    "line,label",
    [
        ("## Work Experience", "experience"),
        ("**Technical Skills**", "skills"),
        ("EDUCATION", "education"),
        ("Skills:", "skills"),
        ("Experience", "experience"),
        ("## Education & Training", "education"),
        ("## Educación", "education"),
        ("## Licenses and Certifications", "certifications"),
    ],
)
def test_heading_variants(line, label):
    heading = classify(line)
    assert heading is not None and heading.label == label


def test_fuzzy_match_only_for_emphasised_lines():
    assert classify("## Experiance").label == "experience"
    assert classify("## Experiance").method == "fuzzy"
    assert classify("Experiance") is None


def test_inline_heading_carries_content():
    heading = classify("Skills: Python, Go, SQL")
    assert (heading.label, heading.method, heading.rest) == ("skills", "inline", "Python, Go, SQL")


def test_languages_prefix_inside_skills_is_not_a_heading():
    assert classify("Languages: Python, Go") is None


@pytest.mark.parametrize(
    "line",
    [
        "- Experience",  # bullet
        "Led the experience team to ship a new checkout flow end to end in Q3.",
        "**Senior Engineer, Acme**",
        "",
    ],
)
def test_content_lines_are_not_headings(line):
    assert classify(line) is None


def test_full_resume_sections():
    secs = split_sections(FULL_MD)
    assert labels(secs) == [
        "header", "summary", "experience", "education", "skills", "certifications",
    ]
    assert secs[0].method == "preamble" and "# Priya Raman" in secs[0].text
    assert all(s.method == "synonym" for s in secs[1:])
    assert "Northwind Payments" in S.by_label(secs)["experience"][0].text


def test_name_line_is_not_a_section():
    secs = split_sections("# Priya Raman\nA short line\n\n## Skills\nPython, Go")
    assert labels(secs) == ["header", "skills"]


def test_unknown_heading_is_flagged_not_merged():
    md = "# Priya Raman\n\n## Experience\ntext here\n\n## Volunteer Work\nfood bank\n"
    secs = split_sections(md)
    unknown = [s for s in secs if s.label == "unknown"]
    assert [(s.heading, s.method) for s in unknown] == [("Volunteer Work", "unknown")]
    assert "food bank" in unknown[0].text
    assert "food bank" not in S.by_label(secs)["experience"][0].text


def test_sidebar_first_resume_keeps_name_out_of_sections():
    secs = split_sections(SIDEBAR_FIRST_MD)
    assert labels(secs) == ["contact", "skills", "education", "summary", "experience"]
    assert "Aisha Rahman" in S.by_label(secs)["education"][0].text  # name trails the sidebar


def test_no_headings_falls_back_to_content_inference():
    secs = split_sections(NO_HEADINGS_MD)
    assert labels(secs) == [
        "header", "summary", "experience", "education", "skills", "certifications",
    ]
    assert {s.method for s in secs[1:]} == {"heuristic"}
    by = S.by_label(secs)
    assert "Pinecone Apps" in by["experience"][0].text
    assert "Cascade Fitness" in by["experience"][0].text
    assert "Puget Sound College" in by["education"][0].text
    assert by["skills"][0].text.startswith("Swift, Kotlin")
    assert "Associate Android Developer" in by["certifications"][0].text


def test_text_without_any_structure_stays_one_header():
    secs = split_sections("just some words\nand nothing else")
    assert labels(secs) == ["header"]


def test_known_token_share():
    assert S.known_token_share(split_sections(FULL_MD)) == 1.0
    md = (
        "# Priya Raman\n\n## Experience\none two\n\n"
        "## Volunteer Work\nthree four five six seven eight nine ten\n"
    )
    assert S.known_token_share(split_sections(md)) < 0.5


def test_repeated_headings_are_both_kept():
    secs = split_sections("## Experience\na\n\n## Skills\nb\n\n## Experience\nc\n")
    assert [s.text for s in S.by_label(secs)["experience"]] == ["a", "c"]
    lines = S.section_lines(secs, "experience")
    assert "a" in lines and "c" in lines


MERGED_LINES_MD = """\
# **Tomas Becker**

_tomas.becker@example.com | +1 555 0104 | Seattle, WA_ Mobile engineer shipping iOS apps.

## **Mobile Engineer, Pinecone Apps**

_Jan 2020 - Dec 2022 | Seattle, WA_

- Rebuilt the onboarding flow in Swift, raising activation by 18 percent.

## **B.S. in Computer Science**

_Puget Sound College, 2019_ Swift, Kotlin, Objective-C, Firebase, Xcode

- Google Associate Android Developer
"""


def test_deeper_or_degree_like_headings_are_not_unknown_sections():
    md = (
        "# Priya Raman\n\n## Experience\ntext\n\n"
        "### M.S. in Statistics\n#### Great Lakes University\n"
    )
    assert "unknown" not in labels(split_sections(md))


def test_school_merged_into_the_skills_line_is_split_apart():
    secs = S.by_label(split_sections(MERGED_LINES_MD))
    assert "Puget Sound College, 2019" in secs["education"][0].text
    assert "Puget" not in secs["skills"][0].text
    assert secs["skills"][0].text.startswith("Swift, Kotlin")
