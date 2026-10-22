import pytest

from src.structure import entries as E

HEADER_ABOVE = """\
**Senior Backend Engineer, Northwind Payments**
Mar 2021 - Dec 2024 | Austin, TX
- Led migration of 40 services to Kubernetes on EKS, cutting deploy time
  from 45 to 8 minutes.
- Designed an idempotent payment API in Go.

**Software Engineer, Brightline Logistics**
Jun 2018 - Feb 2021 | Dallas, TX
- Built a route optimisation service in Python and PostgreSQL.
""".splitlines()


def test_header_above_dates():
    first, second = E.parse_experiences(HEADER_ABOVE)
    assert (first.title, first.company) == ("Senior Backend Engineer", "Northwind Payments")
    assert (first.start, first.end, first.is_current) == ("2021-03", "2024-12", False)
    assert first.location == "Austin, TX"
    assert len(first.bullets) == 2
    assert first.bullets[0].endswith("from 45 to 8 minutes.")  # wrapped line rejoined
    assert (second.title, second.company) == ("Software Engineer", "Brightline Logistics")
    assert (second.start, second.end) == ("2018-06", "2021-02")
    assert second.bullets == ["Built a route optimisation service in Python and PostgreSQL."]


def test_entry_text_stays_with_its_entry():
    first, second = E.parse_experiences(HEADER_ABOVE)
    assert "Kubernetes" in first.raw_text and "Kubernetes" not in second.raw_text
    assert "Brightline" in second.raw_text and "Brightline" not in first.raw_text


def test_header_and_dates_on_one_line():
    lines = [
        "Software Engineer, Acme Corp  Jan 2019 – Dec 2020",
        "- Shipped things",
        "Data Analyst, Globex  Feb 2021 – Present",
        "- Analysed things",
    ]
    a, b = E.parse_experiences(lines)
    assert (a.title, a.company) == ("Software Engineer", "Acme Corp")
    assert (a.start, a.end) == ("2019-01", "2020-12")
    assert (b.title, b.company, b.is_current, b.end) == ("Data Analyst", "Globex", True, None)


def test_dates_first_then_header():
    lines = ["2019 - 2021", "Software Engineer - Acme", "- Shipped things", "2021 - Present",
             "Staff Engineer - Globex", "- Led things"]
    a, b = E.parse_experiences(lines)
    assert (a.title, a.company, a.start, a.end) == ("Software Engineer", "Acme", "2019", "2021")
    assert (b.title, b.company, b.is_current) == ("Staff Engineer", "Globex", True)
    assert a.bullets == ["Shipped things"] and b.bullets == ["Led things"]


def test_company_and_title_on_separate_lines():
    lines = ["Acme Corp", "Senior Engineer", "Jan 2020 - Dec 2021", "- Did work"]
    (exp,) = E.parse_experiences(lines)
    assert (exp.title, exp.company) == ("Senior Engineer", "Acme Corp")


def test_single_word_title_next_to_dates():
    (exp,) = E.parse_experiences(["Consultant  2020 - 2021", "- Advised clients"])
    assert exp.title == "Consultant" and exp.start == "2020"


def test_section_without_dates_is_one_undated_entry():
    (exp,) = E.parse_experiences(["Worked on many things.", "- A bullet"])
    assert exp.start is None and exp.bullets == ["A bullet"]


def test_prose_with_a_year_is_not_an_entry_header():
    prose = "In 2020 - 2021 we rebuilt the entire platform from scratch with a new team of ten."
    entries = E.split_entries(["Senior Engineer, Acme  2019 - 2021", prose])
    assert len(entries) == 1


@pytest.mark.parametrize(
    "texts,title,company,confidence",
    [
        (["Staff SRE, Orchard Streaming"], "Staff SRE", "Orchard Streaming", "high"),
        (["Orchard Streaming", "Staff SRE"], "Staff SRE", "Orchard Streaming", "high"),
        (["Freelance Consultant, Self-employed"], "Freelance Consultant", "Self-employed", "high"),
        (["Senior Engineer, Acme, Inc."], "Senior Engineer", "Acme, Inc.", "high"),
        (["Engineer at Globex"], "Engineer", "Globex", "high"),
        (["Globex | Data Analyst"], "Data Analyst", "Globex", "high"),
        (["Ingeniera de Datos Sénior, Mediterráneo Energía"],
         "Ingeniera de Datos Sénior", "Mediterráneo Energía", "high"),
        (["Acme Corp, Globex"], "Acme Corp", "Globex", "low"),
        (["Software Engineer"], "Software Engineer", "", "low"),
        (["Acme Corp"], "", "Acme Corp", "low"),
        ([], "", "", "none"),
    ],
)
def test_title_company_split(texts, title, company, confidence):
    assert E.classify_header(texts) == (title, company, confidence)


def test_company_words_that_resemble_titles_are_not_titles():
    # "Analytics" is not "analyst"; "Engineering" is not "engineer"
    assert E.classify_header(["Data Engineer, Charles River Analytics"])[:2] == (
        "Data Engineer", "Charles River Analytics")
    assert E.classify_header(["Mechanic, Acme Engineering"])[2] == "low"


def test_education_two_lines():
    lines = ["**B.Tech in Computer Science**", "Anand Institute of Technology, 2018"]
    (edu,) = E.parse_education(lines)
    assert (edu.degree, edu.school, edu.year) == (
        "B.Tech in Computer Science", "Anand Institute of Technology", 2018)


def test_education_multiple_entries_and_year_ranges():
    lines = ["M.S. in Statistics", "Great Lakes University, 2012 - 2014",
             "B.S. in Mathematics", "State College, 2012"]
    ms, bs = E.parse_education(lines)
    assert (ms.degree, ms.school) == ("M.S. in Statistics", "Great Lakes University")
    assert ms.year == 2014
    assert (bs.degree, bs.school, bs.year) == ("B.S. in Mathematics", "State College", 2012)


def test_education_single_line_and_spanish():
    (a,) = E.parse_education(["B.S. in CS, Rocky Mountain University, 2014"])
    assert (a.degree, a.school, a.year) == ("B.S. in CS", "Rocky Mountain University", 2014)
    (b,) = E.parse_education(["Grado en Matemáticas", "Universidad del Levante, 2018"])
    assert (b.degree, b.school) == ("Grado en Matemáticas", "Universidad del Levante")
    assert b.year == 2018


def test_location_abbreviation_is_not_a_degree():
    assert E.parse_education(["Boston, MA"]) == []


def test_certifications():
    certs = E.parse_certifications([
        "- AWS Certified Solutions Architect (2022)",
        "- Certified Kubernetes Administrator, 2021",
        "Tableau Desktop Specialist",
        "- AWS Certified Solutions Architect (2022)",
    ])
    assert [(c.name, c.year) for c in certs] == [
        ("AWS Certified Solutions Architect", 2022),
        ("Certified Kubernetes Administrator", 2021),
        ("Tableau Desktop Specialist", None),
    ]


def test_flattened_ocr_paragraph_is_split_into_header_and_bullets():
    line = (
        "DevOps Engineer, Peachtree Fintech Jul 2019 - Sep 2022 | Atlanta, GA : Automated AWS "
        "infrastructure with Terraform across 15 accounts. . Built Jenkins pipelines that "
        "shortened releases."
    )
    (exp,) = E.parse_experiences([line])
    assert (exp.title, exp.company, exp.location) == (
        "DevOps Engineer", "Peachtree Fintech", "Atlanta, GA")
    assert (exp.start, exp.end) == ("2019-07", "2022-09")
    assert exp.bullets == [
        "Automated AWS infrastructure with Terraform across 15 accounts.",
        "Built Jenkins pipelines that shortened releases.",
    ]


def test_abbreviations_with_periods_are_not_split_as_bullets():
    line = "Engineer, Acme Inc. Jan 2020 - Dec 2021 | Austin, TX"
    (exp,) = E.parse_experiences([line])
    assert exp.company == "Acme Inc." and exp.bullets == []


def test_degree_and_school_on_one_line():
    (a,) = E.parse_education(["B.E. in Information Technology Georgia Valley College, 2019"])
    assert (a.degree, a.school, a.year) == (
        "B.E. in Information Technology", "Georgia Valley College", 2019)
    (b,) = E.parse_education(["B.Tech in Computer Science Anand Institute of Technology"])
    assert (b.degree, b.school) == ("B.Tech in Computer Science", "Anand Institute of Technology")


def test_certification_with_leading_ocr_punctuation():
    (cert,) = E.parse_certifications([": HashiCorp Certified Terraform Associate"])
    assert cert.name == "HashiCorp Certified Terraform Associate"
