from datetime import date

import pytest

from src.metadata.profile import build_profile, seniority
from tests.unit.fakes import FULL_MD, NO_HEADINGS_MD, SIDEBAR_FIRST_MD, FakeLLM

AS_OF = date(2025, 6, 1)


@pytest.fixture(scope="module")
def full():
    return build_profile(FULL_MD, as_of=AS_OF)


def test_contact_and_name(full):
    assert full.name == "Priya Raman"
    assert full.contact.email == "priya.raman@example.com"
    assert full.contact.phone == "+1 555 0101"
    assert full.contact.location == "Austin, TX"
    assert len(full.email_hash) == 64


def test_experiences(full):
    first, second = full.experiences
    assert (first.title, first.company) == ("Senior Backend Engineer", "Northwind Payments")
    assert (first.start, first.end) == ("2021-03", "2024-12")
    assert (second.title, second.company) == ("Software Engineer", "Brightline Logistics")
    assert (second.start, second.end) == ("2018-06", "2021-02")
    assert all(e.confidence == "high" for e in full.experiences)


def test_totals(full):
    assert full.total_yoe == pytest.approx(77 / 12, abs=0.01)
    assert full.gaps == []
    assert full.headline == "Senior Backend Engineer"
    assert full.seniority == "senior"


def test_education_and_certifications(full):
    (edu,) = full.education
    assert (edu.degree, edu.school, edu.year) == (
        "B.Tech in Computer Science", "Anand Institute of Technology", 2018)
    assert [c.name for c in full.certifications] == ["AWS Certified Solutions Architect"]


def test_skills_distinguish_claimed_from_used(full):
    skills = {s.skill: s for s in full.skills}
    assert skills["AWS"].claimed and not skills["AWS"].used  # listed, never shown in a job
    assert skills["GitHub Actions"].used and not skills["GitHub Actions"].claimed
    assert skills["Kubernetes"].claimed and skills["Kubernetes"].used
    assert skills["Kubernetes"].evidence == "used" and skills["AWS"].evidence == "claimed"
    assert skills["Go"].claimed and skills["Go"].used  # summary and skills section agree
    assert skills["Kubernetes"].years == pytest.approx(45 / 12, abs=0.1)  # only the newer job
    assert skills["Python"].years == pytest.approx(32 / 12, abs=0.1)  # only the older job
    assert skills["AWS"].years is None


def test_clean_resume_has_no_warnings(full):
    assert full.warnings == []


def test_sections_are_recorded(full):
    assert [s.label for s in full.sections] == [
        "header", "summary", "experience", "education", "skills", "certifications"]
    assert "Northwind Payments" in next(s for s in full.sections if s.label == "experience").text


def test_resume_without_headings():
    p = build_profile(NO_HEADINGS_MD, as_of=AS_OF)
    assert p.name == "Tomas Becker"
    assert [(e.title, e.company) for e in p.experiences] == [
        ("Mobile Engineer", "Pinecone Apps"), ("Senior Mobile Engineer", "Cascade Fitness")]
    assert p.education[0].degree == "B.S. in Computer Science"
    assert {"Swift", "Kotlin"} <= {s.skill for s in p.skills}
    assert [c.name for c in p.certifications] == ["Google Associate Android Developer"]
    assert p.total_yoe == pytest.approx(61 / 12, abs=0.01)


def test_sidebar_first_layout():
    p = build_profile(SIDEBAR_FIRST_MD, as_of=AS_OF)
    assert p.name == "Aisha Rahman"
    assert p.contact.email == "aisha.rahman@example.com" and p.contact.location == "Boston, MA"
    assert [e.company for e in p.experiences] == ["Beacon Biotech"]
    assert {"Python", "SQL", "Airflow"} <= {s.skill for s in p.skills}


def test_missing_sections_produce_warnings():
    md = "# Jane Doe\n\n## Skills\nPython, Go, SQL\n\n## Summary\nEngineer."
    p = build_profile(md, as_of=AS_OF)
    assert "no experience found" in p.warnings and "no education found" in p.warnings


def test_unknown_section_is_reported():
    md = FULL_MD + "\n## Volunteer Work\nFood bank volunteer for three years running now.\n"
    p = build_profile(md, as_of=AS_OF)
    assert "unknown section: Volunteer Work" in p.warnings


def test_llm_resolves_an_ambiguous_entry():
    md = ("## Experience\nAcme Corp, Globex\nJan 2020 - Dec 2021\n- Built things\n\n"
          "## Skills\nPython, Go, SQL\n")
    rules = build_profile(md, as_of=AS_OF)
    assert rules.experiences[0].confidence == "low"
    assert any("uncertain" in w for w in rules.warnings)

    llm = FakeLLM({"title": "Globex", "company": "Acme Corp"})
    enriched = build_profile(md, as_of=AS_OF, llm=llm)
    exp = enriched.experiences[0]
    assert (exp.title, exp.company, exp.source) == ("Globex", "Acme Corp", "llm")
    assert not any("uncertain" in w for w in enriched.warnings)


def test_llm_is_not_called_when_rules_are_confident():
    llm = FakeLLM({"title": "x", "company": "y"})
    build_profile(FULL_MD, as_of=AS_OF, llm=llm)
    assert llm.prompts == []


@pytest.mark.parametrize(
    "title,yoe,level",
    [
        ("Software Engineer", 1.0, "junior"),
        ("Software Engineer", 3.0, "mid"),
        ("Software Engineer", 8.0, "senior"),
        ("Senior Engineer", 1.0, "senior"),
        ("Engineering Manager", 4.0, "lead"),
        ("Staff SRE", 4.0, "principal"),
        ("Director of Engineering", 12.0, "director"),
        ("", 0.0, "junior"),
    ],
)
def test_seniority(title, yoe, level):
    assert seniority(title, yoe) == level


def test_merged_school_and_skills_line_yields_clean_education_and_skills():
    from tests.unit.test_sections import MERGED_LINES_MD

    p = build_profile(MERGED_LINES_MD, as_of=AS_OF)
    (edu,) = p.education
    assert (edu.degree, edu.school) == ("B.S. in Computer Science", "Puget Sound College")
    assert edu.year == 2019
    skills = {s.skill for s in p.skills}
    assert {"Swift", "Kotlin", "Objective-C"} <= skills
    assert not any("2019" in s or "Puget" in s for s in skills)


def test_ocr_flattened_resume():
    md = (
        "Rahul Verma rahul.verma@example.com | +1 555 0108 | Atlanta, GA\n\n"
        "## Summary\n\nDevOps engineer automating cloud infrastructure for fintech teams.\n\n"
        "## Experience\n\n"
        "DevOps Engineer, Peachtree Fintech Jul 2019 - Sep 2022 | Atlanta, GA : Automated AWS "
        "infrastructure with Terraform. . Built Jenkins pipelines.\n\n"
        "Senior DevOps Engineer, Southern Ledger Oct 2022 - Mar 2025 | Atlanta, GA : Moved "
        "deployments to Docker and Kubernetes.\n\n"
        "### Education\n\nB.E. in Information Technology Georgia Valley College, 2019\n\n"
        "# Skills\n\nAWS, Terraform, Docker, Kubernetes, Jenkins\n\n"
        "### Certifications\n\n: HashiCorp Certified Terraform Associate\n"
    )
    p = build_profile(md, as_of=AS_OF)
    assert p.name == "Rahul Verma" and p.contact.location == "Atlanta, GA"
    assert [(e.title, e.company) for e in p.experiences] == [
        ("DevOps Engineer", "Peachtree Fintech"), ("Senior DevOps Engineer", "Southern Ledger")]
    assert p.total_yoe == pytest.approx(67 / 12, abs=0.01)
    assert (p.education[0].degree, p.education[0].school) == (
        "B.E. in Information Technology", "Georgia Valley College")
    assert [c.name for c in p.certifications] == ["HashiCorp Certified Terraform Associate"]
