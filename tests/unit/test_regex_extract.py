import pytest

from src.metadata import regex_extract as R


def test_email():
    line = "Contact: Priya.Raman@example.com | +1 555 0101"
    assert R.find_email(line) == "Priya.Raman@example.com"
    assert R.find_email("see priya_raman@mail.example.co.uk.") == "priya_raman@mail.example.co.uk"
    assert R.find_email("no address here") == ""


@pytest.mark.parametrize(
    "text,phone",
    [
        ("call +1 555 0101 today", "+1 555 0101"),
        ("(512) 555-0147", "(512) 555-0147"),
        ("512-555-0147", "512-555-0147"),
        ("+34 612 345 678", "+34 612 345 678"),
        ("Mar 2021 - Dec 2024", ""),
        ("2018 - 2021", ""),
        ("Led 40 services handling 3,000 requests", ""),
        ("", ""),
    ],
)
def test_phone(text, phone):
    assert R.find_phone(text) == phone


def test_links():
    text = "https://www.linkedin.com/in/priya-raman/ and github.com/priyaraman"
    assert R.find_linkedin(text) == "https://www.linkedin.com/in/priya-raman"
    assert R.find_github(text) == "github.com/priyaraman"


@pytest.mark.parametrize(
    "lines,location",
    [
        (["priya@example.com | +1 555 0101 | Austin, TX"], "Austin, TX"),
        (["priya@example.com", "+1 555 0106", "Boston, MA"], "Boston, MA"),
        (["*lucia@example.com | +34 555 0105 | Valencia, España*"], "Valencia, España"),
        (["# Priya Raman", "priya@example.com"], ""),
    ],
)
def test_location(lines, location):
    assert R.find_location(lines) == location


@pytest.mark.parametrize(
    "lines,name",
    [
        (["# Priya Raman", "priya@example.com"], "Priya Raman"),
        (["**Priya Raman**", "priya@example.com"], "Priya Raman"),
        (["PRIYA RAMAN", "priya@example.com"], "Priya Raman"),
        (["## Contact", "a@b.com", "# Aisha Rahman", "## Summary"], "Aisha Rahman"),
        (["Curriculum Vitae", "Priya Raman"], "Priya Raman"),
        (["# Lucía Fernández"], "Lucía Fernández"),
        (["# Contact Information", "Mary-Jane O'Neil"], "Mary-Jane O'Neil"),
        (["priya@example.com | Austin, TX"], ""),
        (["Rahul Verma rahul.verma@example.com | +1 555 0108 | Atlanta, GA"], "Rahul Verma"),
    ],
)
def test_name(lines, name):
    assert R.find_name(lines) == name
