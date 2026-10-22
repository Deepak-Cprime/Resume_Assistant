from datetime import date

import pytest

from src.structure.dates import DateValue, find_range, parse_date, strip_range


@pytest.mark.parametrize(
    "text,iso",
    [
        ("Jan 2021", "2021-01"),
        ("January 2021", "2021-01"),
        ("Mar. 2019", "2019-03"),
        ("Sept. 2021", "2021-09"),
        ("07/2018", "2018-07"),
        ("7.2018", "2018-07"),
        ("2021-03", "2021-03"),
        ("2018", "2018"),
        ("enero de 2020", "2020-01"),
        ("Dic 2019", "2019-12"),
    ],
)
def test_parse_date_formats(text, iso):
    assert parse_date(text).iso() == iso


@pytest.mark.parametrize("text", ["", "hello", "1850", "13/2020", "Present"])
def test_parse_date_rejects(text):
    assert parse_date(text) is None


@pytest.mark.parametrize(
    "text,start,end,current",
    [
        ("Mar 2021 - Dec 2024", "2021-03", "2024-12", False),
        ("Jan 2021 – Present", "2021-01", None, True),
        ("Jan 2021 — current", "2021-01", None, True),
        ("07/2018 - 05/2021", "2018-07", "2021-05", False),
        ("2018 - 2021", "2018", "2021", False),
        ("2020-2022", "2020", "2022", False),
        ("2021-03 to 2023-12", "2021-03", "2023-12", False),
        ("2019 – Present", "2019", None, True),
        ("04/2019 - Actualidad", "2019-04", None, True),
        ("March 2020 – Currently", "2020-03", None, True),
        ("enero de 2018 hasta marzo de 2020", "2018-01", "2020-03", False),
    ],
)
def test_find_range(text, start, end, current):
    rng = find_range(f"Software Engineer | {text} | Austin")
    assert rng is not None
    assert rng.start.iso() == start
    assert (rng.end.iso() if rng.end else None) == end
    assert rng.is_current is current


@pytest.mark.parametrize(
    "text",
    [
        "Python 3.11 and 2020 releases",
        "2022 - 2018",
        "+1 555 0101",
        "555-0101",
        "Led 40 services, 3,000 requests",
    ],
)
def test_no_range_in_ordinary_text(text):
    assert find_range(text) is None


def test_strip_range_returns_surrounding_text():
    text = "Senior Engineer, Acme  Mar 2021 - Dec 2024 | Austin, TX"
    before, after = strip_range(text, find_range(text))
    assert (before, after) == ("Senior Engineer, Acme", "Austin, TX")


def test_year_only_counts_as_mid_year():
    assert DateValue(2018).month_index() == 2018 * 12 + 6
    assert DateValue(2018, 3).month_index() == 2018 * 12 + 2


def test_end_or_uses_as_of_for_current():
    rng = find_range("2021 - Present")
    assert rng.end_or(date(2025, 6, 1)) == DateValue(2025, 6)
