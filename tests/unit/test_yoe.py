from datetime import date

import pytest

from src.metadata import yoe
from src.metadata.schema import Experience

AS_OF = date(2025, 6, 1)


def job(start, end=None, current=False):
    return Experience(start=start, end=end, is_current=current)


def test_single_job():
    years, gaps = yoe.compute_yoe([job("2021-03", "2024-12")], AS_OF)
    assert years == pytest.approx(45 / 12, abs=0.01) and gaps == []


def test_overlap_counts_each_month_once():
    jobs = [job("2019-01", "2022-06"), job("2020-06", "2021-12")]  # freelance inside full-time
    years, _ = yoe.compute_yoe(jobs, AS_OF)
    summed = (41 + 18) / 12
    assert years == pytest.approx(41 / 12, abs=0.01)
    assert years < summed


def test_partial_overlap_extends_the_union():
    years, _ = yoe.compute_yoe([job("2018-01", "2019-12"), job("2019-06", "2021-06")], AS_OF)
    assert years == pytest.approx(41 / 12, abs=0.01)  # Jan 2018 -> Jun 2021


def test_order_in_the_resume_does_not_matter():
    a = [job("2015-02", "2018-03"), job("2019-09", "2023-05")]
    assert yoe.compute_yoe(a, AS_OF) == yoe.compute_yoe(list(reversed(a)), AS_OF)


def test_gap_longer_than_three_months_is_reported():
    years, gaps = yoe.compute_yoe([job("2015-02", "2018-03"), job("2019-09", "2023-05")], AS_OF)
    assert years == pytest.approx(81 / 12, abs=0.01)
    assert [(g.start, g.end, g.months) for g in gaps] == [("2018-03", "2019-09", 18)]


def test_short_gap_is_ignored_and_threshold_is_configurable():
    jobs = [job("2018-01", "2018-12"), job("2019-04", "2020-01")]  # 4-month gap
    assert yoe.compute_yoe(jobs, AS_OF)[1][0].months == 4
    assert yoe.compute_yoe(jobs, AS_OF, gap_months=6)[1] == []
    assert yoe.compute_yoe([job("2018-01", "2018-12"), job("2019-02", "2020-01")], AS_OF)[1] == []


def test_current_job_runs_to_as_of():
    years, _ = yoe.compute_yoe([job("2023-11", None, current=True)], AS_OF)
    assert years == pytest.approx(19 / 12, abs=0.01)


def test_year_only_dates_count_mid_year():
    years, _ = yoe.compute_yoe([job("2018", "2021")], AS_OF)
    assert years == pytest.approx(3.0, abs=0.01)


def test_jobs_without_a_start_date_are_skipped():
    years, _ = yoe.compute_yoe([job(None), job("2020-01", "2021-01")], AS_OF)
    assert years == pytest.approx(1.0, abs=0.01)


def test_no_jobs():
    assert yoe.compute_yoe([], AS_OF) == (0.0, [])


def test_months_covered_matches_union():
    spans = yoe.intervals([job("2019-01", "2022-06"), job("2020-06", "2021-12")], AS_OF)
    assert yoe.months_covered(spans) == 41
