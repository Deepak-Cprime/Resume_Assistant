from src.ingestion import validation

GOOD = " ".join(["engineer"] * 60) + " someone@example.com"


def test_good_text_has_no_issues(settings):
    report = validation.validate_markdown(GOOD, 1, settings)
    assert not report.fatal
    assert report.issues == []


def test_empty_is_fatal(settings):
    report = validation.validate_markdown("  \n ", 1, settings)
    assert report.fatal and report.codes == ["empty"]


def test_too_little_text_per_page_is_fatal(settings):
    report = validation.validate_markdown("Jane Doe\nEngineer", 2, settings)
    assert report.fatal
    assert {"no_text", "too_short"} <= set(report.codes)


def test_replacement_characters_are_fatal(settings):
    report = validation.validate_markdown(GOOD + " �" * 40, 1, settings)
    assert report.fatal and "garbage" in report.codes


def test_cid_glyph_ids_count_as_garbage(settings):
    report = validation.validate_markdown(GOOD + " (cid:12)" * 20, 1, settings)
    assert "garbage" in report.codes


def test_small_garbage_is_only_a_warning(settings):
    report = validation.validate_markdown(GOOD + " �" * 8, 1, settings)
    assert not report.fatal and "some_garbage" in report.codes


def test_mostly_symbols_is_fatal(settings):
    report = validation.validate_markdown("1234 5678 " * 100 + "a@b.c", 1, settings)
    assert report.fatal and "low_alpha" in report.codes


def test_missing_email_is_a_warning(settings):
    report = validation.validate_markdown(" ".join(["engineer"] * 60), 1, settings)
    assert not report.fatal and report.codes == ["no_email"]
