import pytest

from src.ingestion import cleaning


def test_ligatures_and_invisible_characters():
    raw = "ﬁnance and ﬂow, o­ffice​ space"
    assert cleaning.clean_markdown(raw) == "finance and flow, office space"


@pytest.mark.parametrize("marker", ["•", "▪", "◦", "●", "➢", "–", "*", "-", ":", "."])
def test_bullets_become_dash(marker):
    assert cleaning.clean_markdown(f"{marker} Built a service") == "- Built a service"


def test_indented_bullet_keeps_indent():
    assert cleaning.clean_markdown("Parent\n  • Nested item") == "Parent\n  - Nested item"


def test_bold_and_plain_dashes_untouched():
    text = "**Senior Engineer** - Acme\n---\nwell-known"
    assert cleaning.clean_markdown(text) == text


def test_hyphenated_line_break_is_rejoined():
    text = "strong exper-\nience in design"
    assert cleaning.clean_markdown(text) == "strong experience in design"


def test_capitalised_hyphen_break_and_ranges_kept():
    text = "Full-\nstack developer\n2019-\n2021"
    assert cleaning.clean_markdown(text) == text


def test_repeated_footer_removed_content_kept():
    pages = [
        "Experience\n- Built APIs\nJane Doe - Page 1 of 3",
        "- Led a team\nJane Doe - Page 2 of 3",
        "Education\nB.S. Computer Science\nJane Doe - Page 3 of 3",
    ]
    out = cleaning.clean_pages(pages)
    assert "Page" not in out
    for kept in ("Built APIs", "Led a team", "B.S. Computer Science"):
        assert kept in out


def test_page_numbers_removed():
    out = cleaning.clean_pages(["Intro text\n1", "More text\n2"])
    assert out.split() == ["Intro", "text", "More", "text"]


def test_single_page_never_drops_lines():
    assert cleaning.clean_pages(["Jane Doe\nJane Doe"]) == "Jane Doe\nJane Doe"


def test_blank_lines_collapsed_and_trailing_space_stripped():
    assert cleaning.clean_markdown("a  \n\n\n\n\nb\n\n") == "a\n\nb"


def test_cleaning_is_idempotent():
    raw = "• ﬁrst  \n\n\n\n– exper-\nience\n"
    once = cleaning.clean_markdown(raw)
    assert cleaning.clean_markdown(once) == once
