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


@pytest.mark.parametrize(
    "broken,fixed",
    [
        ("informa-\ntion design", "information design"),  # typesetter hyphenation: glued
        ("customer-\nvisible downtime", "customer-visible downtime"),  # real hyphen: kept
        ("Full-\nstack developer", "Full-stack developer"),
        ("cross-\nfunctional teams", "cross-functional teams"),
        ("linkedin.com/in/ananya-\nkrishnan-example", "linkedin.com/in/ananya-krishnan-example"),
        ("re-\nusable components", "re-usable components"),
    ],
)
def test_a_hyphen_ending_a_line_loses_the_break_but_only_real_word_splits_lose_the_hyphen(
    broken, fixed
):
    assert cleaning.clean_markdown(broken) == fixed


def test_number_ranges_are_left_alone():
    text = "2019-\n2021 and Jan-\n2020"
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


def test_merged_paragraph_gets_its_bullets_back_on_separate_lines():
    raw = "Web Developer, Self-employed •  Built sites for 11 clients. •  Wrote a checkout plugin."
    out = cleaning.clean_markdown(raw)
    assert out.splitlines() == [
        "Web Developer, Self-employed",
        "- Built sites for 11 clients.",
        "- Wrote a checkout plugin.",
    ]


@pytest.mark.parametrize(
    "line", ["- 2018 - 2019 **Web Developer** - Self-employed", "- 2021 - Present **Engineer**"]
)
def test_date_first_entry_is_not_a_list_item(line):
    assert not cleaning.clean_markdown(line).startswith("-")


def test_a_real_bullet_that_starts_with_a_year_keeps_its_marker():
    assert cleaning.clean_markdown("- 2019 revenue grew 20%") == "- 2019 revenue grew 20%"
