import pymupdf
import pytest

from src.providers.parsers import layout


def page_of(path):
    doc = pymupdf.open(str(path))
    return doc, doc[0]


def test_single_column_has_no_split(fixture_files):
    doc, page = page_of(fixture_files["single_col_backend"])
    assert layout.find_column_split(page) is None
    doc.close()


@pytest.mark.parametrize(
    "spec_id,low,high",
    [("two_col_sidebar", 190, 235), ("two_col_sidebar_right", 340, 400)],
)
def test_split_falls_in_the_gutter(fixture_files, spec_id, low, high):
    doc, page = page_of(fixture_files[spec_id])
    split = layout.find_column_split(page)
    doc.close()
    assert split is not None and low < split < high


def test_column_markdown_reads_main_column_contiguously(fixture_files, specs):
    spec = specs["two_col_sidebar"]
    doc, page = page_of(fixture_files["two_col_sidebar"])
    md = layout.column_markdown(page, layout.find_column_split(page))
    doc.close()
    lines = md.splitlines()
    assert any(ln.startswith("# ") and spec["name"] in ln for ln in lines), "name is level 1"
    assert any(ln.startswith("## ") and "Experience" in ln for ln in lines)
    for job in spec["experience"]:
        for bullet in job["bullets"]:
            assert f"- {bullet}" in md, "wrapped bullet lines should be rejoined"


def test_sidebar_list_items_stay_on_separate_lines(fixture_files, specs):
    doc, page = page_of(fixture_files["two_col_sidebar"])
    md = layout.column_markdown(page, layout.find_column_split(page))
    doc.close()
    lines = [ln.strip() for ln in md.splitlines()]
    for skill in specs["two_col_sidebar"]["skills"]:
        assert skill in lines, f"{skill!r} should be its own line"


def test_order_blocks_puts_full_width_header_first_and_footer_last():
    def block(x0, y0, x1, y1, tag):
        return {"bbox": (x0, y0, x1, y1), "tag": tag}

    blocks = [
        block(300, 100, 550, 200, "right"),
        block(50, 100, 200, 200, "left"),
        block(50, 20, 550, 60, "header"),
        block(50, 700, 550, 740, "footer"),
    ]
    ordered = [b["tag"] for b in layout.order_blocks(blocks, split=250)]
    assert ordered == ["header", "left", "right", "footer"]
