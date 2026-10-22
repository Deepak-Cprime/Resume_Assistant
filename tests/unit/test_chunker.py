import re
from datetime import date

import pytest

from src.chunking import chunker as C
from src.chunking.budgets import Budget, Budgets, compute_budgets
from src.chunking.tokens import estimate_tokens
from src.metadata.profile import build_profile
from src.metadata.schema import CandidateProfile, Experience, SectionInfo
from src.stores import db
from tests.unit.fakes import FULL_MD

AS_OF = date(2025, 6, 1)


@pytest.fixture(scope="module")
def profile():
    return build_profile(FULL_MD, as_of=AS_OF)


@pytest.fixture(scope="module")
def defaults():
    from src.config import load_settings

    cfg = load_settings().chunking
    return compute_budgets({}, n_candidates=1, cfg=cfg)  # fixed defaults: min 64, max 400


def fixed_budgets(min_tok=20, max_tok=60, limit=512) -> Budgets:
    b = Budget(min_tok, max_tok, 0, "default")
    return Budgets({"default": b, "experience": b}, 1, limit, 20, 90, 32)


def build(profile, budgets, *, limit=512, cid=12):
    return C.build_chunks(profile, candidate_id=cid, doc_id="doc1", budgets=budgets,
                          model_limit=limit)


def by_key(chunks):
    return {c.key: c for c in chunks}


# --- the standard resume ---------------------------------------------------------------


def test_chunk_set_for_a_typical_resume(profile, defaults):
    chunks = build(profile, defaults)
    assert [c.key for c in chunks] == [
        "c12-profile",
        "c12-experience",
        "c12-exp-1-p1",
        "c12-exp-2-p1",
        "c12-education-skills-certifications",  # three short sections merged into one
    ]
    assert [c.level for c in chunks] == [0, 1, 2, 2, 1]
    assert [c.embed for c in chunks] == [True, False, True, True, True]


def test_entry_chunk_carries_company_dates_and_tags(profile, defaults):
    entry = by_key(build(profile, defaults))["c12-exp-1-p1"]
    assert entry.embed_text.startswith(
        "Candidate C12 | Experience > Senior Backend Engineer @ Northwind Payments "
        "(2021-03 – 2024-12)\n")
    assert "Kubernetes" in entry.embed_text
    assert entry.embed_text.splitlines()[-1] == "Skills: Kubernetes, Go"
    assert entry.skills == ["Kubernetes", "Go"]
    assert (entry.entry_title, entry.start, entry.end) == (
        "Senior Backend Engineer @ Northwind Payments", "2021-03", "2024-12")


def test_text_and_embed_text_are_separate(profile, defaults):
    entry = by_key(build(profile, defaults))["c12-exp-1-p1"]
    assert entry.text.splitlines()[0] == (
        "Senior Backend Engineer @ Northwind Payments (2021-03 – 2024-12)")
    assert "- Led migration of 40 services" in entry.text
    assert "Candidate C12" not in entry.text and "Skills:" not in entry.text


def test_every_experience_chunk_has_company_and_dates(profile, defaults):
    entries = [c for c in build(profile, defaults) if c.level == 2]
    assert len(entries) == 2
    for chunk, exp in zip(entries, profile.experiences, strict=True):
        assert exp.company in chunk.embed_text
        assert exp.start in chunk.embed_text and exp.end in chunk.embed_text


def test_current_job_is_marked_present(defaults):
    p = CandidateProfile(name="A B", experiences=[
        Experience(title="Engineer", company="Acme", start="2023-11", is_current=True,
                   bullets=["Did things"])])
    entry = by_key(build(p, defaults, cid=1))["c1-exp-1-p1"]
    assert "(2023-11 – present)" in entry.embed_text and entry.end == "present"


def test_no_personal_data_in_embedded_text(profile, defaults):
    for chunk in build(profile, defaults):
        low = chunk.embed_text.lower()
        for secret in ("priya", "raman", "555 0101", "example.com"):
            assert secret not in low, (chunk.key, secret)
        assert not re.search(r"[\w.]+@[\w.]+\.\w+", chunk.embed_text), chunk.key  # 'Title @ Co' ok


def test_parent_links(profile, defaults):
    rows = {c.key: c.to_row() for c in build(profile, defaults)}
    parent_id = rows["c12-experience"]["chunk_id"]
    assert rows["c12-exp-1-p1"]["parent_id"] == parent_id == C.chunk_id_for("c12-experience")
    assert rows["c12-exp-2-p1"]["parent_id"] == parent_id
    assert rows["c12-profile"]["parent_id"] is None
    assert "Northwind Payments" in by_key(build(profile, defaults))["c12-experience"].text


def test_profile_chunk_content(profile, defaults):
    chunk = by_key(build(profile, defaults))["c12-profile"]
    assert chunk.level == 0 and chunk.embed
    for expected in ("Senior Backend Engineer", "Senior level, 6.4 years of experience",
                     "Backend engineer with six years", "Top skills:", "Kubernetes"):
        assert expected in chunk.text
    assert chunk.embed_text.startswith("Candidate C12 | Profile\n")


def test_ids_are_deterministic_unique_and_fit_sqlite_and_faiss(profile, defaults):
    first, second = build(profile, defaults), build(profile, defaults)
    assert [(c.key, c.chunk_id, c.content_hash) for c in first] == [
        (c.key, c.chunk_id, c.content_hash) for c in second]
    ids = [c.chunk_id for c in first]
    assert len(set(ids)) == len(ids) and all(0 < i < 2**63 for i in ids)
    assert build(profile, defaults, cid=13)[0].chunk_id != ids[0]


def test_content_hash_tracks_embedded_text(profile, defaults):
    chunk = by_key(build(profile, defaults))["c12-profile"]
    changed = profile.model_copy(update={"headline": "Other"})
    other = by_key(build(changed, defaults))["c12-profile"]
    assert chunk.content_hash != other.content_hash


def test_rows_match_the_database_columns(profile, defaults):
    row = build(profile, defaults)[0].to_row()
    assert set(row) == set(db._CHUNK_COLUMNS) - {"role_id"}


def test_stats(profile, defaults):
    chunks = build(profile, defaults)
    stats = C.chunk_stats(chunks, defaults, 512)
    assert stats["chunks"] == 5 and stats["embedded"] == 4 and stats["over_limit"] == 0
    assert stats["avg_tokens"] > 0 and 0 <= stats["under_min_share"] <= 1


def test_unit_lengths_cover_entries_and_sections_but_not_summary(profile):
    lengths = C.unit_lengths(profile)
    assert sorted(lengths) == ["certifications", "education", "experience", "skills"]
    assert len(lengths["experience"]) == 2


# --- splitting, packing, merging --------------------------------------------------------


def long_job(n_bullets=12):
    bullets = [f"Built service number {i} that handled many requests per second reliably"
               for i in range(n_bullets)]
    return Experience(title="Engineer", company="Acme", start="2020-01", end="2021-01",
                      bullets=bullets)


def test_long_entry_splits_on_bullets_and_repeats_the_header():
    p = CandidateProfile(name="A B", experiences=[long_job()])
    chunks = [c for c in build(p, fixed_budgets(), cid=1) if c.level == 2]
    assert len(chunks) >= 4
    assert [c.key for c in chunks] == [f"c1-exp-1-p{i}" for i in range(1, len(chunks) + 1)]
    prefix = "Candidate C1 | Experience > Engineer @ Acme (2020-01 – 2021-01)"
    for c in chunks:
        assert c.embed_text.startswith(prefix + "\n")
        assert c.text.startswith("Engineer @ Acme (2020-01 – 2021-01)\n")
    bullets = [line for c in chunks for line in c.text.splitlines() if line.startswith("- ")]
    assert bullets == [f"- {b}" for b in long_job().bullets]  # each bullet once, in order


def test_split_parts_respect_the_body_budget():
    p = CandidateProfile(name="A B", experiences=[long_job()])
    for c in (c for c in build(p, fixed_budgets(max_tok=60), cid=1) if c.level == 2):
        body = "\n".join(c.text.splitlines()[1:])
        assert estimate_tokens(body) <= 60


def test_nothing_exceeds_a_small_model_limit(profile):
    cfg_budgets = fixed_budgets(min_tok=20, max_tok=96, limit=128)
    chunks = build(profile, cfg_budgets, limit=128)
    assert chunks and all(c.n_tokens <= 128 for c in chunks if c.embed)
    p = CandidateProfile(name="A B", experiences=[long_job(40)])
    assert all(c.n_tokens <= 128 for c in build(p, cfg_budgets, limit=128, cid=1) if c.embed)


def test_one_huge_bullet_is_split_without_losing_words():
    text = " ".join(f"word{i}" for i in range(300))
    parts = C.split_oversize(text, 60, estimate_tokens)
    assert len(parts) > 1 and all(estimate_tokens(p) <= 60 for p in parts)
    assert " ".join(parts).split() == text.split()


def test_oversize_unit_prefers_sentence_boundaries():
    unit = "First sentence here. Second sentence here. Third sentence here."
    assert C.split_oversize(unit, 8, estimate_tokens) == [
        "First sentence here.", "Second sentence here.", "Third sentence here."]


def words(s):
    return len(s.split())


def test_pack_keeps_units_whole_and_fills_greedily():
    units = ["a " * 10] * 5
    parts = C.pack([u.strip() for u in units], budget=25, min_tok=5, count=words)
    assert [words(p) for p in parts] == [20, 20, 10]


def test_pack_rebalances_a_tiny_last_part():
    units = ["a " * 10, "b " * 10, "c " * 10, "d " * 10, "e " * 2]
    parts = C.pack([u.strip() for u in units], budget=40, min_tok=8, count=words)
    assert [words(p) for p in parts] == [30, 12]  # last bullet group moved to avoid a 2-word tail


def test_pack_does_not_starve_the_previous_part():
    units = ["a " * 10, "b " * 2]
    parts = C.pack([u.strip() for u in units], budget=10, min_tok=5, count=words)
    assert [words(p) for p in parts] == [10, 2]  # moving would drop the first part below min


def test_short_sections_merge_but_long_ones_stand_alone():
    sections = [
        SectionInfo(label="education", method="synonym", text="B.S. CS, State University, 2012"),
        SectionInfo(label="skills", method="synonym",
                    text=", ".join(f"Skill{i}" for i in range(12))),
        SectionInfo(label="certifications", method="synonym", text="AWS Certified Architect"),
        SectionInfo(label="languages", method="synonym", text="English, Spanish"),
    ]
    p = CandidateProfile(name="A B", sections=sections)
    chunks = [c for c in build(p, fixed_budgets(min_tok=20), cid=1) if c.level == 1]
    assert [c.key for c in chunks] == [
        "c1-education", "c1-skills", "c1-certifications-languages"]
    merged = chunks[-1]
    assert merged.section == "certifications+languages"
    assert "Certifications:\nAWS Certified Architect" in merged.text
    assert "Languages:\nEnglish, Spanish" in merged.text
    assert merged.embed_text.startswith("Candidate C1 | Certifications + Languages\n")


def test_experience_entries_are_never_merged_with_each_other(defaults):
    p = CandidateProfile(name="A B", experiences=[
        Experience(title="Engineer", company="A", start="2020-01", end="2020-02", bullets=["x"]),
        Experience(title="Engineer", company="B", start="2020-03", end="2020-04", bullets=["y"]),
    ])
    entries = [c for c in build(p, defaults, cid=1) if c.level == 2]
    assert [c.key for c in entries] == ["c1-exp-1-p1", "c1-exp-2-p1"]
    assert [c.entry_title for c in entries] == ["Engineer @ A", "Engineer @ B"]
    assert "- x" in entries[0].text and "- y" not in entries[0].text


def test_unrecognised_sections_become_other(defaults):
    p = CandidateProfile(name="A B", sections=[
        SectionInfo(label="unknown", heading="Hobbies Club", method="unknown", text="Chess")])
    (chunk,) = [c for c in build(p, defaults, cid=1) if c.level == 1]
    assert chunk.section == "other" and chunk.key == "c1-other"


def test_header_contact_and_reference_sections_are_not_chunked(defaults):
    sections = [SectionInfo(label=lbl, method="synonym", text="some text here")
                for lbl in ("header", "contact", "references")]
    p = CandidateProfile(name="A B", sections=sections)
    assert build(p, defaults, cid=1) == []


def test_undated_entry_and_empty_resume(defaults):
    p = CandidateProfile(name="A B", experiences=[
        Experience(title="Engineer", company="Acme", bullets=["Did things"])])
    entry = by_key(build(p, defaults, cid=1))["c1-exp-1-p1"]
    assert entry.embed_text.splitlines()[0] == "Candidate C1 | Experience > Engineer @ Acme"
    assert build(CandidateProfile(name="A B"), defaults) == []


def test_entry_without_bullets_still_gets_a_chunk(defaults):
    p = CandidateProfile(name="A B", experiences=[
        Experience(title="Engineer", company="Acme", start="2020-01", end="2021-01")])
    entry = by_key(build(p, defaults, cid=1))["c1-exp-1-p1"]
    assert entry.text == "Engineer @ Acme (2020-01 – 2021-01)"
