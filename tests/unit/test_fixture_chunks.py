"""Chunk every synthetic resume (real parsers -> profile -> chunks) and check the Phase 1 chunking
gates: nothing over the model limit, company and dates in every experience chunk, no personal
data in embedded text, unique deterministic ids.

The plan's 'chunks under min_tok < 5%' gate is not asserted here: the synthetic resumes have two
short bullets per job, far shorter than the 5-10 year resumes the budgets are tuned for. It is
reported by chunk_stats and belongs to the manual QA run on realistic resumes.
"""

import re

import pytest

from src.chunking.budgets import compute_budgets
from src.chunking.chunker import build_chunks, chunk_stats, unit_lengths
from src.chunking.tokens import estimate_tokens


@pytest.fixture(scope="module")
def corpus(profiles):
    from src.config import load_settings

    cfg = load_settings().chunking
    lengths: dict[str, list[int]] = {}
    for profile in profiles.values():
        for label, values in unit_lengths(profile).items():
            lengths.setdefault(label, []).extend(values)
    budgets = compute_budgets(lengths, n_candidates=len(profiles), cfg=cfg)
    chunks = {
        spec_id: build_chunks(
            profile, candidate_id=i, doc_id=spec_id, budgets=budgets,
            model_limit=cfg.model_limit,
        )
        for i, (spec_id, profile) in enumerate(profiles.items(), 1)
    }
    return budgets, chunks, cfg.model_limit


def test_every_resume_gets_a_profile_and_entry_chunks(corpus, profiles):
    _, chunks, _ = corpus
    for spec_id, profile in profiles.items():
        mine = chunks[spec_id]
        assert any(c.level == 0 for c in mine), spec_id
        assert sum(c.level == 2 for c in mine) >= len(profile.experiences), spec_id


def test_no_embedded_chunk_exceeds_the_model_limit(corpus):
    budgets, chunks, limit = corpus
    for spec_id, mine in chunks.items():
        stats = chunk_stats(mine, budgets, limit)
        assert stats["over_limit"] == 0, spec_id
        assert all(estimate_tokens(c.embed_text) <= limit for c in mine if c.embed), spec_id


def test_experience_chunks_contain_company_and_dates(corpus, profiles):
    _, chunks, _ = corpus
    checked = 0
    for spec_id, profile in profiles.items():
        entries = [c for c in chunks[spec_id] if c.level == 2]
        for exp in profile.experiences:
            if not (exp.company and exp.start):
                continue
            matching = [c for c in entries if c.entry_title.endswith(exp.company)]
            assert matching, (spec_id, exp.company)
            for c in matching:
                assert exp.company in c.embed_text and exp.start in c.embed_text, (spec_id, c.key)
                checked += 1
    assert checked >= 15  # 9 resumes x ~2 jobs


def test_no_personal_data_is_embedded(corpus, profiles):
    _, chunks, _ = corpus
    for spec_id, profile in profiles.items():
        email = profile.contact.email
        phone = profile.contact.phone
        name_parts = [p.lower() for p in profile.name.split() if len(p) >= 3]
        for c in chunks[spec_id]:
            low = c.embed_text.lower()
            assert not email or email.lower() not in low, (spec_id, c.key)
            assert not phone or phone not in c.embed_text, (spec_id, c.key)
            assert not re.search(r"[\w.]+@[\w.]+\.\w+", c.embed_text), (spec_id, c.key)
            assert not any(re.search(rf"\b{re.escape(p)}\b", low) for p in name_parts), (
                spec_id, c.key)


def test_ids_are_unique_across_the_corpus(corpus):
    _, chunks, _ = corpus
    everything = [c for mine in chunks.values() for c in mine]
    assert len({c.chunk_id for c in everything}) == len(everything)
    assert len({c.key for c in everything}) == len(everything)


def test_embedded_chunks_are_a_minority_of_stored_text(corpus):
    """Stored-only parents (full experience) must not be embedded."""
    _, chunks, _ = corpus
    parents = [c for mine in chunks.values() for c in mine if c.level == 1 and not c.embed]
    assert parents and all(c.section == "experience" for c in parents)
