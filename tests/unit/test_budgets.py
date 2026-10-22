from datetime import UTC, datetime

import pytest

from src.chunking.budgets import (
    Budget,
    Budgets,
    clamp,
    compute_budgets,
    needs_recompute,
)


@pytest.fixture
def cfg(settings):
    return settings.chunking


def spread(n=100, start=50, step=10):
    return list(range(start, start + n * step, step))


def test_clamp():
    assert clamp(10, 64, 128) == 64
    assert clamp(500, 64, 128) == 128
    assert clamp(99.6, 64, 128) == 100


def test_percentiles_are_clamped(cfg):
    b = compute_budgets({"experience": spread()}, n_candidates=40, cfg=cfg)
    exp = b.types["experience"]
    # p20 = 248 -> clamped to 128; p90 = 941 -> clamped to model_limit - reserve = 480
    assert (exp.min_tok, exp.max_tok, exp.n_units, exp.source) == (128, 480, 100, "percentile")


def test_short_units_still_get_sane_budgets(cfg):
    b = compute_budgets({"skills": [40] * 20}, n_candidates=20, cfg=cfg).types["skills"]
    assert (b.min_tok, b.max_tok) == (64, 256)  # lower clamps


def test_mid_range_percentiles_are_used_as_is(cfg):
    units = [100] * 5 + [200] * 5 + [300] * 10
    b = compute_budgets({"experience": units}, n_candidates=10, cfg=cfg).types["experience"]
    assert 64 <= b.min_tok <= 128 and 256 <= b.max_tok <= 480
    assert b.max_tok == 300  # p90 of this sample


def test_too_few_units_use_fixed_defaults(cfg):
    b = compute_budgets({"experience": [100, 900]}, n_candidates=1, cfg=cfg).types["experience"]
    assert (b.min_tok, b.max_tok, b.source) == (64, 400, "default")


def test_upper_clamp_follows_the_model_limit(cfg):
    small = cfg.model_copy(update={"model_limit": 256})
    b = compute_budgets({"experience": spread()}, n_candidates=40, cfg=small).types["experience"]
    assert b.max_tok == 256 - small.budgets.prefix_reserve  # 224, below the usual 256 floor
    assert b.min_tok < b.max_tok


def test_pooled_default_and_fallback_for_other_sections(cfg):
    b = compute_budgets({"experience": spread(), "skills": spread(20)}, n_candidates=40, cfg=cfg)
    assert b.types["default"].n_units == 120
    assert b.for_section("projects") == b.types["default"]
    assert b.for_section("skills") is b.types["skills"]


def test_roundtrip_and_fingerprint(cfg):
    b = compute_budgets({"experience": spread()}, n_candidates=40, cfg=cfg)
    again = Budgets.from_dict(b.to_dict())
    assert again == b and again.fingerprint() == b.fingerprint()


def test_fingerprint_tracks_boundaries_only(cfg):
    b = compute_budgets({"experience": spread()}, n_candidates=40, cfg=cfg)
    later = compute_budgets(
        {"experience": spread()}, n_candidates=99, cfg=cfg, now=datetime(2030, 1, 1, tzinfo=UTC)
    )
    assert b.fingerprint() == later.fingerprint()
    changed = Budgets.from_dict(b.to_dict())
    changed.types["experience"] = Budget(64, 300, 100, "percentile")
    assert changed.fingerprint() != b.fingerprint()


def test_recompute_rules(cfg):
    lengths = {"experience": spread(30)}
    stored = compute_budgets(lengths, n_candidates=10, cfg=cfg)

    assert needs_recompute(None, lengths, 10, cfg)
    assert not needs_recompute(stored, lengths, 10, cfg)
    assert not needs_recompute(stored, lengths, 15, cfg)  # exactly +50%: keep
    assert needs_recompute(stored, lengths, 16, cfg)  # more than +50%: recompute

    other = cfg.model_copy(update={"budgets": cfg.budgets.model_copy(update={"p_high": 95})})
    assert needs_recompute(stored, lengths, 10, other)  # configuration changed


def test_recompute_when_a_type_gains_enough_samples(cfg):
    few = compute_budgets({"experience": [100, 200, 300]}, n_candidates=2, cfg=cfg)
    assert few.types["experience"].source == "default"
    more = {"experience": spread(12)}
    assert needs_recompute(few, more, 2, cfg)  # 12 >= min_units, stored had 3

    enough = compute_budgets(more, n_candidates=6, cfg=cfg)
    assert not needs_recompute(enough, {"experience": spread(40)}, 6, cfg)  # already percentile


def test_new_section_type_with_enough_samples_triggers_recompute(cfg):
    stored = compute_budgets({"experience": spread(30)}, n_candidates=10, cfg=cfg)
    assert needs_recompute(stored, {"experience": spread(30), "skills": spread(12)}, 10, cfg)
