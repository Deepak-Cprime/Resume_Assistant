"""Adaptive per-section chunk budgets, computed once per role and then kept.

For every section type (experience entries, skills, education, ...) the min budget is the
p_low percentile of unit lengths and the max budget the p_high percentile, each clamped:
    min_tok = clamp(P_low,  min_clamp)
    max_tok = clamp(P_high, max_clamp[0], min(max_clamp[1], model_limit - prefix_reserve))
A section type with fewer than `min_units` samples falls back to fixed defaults. Budgets are
stored on the role and recomputed only when the corpus has grown by `growth_trigger`, or a
type has just gathered enough samples, so a resume is not chunked differently tomorrow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src.config import BudgetsConfig, ChunkingConfig

# Pooled over every unit; used for section types that have no budget of their own.
DEFAULT_KEY = "default"
SOURCE_PERCENTILE, SOURCE_DEFAULT = "percentile", "default"
MIN_BODY_TOKENS = 16  # a chunk body budget never shrinks below this, whatever the prefix costs


@dataclass(frozen=True)
class Budget:
    min_tok: int
    max_tok: int
    n_units: int
    source: str  # percentile | default


@dataclass
class Budgets:
    types: dict[str, Budget]
    n_candidates: int
    model_limit: int
    p_low: int
    p_high: int
    prefix_reserve: int
    computed_at: str = ""

    def for_section(self, label: str) -> Budget:
        return self.types.get(label) or self.types[DEFAULT_KEY]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "computed_at": self.computed_at,
            "n_candidates": self.n_candidates,
            "model_limit": self.model_limit,
            "p_low": self.p_low,
            "p_high": self.p_high,
            "prefix_reserve": self.prefix_reserve,
            "types": {
                k: {"min_tok": b.min_tok, "max_tok": b.max_tok, "n_units": b.n_units,
                    "source": b.source}
                for k, b in sorted(self.types.items())
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Budgets:
        return cls(
            types={k: Budget(**v) for k, v in raw["types"].items()},
            n_candidates=raw["n_candidates"],
            model_limit=raw["model_limit"],
            p_low=raw["p_low"],
            p_high=raw["p_high"],
            prefix_reserve=raw["prefix_reserve"],
            computed_at=raw.get("computed_at", ""),
        )

    def fingerprint(self) -> str:
        """Hash of what affects chunk boundaries (not timestamps or sample counts)."""
        core = {
            "limit": self.model_limit,
            "reserve": self.prefix_reserve,
            "types": {k: [b.min_tok, b.max_tok] for k, b in sorted(self.types.items())},
        }
        return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def clamp(value: float, low: float, high: float) -> int:
    return int(round(max(low, min(high, value))))


def _budget(units: list[int], cfg: BudgetsConfig, model_limit: int) -> Budget:
    cap = model_limit - cfg.prefix_reserve
    upper = min(cfg.max_clamp[1] or cap, cap)
    lower = min(cfg.max_clamp[0], upper)
    if len(units) >= cfg.min_units:
        low_value = float(np.percentile(units, cfg.p_low))
        high_value = float(np.percentile(units, cfg.p_high))
        source = SOURCE_PERCENTILE
    else:
        low_value, high_value, source = cfg.min_clamp[0], cfg.default_max, SOURCE_DEFAULT
    max_tok = clamp(high_value, lower, upper)
    min_tok = min(clamp(low_value, *cfg.min_clamp), max_tok // 2)
    return Budget(min_tok, max_tok, len(units), source)


def compute_budgets(
    lengths: dict[str, list[int]],
    *,
    n_candidates: int,
    cfg: ChunkingConfig,
    now: datetime | None = None,
) -> Budgets:
    """Budgets for every section type in `lengths`, plus a pooled `default`."""
    pooled = [n for values in lengths.values() for n in values]
    types = {k: _budget(v, cfg.budgets, cfg.model_limit) for k, v in lengths.items() if v}
    types[DEFAULT_KEY] = _budget(pooled, cfg.budgets, cfg.model_limit)
    return Budgets(
        types=types,
        n_candidates=n_candidates,
        model_limit=cfg.model_limit,
        p_low=cfg.budgets.p_low,
        p_high=cfg.budgets.p_high,
        prefix_reserve=cfg.budgets.prefix_reserve,
        computed_at=(now or datetime.now(UTC)).isoformat(timespec="seconds"),
    )


def needs_recompute(
    stored: Budgets | None,
    lengths: dict[str, list[int]],
    n_candidates: int,
    cfg: ChunkingConfig,
) -> bool:
    if stored is None:
        return True
    b = cfg.budgets
    if (stored.model_limit, stored.p_low, stored.p_high, stored.prefix_reserve) != (
        cfg.model_limit, b.p_low, b.p_high, b.prefix_reserve,
    ):
        return True  # the configuration changed
    if n_candidates > stored.n_candidates * (1 + b.growth_trigger):
        return True
    def had_too_few(label: str) -> bool:  # was a default last time, now has enough samples
        previous = stored.types.get(label)
        return previous is None or previous.n_units < b.min_units

    return any(len(values) >= b.min_units and had_too_few(k) for k, values in lengths.items())
