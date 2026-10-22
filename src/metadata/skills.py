"""Skill vocabulary: normalise aliases (k8s -> Kubernetes) and find skills in free text.

The same configs/skills_aliases.yaml feeds the BM25 tokenizer later, so a query for
"golang" and a resume that says "Go" meet on one canonical token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT
from src.structure.text import strip_md

DEFAULT_ALIASES_PATH = PROJECT_ROOT / "configs" / "skills_aliases.yaml"
# A skill never starts after a word character or one of + # . and never continues into
# word characters or + # (so "Java" does not match inside "JavaScript" nor "C" inside "C++").
_BEFORE = r"(?<![\w+#.])"
_AFTER = r"(?![\w+#]|\.\w)"
_LIST_SPLIT = re.compile(r"[,;|•·\n]+")
_CATEGORY_PREFIX = re.compile(r"^\s*[\w &/+#.\-]{2,30}:\s+")
MAX_ITEM_WORDS = 4


@dataclass(frozen=True)
class SkillVocabulary:
    canonical: tuple[str, ...]
    by_term: dict[str, str]  # lowercase term -> canonical name
    exact_case: dict[str, str]  # lowercase term -> the one spelling that counts ("go" -> "Go")
    _exact_re: re.Pattern[str] | None
    _loose_re: re.Pattern[str] | None

    def normalize(self, token: str) -> str | None:
        """Canonical name for one token ('k8s' -> 'Kubernetes'), or None if unknown."""
        token = token.strip(" *_()[]:;,").rstrip(".")
        key = token.lower()
        if key not in self.by_term:
            return None
        if key in self.exact_case and token != self.exact_case[key]:
            return None  # "go" the verb is not the language "Go"
        return self.by_term[key]

    def find(self, text: str) -> list[str]:
        """Canonical skills mentioned in `text`, in order of first occurrence."""
        hits: list[tuple[int, str]] = []
        for rx in (self._exact_re, self._loose_re):
            if rx is not None:
                hits += [(m.start(), self.by_term[m.group(0).lower()]) for m in rx.finditer(text)]
        return list(dict.fromkeys(canon for _, canon in sorted(hits)))

    def split_list(self, text: str) -> list[str]:
        """Items of a skills section ('Languages: Python, Go' -> ['Python', 'Go']).

        Known skills are normalised; unknown short items are kept as written."""
        items: list[str] = []
        for line in text.splitlines():
            line = _CATEGORY_PREFIX.sub("", strip_md(line))
            for part in _LIST_SPLIT.split(line):
                part = part.strip(" *_").rstrip(".")
                if part and len(part.split()) <= MAX_ITEM_WORDS:
                    items.append(self.normalize(part) or part)
        return list(dict.fromkeys(items))

    def looks_like_skill_list(self, line: str, min_items: int = 4) -> bool:
        """A comma-separated line of mostly known skills (for resumes without headings)."""
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < min_items or line.rstrip().endswith("."):
            return False
        return sum(self.normalize(p) is not None for p in parts) / len(parts) >= 0.5


def _compile(terms: set[str], flags: int) -> re.Pattern[str] | None:
    if not terms:
        return None
    ordered = sorted(terms, key=len, reverse=True)  # longest first: "C++" before "C"
    return re.compile(
        f"{_BEFORE}(?:{'|'.join(re.escape(t) for t in ordered)}){_AFTER}", flags
    )


def load_vocabulary(path: Path | str | None = None) -> SkillVocabulary:
    raw = yaml.safe_load(Path(path or DEFAULT_ALIASES_PATH).read_text(encoding="utf-8"))
    by_term: dict[str, str] = {}
    for canon, aliases in raw["skills"].items():
        for term in [canon, *(aliases or [])]:
            by_term[str(term).lower()] = str(canon)
    exact_case = {str(t).lower(): str(t) for t in raw.get("case_sensitive", [])}
    return SkillVocabulary(
        canonical=tuple(dict.fromkeys(by_term.values())),
        by_term=by_term,
        exact_case=exact_case,
        _exact_re=_compile(set(exact_case.values()), 0),
        _loose_re=_compile({t for t in by_term if t not in exact_case}, re.IGNORECASE),
    )


@lru_cache(maxsize=1)
def default_vocabulary() -> SkillVocabulary:
    return load_vocabulary()
