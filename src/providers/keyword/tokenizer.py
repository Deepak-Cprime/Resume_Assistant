"""Tokenizers for the keyword index.

`tech_aware` is built for resumes. A generic tokenizer splits "C#", "C++", ".NET" and "Node.js"
into "c", "net", "node", "js" and loses what a recruiter is searching for. Here every skill
mention (from configs/skills_aliases.yaml) becomes ONE token named after its canonical skill,
so "k8s", "kube" and "Kubernetes" all index as `kubernetes`, and `c#`, `c++` and `.net` stay
distinct. The same function tokenizes documents and queries; the indexer and the searcher
therefore always agree on aliases.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from src.metadata.skills import DEFAULT_ALIASES_PATH, SkillVocabulary, default_vocabulary

TOKENIZER_VERSION = "1"  # bump when tokenizing rules change: indexes built before are stale

# "3,000" and "1.2" stay whole; words keep inner hyphens/apostrophes ("full-stack", "o'neil")
_TOKEN = re.compile(r"\d+(?:[.,]\d+)+|[^\W_]+(?:['’-][^\W_]+)*")
MIN_PART_CHARS = 2
STOPWORDS = frozenset(
    "a an and the of to in on for with by at is are was were be as or from that this it its "
    "into than then who whom has have had which what do does did any all me".split()
)

Tokenizer = Callable[..., list[str]]


def tech_tokenize(
    text: str, *, is_query: bool = False, vocab: SkillVocabulary | None = None
) -> list[str]:
    """Skill tokens first (canonical, lowercase), then ordinary lowercase word tokens.

    In documents a lowercase "go" or "swift" is an ordinary word, so it is dropped rather than
    mistaken for the language; in queries (`is_query`) people type "go developer" in lowercase,
    so those words are accepted as skills."""
    vocab = vocab or default_vocabulary()
    tokens: list[str] = []
    rest: list[str] = []
    cursor = 0
    for start, end, canonical in vocab.find_spans(text, ignore_case=is_query):
        rest.append(text[cursor:start])
        tokens.append(canonical.lower())
        cursor = end
    rest.append(text[cursor:])

    ambiguous = frozenset() if is_query else frozenset(vocab.exact_case)
    for match in _TOKEN.finditer(" ".join(rest).lower()):
        word = match.group(0)
        if word in STOPWORDS or word in ambiguous:
            continue
        tokens.append(word)
        if "-" in word:  # "full-stack" is searchable as itself and as "full" / "stack"
            tokens += [
                part for part in word.split("-")
                if len(part) >= MIN_PART_CHARS and part not in STOPWORDS
            ]
    return tokens


def simple_tokenize(text: str, *, is_query: bool = False) -> list[str]:
    """Plain lowercase word split: the baseline `tech_aware` is compared against."""
    return re.findall(r"\w+", text.lower())


TOKENIZERS: dict[str, Tokenizer] = {"tech_aware": tech_tokenize, "simple": simple_tokenize}


def get_tokenizer(name: str) -> Tokenizer:
    try:
        return TOKENIZERS[name]
    except KeyError:
        raise ValueError(f"unknown tokenizer '{name}' (known: {', '.join(TOKENIZERS)})") from None


def tokenizer_fingerprint(name: str) -> str:
    """Identity of a tokenizer's behaviour: name, code version and, for `tech_aware`, a hash of
    the alias file. An index built under a different fingerprint no longer matches queries."""
    if name != "tech_aware":
        return f"{name}:{TOKENIZER_VERSION}"
    aliases = hashlib.sha256(DEFAULT_ALIASES_PATH.read_bytes()).hexdigest()[:12]
    return f"{name}:{TOKENIZER_VERSION}:{aliases}"
