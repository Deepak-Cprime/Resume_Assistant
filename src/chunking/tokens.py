"""A model-free token estimate for chunk budgets.

Chunks are shared by several embedding models with different tokenizers, so budgets use one
conservative estimate (it over-counts word-piece tokenizers on purpose) and each model's real
tokenizer verifies the 'no chunk over the limit' rule when vectors are built (M6).
"""

from __future__ import annotations

import math
import re

_WORD = re.compile(r"\w+")
_PUNCTUATION = re.compile(r"[^\w\s]")
CHARS_PER_TOKEN = 4  # word-piece tokenizers average well below this on technical text


def estimate_tokens(text: str) -> int:
    """A word costs one token per ~4 characters (at least one); each punctuation mark costs one.

    Additive over whitespace: estimate(a + ' ' + b) == estimate(a) + estimate(b), which keeps
    packing arithmetic exact."""
    words = sum(max(1, math.ceil(len(w) / CHARS_PER_TOKEN)) for w in _WORD.findall(text))
    return words + len(_PUNCTUATION.findall(text))
