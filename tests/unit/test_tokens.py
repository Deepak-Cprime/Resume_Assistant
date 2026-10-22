import pytest

from src.chunking.tokens import estimate_tokens


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", 0),
        ("a b c", 3),
        ("Kubernetes", 3),  # 10 characters -> 3 tokens, like a word-piece split
        ("Hello, world!", 6),  # 2 + comma + 2 + bang
        ("C++", 3),
        ("3,000", 3),
    ],
)
def test_estimates(text, expected):
    assert estimate_tokens(text) == expected


def test_additive_over_whitespace():
    a, b = "Led migration of 40 services", "to Kubernetes on EKS, cutting deploy time"
    assert estimate_tokens(f"{a} {b}") == estimate_tokens(a) + estimate_tokens(b)
    assert estimate_tokens(f"{a}\n{b}") == estimate_tokens(a) + estimate_tokens(b)


def test_never_below_word_count():
    text = "Built a route optimisation service in Python and PostgreSQL."
    assert estimate_tokens(text) >= len(text.split())
