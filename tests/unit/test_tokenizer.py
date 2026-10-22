import pytest

from src.providers.keyword import tokenizer as T
from src.providers.keyword.tokenizer import simple_tokenize, tech_tokenize


def doc(text):
    return tech_tokenize(text)


def query(text):
    return tech_tokenize(text, is_query=True)


def test_c_sharp_c_plus_plus_and_dot_net_stay_distinct():
    tokens = doc("Built services in C#, C++ and .NET")
    assert sorted(tokens) == sorted(["c#", "c++", ".net", "built", "services"])


def test_a_generic_tokenizer_would_have_merged_them():
    assert simple_tokenize("C# and C++ and .NET") == ["c", "and", "c", "and", "net"]


def test_dotted_names_are_single_tokens():
    assert sorted(doc("Node.js and Next.js apps")) == ["apps", "next.js", "node.js"]
    assert doc("Objective-C, scikit-learn and CI/CD") == ["objective-c", "scikit-learn", "ci/cd"]


@pytest.mark.parametrize(
    "text,token",
    [
        ("k8s", "kubernetes"),
        ("kube", "kubernetes"),
        ("Kubernetes", "kubernetes"),
        ("golang", "go"),
        ("Amazon Web Services", "aws"),
        ("Apache Airflow", "airflow"),
        ("postgres", "postgresql"),
        ("PySpark", "spark"),
    ],
)
def test_aliases_collapse_to_one_token(text, token):
    assert doc(text) == [token] == query(text)


def test_lowercase_go_in_a_resume_is_not_the_language():
    assert doc("We go to market quickly") == ["we", "market", "quickly"]
    assert doc("Wrote services in Go") == ["go", "wrote", "services"]
    assert doc("go live next week") == ["live", "next", "week"]


def test_lowercase_go_in_a_query_is_the_language():
    assert query("go developer") == ["go", "developer"]
    assert query("swift or ruby experience") == ["swift", "ruby", "experience"]


def test_question_words_and_stopwords_are_dropped():
    assert query("who has run Kubernetes in production?") == ["kubernetes", "run", "production"]
    assert query("the and of") == []


def test_hyphenated_words_are_kept_whole_and_split():
    assert doc("full-stack engineer") == ["full-stack", "full", "stack", "engineer"]


def test_numbers_dates_and_apostrophes():
    assert doc("3,000 requests") == ["3,000", "requests"]
    assert doc("2021-03") == ["2021-03", "2021", "03"]
    assert doc("O'Neil") == ["o'neil"]


def test_non_english_letters_survive():
    assert doc("Ingeniería de datos") == ["ingeniería", "de", "datos"]


def test_output_is_lowercase_and_deterministic():
    text = "Senior DevOps Engineer: Docker, Terraform, Python."
    tokens = doc(text)
    assert tokens == doc(text) and all(t == t.lower() for t in tokens)
    assert {"docker", "terraform", "python", "devops", "engineer"} <= set(tokens)


def test_empty_and_symbol_only_text():
    assert doc("") == [] and doc("  |  ---  ") == []


def test_registry():
    assert T.get_tokenizer("tech_aware") is tech_tokenize
    assert T.get_tokenizer("simple") is simple_tokenize
    with pytest.raises(ValueError, match="unknown tokenizer 'nope'"):
        T.get_tokenizer("nope")


def test_fingerprint_names_the_tokenizer_version_and_aliases():
    tech = T.tokenizer_fingerprint("tech_aware")
    assert tech.startswith(f"tech_aware:{T.TOKENIZER_VERSION}:") and len(tech.split(":")) == 3
    assert tech == T.tokenizer_fingerprint("tech_aware")
    assert T.tokenizer_fingerprint("simple") == f"simple:{T.TOKENIZER_VERSION}"
