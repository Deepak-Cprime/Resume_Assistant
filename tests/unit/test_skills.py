import pytest

from src.metadata.skills import default_vocabulary

V = default_vocabulary()


@pytest.mark.parametrize(
    "token,canonical",
    [
        ("k8s", "Kubernetes"),
        ("K8S", "Kubernetes"),
        ("golang", "Go"),
        ("Go", "Go"),
        ("postgres", "PostgreSQL"),
        ("Apache Airflow", "Airflow"),
        ("nodejs", "Node.js"),
        ("sklearn", "scikit-learn"),
        ("Amazon Web Services", "AWS"),
        ("C#", "C#"),
        ("c sharp", "C#"),
        (".NET", ".NET"),
        ("ASP.NET", ".NET"),
        ("Python.", "Python"),
    ],
)
def test_normalize(token, canonical):
    assert V.normalize(token) == canonical


@pytest.mark.parametrize("token", ["go", "swift", "excel", "ml", "unknown tool", ""])
def test_normalize_rejects(token):
    assert V.normalize(token) is None


def test_distinct_tokens_stay_distinct():
    found = V.find("Built in C# and C++ with .NET, then JavaScript, Java and Node.js services")
    assert found == ["C#", "C++", ".NET", "JavaScript", "Java", "Node.js"]


def test_find_uses_word_boundaries_and_case_rules():
    assert V.find("JavaScript developer") == ["JavaScript"]
    assert V.find("We go to market quickly and excel at delivery") == []
    assert V.find("Wrote services in Go and golang") == ["Go"]
    assert V.find("POSTGRES, k8s and CI/CD pipelines") == ["PostgreSQL", "Kubernetes", "CI/CD"]


def test_find_orders_by_first_occurrence_without_duplicates():
    assert V.find("Terraform, then Python, then Terraform again") == ["Terraform", "Python"]


def test_split_list_handles_categories_and_separators():
    text = "Languages: Python, Go\nTools: k8s | Terraform\n- Docker\n• Kafka; Spark"
    assert V.split_list(text) == [
        "Python", "Go", "Kubernetes", "Terraform", "Docker", "Kafka", "Spark",
    ]


def test_split_list_keeps_unknown_short_items_and_drops_sentences():
    sentence = "I have worked with many tools every day"
    items = V.split_list(f"Python, Underwater Basket Weaving, {sentence}")
    assert items == ["Python", "Underwater Basket Weaving"]


def test_markdown_emphasis_is_ignored():
    assert V.split_list("_Python, Go_\n**Docker**") == ["Python", "Go", "Docker"]


def test_sidebar_one_per_line():
    assert V.split_list("Python\nSQL\nApache Airflow") == ["Python", "SQL", "Airflow"]


def test_skill_list_detection():
    assert V.looks_like_skill_list("Swift, Kotlin, Objective-C, Firebase, Xcode")
    assert not V.looks_like_skill_list("Austin, TX, USA")
    assert not V.looks_like_skill_list("Python, Go, SQL")  # too few items
    assert not V.looks_like_skill_list("Mobile engineer, Pinecone Apps, Seattle, WA, USA, remote")


def test_find_spans_gives_positions_and_resolves_overlaps():
    text = "Built in Go and golang with Apache Airflow"
    spans = V.find_spans(text)
    assert [(text[s:e], canon) for s, e, canon in spans] == [
        ("Go", "Go"), ("golang", "Go"), ("Apache Airflow", "Airflow")]  # not also bare "Airflow"
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))


def test_find_spans_can_ignore_case_for_queries():
    assert V.find_spans("go developer") == []
    assert V.find_spans("go developer", ignore_case=True) == [(0, 2, "Go")]
