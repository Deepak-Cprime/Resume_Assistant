import pytest

from src.providers.base import KeywordIndex
from src.providers.keyword.bm25_index import Bm25Index

pytest.importorskip("bm25s")

DOCS = {
    101: "Candidate C1 | Experience > Developer @ Acme. Built desktop apps in C# for Windows.",
    102: "Candidate C2 | Experience > Engineer @ Beta. Wrote embedded firmware in C++ for devices.",
    103: "Candidate C3 | Experience > Developer @ Gamma. Maintained .NET web services.",
    104: "Candidate C4 | Experience > Engineer @ Delta. Built Java backends with Spring Boot.",
    105: "Candidate C5 | Skills: Python, Terraform, Ansible. Ran k8s with Terraform modules.",
}


def build(tokenizer="tech_aware", docs=None) -> Bm25Index:
    docs = docs or DOCS
    index = Bm25Index(tokenizer=tokenizer)
    index.build(list(docs), list(docs.values()))
    return index


def found(index, query, k=10):
    return [h.chunk_id for h in index.search(query, k)]


def test_is_a_keyword_index():
    assert isinstance(Bm25Index(), KeywordIndex)


@pytest.mark.parametrize(
    "query,expected",
    [("C#", [101]), ("C++", [102]), (".NET", [103]), ("Java", [104])],
)
def test_c_sharp_c_plus_plus_and_dot_net_are_searched_separately(query, expected):
    assert found(build(), query) == expected


def test_the_simple_tokenizer_confuses_c_sharp_with_c_plus_plus():
    assert set(found(build("simple"), "C#")) == {101, 102}  # both reduce to the word "c"


def test_the_rare_skill_ranks_first():
    hits = build().search("developer Terraform", 5)
    assert hits[0].chunk_id == 105  # Terraform appears in one chunk, "developer" in two
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(h.score > 0 for h in hits)


@pytest.mark.parametrize("query", ["kubernetes", "K8S", "kube cluster"])
def test_aliases_in_either_direction(query):
    assert found(build(), query)[0] == 105  # the chunk says "k8s"


def test_lowercase_go_only_matches_the_language():
    docs = {201: "Built tools in Go for Kubernetes.", 202: "We go to market quickly."}
    assert found(build(docs=docs), "go") == [201]
    assert found(build(docs=docs), "who has Go experience?") == [201]


def test_queries_without_known_terms_find_nothing():
    index = build()
    assert found(index, "zzzunknownterm") == []
    assert found(index, "the and of") == []
    assert found(index, "") == []


def test_k_limits_results_and_may_exceed_the_corpus():
    index = build()
    assert len(found(index, "developer", k=1)) == 1
    assert len(found(index, "candidate", k=50)) == 5
    assert index.search("developer", 0) == []


def test_documents_without_tokens_do_not_break_indexing():
    index = Bm25Index()
    index.build([1, 2], ["", "kubernetes cluster"])
    assert found(index, "kubernetes") == [2]


def test_save_and_load_give_identical_results(tmp_path):
    index = build()
    index.save(tmp_path / "bm25")
    loaded = Bm25Index()
    loaded.load(tmp_path / "bm25")
    assert loaded.ids() == index.ids() and loaded.count() == 5
    for q in ["C#", "developer Terraform", "k8s", "web services", "Java"]:
        before, after = index.search(q, 5), loaded.search(q, 5)
        assert [h.chunk_id for h in after] == [h.chunk_id for h in before], q
        assert [h.score for h in after] == pytest.approx([h.score for h in before], rel=1e-5), q


def test_saving_again_replaces_the_old_index(tmp_path):
    first = build(docs={1: "kubernetes cluster", 2: "terraform modules"})
    first.save(tmp_path / "bm25")
    second = build(docs={9: "ansible roles"})
    second.save(tmp_path / "bm25")
    assert not (tmp_path / "bm25.tmp").exists()
    loaded = Bm25Index()
    loaded.load(tmp_path / "bm25")
    assert loaded.ids() == [9]
    assert found(loaded, "kubernetes") == [] and found(loaded, "ansible") == [9]


def test_building_replaces_the_whole_corpus():
    index = build(docs={1: "kubernetes cluster"})
    index.build([2], ["terraform modules"])
    assert index.ids() == [2] and found(index, "kubernetes") == []


def test_empty_index_round_trips(tmp_path):
    index = Bm25Index()
    index.build([], [])
    assert index.count() == 0 and index.search("anything", 5) == []
    index.save(tmp_path / "bm25")
    loaded = Bm25Index()
    loaded.load(tmp_path / "bm25")
    assert loaded.count() == 0 and loaded.search("anything", 5) == []


def test_bad_input():
    index = Bm25Index()
    with pytest.raises(ValueError, match="ids for"):
        index.build([1, 2], ["only one text"])
    with pytest.raises(ValueError, match="duplicate"):
        index.build([1, 1], ["a", "b"])


def test_loading_a_missing_index_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        Bm25Index().load(tmp_path / "nope")
