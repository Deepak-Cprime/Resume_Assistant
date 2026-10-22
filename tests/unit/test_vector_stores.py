"""One behaviour spec for every VectorStore. FAISS tests are skipped where faiss-cpu is missing."""

import importlib.util

import numpy as np
import pytest

from src.providers.base import VectorStore
from src.providers.vectorstores.numpy_store import NumpyStore

DIM = 8
HAS_FAISS = importlib.util.find_spec("faiss") is not None


def make_store(kind: str, dim: int = DIM):
    if kind == "faiss":
        from src.providers.vectorstores.faiss_store import FaissStore

        return FaissStore(dim)
    return NumpyStore(dim)


@pytest.fixture(
    params=[
        "numpy",
        pytest.param("faiss", marks=pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu missing")),
    ]
)
def kind(request):
    return request.param


def unit_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_is_a_vector_store(kind):
    store = make_store(kind)
    assert isinstance(store, VectorStore)
    assert store.suffix.startswith(".") and store.dim == DIM and store.count() == 0


def test_search_ranks_by_cosine(kind):
    store = make_store(kind)
    vectors = unit_vectors(20)
    store.add(list(range(100, 120)), vectors)
    query = vectors[7]
    hits = store.search(query, 5)
    assert hits[0].chunk_id == 107 and hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    expected = np.argsort(-(vectors @ query))[:5] + 100
    assert [h.chunk_id for h in hits] == list(expected)


def test_k_larger_than_the_store_and_empty_store(kind):
    store = make_store(kind)
    assert store.search(unit_vectors(1)[0], 5) == []
    store.add([1, 2], unit_vectors(2))
    assert len(store.search(unit_vectors(1)[0], 50)) == 2
    assert store.search(unit_vectors(1)[0], 0) == []


def test_adding_an_existing_id_replaces_it(kind):
    store = make_store(kind)
    first, second = unit_vectors(2, seed=1)
    store.add([5], first[None, :])
    store.add([5], second[None, :])
    assert store.count() == 1 and store.ids() == [5]
    (hit,) = store.search(second, 1)
    assert hit.chunk_id == 5 and hit.score == pytest.approx(1.0, abs=1e-5)


def test_remove(kind):
    store = make_store(kind)
    store.add([1, 2, 3], unit_vectors(3))
    store.remove([2, 99])  # unknown ids are ignored
    store.remove([])
    assert sorted(store.ids()) == [1, 3] and store.count() == 2
    assert 2 not in {h.chunk_id for h in store.search(unit_vectors(1)[0], 10)}


def test_save_and_load_give_identical_results(kind, tmp_path):
    store = make_store(kind)
    vectors = unit_vectors(30, seed=3)
    store.add(list(range(30)), vectors)
    path = tmp_path / "deep" / "dir" / f"vectors{store.suffix}"
    store.save(path)
    assert path.exists() and not list(path.parent.glob("*.tmp"))

    loaded = make_store(kind)
    loaded.load(path)
    assert sorted(loaded.ids()) == sorted(store.ids()) and loaded.count() == 30
    query = unit_vectors(1, seed=9)[0]
    assert loaded.search(query, 10) == store.search(query, 10)


def test_load_rejects_a_different_dimension(kind, tmp_path):
    store = make_store(kind, dim=DIM)
    store.add([1], unit_vectors(1))
    path = tmp_path / f"v{store.suffix}"
    store.save(path)
    with pytest.raises(ValueError, match="dimension"):
        make_store(kind, dim=DIM + 1).load(path)


def test_empty_store_round_trips(kind, tmp_path):
    store = make_store(kind)
    path = tmp_path / f"v{store.suffix}"
    store.save(path)
    loaded = make_store(kind)
    loaded.load(path)
    assert loaded.count() == 0 and loaded.search(unit_vectors(1)[0], 3) == []


@pytest.mark.parametrize(
    "ids,vectors,message",
    [
        ([1], np.ones((1, DIM), dtype=np.float32), "L2-normalised"),
        ([1], unit_vectors(1, dim=DIM + 1), "shape"),
        ([1, 2], unit_vectors(1), "ids for"),
        ([1, 1], unit_vectors(2), "duplicate"),
        ([1], np.full((1, DIM), np.nan, dtype=np.float32), "NaN"),
    ],
)
def test_bad_batches_are_rejected(kind, ids, vectors, message):
    store = make_store(kind)
    with pytest.raises(ValueError, match=message):
        store.add(ids, vectors)
    assert store.count() == 0


def test_query_of_the_wrong_size_is_rejected(kind):
    store = make_store(kind)
    store.add([1], unit_vectors(1))
    with pytest.raises(ValueError, match="dimensions"):
        store.search(np.ones(DIM + 3, dtype=np.float32), 1)


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu missing")
def test_faiss_and_numpy_agree():
    vectors = unit_vectors(200, dim=16, seed=5)
    ids = list(range(1000, 1200))
    a, b = make_store("faiss", 16), make_store("numpy", 16)
    for store in (a, b):
        store.add(ids, vectors)
    for seed in range(5):
        query = unit_vectors(1, dim=16, seed=100 + seed)[0]
        hits_a, hits_b = a.search(query, 10), b.search(query, 10)
        assert [h.chunk_id for h in hits_a] == [h.chunk_id for h in hits_b]
        assert [h.score for h in hits_a] == pytest.approx([h.score for h in hits_b], abs=1e-5)
