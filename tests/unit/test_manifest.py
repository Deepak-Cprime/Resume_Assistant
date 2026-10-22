import pytest

from src.pipeline import manifest as mf
from tests.unit.fakes import FakeEmbedder


def sample(**overrides) -> mf.IndexManifest:
    base = dict(
        emb_model="fake-embed", dim=32, n_vectors=4, index_version="abc",
        doc_prefix="doc: ", query_prefix="query: ", budgets_fingerprint="fp",
        chunk_set_hash="h", created_at="2025-01-01T00:00:00+00:00", vector_store="numpy",
    )
    return mf.IndexManifest(**{**base, **overrides})


def test_json_round_trip():
    manifest = sample()
    assert mf.IndexManifest.from_json(manifest.to_json()) == manifest


def test_chunk_set_hash_ignores_order_but_not_content():
    pairs = [(1, "a"), (2, "b"), (3, "c")]
    assert mf.chunk_set_hash(pairs) == mf.chunk_set_hash(reversed(pairs))
    assert mf.chunk_set_hash(pairs) != mf.chunk_set_hash([(1, "a"), (2, "b"), (3, "X")])
    assert mf.chunk_set_hash(pairs) != mf.chunk_set_hash(pairs[:2])


def test_index_version_changes_with_every_input():
    base = ("m", 384, "d", "q", "fp", "hash")
    version = mf.index_version(*base)
    assert version == mf.index_version(*base) and len(version) == 12
    for position, other in enumerate(["m2", 768, "d2", "q2", "fp2", "hash2"]):
        changed = list(base)
        changed[position] = other
        assert mf.index_version(*changed) != version, position


def test_query_embedder_must_match_the_index():
    manifest = sample()
    mf.check_query_embedder(manifest, FakeEmbedder("fake-embed", dim=32))  # fine

    with pytest.raises(mf.IndexMismatch, match="model is 'other'"):
        mf.check_query_embedder(manifest, FakeEmbedder("other", dim=32))
    with pytest.raises(mf.IndexMismatch, match="dimension is 64"):
        mf.check_query_embedder(manifest, FakeEmbedder("fake-embed", dim=64))
    with pytest.raises(mf.IndexMismatch, match="query prefix"):
        mf.check_query_embedder(manifest, FakeEmbedder("fake-embed", dim=32, query_prefix="x: "))


def test_every_problem_is_reported_together():
    with pytest.raises(mf.IndexMismatch) as err:
        mf.check_query_embedder(sample(), FakeEmbedder("other", dim=64, query_prefix=""))
    assert str(err.value).count(";") == 2


def test_load_index_without_an_index(settings):
    with pytest.raises(FileNotFoundError, match="no index has been built"):
        mf.load_index(settings, FakeEmbedder("never-built"))


def test_missing_files_read_as_empty(settings):
    assert mf.read_manifest(settings, "fake-embed") is None
    assert mf.read_hashes(settings, "fake-embed") == {}
