import sys
import types

import numpy as np
import pytest

from src.chunking.tokens import estimate_tokens
from src.providers import factory
from src.providers.base import EmbeddingProvider
from src.providers.embedders import common
from src.providers.embedders.ollama_embedder import OllamaEmbedder
from src.providers.embedders.st_embedder import SentenceTransformerEmbedder
from tests.unit.embedder_contract import bag_vector, check_embedder_contract
from tests.unit.fakes import FakeEmbedder

BGE = "BAAI/bge-small-en-v1.5"


# --- the contract, against the fake and against stubbed backends ------------------------


def test_fake_embedder_meets_the_contract():
    embedder = FakeEmbedder(dim=256)
    assert isinstance(embedder, EmbeddingProvider)
    check_embedder_contract(embedder)


class StubSentenceModel:
    """Stands in for a SentenceTransformer: word-bag vectors, deliberately NOT normalised."""

    max_seq_length = 512

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.encoded: list[list[str]] = []
        self.tokenizer = types.SimpleNamespace(
            encode=lambda text, add_special_tokens=True: ["[CLS]", *text.split(), "[SEP]"]
        )

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.encoded.append(list(texts))
        return np.array([bag_vector(t, self.dim) for t in texts]) * 3.0  # un-normalised on purpose


def test_sentence_transformer_wrapper_meets_the_contract_with_a_stub():
    embedder = SentenceTransformerEmbedder(BGE, model=StubSentenceModel())
    assert isinstance(embedder, EmbeddingProvider)
    check_embedder_contract(embedder)


def test_sentence_transformer_wrapper_applies_prefixes_and_reads_model_facts():
    stub = StubSentenceModel()
    embedder = SentenceTransformerEmbedder(BGE, batch_size=7, model=stub)
    embedder.embed_documents(["alpha"])
    embedder.embed_query("beta")
    assert stub.encoded[0] == ["alpha"]  # bge: no document prefix
    assert stub.encoded[1] == [f"{embedder.query_prefix}beta"]
    assert embedder.query_prefix.startswith("Represent this sentence")
    assert (embedder.dim, embedder.max_tokens) == (512, 512)
    assert embedder.count_tokens("one two three") == 5  # + [CLS] and [SEP]


def test_sentence_transformer_is_not_loaded_until_used():
    embedder = SentenceTransformerEmbedder(BGE)
    assert embedder._model is None  # constructing is free; no torch import, no download


def test_load_failure_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import raises
    with pytest.raises(common.EmbeddingError, match="could not load embedding model 'org/m'"):
        SentenceTransformerEmbedder("org/m").dim  # noqa: B018


class StubOllama:
    def __init__(self, dim: int = 256, error: Exception | None = None, drop: int = 0):
        self.dim, self.error, self.drop = dim, error, drop
        self.calls: list[tuple[str, list[str]]] = []

    def embed(self, model, input):  # noqa: A002 - mirrors the ollama client signature
        if self.error:
            raise self.error
        self.calls.append((model, list(input)))
        rows = [bag_vector(t, self.dim) for t in input]
        return {"embeddings": rows[self.drop :]}


def test_ollama_wrapper_meets_the_contract_with_a_stub():
    embedder = OllamaEmbedder("nomic-embed-text", client=StubOllama())
    assert isinstance(embedder, EmbeddingProvider)
    check_embedder_contract(embedder)


def test_ollama_wrapper_prefixes_batching_and_normalisation():
    client = StubOllama()
    embedder = OllamaEmbedder("nomic-embed-text", batch_size=2, client=client)
    vectors = embedder.embed_documents([f"text {i}" for i in range(5)])
    assert [len(c[1]) for c in client.calls] == [2, 2, 1]
    assert all(t.startswith("search_document: ") for _, batch in client.calls for t in batch)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    embedder.embed_query("q")
    assert client.calls[-1] == ("nomic-embed-text", ["search_query: q"])


def test_ollama_dimension_is_probed_once():
    client = StubOllama(dim=9)
    embedder = OllamaEmbedder(client=client)
    assert (embedder.dim, embedder.dim) == (9, 9)
    assert len(client.calls) == 1
    assert embedder.max_tokens == 8192


def test_ollama_failures_explain_what_to_do():
    embedder = OllamaEmbedder(client=StubOllama(error=ConnectionError("refused")))
    with pytest.raises(common.EmbeddingError, match="ollama pull nomic-embed-text"):
        embedder.embed_documents(["x"])


def test_ollama_wrong_vector_count_is_an_error():
    embedder = OllamaEmbedder(client=StubOllama(drop=1))
    with pytest.raises(common.EmbeddingError, match="1 vectors for 2 inputs"):
        embedder.embed_documents(["a", "b"])


def test_ollama_token_count_is_the_conservative_estimate_with_prefix():
    embedder = OllamaEmbedder(client=StubOllama())
    assert embedder.count_tokens("hello world") == estimate_tokens("search_document: hello world")


# --- helpers -----------------------------------------------------------------------------


def test_l2_normalize():
    out = common.l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])  # a zero row stays zero, no NaN


@pytest.mark.parametrize(
    "name,query,document",
    [
        (BGE, "Represent this sentence for searching relevant passages: ", ""),
        ("nomic-embed-text", "search_query: ", "search_document: "),
        ("nomic-embed-text:latest", "search_query: ", "search_document: "),
        ("intfloat/e5-base-v2", "query: ", "passage: "),
        ("some/unknown-model", "", ""),
    ],
)
def test_prefix_registry(name, query, document):
    prefixes = common.prefixes_for(name)
    assert (prefixes.query, prefixes.document) == (query, document)


def test_context_registry():
    assert common.context_for("nomic-embed-text:latest") == 8192
    assert common.context_for("unknown") == common.DEFAULT_CONTEXT


def test_factory_builds_embedders_without_loading_them(settings):
    embedder = factory.create_embedder(BGE, settings)
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert (embedder.batch_size, embedder.device, embedder._model) == (32, None, None)


# --- the real models (opt in: python -m pytest -m models / -m ollama) ----------------------


@pytest.mark.models
def test_bge_small_real_model():
    embedder = SentenceTransformerEmbedder(BGE)
    assert (embedder.dim, embedder.max_tokens) == (384, 512)
    check_embedder_contract(embedder)
    # the real tokenizer counts more than words (word pieces + [CLS]/[SEP])
    assert embedder.count_tokens("Kubernetes") > 1


@pytest.mark.ollama
def test_nomic_real_model():
    embedder = OllamaEmbedder("nomic-embed-text")
    assert embedder.dim == 768
    check_embedder_contract(embedder)
