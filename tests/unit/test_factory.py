import pytest

from src.providers import factory
from src.providers.base import EmbeddingProvider, KeywordIndex, LLMProvider


def test_import_src():
    import src

    assert src.__version__


def test_unknown_provider_lists_known(settings):
    with pytest.raises(factory.ProviderError, match="known: .*pymupdf4llm"):
        factory.create_parser("nope", settings)


def test_unimplemented_provider_gives_clear_error(settings, monkeypatch):
    # Implementations arrive in later milestones; until then the error must be explicit.
    monkeypatch.setitem(factory._REGISTRY["keyword"], "bm25s", "src.providers.keyword.missing:X")
    with pytest.raises(factory.ProviderError, match="not available yet"):
        factory.create_keyword_index(settings)


def test_register_and_create_fake(settings, monkeypatch):
    monkeypatch.setitem(factory._REGISTRY, "llm", dict(factory._REGISTRY["llm"]))

    class Fake:
        name = "fake"

        def __init__(self, model, temperature):
            self.model = model

        def generate(self, prompt, *, system=None):
            return "ok"

        def generate_json(self, prompt, schema, *, system=None):
            return {}

    factory.register("llm", "ollama", Fake)
    llm = factory.create_llm(settings)
    assert isinstance(llm, LLMProvider)
    assert llm.model == settings.llm.model


def test_protocols_are_runtime_checkable():
    assert not isinstance(object(), EmbeddingProvider)
    assert not isinstance(object(), KeywordIndex)
