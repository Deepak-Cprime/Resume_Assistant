"""The behaviour every EmbeddingProvider must have, whatever model is behind it."""

import numpy as np

SAMPLES = [
    "Led migration of 40 services to Kubernetes on EKS, cutting deploy time to 8 minutes.",
    "Designed a payment API in Go handling 3,000 requests per second.",
]


def check_embedder_contract(embedder) -> None:
    docs = embedder.embed_documents(SAMPLES)
    assert docs.dtype == np.float32
    assert docs.shape == (len(SAMPLES), embedder.dim)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-4), "documents not unit length"

    # deterministic, and independent of what else is in the batch
    assert np.allclose(docs, embedder.embed_documents(SAMPLES), atol=1e-4)
    assert np.allclose(docs[0], embedder.embed_documents(SAMPLES[:1])[0], atol=1e-3)
    assert embedder.embed_documents([]).shape == (0, embedder.dim)

    query = embedder.embed_query("Kubernetes migration")
    assert query.dtype == np.float32 and query.shape == (embedder.dim,)
    assert np.isclose(np.linalg.norm(query), 1.0, atol=1e-4), "queries must be unit length"
    assert float(query @ docs[0]) > float(query @ docs[1]), "the relevant document should win"

    # the model's query prefix is applied: when prefixes differ, query != document embedding
    if getattr(embedder, "query_prefix", "") != getattr(embedder, "doc_prefix", ""):
        same_text = embedder.embed_query(SAMPLES[0])
        assert not np.allclose(same_text, docs[0], atol=1e-4)

    assert embedder.max_tokens > 0
    assert embedder.count_tokens(SAMPLES[0]) >= len(SAMPLES[0].split()) // 2
    assert embedder.count_tokens(SAMPLES[0]) < embedder.max_tokens


def bag_vector(text: str, dim: int) -> list[float]:
    """Word counts hashed into `dim` bins (not normalised): texts sharing words get similar
    vectors, which is all the stub backends need to pass the 'relevant document wins' check."""
    import hashlib
    import re

    out = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        out[int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % dim] += 1.0
    return out
