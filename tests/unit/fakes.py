"""Deterministic stand-ins for external services, so default test runs need no Ollama."""

from typing import Any

FULL_MD = """\
# Priya Raman

priya.raman@example.com | +1 555 0101 | Austin, TX

## Summary
Backend engineer with six years of experience building payment services in Go and Python.

## Experience
**Senior Backend Engineer, Northwind Payments**
Mar 2021 - Dec 2024 | Austin, TX
- Led migration of 40 services to Kubernetes on EKS, cutting deploy time from 45 to 8 minutes.
- Designed an idempotent payment API in Go handling 3,000 requests per second.

**Software Engineer, Brightline Logistics**
Jun 2018 - Feb 2021 | Dallas, TX
- Built a route optimisation service in Python and PostgreSQL.
- Introduced CI pipelines with GitHub Actions and Terraform.

## Education
**B.Tech in Computer Science**
Anand Institute of Technology, 2018

## Skills
Go, Python, Kubernetes, PostgreSQL, Terraform, AWS

## Certifications
- AWS Certified Solutions Architect
"""

NO_HEADINGS_MD = """\
**Tomas Becker**
tomas.becker@example.com | +1 555 0104 | Seattle, WA

Mobile engineer shipping iOS and Android apps for consumer products.

**Mobile Engineer, Pinecone Apps**
Jan 2020 - Dec 2022 | Seattle, WA
- Rebuilt the onboarding flow in Swift, raising activation by 18 percent.
- Owned the release train for iOS and Android.

**Senior Mobile Engineer, Cascade Fitness**
Feb 2023 - Apr 2025 | Seattle, WA
- Introduced Kotlin Multiplatform for shared business logic.
- Mentored four junior engineers.

**B.S. in Computer Science**
Puget Sound College, 2019

Swift, Kotlin, Objective-C, Firebase, Xcode

- Google Associate Android Developer
"""

# Sidebar read first, so the name and summary come after Education.
SIDEBAR_FIRST_MD = """\
## Contact
aisha.rahman@example.com
+1 555 0106
Boston, MA

## Skills
Python
SQL
Apache Airflow

## Education
**B.S. in Computer Engineering**
Northeastern Bay University, 2020

# Aisha Rahman

## Summary
Data engineer building batch and streaming pipelines on Spark, Kafka and Snowflake.

## Experience
**Data Engineer, Beacon Biotech**
May 2020 - Oct 2023 | Boston, MA
- Built Airflow pipelines loading 2 TB of lab data per day into Snowflake.
"""


class FakeLLM:
    """Returns a canned JSON object (or raises) and records every prompt it receives."""

    name = "fake"

    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None):
        self.response = response or {}
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return ""

    def generate_json(
        self, prompt: str, schema: dict[str, Any], *, system: str | None = None
    ) -> dict[str, Any]:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


class FakeEmbedder:
    """Bag-of-words embedder: every word maps to a fixed pseudo-random vector, so texts sharing
    words score higher. Deterministic, dependency-free, and applies prefixes like a real model."""

    def __init__(
        self,
        name: str = "fake-embed",
        dim: int = 32,
        max_tokens: int = 512,
        query_prefix: str = "query: ",
        doc_prefix: str = "doc: ",
    ) -> None:
        self.name, self.dim, self.max_tokens = name, dim, max_tokens
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.document_calls = 0

    def _vector(self, text: str):
        import hashlib
        import re

        import numpy as np

        total = np.zeros(self.dim, dtype=np.float64)
        for word in re.findall(r"\w+", text.lower()):
            seed = int(hashlib.md5(f"{self.name}:{word}".encode()).hexdigest()[:8], 16)
            total += np.random.default_rng(seed).standard_normal(self.dim)
        norm = np.linalg.norm(total)
        return (total / norm if norm else total).astype(np.float32)

    def embed_documents(self, texts):
        import numpy as np

        self.document_calls += len(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(self.doc_prefix + t) for t in texts])

    def embed_query(self, text):
        return self._vector(self.query_prefix + text)

    def count_tokens(self, text):
        return len((self.doc_prefix + text).split())
