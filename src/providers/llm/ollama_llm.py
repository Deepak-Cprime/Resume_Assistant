"""LLM provider backed by a local Ollama server."""

from __future__ import annotations

import json
import os
from typing import Any


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str, temperature: float = 0, host: str | None = None) -> None:
        self.model = model
        self.temperature = temperature
        self.host = host or os.environ.get("OLLAMA_HOST")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host) if self.host else ollama.Client()
        return self._client

    def _chat(self, prompt: str, system: str | None, fmt: Any = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        response = self._get_client().chat(
            model=self.model,
            messages=messages,
            format=fmt,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self._chat(prompt, system)

    def generate_json(
        self, prompt: str, schema: dict[str, Any], *, system: str | None = None
    ) -> dict[str, Any]:
        text = self._chat(prompt, system, fmt=schema)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"model did not return valid JSON: {text[:200]!r}") from e
        if not isinstance(data, dict):
            raise ValueError("model returned JSON that is not an object")
        return data
