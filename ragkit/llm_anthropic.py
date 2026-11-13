"""Optional adapter for a real model. Not exercised by the test suite (it needs an API key).

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    RagPipeline(retriever, AnthropicLLM())
"""
from __future__ import annotations

import os


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 400) -> None:
        import anthropic  # imported lazily so the package works without it

        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("RAG_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def complete(self, system: str, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
