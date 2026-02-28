from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMClient(ABC):
    """
    Abstract interface for LLM backends.

    This keeps RAG code decoupled from specific providers. Future backends can
    implement this interface for:
    - OpenRouter-hosted models
    - Local models (e.g., Ollama/vLLM/Transformers runtime)
    """

    @abstractmethod
    def generate(self, system: str, user: str, context: List[Dict], **kwargs: Any) -> str:
        """Generate an answer string from system/user prompts and retrieval context."""


class NoneLLMClient(LLMClient):
    """
    Placeholder backend used when no model is configured.

    This intentionally avoids network calls or model dependencies and lets the
    app run in "retrieval-only" mode until a real backend is configured.
    """

    def generate(self, system: str, user: str, context: List[Dict], **kwargs: Any) -> str:
        return "LLM not configured. Showing retrieved evidence only."


class ConfigErrorLLMClient(LLMClient):
    """Client that returns a deterministic configuration error message."""

    def __init__(self, message: str) -> None:
        self.message = message

    def generate(self, system: str, user: str, context: List[Dict], **kwargs: Any) -> str:
        return self.message


def get_llm_client() -> LLMClient:
    """
    Return configured LLM client based on LLM_BACKEND env var.

    Supported backends now:
    - none (default)
    - openrouter

    Planned:
    - local
    """
    backend = os.getenv("LLM_BACKEND", "none").strip().lower()
    if backend in {"", "none"}:
        return NoneLLMClient()
    if backend == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            return ConfigErrorLLMClient(
                "OpenRouter backend selected but OPENROUTER_API_KEY is missing."
            )
        from app.llm.openrouter_client import OpenRouterLLMClient

        return OpenRouterLLMClient()
    # Unknown backends fall back safely to retrieval-only mode.
    return NoneLLMClient()
