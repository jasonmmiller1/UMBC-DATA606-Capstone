from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def _llm_metadata(
    *,
    backend: str,
    mode: str,
    requested_model: Optional[str] = None,
    used_model: Optional[str] = None,
    fallback_triggered: bool = False,
    retries: int = 0,
    latency_ms: Optional[int] = None,
    status: str = "unknown",
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "backend": backend,
        "mode": mode,
        "requested_model": requested_model,
        "used_model": used_model,
        "fallback_triggered": fallback_triggered,
        "retries": retries,
        "latency_ms": latency_ms,
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        "note": note,
    }


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

    def describe_backend(self) -> Dict[str, Any]:
        return _llm_metadata(backend="unknown", mode="unknown")

    def last_call_metadata(self) -> Dict[str, Any]:
        return dict(self.describe_backend())


class NoneLLMClient(LLMClient):
    """
    Placeholder backend used when no model is configured.

    This intentionally avoids network calls or model dependencies and lets the
    app run in "retrieval-only" mode until a real backend is configured.
    """

    def __init__(self) -> None:
        self._backend_info = _llm_metadata(
            backend="none",
            mode="retrieval_only",
            status="ready",
            note="LLM backend disabled; retrieval-only mode is active.",
        )
        self._last_call = dict(self._backend_info)

    def generate(self, system: str, user: str, context: List[Dict], **kwargs: Any) -> str:
        self._last_call = _llm_metadata(
            backend="none",
            mode="retrieval_only",
            status="unavailable",
            error_type="disabled",
            error_message="LLM backend disabled; retrieval-only mode is active.",
            note="Returning retrieval-only fallback response.",
        )
        return "LLM not configured. Showing retrieved evidence only."

    def describe_backend(self) -> Dict[str, Any]:
        return dict(self._backend_info)

    def last_call_metadata(self) -> Dict[str, Any]:
        return dict(self._last_call)


class ConfigErrorLLMClient(LLMClient):
    """Client that returns a deterministic configuration error message."""

    def __init__(
        self,
        message: str,
        *,
        backend: str = "openrouter",
        requested_model: Optional[str] = None,
    ) -> None:
        self.message = message
        self.backend = backend
        self.requested_model = requested_model
        self._backend_info = _llm_metadata(
            backend=backend,
            mode="retrieval_only",
            requested_model=requested_model,
            status="unavailable",
            error_type="configuration",
            error_message=message,
            note="Configured LLM backend is unavailable; retrieval-only fallback will be used.",
        )
        self._last_call = dict(self._backend_info)

    def generate(self, system: str, user: str, context: List[Dict], **kwargs: Any) -> str:
        self._last_call = dict(self._backend_info)
        return self.message

    def describe_backend(self) -> Dict[str, Any]:
        return dict(self._backend_info)

    def last_call_metadata(self) -> Dict[str, Any]:
        return dict(self._last_call)


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
        selected_model = os.getenv("OPENROUTER_MODEL", "").strip() or None
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            return ConfigErrorLLMClient(
                "OpenRouter backend selected but OPENROUTER_API_KEY is missing.",
                backend="openrouter",
                requested_model=selected_model,
            )
        from app.llm.openrouter_client import OpenRouterLLMClient

        return OpenRouterLLMClient(model=selected_model)
    # Unknown backends fall back safely to retrieval-only mode.
    return ConfigErrorLLMClient(
        f"Unknown LLM_BACKEND '{backend}'. Falling back to retrieval-only mode.",
        backend=backend or "unknown",
        requested_model=None,
    )
