from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import requests


class OpenRouterLLMClient:
    """
    OpenRouter Chat Completions client.

    This backend uses the OpenAI-compatible chat endpoint and can be plugged
    into the shared LLM abstraction through get_llm_client().
    """

    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    default_fallback_models = [
        "nvidia/nemotron-nano-9b-v2:free",
        "google/gemma-3-12b-it:free",
        "qwen/qwen3-4b:free",
    ]

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = os.getenv(
            "OPENROUTER_MODEL",
            "mistralai/mistral-small-3.1-24b-instruct:free",
        ).strip()
        fallback_models_env = os.getenv("OPENROUTER_FALLBACK_MODELS", "").strip()
        if fallback_models_env:
            parsed = [m.strip() for m in fallback_models_env.split(",") if m.strip()]
            self.fallback_models = parsed or list(self.default_fallback_models)
        else:
            self.fallback_models = list(self.default_fallback_models)
        self.app_url = os.getenv("OPENROUTER_APP_URL", "").strip()
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip()

    def _format_context(self, context: List[Dict]) -> str:
        if not context:
            return "(no retrieved context)"

        lines: List[str] = []
        for idx, item in enumerate(context, start=1):
            source_type = item.get("source_type", "unknown")
            control_id = item.get("control_id")
            doc_id = item.get("doc_id")
            section = item.get("section_path", "n/a")
            page_start = item.get("page_start")
            page_end = item.get("page_end")
            page = ""
            if page_start is not None and page_end is not None:
                page = f" pages={page_start}-{page_end}"
            elif page_start is not None:
                page = f" page={page_start}"

            meta = f"[{idx}] source={source_type}"
            if control_id:
                meta += f" control_id={control_id}"
            if doc_id:
                meta += f" doc_id={doc_id}"
            meta += f" section={section}{page}"

            text = str(item.get("chunk_text", "")).replace("\n", " ").strip()
            lines.append(f"{meta}\n{text[:1000]}")
        return "\n\n".join(lines)

    def generate(self, system: str, user: str, context: List[Dict], **kwargs: Any) -> str:
        try:
            if not self.api_key:
                return "OpenRouter error: missing OPENROUTER_API_KEY."

            temperature = float(kwargs.get("temperature", 0.2))
            max_tokens = int(kwargs.get("max_tokens", 600))
            timeout = float(kwargs.get("timeout", 60))

            context_block = self._format_context(context)
            user_content = (
                f"User question:\n{user}\n\n"
                f"Retrieved context:\n{context_block}\n\n"
                "Answer using only retrieved context."
            )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.app_url:
                headers["HTTP-Referer"] = self.app_url
            if self.app_title:
                headers["X-Title"] = self.app_title

            model_candidates: List[str] = [self.model]
            for m in self.fallback_models:
                if m and m not in model_candidates:
                    model_candidates.append(m)

            max_attempts = min(3, len(model_candidates))
            last_status: int | None = None
            last_msg = ""

            for attempt in range(max_attempts):
                model_name = model_candidates[attempt]
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                try:
                    resp = requests.post(
                        self.endpoint,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    )
                except requests.RequestException as exc:
                    last_status = None
                    last_msg = str(exc)
                    if attempt < max_attempts - 1:
                        time.sleep(0.5 * (2**attempt))
                        continue
                    return f"OpenRouter error: request failed: {exc}"

                response_text = resp.text or ""
                body_text = response_text[:200]

                try:
                    data = resp.json()
                except ValueError:
                    return f"OpenRouter error: non-JSON response (code={resp.status_code})"

                if isinstance(data, dict) and "error" in data:
                    err = data.get("error") or {}
                    if isinstance(err, dict):
                        err_msg = str(err.get("message", "unknown error"))
                        err_code = err.get("code")
                    else:
                        err_msg = str(err)
                        err_code = None
                    error_line = f"OpenRouter error: {err_msg} (code={err_code})"

                    # Retry on explicit rate-limit signals only.
                    low = err_msg.lower()
                    rate_limited = (
                        resp.status_code == 429
                        or "rate limit" in low
                        or "too many requests" in low
                    )
                    last_status = resp.status_code
                    last_msg = err_msg
                    if rate_limited and attempt < max_attempts - 1:
                        time.sleep(0.5 * (2**attempt))
                        continue
                    return error_line

                if resp.status_code in (401, 403):
                    return "OpenRouter error: unauthorized (check OPENROUTER_API_KEY and model access)."

                choices = data.get("choices") if isinstance(data, dict) else None
                if not isinstance(choices, list) or not choices:
                    return (
                        "OpenRouter error: malformed response "
                        f"(missing choices/message/content, code={resp.status_code})"
                    )
                first = choices[0] if isinstance(choices[0], dict) else None
                msg = first.get("message") if isinstance(first, dict) else None
                content = msg.get("content") if isinstance(msg, dict) else None
                if content is None:
                    return (
                        "OpenRouter error: malformed response "
                        f"(missing choices/message/content, code={resp.status_code})"
                    )

                if resp.status_code != 200:
                    # Non-200 but parseable response without explicit error block.
                    last_status = resp.status_code
                    last_msg = body_text
                    return f"OpenRouter error: status={resp.status_code}, message={body_text}"

                if isinstance(content, list):
                    # Some providers return content parts. Flatten text parts conservatively.
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(str(part.get("text", "")))
                    return "\n".join([p for p in text_parts if p]).strip() or str(content)
                return str(content).strip()

            status_str = str(last_status) if last_status is not None else "n/a"
            msg = (last_msg or "unknown provider error").strip()
            return f"OpenRouter error: status={status_str}, message={msg}"
        except Exception as exc:
            return f"OpenRouter error: unexpected failure ({exc})"
