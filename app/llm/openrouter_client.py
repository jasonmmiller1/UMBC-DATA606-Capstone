from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

import requests


logger = logging.getLogger(__name__)


class OpenRouterLLMClient:
    """
    OpenRouter Chat Completions client.

    This backend uses the OpenAI-compatible chat endpoint and can be plugged
    into the shared LLM abstraction through get_llm_client().
    """

    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    default_retry_count = 2
    retry_backoff_seconds = 0.5
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

    def _retry_count(self, **kwargs: Any) -> int:
        if "retry_count" in kwargs:
            raw = str(kwargs.get("retry_count", self.default_retry_count))
        else:
            raw = os.getenv("OPENROUTER_RETRY_COUNT", str(self.default_retry_count))
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return self.default_retry_count

    def generate(self, system: str, user: str, context: List[Dict], **kwargs: Any) -> str:
        try:
            if not self.api_key:
                return "OpenRouter error: missing OPENROUTER_API_KEY."

            temperature = float(kwargs.get("temperature", 0.2))
            max_tokens = int(kwargs.get("max_tokens", 600))
            timeout = float(kwargs.get("timeout", 60))
            retry_count = self._retry_count(**kwargs)

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

            last_status: int | None = None
            last_msg = ""
            last_model = ""
            any_retry = False

            for model_index, model_name in enumerate(model_candidates):
                last_model = model_name
                retried_for_model = False
                attempts_for_model = retry_count + 1

                for attempt in range(attempts_for_model):
                    is_retry = attempt > 0
                    retried_for_model = retried_for_model or is_retry
                    any_retry = any_retry or is_retry

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
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter transient request failure model=%s attempt=%d/%d retrying: %s",
                                model_name,
                                attempt + 1,
                                attempts_for_model,
                                exc,
                            )
                            time.sleep(self.retry_backoff_seconds * (2**attempt))
                            continue
                        logger.warning(
                            "OpenRouter request failure model=%s retries_exhausted=%s switching_model=%s error=%s",
                            model_name,
                            retried_for_model,
                            model_index < (len(model_candidates) - 1),
                            exc,
                        )
                        break

                    response_text = resp.text or ""
                    body_text = response_text[:200]

                    if not response_text.strip():
                        last_status = resp.status_code
                        last_msg = "empty response body"
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter transient empty body model=%s status=%s attempt=%d/%d retrying",
                                model_name,
                                resp.status_code,
                                attempt + 1,
                                attempts_for_model,
                            )
                            time.sleep(self.retry_backoff_seconds * (2**attempt))
                            continue
                        logger.warning(
                            "OpenRouter empty body model=%s retries_exhausted=%s switching_model=%s",
                            model_name,
                            retried_for_model,
                            model_index < (len(model_candidates) - 1),
                        )
                        break

                    try:
                        data = resp.json()
                    except ValueError:
                        last_status = resp.status_code
                        last_msg = "non-JSON response"
                        # Non-JSON is not classified as transient by default.
                        break

                    if isinstance(data, dict) and "error" in data:
                        err = data.get("error") or {}
                        if isinstance(err, dict):
                            err_msg = str(err.get("message", "unknown error"))
                            err_code = err.get("code")
                        else:
                            err_msg = str(err)
                            err_code = None
                        error_line = f"OpenRouter error: {err_msg} (code={err_code})"

                        if resp.status_code in (401, 403):
                            return "OpenRouter error: unauthorized (check OPENROUTER_API_KEY and model access)."

                        is_transient_error = resp.status_code in {429, 502}
                        last_status = resp.status_code
                        last_msg = err_msg
                        should_retry = is_transient_error and attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter transient provider error model=%s status=%s attempt=%d/%d retrying: %s",
                                model_name,
                                resp.status_code,
                                attempt + 1,
                                attempts_for_model,
                                err_msg,
                            )
                            time.sleep(self.retry_backoff_seconds * (2**attempt))
                            continue
                        if is_transient_error:
                            logger.warning(
                                "OpenRouter transient error retries exhausted model=%s status=%s switching_model=%s",
                                model_name,
                                resp.status_code,
                                model_index < (len(model_candidates) - 1),
                            )
                            break
                        if model_index < (len(model_candidates) - 1):
                            logger.warning(
                                "OpenRouter provider error model=%s status=%s switching to fallback model: %s",
                                model_name,
                                resp.status_code,
                                err_msg,
                            )
                            break
                        return error_line

                    if resp.status_code in (401, 403):
                        return "OpenRouter error: unauthorized (check OPENROUTER_API_KEY and model access)."

                    if resp.status_code in {429, 502}:
                        last_status = resp.status_code
                        last_msg = body_text
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter transient HTTP status model=%s status=%s attempt=%d/%d retrying",
                                model_name,
                                resp.status_code,
                                attempt + 1,
                                attempts_for_model,
                            )
                            time.sleep(self.retry_backoff_seconds * (2**attempt))
                            continue
                        logger.warning(
                            "OpenRouter transient status retries exhausted model=%s status=%s switching_model=%s",
                            model_name,
                            resp.status_code,
                            model_index < (len(model_candidates) - 1),
                        )
                        break

                    choices = data.get("choices") if isinstance(data, dict) else None
                    if not isinstance(choices, list) or not choices:
                        last_status = resp.status_code
                        last_msg = "missing choices"
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter transient malformed response model=%s attempt=%d/%d retrying: missing choices",
                                model_name,
                                attempt + 1,
                                attempts_for_model,
                            )
                            time.sleep(self.retry_backoff_seconds * (2**attempt))
                            continue
                        logger.warning(
                            "OpenRouter malformed response retries exhausted model=%s switching_model=%s",
                            model_name,
                            model_index < (len(model_candidates) - 1),
                        )
                        break
                    first = choices[0] if isinstance(choices[0], dict) else None
                    msg = first.get("message") if isinstance(first, dict) else None
                    content = msg.get("content") if isinstance(msg, dict) else None
                    if content is None:
                        last_status = resp.status_code
                        last_msg = "missing content"
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter transient malformed response model=%s attempt=%d/%d retrying: missing content",
                                model_name,
                                attempt + 1,
                                attempts_for_model,
                            )
                            time.sleep(self.retry_backoff_seconds * (2**attempt))
                            continue
                        logger.warning(
                            "OpenRouter malformed content retries exhausted model=%s switching_model=%s",
                            model_name,
                            model_index < (len(model_candidates) - 1),
                        )
                        break

                    if resp.status_code != 200:
                        # Non-transient non-200 response.
                        last_status = resp.status_code
                        last_msg = body_text
                        if model_index < (len(model_candidates) - 1):
                            logger.warning(
                                "OpenRouter non-200 response model=%s status=%s switching to fallback model",
                                model_name,
                                resp.status_code,
                            )
                            break
                        return f"OpenRouter error: status={resp.status_code}, message={body_text}"

                    if isinstance(content, list):
                        # Some providers return content parts. Flatten text parts conservatively.
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(str(part.get("text", "")))
                        output = "\n".join([p for p in text_parts if p]).strip() or str(content)
                    else:
                        output = str(content).strip()

                    logger.info(
                        "OpenRouter success model=%s used_fallback=%s retried=%s attempts=%d",
                        model_name,
                        model_index > 0,
                        retried_for_model,
                        attempt + 1,
                    )
                    return output

            status_str = str(last_status) if last_status is not None else "n/a"
            msg = (last_msg or "unknown provider error").strip()
            logger.error(
                "OpenRouter failed after retries and fallbacks last_model=%s last_status=%s retried=%s error=%s",
                last_model or "n/a",
                status_str,
                any_retry,
                msg,
            )
            return f"OpenRouter error: status={status_str}, message={msg}"
        except Exception as exc:
            return f"OpenRouter error: unexpected failure ({exc})"
