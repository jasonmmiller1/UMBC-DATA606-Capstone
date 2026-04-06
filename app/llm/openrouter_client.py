from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.llm.client import LLMClient, _llm_metadata


logger = logging.getLogger(__name__)


class OpenRouterLLMClient(LLMClient):
    """
    OpenRouter Chat Completions client.

    This backend uses the OpenAI-compatible chat endpoint and reports
    per-call metadata so the app can surface mode/model/fallback behavior
    without loosening grounded retrieval guardrails.
    """

    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    default_retry_count = 3
    retry_backoff_seconds = 0.5
    retry_jitter_seconds = 0.15
    default_fallback_models = [
        "nvidia/nemotron-nano-9b-v2:free",
        "google/gemma-3-12b-it:free",
        "qwen/qwen3-4b:free",
    ]

    def __init__(self, model: str | None = None) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        env_model = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free").strip()
        self.model = (model or env_model).strip()
        fallback_models_env = os.getenv("OPENROUTER_FALLBACK_MODELS", "").strip()
        if fallback_models_env:
            parsed = [m.strip() for m in fallback_models_env.split(",") if m.strip()]
            self.fallback_models = parsed or list(self.default_fallback_models)
        else:
            self.fallback_models = list(self.default_fallback_models)
        self.app_url = os.getenv("OPENROUTER_APP_URL", "").strip()
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
        self._backend_info = _llm_metadata(
            backend="openrouter",
            mode="retrieval_plus_llm",
            requested_model=self.model or None,
            status="ready",
            note="OpenRouter backend configured.",
        )
        self._last_call = dict(self._backend_info)

    def describe_backend(self) -> Dict[str, Any]:
        return dict(self._backend_info)

    def last_call_metadata(self) -> Dict[str, Any]:
        return dict(self._last_call)

    def _set_last_call(
        self,
        *,
        used_model: Optional[str],
        fallback_triggered: bool,
        retries: int,
        latency_ms: Optional[int],
        status: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        self._last_call = _llm_metadata(
            backend="openrouter",
            mode="retrieval_plus_llm",
            requested_model=self.model or None,
            used_model=used_model,
            fallback_triggered=fallback_triggered,
            retries=retries,
            latency_ms=latency_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            note=note,
        )

    def _elapsed_ms(self, start_time: float) -> int:
        return int(round((time.monotonic() - start_time) * 1000))

    def _sleep_for_retry(self, attempt: int) -> None:
        base_delay = self.retry_backoff_seconds * (2**attempt)
        jitter = random.uniform(0.0, self.retry_jitter_seconds)
        time.sleep(base_delay + jitter)

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

    def _env_float(self, name: str, default: float) -> float:
        value = os.getenv(name, "").strip()
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def _env_int(self, name: str, default: int) -> int:
        value = os.getenv(name, "").strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _retry_count(self, **kwargs: Any) -> int:
        if "retry_count" in kwargs:
            raw = str(kwargs.get("retry_count", self.default_retry_count))
        else:
            raw = os.getenv("OPENROUTER_RETRY_COUNT", str(self.default_retry_count))
        try:
            return min(3, max(0, int(raw)))
        except (TypeError, ValueError):
            return self.default_retry_count

    def _temperature(self, **kwargs: Any) -> float:
        if "temperature" in kwargs:
            try:
                return float(kwargs.get("temperature", 0.2))
            except (TypeError, ValueError):
                return 0.2
        return self._env_float("OPENROUTER_TEMPERATURE", 0.2)

    def _max_tokens(self, **kwargs: Any) -> int:
        if "max_tokens" in kwargs:
            try:
                return int(kwargs.get("max_tokens", 600))
            except (TypeError, ValueError):
                return 600
        return self._env_int("OPENROUTER_MAX_TOKENS", 600)

    def _timeout(self, **kwargs: Any) -> float:
        if "timeout" in kwargs:
            try:
                return float(kwargs.get("timeout", 60))
            except (TypeError, ValueError):
                return 60.0
        return self._env_float("OPENROUTER_TIMEOUT_SECONDS", 60.0)

    def _classify_error(
        self,
        status_code: Optional[int],
        message: str,
        *,
        err_code: Any = None,
    ) -> Tuple[str, str]:
        normalized = (message or "").strip()
        lowered = normalized.lower()
        err_code_text = str(err_code or "").strip().lower()

        if status_code in {401, 403}:
            return "auth", "OpenRouter error: authentication failed (check OPENROUTER_API_KEY and model access)."
        if "quota" in lowered or "credit" in lowered or "insufficient" in lowered and "balance" in lowered:
            return "quota", "OpenRouter error: quota or credit limit reached."
        if status_code == 429:
            return "rate_limit", "OpenRouter error: rate limit reached. Please retry shortly."
        if "timeout" in lowered or "timed out" in lowered or "deadline exceeded" in lowered:
            return "timeout", "OpenRouter error: request timed out."
        if status_code in {400, 404, 422}:
            if "model" in lowered or "provider" in lowered or "not found" in lowered or "invalid model" in lowered:
                return "model_error", f"OpenRouter error: model unavailable or invalid ({normalized or err_code_text or 'provider response'})."
            return "invalid_request", f"OpenRouter error: invalid request ({normalized or err_code_text or 'provider response'})."
        if status_code is not None and status_code >= 500:
            return "provider_error", f"OpenRouter error: provider request failed ({normalized or f'status {status_code}'})."
        if err_code_text in {"rate_limit", "quota_exceeded"}:
            if err_code_text == "quota_exceeded":
                return "quota", "OpenRouter error: quota or credit limit reached."
            return "rate_limit", "OpenRouter error: rate limit reached. Please retry shortly."
        return "backend_error", f"OpenRouter error: backend request failed ({normalized or 'unknown provider error'})."

    def generate(self, system: str, user: str, context: List[Dict], **kwargs: Any) -> str:
        start_time = time.monotonic()
        fallback_attempted = False
        total_retries = 0

        try:
            if not self.api_key:
                message = "OpenRouter backend selected but OPENROUTER_API_KEY is missing."
                self._set_last_call(
                    used_model=None,
                    fallback_triggered=False,
                    retries=0,
                    latency_ms=self._elapsed_ms(start_time),
                    status="unavailable",
                    error_type="missing_api_key",
                    error_message=message,
                    note="Falling back to retrieval-only behavior because no API key is configured.",
                )
                logger.warning("OpenRouter unavailable requested_model=%s reason=missing_api_key", self.model or "n/a")
                return message

            temperature = self._temperature(**kwargs)
            max_tokens = self._max_tokens(**kwargs)
            timeout = self._timeout(**kwargs)
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
            for model_name in self.fallback_models:
                if model_name and model_name not in model_candidates:
                    model_candidates.append(model_name)

            last_status: Optional[int] = None
            last_message = ""
            last_model = self.model
            last_error_type = "backend_error"

            for model_index, model_name in enumerate(model_candidates):
                if model_index > 0:
                    fallback_attempted = True
                last_model = model_name
                attempts_for_model = retry_count + 1

                for attempt in range(attempts_for_model):
                    is_retry = attempt > 0
                    if is_retry:
                        total_retries += 1

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
                    except requests.Timeout as exc:
                        last_status = None
                        last_message = str(exc) or "request timed out"
                        last_error_type = "timeout"
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter timeout model=%s attempt=%d/%d retrying",
                                model_name,
                                attempt + 1,
                                attempts_for_model,
                            )
                            self._sleep_for_retry(attempt)
                            continue
                        logger.warning(
                            "OpenRouter timeout model=%s retries_exhausted switching_model=%s",
                            model_name,
                            model_index < (len(model_candidates) - 1),
                        )
                        break
                    except requests.RequestException as exc:
                        last_status = None
                        last_message = str(exc)
                        last_error_type = "backend_error"
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter request failure model=%s attempt=%d/%d retrying: %s",
                                model_name,
                                attempt + 1,
                                attempts_for_model,
                                exc,
                            )
                            self._sleep_for_retry(attempt)
                            continue
                        logger.warning(
                            "OpenRouter request failure model=%s retries_exhausted switching_model=%s error=%s",
                            model_name,
                            model_index < (len(model_candidates) - 1),
                            exc,
                        )
                        break

                    response_text = resp.text or ""
                    body_text = response_text[:200].strip()

                    if not response_text.strip():
                        last_status = resp.status_code
                        last_message = "empty response body"
                        last_error_type = "provider_error"
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter empty body model=%s status=%s attempt=%d/%d retrying",
                                model_name,
                                resp.status_code,
                                attempt + 1,
                                attempts_for_model,
                            )
                            self._sleep_for_retry(attempt)
                            continue
                        logger.warning(
                            "OpenRouter empty body model=%s retries_exhausted switching_model=%s",
                            model_name,
                            model_index < (len(model_candidates) - 1),
                        )
                        break

                    try:
                        data = resp.json()
                    except ValueError:
                        last_status = resp.status_code
                        last_message = "non-JSON response"
                        last_error_type = "provider_error"
                        if resp.status_code in {502, 503, 504} and attempt < retry_count:
                            logger.warning(
                                "OpenRouter non-JSON transient response model=%s status=%s attempt=%d/%d retrying",
                                model_name,
                                resp.status_code,
                                attempt + 1,
                                attempts_for_model,
                            )
                            self._sleep_for_retry(attempt)
                            continue
                        if resp.status_code >= 500 and model_index < (len(model_candidates) - 1):
                            logger.warning(
                                "OpenRouter non-JSON provider response model=%s status=%s switching to fallback model",
                                model_name,
                                resp.status_code,
                            )
                            break
                        final_message = self._classify_error(resp.status_code, last_message)[1]
                        self._set_last_call(
                            used_model=model_name,
                            fallback_triggered=fallback_attempted,
                            retries=total_retries,
                            latency_ms=self._elapsed_ms(start_time),
                            status="error",
                            error_type=last_error_type,
                            error_message=final_message,
                            note="OpenRouter returned a malformed non-JSON response.",
                        )
                        return final_message

                    if isinstance(data, dict) and "error" in data:
                        err = data.get("error") or {}
                        if isinstance(err, dict):
                            err_message = str(err.get("message", "unknown error"))
                            err_code = err.get("code")
                        else:
                            err_message = str(err)
                            err_code = None

                        error_type, final_message = self._classify_error(
                            resp.status_code,
                            err_message,
                            err_code=err_code,
                        )
                        last_status = resp.status_code
                        last_message = err_message
                        last_error_type = error_type

                        if error_type in {"auth", "quota"}:
                            logger.error(
                                "OpenRouter terminal error requested_model=%s used_model=%s error_type=%s status=%s",
                                self.model,
                                model_name,
                                error_type,
                                resp.status_code,
                            )
                            self._set_last_call(
                                used_model=model_name,
                                fallback_triggered=fallback_attempted,
                                retries=total_retries,
                                latency_ms=self._elapsed_ms(start_time),
                                status="error",
                                error_type=error_type,
                                error_message=final_message,
                                note="OpenRouter returned a terminal authentication or quota error.",
                            )
                            return final_message

                        is_transient = error_type in {"rate_limit", "timeout"}
                        should_retry = is_transient and attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter transient error model=%s type=%s status=%s attempt=%d/%d retrying",
                                model_name,
                                error_type,
                                resp.status_code,
                                attempt + 1,
                                attempts_for_model,
                            )
                            self._sleep_for_retry(attempt)
                            continue

                        should_switch_model = error_type in {"rate_limit", "model_error", "provider_error"} and model_index < (
                            len(model_candidates) - 1
                        )
                        if should_switch_model:
                            logger.warning(
                                "OpenRouter switching to fallback model requested_model=%s failed_model=%s error_type=%s",
                                self.model,
                                model_name,
                                error_type,
                            )
                            break

                        self._set_last_call(
                            used_model=model_name,
                            fallback_triggered=fallback_attempted,
                            retries=total_retries,
                            latency_ms=self._elapsed_ms(start_time),
                            status="timeout" if error_type == "timeout" else "error",
                            error_type=error_type,
                            error_message=final_message,
                            note="OpenRouter did not return a usable answer.",
                        )
                        return final_message

                    if resp.status_code in {429, 502, 503, 504}:
                        error_type, final_message = self._classify_error(resp.status_code, body_text or f"status {resp.status_code}")
                        last_status = resp.status_code
                        last_message = body_text or f"status {resp.status_code}"
                        last_error_type = error_type
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter transient HTTP status model=%s type=%s status=%s attempt=%d/%d retrying",
                                model_name,
                                error_type,
                                resp.status_code,
                                attempt + 1,
                                attempts_for_model,
                            )
                            self._sleep_for_retry(attempt)
                            continue
                        if model_index < (len(model_candidates) - 1):
                            logger.warning(
                                "OpenRouter transient status exhausted requested_model=%s failed_model=%s switching to fallback",
                                self.model,
                                model_name,
                            )
                            break
                        self._set_last_call(
                            used_model=model_name,
                            fallback_triggered=fallback_attempted,
                            retries=total_retries,
                            latency_ms=self._elapsed_ms(start_time),
                            status="error",
                            error_type=error_type,
                            error_message=final_message,
                            note="OpenRouter exhausted retries on a transient HTTP response.",
                        )
                        return final_message

                    choices = data.get("choices") if isinstance(data, dict) else None
                    if not isinstance(choices, list) or not choices:
                        last_status = resp.status_code
                        last_message = "missing choices"
                        last_error_type = "provider_error"
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter malformed response model=%s attempt=%d/%d retrying: missing choices",
                                model_name,
                                attempt + 1,
                                attempts_for_model,
                            )
                            self._sleep_for_retry(attempt)
                            continue
                        if model_index < (len(model_candidates) - 1):
                            logger.warning(
                                "OpenRouter malformed response requested_model=%s failed_model=%s switching to fallback",
                                self.model,
                                model_name,
                            )
                            break
                        final_message = "OpenRouter error: provider returned a malformed response (missing choices)."
                        self._set_last_call(
                            used_model=model_name,
                            fallback_triggered=fallback_attempted,
                            retries=total_retries,
                            latency_ms=self._elapsed_ms(start_time),
                            status="error",
                            error_type=last_error_type,
                            error_message=final_message,
                            note="OpenRouter returned a malformed response without choices.",
                        )
                        return final_message

                    first = choices[0] if isinstance(choices[0], dict) else None
                    msg = first.get("message") if isinstance(first, dict) else None
                    content = msg.get("content") if isinstance(msg, dict) else None
                    if content is None:
                        last_status = resp.status_code
                        last_message = "missing content"
                        last_error_type = "provider_error"
                        should_retry = attempt < retry_count
                        if should_retry:
                            logger.warning(
                                "OpenRouter malformed content model=%s attempt=%d/%d retrying: missing content",
                                model_name,
                                attempt + 1,
                                attempts_for_model,
                            )
                            self._sleep_for_retry(attempt)
                            continue
                        if model_index < (len(model_candidates) - 1):
                            logger.warning(
                                "OpenRouter missing content requested_model=%s failed_model=%s switching to fallback",
                                self.model,
                                model_name,
                            )
                            break
                        final_message = "OpenRouter error: provider returned a malformed response (missing content)."
                        self._set_last_call(
                            used_model=model_name,
                            fallback_triggered=fallback_attempted,
                            retries=total_retries,
                            latency_ms=self._elapsed_ms(start_time),
                            status="error",
                            error_type=last_error_type,
                            error_message=final_message,
                            note="OpenRouter returned a malformed response without content.",
                        )
                        return final_message

                    if resp.status_code != 200:
                        error_type, final_message = self._classify_error(resp.status_code, body_text or f"status {resp.status_code}")
                        last_status = resp.status_code
                        last_message = body_text or f"status {resp.status_code}"
                        last_error_type = error_type
                        if model_index < (len(model_candidates) - 1) and error_type in {"model_error", "provider_error"}:
                            logger.warning(
                                "OpenRouter non-200 response requested_model=%s failed_model=%s switching to fallback status=%s",
                                self.model,
                                model_name,
                                resp.status_code,
                            )
                            break
                        self._set_last_call(
                            used_model=model_name,
                            fallback_triggered=fallback_attempted,
                            retries=total_retries,
                            latency_ms=self._elapsed_ms(start_time),
                            status="error",
                            error_type=error_type,
                            error_message=final_message,
                            note="OpenRouter returned a non-200 response with content.",
                        )
                        return final_message

                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(str(part.get("text", "")))
                        output = "\n".join([p for p in text_parts if p]).strip() or str(content)
                    else:
                        output = str(content).strip()

                    latency_ms = self._elapsed_ms(start_time)
                    self._set_last_call(
                        used_model=model_name,
                        fallback_triggered=fallback_attempted,
                        retries=total_retries,
                        latency_ms=latency_ms,
                        status="ok",
                        note="OpenRouter answer generated successfully.",
                    )
                    logger.info(
                        "OpenRouter success model=%s requested_model=%s used_fallback=%s retried=%s retries=%d latency_ms=%d",
                        model_name,
                        self.model,
                        fallback_attempted,
                        total_retries > 0,
                        total_retries,
                        latency_ms,
                    )
                    return output

            error_type = last_error_type or "backend_error"
            final_message = self._classify_error(last_status, last_message or "unknown provider error")[1]
            latency_ms = self._elapsed_ms(start_time)
            self._set_last_call(
                used_model=last_model or None,
                fallback_triggered=fallback_attempted,
                retries=total_retries,
                latency_ms=latency_ms,
                status="timeout" if error_type == "timeout" else "error",
                error_type=error_type,
                error_message=final_message,
                note="OpenRouter exhausted retries and fallbacks.",
            )
            logger.error(
                "OpenRouter failed requested_model=%s last_model=%s error_type=%s last_status=%s fallback_triggered=%s retries=%d latency_ms=%d",
                self.model,
                last_model or "n/a",
                error_type,
                str(last_status) if last_status is not None else "n/a",
                fallback_attempted,
                total_retries,
                latency_ms,
            )
            return final_message
        except Exception as exc:
            message = f"OpenRouter error: unexpected failure ({exc})"
            latency_ms = self._elapsed_ms(start_time)
            self._set_last_call(
                used_model=None,
                fallback_triggered=fallback_attempted,
                retries=total_retries,
                latency_ms=latency_ms,
                status="error",
                error_type="unexpected_failure",
                error_message=message,
                note="Unexpected OpenRouter client failure.",
            )
            logger.exception("OpenRouter unexpected failure requested_model=%s", self.model)
            return message
