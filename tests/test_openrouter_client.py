from __future__ import annotations

import os
import requests
import unittest
from unittest.mock import patch

from app.llm.openrouter_client import OpenRouterLLMClient


class _FakeResponse:
    def __init__(self, status_code: int, text: str, json_data=None, json_error: bool = False):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("invalid json")
        return self._json_data


class OpenRouterClientRetryTest(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "primary/model",
            "OPENROUTER_FALLBACK_MODELS": "nvidia/nemotron-nano-9b-v2:free",
        },
        clear=False,
    )
    @patch("app.llm.openrouter_client.time.sleep", return_value=None)
    @patch("app.llm.openrouter_client.requests.post")
    def test_retries_transient_429_before_success(self, mock_post, _sleep) -> None:
        mock_post.side_effect = [
            _FakeResponse(429, '{"error":{"message":"rate limited"}}', {"error": {"message": "rate limited"}}),
            _FakeResponse(429, '{"error":{"message":"rate limited"}}', {"error": {"message": "rate limited"}}),
            _FakeResponse(200, '{"choices":[{"message":{"content":"ok"}}]}', {"choices": [{"message": {"content": "ok"}}]}),
        ]

        client = OpenRouterLLMClient()
        with self.assertLogs("app.llm.openrouter_client", level="INFO") as captured:
            out = client.generate("sys", "usr", [], retry_count=2, timeout=1)

        self.assertEqual(out, "ok")
        self.assertEqual(mock_post.call_count, 3)
        models = [call.kwargs["json"]["model"] for call in mock_post.call_args_list]
        self.assertEqual(models, ["primary/model", "primary/model", "primary/model"])
        self.assertTrue(any("retried=True" in line and "model=primary/model" in line for line in captured.output))
        self.assertEqual(client.last_call_metadata().get("status"), "ok")
        self.assertEqual(client.last_call_metadata().get("used_model"), "primary/model")

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "primary/model",
            "OPENROUTER_FALLBACK_MODELS": "nvidia/nemotron-nano-9b-v2:free",
        },
        clear=False,
    )
    @patch("app.llm.openrouter_client.time.sleep", return_value=None)
    @patch("app.llm.openrouter_client.requests.post")
    def test_switches_to_fallback_model_after_retry_exhaustion(self, mock_post, _sleep) -> None:
        mock_post.side_effect = [
            _FakeResponse(502, "bad gateway", {}),
            _FakeResponse(502, "bad gateway", {}),
            _FakeResponse(502, "bad gateway", {}),
            _FakeResponse(
                200,
                '{"choices":[{"message":{"content":"fallback ok"}}]}',
                {"choices": [{"message": {"content": "fallback ok"}}]},
            ),
        ]

        client = OpenRouterLLMClient()
        with self.assertLogs("app.llm.openrouter_client", level="INFO") as captured:
            out = client.generate("sys", "usr", [], retry_count=2, timeout=1)

        self.assertEqual(out, "fallback ok")
        models = [call.kwargs["json"]["model"] for call in mock_post.call_args_list]
        self.assertEqual(
            models,
            [
                "primary/model",
                "primary/model",
                "primary/model",
                "nvidia/nemotron-nano-9b-v2:free",
            ],
        )
        self.assertTrue(any("used_fallback=True" in line for line in captured.output))
        self.assertTrue(bool(client.last_call_metadata().get("fallback_triggered")))
        self.assertEqual(client.last_call_metadata().get("used_model"), "nvidia/nemotron-nano-9b-v2:free")

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "primary/model",
            "OPENROUTER_FALLBACK_MODELS": "nvidia/nemotron-nano-9b-v2:free",
        },
        clear=False,
    )
    @patch("app.llm.openrouter_client.time.sleep", return_value=None)
    @patch("app.llm.openrouter_client.requests.post")
    def test_retries_on_missing_choices_and_empty_body(self, mock_post, _sleep) -> None:
        mock_post.side_effect = [
            _FakeResponse(200, "", {}),  # empty body -> transient retry
            _FakeResponse(200, "{}", {}),  # missing choices -> transient retry
            _FakeResponse(
                200,
                '{"choices":[{"message":{"content":"final ok"}}]}',
                {"choices": [{"message": {"content": "final ok"}}]},
            ),
        ]

        client = OpenRouterLLMClient()
        out = client.generate("sys", "usr", [], retry_count=2, timeout=1)
        self.assertEqual(out, "final ok")
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(client.last_call_metadata().get("status"), "ok")

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "",
            "OPENROUTER_MODEL": "primary/model",
        },
        clear=False,
    )
    def test_missing_api_key_reports_unavailable_metadata(self) -> None:
        client = OpenRouterLLMClient()
        out = client.generate("sys", "usr", [], retry_count=0, timeout=1)

        self.assertIn("OPENROUTER_API_KEY is missing", out)
        self.assertEqual(client.last_call_metadata().get("status"), "unavailable")
        self.assertEqual(client.last_call_metadata().get("error_type"), "missing_api_key")

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "primary/model",
        },
        clear=False,
    )
    @patch("app.llm.openrouter_client.requests.post")
    def test_auth_failure_is_classified(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            401,
            '{"error":{"message":"invalid auth"}}',
            {"error": {"message": "invalid auth"}},
        )

        client = OpenRouterLLMClient()
        out = client.generate("sys", "usr", [], retry_count=0, timeout=1)

        self.assertIn("authentication failed", out.lower())
        self.assertEqual(client.last_call_metadata().get("status"), "error")
        self.assertEqual(client.last_call_metadata().get("error_type"), "auth")

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "primary/model",
        },
        clear=False,
    )
    @patch("app.llm.openrouter_client.time.sleep", return_value=None)
    @patch("app.llm.openrouter_client.requests.post", side_effect=requests.Timeout("read timed out"))
    def test_timeout_is_classified(self, _mock_post, _sleep) -> None:
        client = OpenRouterLLMClient()
        out = client.generate("sys", "usr", [], retry_count=1, timeout=1)

        self.assertIn("timed out", out.lower())
        self.assertEqual(client.last_call_metadata().get("status"), "timeout")
        self.assertEqual(client.last_call_metadata().get("error_type"), "timeout")

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "primary/model",
        },
        clear=False,
    )
    @patch("app.llm.openrouter_client.requests.post")
    def test_quota_error_is_classified_without_retry_looping(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            429,
            '{"error":{"message":"quota exceeded"}}',
            {"error": {"message": "quota exceeded"}},
        )

        client = OpenRouterLLMClient()
        out = client.generate("sys", "usr", [], retry_count=2, timeout=1)

        self.assertIn("quota or credit limit", out.lower())
        self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(client.last_call_metadata().get("status"), "error")
        self.assertEqual(client.last_call_metadata().get("error_type"), "quota")


if __name__ == "__main__":
    unittest.main()
