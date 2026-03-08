from __future__ import annotations

import os
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


if __name__ == "__main__":
    unittest.main()
