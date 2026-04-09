from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.llm.client import ConfigErrorLLMClient, NoneLLMClient, get_llm_client


class LLMClientConfigTest(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "LLM_BACKEND": "openrouter",
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_MODEL": "openai/gpt-4.1",
        },
        clear=False,
    )
    @patch("app.llm.openrouter_client.OpenRouterLLMClient")
    def test_get_llm_client_passes_model_env_to_openrouter(self, mock_client_cls) -> None:
        instance = mock_client_cls.return_value
        out = get_llm_client()
        mock_client_cls.assert_called_once_with(model="openai/gpt-4.1")
        self.assertIs(out, instance)

    @patch.dict(
        os.environ,
        {
            "LLM_BACKEND": "openrouter",
            "OPENROUTER_API_KEY": "",
            "OPENROUTER_MODEL": "openai/gpt-4.1",
        },
        clear=False,
    )
    def test_missing_openrouter_key_returns_config_error_client(self) -> None:
        out = get_llm_client()

        self.assertIsInstance(out, ConfigErrorLLMClient)
        self.assertEqual(out.describe_backend().get("backend"), "openrouter")
        self.assertEqual(out.describe_backend().get("mode"), "retrieval_only")

    @patch.dict(
        os.environ,
        {
            "LLM_BACKEND": "mystery-backend",
        },
        clear=False,
    )
    def test_unknown_backend_falls_back_to_config_error_client(self) -> None:
        out = get_llm_client()

        self.assertIsInstance(out, ConfigErrorLLMClient)
        self.assertNotIsInstance(out, NoneLLMClient)
        self.assertEqual(out.describe_backend().get("backend"), "mystery-backend")


if __name__ == "__main__":
    unittest.main()
