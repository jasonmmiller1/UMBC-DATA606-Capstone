from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.llm.client import get_llm_client


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


if __name__ == "__main__":
    unittest.main()
