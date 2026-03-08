from __future__ import annotations

import unittest
from unittest.mock import patch

from app.rag.answer import answer_question


class _ExplodingLLMClient:
    def generate(self, system: str, user: str, context, **kwargs) -> str:
        raise RuntimeError("simulated llm failure")


def _fake_framework_chunks() -> list[dict]:
    return [
        {
            "rank": 1,
            "chunk_id": "oscal::chunk-1",
            "rrf_score": 0.08,
            "dense_score": None,
            "bm25_score": None,
            "payload": {
                "source_type": "oscal_control",
                "control_id": "AC-2",
                "chunk_text": "AC-2 requires account management and approvals.",
                "section_path": "AC-2 Statement",
            },
        },
        {
            "rank": 2,
            "chunk_id": "oscal::chunk-2",
            "rrf_score": 0.07,
            "dense_score": None,
            "bm25_score": None,
            "payload": {
                "source_type": "oscal_control",
                "control_id": "AC-2",
                "chunk_text": "AC-2 includes provisioning and deprovisioning safeguards.",
                "section_path": "AC-2 Guidance",
            },
        },
    ]


class AnswerFallbackTest(unittest.TestCase):
    @patch("app.rag.answer.get_llm_client", return_value=_ExplodingLLMClient())
    @patch("app.rag.answer._retrieve_oscal_control_chunks", return_value=_fake_framework_chunks())
    def test_llm_failure_with_sufficient_evidence_does_not_abstain(self, *_mocks) -> None:
        result = answer_question(
            "Summarize the requirement in AC-2.",
            eval_mode="framework",
            expected={"expected_coverage": "covered"},
            top_k=5,
        )

        self.assertFalse(result["abstained"])
        self.assertAlmostEqual(float(result["confidence"]), 0.10, places=3)
        self.assertIn("Top evidence excerpts", str(result["draft_answer"]))
        self.assertGreaterEqual(len(result.get("retrieved_chunks", [])), 2)


if __name__ == "__main__":
    unittest.main()
