from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app.rag.answer import answer_question


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_QUESTIONS_PATH = REPO_ROOT / "data" / "eval" / "golden_questions.jsonl"
CONTROLS_TRUTH_PATH = REPO_ROOT / "data" / "truth_table" / "controls_truth.csv"

EXPECTED_TRUTH_COVERAGE = {
    "AU-2": "covered",
    "IR-4": "partial",
    "IR-6": "partial",
    "CM-2": "missing",
}


class _UnavailableLLMClient:
    def generate(self, system: str, user: str, context, **kwargs) -> str:
        return "LLM not configured for this environment."


class _OpenRouterErrorLLMClient:
    def generate(self, system: str, user: str, context, **kwargs) -> str:
        return "OpenRouter error: simulated upstream failure"


def _mixed_retrieval_side_effect_for(control_id: str):
    normalized = control_id.upper()
    control_chunks = [
        {
            "rank": 1,
            "chunk_id": f"oscal::{normalized}::1",
            "rrf_score": 0.08,
            "dense_score": None,
            "bm25_score": None,
            "payload": {
                "source_type": "oscal_control",
                "control_id": normalized,
                "chunk_text": f"{normalized} control statement sample.",
                "section_path": f"{normalized} Statement",
                "doc_id": normalized,
            },
        },
        {
            "rank": 2,
            "chunk_id": f"oscal::{normalized}::2",
            "rrf_score": 0.07,
            "dense_score": None,
            "bm25_score": None,
            "payload": {
                "source_type": "oscal_control",
                "control_id": normalized,
                "chunk_text": f"{normalized} control guidance sample.",
                "section_path": f"{normalized} Guidance",
                "doc_id": normalized,
            },
        },
    ]
    policy_chunks = [
        {
            "rank": 1,
            "chunk_id": f"policy::{normalized}::1",
            "rrf_score": 0.06,
            "dense_score": None,
            "bm25_score": None,
            "payload": {
                "source_type": "policy_pdf",
                "control_id": None,
                "doc_id": "policy_doc_1",
                "section_path": "4.0 Policy Statements",
                "chunk_text": "Policy statement excerpt.",
            },
        },
        {
            "rank": 2,
            "chunk_id": f"policy::{normalized}::2",
            "rrf_score": 0.05,
            "dense_score": None,
            "bm25_score": None,
            "payload": {
                "source_type": "policy_pdf",
                "control_id": None,
                "doc_id": "policy_doc_2",
                "section_path": "5.0 Procedures",
                "chunk_text": "Procedure excerpt.",
            },
        },
    ]

    def _side_effect(query: str, top_k: int = 10, **kwargs):
        source_type = kwargs.get("source_type")
        if source_type == "oscal_control":
            return control_chunks
        if source_type in {"policy_pdf", "policy_md"}:
            return policy_chunks
        return []

    return _side_effect


class PolicyVsControlCoverageTest(unittest.TestCase):
    def test_single_control_golden_coverage_matches_truth_table(self) -> None:
        with CONTROLS_TRUTH_PATH.open(newline="", encoding="utf-8") as truth_file:
            truth_coverage_by_control = {
                row["control_id"]: row["expected_coverage"]
                for row in csv.DictReader(truth_file)
            }

        with GOLDEN_QUESTIONS_PATH.open(encoding="utf-8") as golden_file:
            for line in golden_file:
                golden = json.loads(line)
                expected = golden.get("expected", {})
                if golden.get("mode") != "policy_vs_control":
                    continue
                if golden.get("intent") != "coverage_assessment":
                    continue
                control_ids = expected.get("expected_control_ids") or []
                expected_coverage = expected.get("expected_coverage")
                if len(control_ids) != 1 or expected_coverage is None:
                    continue

                control_id = control_ids[0]
                with self.subTest(question_id=golden["id"], control_id=control_id):
                    self.assertIn(control_id, truth_coverage_by_control)
                    self.assertEqual(
                        expected_coverage,
                        truth_coverage_by_control[control_id],
                    )

    def test_truth_based_coverage_when_llm_unavailable(self) -> None:
        query = "How does our current policy coverage align right now?"
        for control_id, expected_coverage in EXPECTED_TRUTH_COVERAGE.items():
            with self.subTest(control_id=control_id):
                with patch("app.rag.answer.get_llm_client", return_value=_UnavailableLLMClient()):
                    with patch("app.rag.answer.hybrid_search", side_effect=_mixed_retrieval_side_effect_for(control_id)):
                        result = answer_question(
                            query,
                            eval_mode="policy_vs_control",
                            eval_intent="coverage_assessment",
                            top_k=6,
                        )
                self.assertEqual(result.get("predicted_coverage"), expected_coverage)
                self.assertTrue(str(result.get("draft_answer", "")).startswith(f"Coverage: {expected_coverage}"))

    def test_truth_based_coverage_when_llm_errors(self) -> None:
        query = "How does our current policy coverage align right now?"
        for control_id, expected_coverage in EXPECTED_TRUTH_COVERAGE.items():
            with self.subTest(control_id=control_id):
                with patch("app.rag.answer.get_llm_client", return_value=_OpenRouterErrorLLMClient()):
                    with patch("app.rag.answer.hybrid_search", side_effect=_mixed_retrieval_side_effect_for(control_id)):
                        result = answer_question(
                            query,
                            eval_mode="policy_vs_control",
                            eval_intent="coverage_assessment",
                            top_k=6,
                        )
                self.assertEqual(result.get("predicted_coverage"), expected_coverage)
                self.assertTrue(str(result.get("draft_answer", "")).startswith(f"Coverage: {expected_coverage}"))

    def test_policy_query_is_expanded_in_mixed_mode(self) -> None:
        control_id = "AU-2"
        normalized = control_id.upper()
        control_chunks = [
            {
                "rank": 1,
                "chunk_id": f"oscal::{normalized}::1",
                "rrf_score": 0.08,
                "dense_score": None,
                "bm25_score": None,
                "payload": {
                    "source_type": "oscal_control",
                    "control_id": normalized,
                    "chunk_text": f"{normalized} statement includes audit event logging requirements.",
                    "section_path": f"{normalized} Statement",
                    "doc_id": normalized,
                },
            },
            {
                "rank": 2,
                "chunk_id": f"oscal::{normalized}::2",
                "rrf_score": 0.07,
                "dense_score": None,
                "bm25_score": None,
                "payload": {
                    "source_type": "oscal_control",
                    "control_id": normalized,
                    "chunk_text": f"{normalized} guidance mentions audit function and event types.",
                    "section_path": f"{normalized} Guidance",
                    "doc_id": normalized,
                },
            },
        ]
        policy_chunks = [
            {
                "rank": 1,
                "chunk_id": f"policy::{normalized}::1",
                "rrf_score": 0.06,
                "dense_score": None,
                "bm25_score": None,
                "payload": {
                    "source_type": "policy_pdf",
                    "control_id": None,
                    "doc_id": "04_logging_monitoring_standard",
                    "section_path": "4.0 Logging",
                    "chunk_text": "Logging policy excerpt.",
                },
            },
            {
                "rank": 2,
                "chunk_id": f"policy::{normalized}::2",
                "rrf_score": 0.05,
                "dense_score": None,
                "bm25_score": None,
                "payload": {
                    "source_type": "policy_pdf",
                    "control_id": None,
                    "doc_id": "04_logging_monitoring_standard",
                    "section_path": "4.2 Retention",
                    "chunk_text": "Retention policy excerpt.",
                },
            },
        ]
        captured_policy_query: dict[str, str] = {}

        def _side_effect(query: str, top_k: int = 10, **kwargs):
            source_type = kwargs.get("source_type")
            if source_type == "oscal_control":
                return control_chunks
            if source_type in {"policy_pdf", "policy_md"}:
                captured_policy_query["value"] = query
                return policy_chunks
            return []

        with patch("app.rag.answer.get_llm_client", return_value=_UnavailableLLMClient()):
            with patch("app.rag.answer.hybrid_search", side_effect=_side_effect):
                result = answer_question(
                    "How well do our policies cover this control today?",
                    eval_mode="policy_vs_control",
                    eval_intent="coverage_assessment",
                    top_k=6,
                )

        expanded = captured_policy_query.get("value", "")
        self.assertIn("Control context:", expanded)
        self.assertIn("audit event logging requirements", expanded.lower())
        debug = result.get("debug") or {}
        self.assertTrue(bool(debug.get("expanded_query_used")))
        self.assertIn("Control context:", str(debug.get("expanded_query_preview", "")))


if __name__ == "__main__":
    unittest.main()
