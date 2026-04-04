from __future__ import annotations

import unittest

from app.rag.answer_state import derive_answer_view_state


def _base_result(**overrides):
    result = {
        "draft_answer": "Evidence:\n- Access requires approval [C1]\nCitations:\n- [C1] Policy excerpt",
        "abstained": False,
        "confidence": 0.82,
        "citations": [{"citation_id": "C1"}],
        "retrieved_chunks": [
            {
                "citation_id": "C1",
                "doc_id": "02_access_control_policy",
                "chunk_text": "Access shall be approved before provisioning.",
                "source_type": "policy_pdf",
                "section_path": "4.0 Policy Statements",
            },
            {
                "citation_id": "C2",
                "doc_id": "02_access_control_policy",
                "chunk_text": "Managers approve access requests before provisioning.",
                "source_type": "policy_pdf",
                "section_path": "5.0 Procedures",
            },
        ],
        "weak_retrieval": False,
        "retrieval_status": "ok",
        "retrieval_error_type": None,
        "llm_status": "ok",
        "llm_error_type": None,
    }
    result.update(overrides)
    return result


class AnswerStateViewTest(unittest.TestCase):
    def test_strong_answer_state_and_display_body(self) -> None:
        view = derive_answer_view_state(_base_result())

        self.assertEqual(view.state, "strong_answer")
        self.assertEqual(view.answer_label, "Generated explanation")
        self.assertEqual(view.answer_body, "- Access requires approval [C1]")

    def test_partial_answer_state_when_abstaining_with_related_evidence(self) -> None:
        view = derive_answer_view_state(
            _base_result(
                draft_answer="insufficient evidence. Related evidence was found but it is indirect.",
                abstained=True,
                confidence=0.2,
                weak_retrieval=True,
            )
        )

        self.assertEqual(view.state, "partial_answer")
        self.assertEqual(view.answer_label, "System response")

    def test_no_evidence_state(self) -> None:
        view = derive_answer_view_state(
            _base_result(
                draft_answer="insufficient evidence. No retrieved evidence chunks were found.",
                citations=[],
                retrieved_chunks=[],
                retrieval_status="no_evidence",
                abstained=True,
                confidence=0.0,
            )
        )

        self.assertEqual(view.state, "no_evidence")
        self.assertEqual(view.tone, "warning")

    def test_conflicting_evidence_state(self) -> None:
        view = derive_answer_view_state(
            _base_result(
                draft_answer="The sources are conflicting and do not support a single conclusion.",
                abstained=True,
            )
        )

        self.assertEqual(view.state, "conflicting_evidence")
        self.assertEqual(view.tone, "warning")

    def test_retrieval_only_state(self) -> None:
        view = derive_answer_view_state(
            _base_result(
                draft_answer="LLM temporarily unavailable. Showing retrieved evidence excerpts.\n\nTop evidence excerpts:\n- [C1] Access shall be approved before provisioning.",
                llm_status="unavailable",
                abstained=False,
            )
        )

        self.assertEqual(view.state, "retrieval_only")
        self.assertEqual(view.answer_label, "Retrieved evidence summary")
        self.assertTrue(view.answer_body.startswith("Top evidence excerpts"))

    def test_backend_error_state(self) -> None:
        view = derive_answer_view_state(
            _base_result(
                draft_answer="insufficient evidence. Backend request timed out before a reliable answer was available.",
                citations=[],
                retrieved_chunks=[],
                retrieval_status="timeout",
                retrieval_error_type="timeout",
                llm_status="not_requested",
                abstained=True,
                confidence=0.0,
            )
        )

        self.assertEqual(view.state, "backend_error")
        self.assertEqual(view.tone, "error")


if __name__ == "__main__":
    unittest.main()
