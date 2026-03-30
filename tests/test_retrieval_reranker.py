from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from app.retrieval.retrieve import _rerank_candidates, get_retrieval_config_snapshot, hybrid_retrieve, rrf_fuse


class RetrievalRerankerTest(unittest.TestCase):
    def test_rrf_fuse_respects_dense_and_bm25_weights(self) -> None:
        dense_ids = ["dense-first", "bm25-first"]
        bm25_ids = ["bm25-first", "dense-first"]

        dense_lean = rrf_fuse(dense_ids, bm25_ids, k=60, dense_weight=1.2, bm25_weight=0.8)
        bm25_lean = rrf_fuse(dense_ids, bm25_ids, k=60, dense_weight=0.8, bm25_weight=1.2)

        self.assertEqual(dense_lean[0][0], "dense-first")
        self.assertEqual(bm25_lean[0][0], "bm25-first")

    def test_config_snapshot_reads_env_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RETRIEVAL_DENSE_K": "24",
                "RETRIEVAL_BM25_K": "18",
                "RETRIEVAL_DENSE_WEIGHT": "1.15",
                "RETRIEVAL_BM25_WEIGHT": "0.85",
                "RERANK_ENABLED": "true",
                "RERANK_CANDIDATES": "30",
            },
            clear=False,
        ):
            snapshot = get_retrieval_config_snapshot(top_k=9)

        self.assertEqual(snapshot["top_k"], 9)
        self.assertEqual(snapshot["dense_k"], 24)
        self.assertEqual(snapshot["bm25_k"], 18)
        self.assertAlmostEqual(float(snapshot["dense_weight"]), 1.15)
        self.assertAlmostEqual(float(snapshot["bm25_weight"]), 0.85)
        self.assertTrue(bool(snapshot["rerank_enabled"]))
        self.assertEqual(int(snapshot["rerank_candidates"]), 30)

    def test_reranker_boosts_control_id_and_heading_overlap(self) -> None:
        query = "What are the AC-2 retention requirements?"
        candidates = [
            {
                "chunk_id": "generic-purpose",
                "rrf_score": 0.22,
                "base_rrf_score": 0.22,
                "payload": {
                    "source_type": "policy_pdf",
                    "section_type": "purpose",
                    "heading": "1.0 Purpose",
                    "section_path": "Policy > 1.0 Purpose",
                    "control_id": "",
                },
            },
            {
                "chunk_id": "ac2-retention",
                "rrf_score": 0.20,
                "base_rrf_score": 0.20,
                "payload": {
                    "source_type": "policy_pdf",
                    "section_type": "requirements",
                    "heading": "4.2 Retention Requirements",
                    "section_path": "Policy > 4.0 Logging Requirements > 4.2 Retention Requirements",
                    "control_id": "AC-2",
                },
            },
        ]

        reranked = _rerank_candidates(query=query, candidates=candidates, top_k=2)
        self.assertEqual(reranked[0]["chunk_id"], "ac2-retention")
        self.assertGreater(float(reranked[0]["rerank_delta"]), 0.0)
        self.assertLess(float(reranked[1]["rerank_delta"]), 0.0)

    def test_reranker_does_not_penalize_scope_when_query_asks_scope(self) -> None:
        query = "What is the scope of this policy?"
        candidates = [
            {
                "chunk_id": "scope-section",
                "rrf_score": 0.10,
                "base_rrf_score": 0.10,
                "payload": {
                    "source_type": "policy_pdf",
                    "section_type": "scope",
                    "heading": "2.0 Scope",
                    "section_path": "Policy > 2.0 Scope",
                    "control_id": "",
                },
            }
        ]

        reranked = _rerank_candidates(query=query, candidates=candidates, top_k=1)
        self.assertGreaterEqual(float(reranked[0]["rerank_delta"]), 0.0)

    @patch("app.retrieval.retrieve._chunks_lookup")
    @patch("app.retrieval.retrieve.retrieve_bm25")
    @patch("app.retrieval.retrieve.retrieve_dense")
    def test_hybrid_retrieve_respects_enable_reranker_flag(
        self,
        mock_retrieve_dense,
        mock_retrieve_bm25,
        mock_chunks_lookup,
    ) -> None:
        # Baseline ranks "generic-purpose" higher than "ac2-retention".
        mock_retrieve_dense.return_value = [
            {"chunk_id": "generic-purpose", "score": 0.9, "payload": {"chunk_id": "generic-purpose"}},
            {"chunk_id": "ac2-retention", "score": 0.8, "payload": {"chunk_id": "ac2-retention"}},
        ]
        mock_retrieve_bm25.return_value = [("generic-purpose", 8.0), ("ac2-retention", 7.0)]
        mock_chunks_lookup.return_value = {
            "generic-purpose": {
                "chunk_id": "generic-purpose",
                "chunk_text": "General purpose statement.",
                "source_type": "oscal_control",
                "section_type": "purpose",
                "heading": "1.0 Purpose",
                "section_path": "Policy > 1.0 Purpose",
                "control_id": "AC-1",
            },
            "ac2-retention": {
                "chunk_id": "ac2-retention",
                "chunk_text": "AC-2 retention and account management requirements.",
                "source_type": "oscal_control",
                "section_type": "requirements",
                "heading": "4.2 Retention Requirements",
                "section_path": "Policy > 4.0 Logging Requirements > 4.2 Retention Requirements",
                "control_id": "AC-2",
            },
        }

        query = "What are AC-2 retention requirements?"
        fake_client = object()
        fake_path = Path("unused")

        with patch.dict(os.environ, {"RERANK_ENABLED": "false", "ENABLE_RERANKER": "false"}, clear=False):
            baseline = hybrid_retrieve(
                query=query,
                client=fake_client,
                bm25_index_path=fake_path,
                chunks_path=fake_path,
                top_k=1,
                dense_k=2,
                bm25_k=2,
                intent="policy",
            )
        self.assertEqual(baseline[0]["chunk_id"], "generic-purpose")

        with patch.dict(os.environ, {"RERANK_ENABLED": "true", "ENABLE_RERANKER": "true"}, clear=False):
            reranked = hybrid_retrieve(
                query=query,
                client=fake_client,
                bm25_index_path=fake_path,
                chunks_path=fake_path,
                top_k=1,
                dense_k=2,
                bm25_k=2,
                intent="policy",
            )
        self.assertEqual(reranked[0]["chunk_id"], "ac2-retention")

    @patch("app.retrieval.retrieve._chunks_lookup")
    @patch("app.retrieval.retrieve.retrieve_bm25")
    @patch("app.retrieval.retrieve.retrieve_dense")
    def test_hybrid_retrieve_skips_reranker_for_non_policy_intent(
        self,
        mock_retrieve_dense,
        mock_retrieve_bm25,
        mock_chunks_lookup,
    ) -> None:
        mock_retrieve_dense.return_value = [
            {"chunk_id": "generic-purpose", "score": 0.9, "payload": {"chunk_id": "generic-purpose"}},
            {"chunk_id": "ac2-retention", "score": 0.8, "payload": {"chunk_id": "ac2-retention"}},
        ]
        mock_retrieve_bm25.return_value = [("generic-purpose", 8.0), ("ac2-retention", 7.0)]
        mock_chunks_lookup.return_value = {
            "generic-purpose": {
                "chunk_id": "generic-purpose",
                "chunk_text": "General purpose statement.",
                "source_type": "oscal_control",
                "section_type": "purpose",
                "heading": "1.0 Purpose",
                "section_path": "Policy > 1.0 Purpose",
                "control_id": "AC-1",
            },
            "ac2-retention": {
                "chunk_id": "ac2-retention",
                "chunk_text": "AC-2 retention and account management requirements.",
                "source_type": "oscal_control",
                "section_type": "requirements",
                "heading": "4.2 Retention Requirements",
                "section_path": "Policy > 4.0 Logging Requirements > 4.2 Retention Requirements",
                "control_id": "AC-2",
            },
        }

        query = "What are AC-2 retention requirements?"
        fake_client = object()
        fake_path = Path("unused")
        with patch.dict(os.environ, {"RERANK_ENABLED": "true"}, clear=False):
            results = hybrid_retrieve(
                query=query,
                client=fake_client,
                bm25_index_path=fake_path,
                chunks_path=fake_path,
                top_k=1,
                dense_k=2,
                bm25_k=2,
                intent="framework",
            )
        self.assertEqual(results[0]["chunk_id"], "generic-purpose")

    @patch("app.retrieval.retrieve._chunks_lookup")
    @patch("app.retrieval.retrieve.retrieve_bm25")
    @patch("app.retrieval.retrieve.retrieve_dense")
    @patch("app.retrieval.retrieve._rerank_candidates", side_effect=RuntimeError("boom"))
    def test_hybrid_retrieve_falls_back_when_reranker_errors(
        self,
        _mock_rerank,
        mock_retrieve_dense,
        mock_retrieve_bm25,
        mock_chunks_lookup,
    ) -> None:
        mock_retrieve_dense.return_value = [
            {"chunk_id": "generic-purpose", "score": 0.9, "payload": {"chunk_id": "generic-purpose"}},
            {"chunk_id": "ac2-retention", "score": 0.8, "payload": {"chunk_id": "ac2-retention"}},
        ]
        mock_retrieve_bm25.return_value = [("generic-purpose", 8.0), ("ac2-retention", 7.0)]
        mock_chunks_lookup.return_value = {
            "generic-purpose": {
                "chunk_id": "generic-purpose",
                "chunk_text": "General purpose statement.",
                "source_type": "oscal_control",
                "section_type": "purpose",
                "heading": "1.0 Purpose",
                "section_path": "Policy > 1.0 Purpose",
                "control_id": "AC-1",
            },
            "ac2-retention": {
                "chunk_id": "ac2-retention",
                "chunk_text": "AC-2 retention and account management requirements.",
                "source_type": "oscal_control",
                "section_type": "requirements",
                "heading": "4.2 Retention Requirements",
                "section_path": "Policy > 4.0 Logging Requirements > 4.2 Retention Requirements",
                "control_id": "AC-2",
            },
        }

        query = "What are AC-2 retention requirements?"
        fake_client = object()
        fake_path = Path("unused")
        with patch.dict(os.environ, {"RERANK_ENABLED": "true"}, clear=False):
            with self.assertLogs("app.retrieval.retrieve", level="WARNING"):
                results = hybrid_retrieve(
                    query=query,
                    client=fake_client,
                    bm25_index_path=fake_path,
                    chunks_path=fake_path,
                    top_k=1,
                    dense_k=2,
                    bm25_k=2,
                    intent="policy",
                )
        self.assertEqual(results[0]["chunk_id"], "generic-purpose")


if __name__ == "__main__":
    unittest.main()
