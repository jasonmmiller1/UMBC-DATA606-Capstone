from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from app.ingest.chunkers import build_chunk_diagnostics, chunk_policy_markdown_files


class PolicyChunkingTest(unittest.TestCase):
    def test_policy_chunks_include_hierarchy_and_chunk_metadata(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        md_path = repo_root / "data/policies_synth_md_v2/04_logging_monitoring_standard.md"
        self.assertTrue(md_path.exists(), f"Missing policy markdown fixture: {md_path}")

        chunks = chunk_policy_markdown_files([md_path])
        self.assertFalse(chunks.empty, "Expected at least one policy chunk")

        expected_columns = {
            "section_path",
            "heading",
            "heading_level",
            "section_type",
            "chunk_type",
            "chunk_index",
            "source_file",
            "source_document_id",
            "title",
        }
        self.assertTrue(expected_columns.issubset(set(chunks.columns)))

        section_path = chunks["section_path"].fillna("").astype(str)
        self.assertTrue((section_path.str.len() > 0).all(), "section_path should be non-empty")

        headings = chunks["heading"].fillna("").astype(str)
        self.assertTrue((headings.str.len() > 0).all(), "heading should be populated")
        self.assertTrue(chunks["heading_level"].notna().all(), "heading_level should be populated")
        section_types = chunks["section_type"].fillna("").astype(str)
        self.assertTrue((section_types.str.len() > 0).all(), "section_type should be populated")
        chunk_types = chunks["chunk_type"].fillna("").astype(str)
        self.assertTrue((chunk_types.str.len() > 0).all(), "chunk_type should be populated")
        self.assertTrue(chunks["chunk_index"].notna().all(), "chunk_index should be populated")
        self.assertTrue(chunks["source_file"].fillna("").astype(str).str.endswith(".md").all())
        self.assertTrue(chunks["source_document_id"].fillna("").astype(str).str.len().gt(0).all())
        self.assertTrue(chunks["title"].fillna("").astype(str).str.len().gt(0).all())

        lengths = chunks["chunk_len_chars"].fillna(chunks["chunk_text"].astype(str).str.len()).astype(int)
        self.assertTrue((lengths > 0).all(), "chunk lengths should be positive")
        self.assertLessEqual(int(lengths.max()), 3000, "policy chunks should stay under the chunk cap")

    def test_chunks_do_not_merge_across_sibling_section_boundaries(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        md_path = repo_root / "data/policies_synth_md/_docling_test.md"
        self.assertTrue(md_path.exists(), f"Missing policy markdown fixture: {md_path}")

        chunks = chunk_policy_markdown_files([md_path])
        scope_chunks = chunks[chunks["heading"] == "2.0 Scope"]
        self.assertFalse(scope_chunks.empty, "Expected a chunk for 2.0 Scope")
        self.assertTrue(
            scope_chunks["chunk_text"].astype(str).map(lambda text: "3.0 Roles and Responsibilities" not in text).all()
        )

        procedure_chunks = chunks[chunks["heading"] == "5.0 Procedures"]
        self.assertFalse(procedure_chunks.empty, "Expected a chunk for 5.0 Procedures")
        self.assertTrue(
            procedure_chunks["chunk_text"].astype(str).map(lambda text: "6.0 Exceptions" not in text).all()
        )

    def test_nested_headings_preserve_hierarchical_section_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "policy.md"
            md_text = (
                "# Doc Title\n\n"
                "## 4.0 Logging Requirements\n\n"
                + ("A" * 420)
                + "\n\n"
                "### 4.2 Retention\n\n"
                + ("B" * 420)
                + "\n"
            )
            md_path.write_text(md_text, encoding="utf-8")

            chunks = chunk_policy_markdown_files([md_path])
            section_paths = set(chunks["section_path"].fillna("").astype(str).tolist())
            self.assertIn("Doc Title > 4.0 Logging Requirements", section_paths)
            self.assertIn("Doc Title > 4.0 Logging Requirements > 4.2 Retention", section_paths)

    def test_procedure_steps_split_on_item_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "procedure_policy.md"
            steps = "\n".join(
                [
                    f"{idx}. Complete step {idx:02d} with manager approval, role verification, and evidence capture."
                    for idx in range(1, 40)
                ]
            )
            md_text = "# Procedure Policy\n\n## 5.0 Procedures\n\n" + steps + "\n"
            md_path.write_text(md_text, encoding="utf-8")

            chunks = chunk_policy_markdown_files([md_path])
            procedure_chunks = chunks[chunks["section_path"] == "Procedure Policy > 5.0 Procedures"]
            self.assertGreaterEqual(len(procedure_chunks), 2, "Expected long procedures section to split")

            for chunk_text in procedure_chunks["chunk_text"].astype(str).tolist():
                lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
                self.assertEqual(lines[0], "5.0 Procedures")
                for body_line in lines[1:]:
                    self.assertRegex(body_line, r"^\d+\.")

            for idx in range(1, 40):
                needle = f"{idx}. Complete step {idx:02d}"
                self.assertTrue(
                    any(needle in chunk_text for chunk_text in procedure_chunks["chunk_text"].astype(str).tolist()),
                    f"Expected full procedure step to remain intact: {needle}",
                )

    def test_markdown_table_is_preserved_and_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "table_policy.md"
            table = "\n".join(
                [
                    "| Control | Requirement |",
                    "|---|---|",
                    "| AC-2 | Account provisioning approval required |",
                    "| AC-6 | Enforce least privilege by role |",
                    "| AU-12 | Centralized audit logging enabled |",
                ]
            )
            md_text = (
                "# Table Policy\n\n"
                "## 4.0 Control Mapping\n\n"
                "The control mapping table is authoritative for this policy.\n\n"
                f"{table}\n\n"
                + ("Supplementary narrative. " * 20)
            )
            md_path.write_text(md_text, encoding="utf-8")

            chunks = chunk_policy_markdown_files([md_path])
            table_chunks = chunks[chunks["chunk_type"] == "table"]
            self.assertFalse(table_chunks.empty, "Expected at least one chunk with table_present=True")
            self.assertTrue(table_chunks["table_present"].astype(bool).all())

            markdown_tables = table_chunks["table_markdown"].fillna("").astype(str).tolist()
            self.assertTrue(any(table in markdown for markdown in markdown_tables), "Expected intact markdown table text")

            chunk_texts = table_chunks["chunk_text"].astype(str).tolist()
            self.assertTrue(any("Table columns: Control, Requirement" in text for text in chunk_texts))
            self.assertTrue(any("Row 1: Control: AC-2" in text for text in chunk_texts))

    def test_large_table_splits_by_rows_with_header_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "large_table_policy.md"
            header = "| Control | Requirement | Evidence |"
            separator = "|---|---|---|"
            rows = [
                f"| AC-{idx:02d} | {'Requirement text ' * 8}{idx} | {'Evidence note ' * 8}{idx} |"
                for idx in range(1, 45)
            ]
            table = "\n".join([header, separator] + rows)
            md_text = "# Large Table Policy\n\n## 5.0 Mappings\n\n" + table + "\n"
            md_path.write_text(md_text, encoding="utf-8")

            chunks = chunk_policy_markdown_files([md_path])
            table_chunks = chunks[chunks["chunk_type"] == "table"]
            self.assertGreaterEqual(len(table_chunks), 2, "Expected oversized table to split into multiple chunks")

            for markdown in table_chunks["table_markdown"].fillna("").astype(str).tolist():
                self.assertIn(header, markdown)
                self.assertIn(separator, markdown)

            for text in table_chunks["chunk_text"].astype(str).tolist():
                self.assertIn("Table columns: Control, Requirement, Evidence", text)

    def test_chunk_diagnostics_report_source_counts_lengths_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "table_policy.md"
            md_text = (
                "# Diagnostics Policy\n\n"
                "## 4.0 Control Mapping\n\n"
                "| Control | Requirement |\n"
                "|---|---|\n"
                "| AC-2 | Approval required |\n"
                "| AC-6 | Least privilege enforced |\n"
            )
            md_path.write_text(md_text, encoding="utf-8")

            chunks = chunk_policy_markdown_files([md_path])
            diagnostics = build_chunk_diagnostics(chunks, sample_per_source=1)

            self.assertEqual(diagnostics["total_chunks"], len(chunks))
            self.assertEqual(
                diagnostics["table_chunks"],
                int((chunks["chunk_type"].fillna("").astype(str) == "table").sum()),
            )
            self.assertEqual(len(diagnostics["chunks_per_source_file"]), 1)
            self.assertEqual(diagnostics["chunks_per_source_file"][0]["total_chunks"], len(chunks))
            self.assertIn("median", diagnostics["chunk_length_chars"])
            self.assertEqual(len(diagnostics["sample_chunks"]), 1)
            self.assertEqual(diagnostics["sample_chunks"][0]["source_file"], str(md_path))


if __name__ == "__main__":
    unittest.main()
