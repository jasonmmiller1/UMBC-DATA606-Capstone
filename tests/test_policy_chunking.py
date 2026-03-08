from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from app.ingest.chunkers import chunk_policy_markdown_files


class PolicyChunkingTest(unittest.TestCase):
    def test_policy_chunks_include_hierarchy_and_heading_metadata(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        md_path = repo_root / "data/policies_synth_md_v2/04_logging_monitoring_standard.md"
        self.assertTrue(md_path.exists(), f"Missing policy markdown fixture: {md_path}")

        chunks = chunk_policy_markdown_files([md_path])
        self.assertFalse(chunks.empty, "Expected at least one policy chunk")

        section_path = chunks["section_path"].fillna("").astype(str)
        self.assertTrue((section_path.str.len() > 0).all(), "section_path should be non-empty")

        headings = chunks["heading"].fillna("").astype(str)
        self.assertTrue((headings.str.len() > 0).all(), "heading should be populated")
        self.assertTrue(chunks["heading_level"].notna().all(), "heading_level should be populated")
        section_types = chunks["section_type"].fillna("").astype(str)
        self.assertTrue((section_types.str.len() > 0).all(), "section_type should be populated")

        lengths = chunks["chunk_len_chars"].fillna(chunks["chunk_text"].astype(str).str.len()).astype(int)
        in_range = (lengths >= 400) & (lengths <= 3000)
        ratio = float(in_range.mean())
        self.assertGreaterEqual(
            ratio,
            0.75,
            f"Expected most chunks in [400, 3000] chars, got ratio={ratio:.2f}",
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
            table_chunks = chunks[chunks["table_present"] == True]
            self.assertFalse(table_chunks.empty, "Expected at least one chunk with table_present=True")

            chunk_texts = table_chunks["chunk_text"].astype(str).tolist()
            self.assertTrue(any(table in text for text in chunk_texts), "Expected intact markdown table text")

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
            table_chunks = chunks[chunks["table_present"] == True]
            self.assertGreaterEqual(len(table_chunks), 2, "Expected oversized table to split into multiple chunks")

            for text in table_chunks["chunk_text"].astype(str).tolist():
                self.assertIn(header, text)
                self.assertIn(separator, text)


if __name__ == "__main__":
    unittest.main()
