from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.source_view import extract_markdown_section, is_policy_chunk, resolve_policy_source_view


class SourceViewTest(unittest.TestCase):
    def test_is_policy_chunk_detects_policy_sources(self) -> None:
        self.assertTrue(is_policy_chunk({"source_type": "policy_pdf"}))
        self.assertTrue(is_policy_chunk({"source_type": "policy_md"}))
        self.assertFalse(is_policy_chunk({"source_type": "oscal_control"}))

    def test_extract_markdown_section_returns_matching_heading_block(self) -> None:
        md_text = (
            "# Demo Policy\n\n"
            "## 1.0 Purpose\n\n"
            "Purpose text.\n\n"
            "## 4.0 Logging Requirements\n\n"
            "Overview text.\n\n"
            "### 4.2 Retention Requirements\n\n"
            "Retention text.\n\n"
            "## 5.0 Exceptions\n\n"
            "Exception text.\n"
        )

        label, section_text = extract_markdown_section(
            md_text,
            section_path="Demo Policy > 4.0 Logging Requirements > 4.2 Retention Requirements",
        )

        self.assertEqual(label, "4.2 Retention Requirements")
        self.assertIn("Retention text.", section_text or "")
        self.assertNotIn("Exception text.", section_text or "")

    def test_resolve_policy_source_view_prefers_existing_source_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            source_path = repo_root / "custom" / "policy.md"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("# Policy\n\n## 4.0 Policy Statements\n\nLeast privilege.\n", encoding="utf-8")

            view = resolve_policy_source_view(
                {
                    "source_type": "policy_pdf",
                    "doc_id": "02_access_control_policy",
                    "doc_title": "Access Control Policy",
                    "section_path": "Access Control Policy > 4.0 Policy Statements",
                    "source_file": str(source_path),
                    "chunk_text": "Least privilege.",
                },
                repo_root=repo_root,
                upload_md_dir=repo_root / "uploads_md",
                upload_pdf_dir=repo_root / "uploads_pdf",
            )

            self.assertIsNotNone(view)
            self.assertEqual(view.source_path, source_path)
            self.assertEqual(view.source_kind, "markdown")
            self.assertIn("Least privilege.", view.full_text or "")
            self.assertIn("4.0 Policy Statements", view.matched_section_text or "")

    def test_resolve_policy_source_view_falls_back_to_doc_id_markdown(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            policy_dir = repo_root / "data" / "policies_synth_md_v2"
            policy_dir.mkdir(parents=True, exist_ok=True)
            policy_path = policy_dir / "01_mini_ssp.md"
            policy_path.write_text("# Mini SSP\n\n## 5.0 Data Types\n\nAudit logs.\n", encoding="utf-8")

            view = resolve_policy_source_view(
                {
                    "source_type": "policy_pdf",
                    "doc_id": "01_mini_ssp",
                    "doc_title": "Mini SSP",
                    "section_path": "Mini SSP > 5.0 Data Types",
                    "chunk_text": "Audit logs.",
                },
                repo_root=repo_root,
                upload_md_dir=repo_root / "uploads_md",
                upload_pdf_dir=repo_root / "uploads_pdf",
            )

            self.assertIsNotNone(view)
            self.assertEqual(view.source_path, policy_path)
            self.assertEqual(view.source_kind, "markdown")
            self.assertIn("Audit logs.", view.full_text or "")

    def test_resolve_policy_source_view_falls_back_to_pdf_when_text_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            upload_pdf_dir = repo_root / "uploads_pdf"
            upload_pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = upload_pdf_dir / "upload_policy.pdf"
            pdf_path.write_bytes(b"%PDF-demo")

            view = resolve_policy_source_view(
                {
                    "source_type": "policy_pdf",
                    "doc_id": "upload_policy",
                    "doc_title": "Upload Policy",
                    "source_file": str(pdf_path),
                },
                repo_root=repo_root,
                upload_md_dir=repo_root / "uploads_md",
                upload_pdf_dir=upload_pdf_dir,
            )

            self.assertIsNotNone(view)
            self.assertEqual(view.source_path, pdf_path)
            self.assertEqual(view.source_kind, "pdf")
            self.assertIsNone(view.full_text)


if __name__ == "__main__":
    unittest.main()
