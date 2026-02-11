#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import html
import re

def clean_markdown(md: str) -> str:
    # Unescape HTML entities (&gt; etc.)
    md = html.unescape(md)

    # Fix duplicated headings like "## ## 1.0 Purpose" -> "## 1.0 Purpose"
    md = re.sub(r"^(#{1,6})\s+\1\s+", r"\1 ", md, flags=re.MULTILINE)

    # Normalize weird spacing
    md = re.sub(r"\n{3,}", "\n\n", md)

    return md


def main():
    if len(sys.argv) != 3:
        print("Usage: python pdf_to_md.py <input.pdf> <output.md>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    md_path  = Path(sys.argv[2]).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    md_path.parent.mkdir(parents=True, exist_ok=True)

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    md = result.document.export_to_markdown()
    md = clean_markdown(md)
    md_path.write_text(md, encoding="utf-8")


    print(f"Wrote markdown: {md_path}")
    print(f"Chars: {len(md)}")

if __name__ == "__main__":
    main()