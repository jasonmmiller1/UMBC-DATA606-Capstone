#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python md_to_pdf.py <input.md> <output.pdf>")
        sys.exit(1)

    md_path = Path(sys.argv[1]).expanduser().resolve()
    pdf_path = Path(sys.argv[2]).expanduser().resolve()

    if not md_path.exists():
        raise FileNotFoundError(f"MD not found: {md_path}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    import markdown
    from weasyprint import HTML

    md_text = md_path.read_text(encoding="utf-8")
    html = markdown.markdown(md_text, extensions=["tables"])

    # Basic wrapper so it renders nicely
    html_doc = f"""
    <html>
      <head><meta charset="utf-8"></head>
      <body>{html}</body>
    </html>
    """

    HTML(string=html_doc).write_pdf(str(pdf_path))
    print(f"Wrote PDF: {pdf_path}")

if __name__ == "__main__":
    main()