#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def main():
    if len(sys.argv) != 3:
        print("Usage: python md_to_pdf_simple.py <input.md> <output.pdf>")
        sys.exit(1)

    md_path = Path(sys.argv[1]).expanduser().resolve()
    pdf_path = Path(sys.argv[2]).expanduser().resolve()

    if not md_path.exists():
        raise FileNotFoundError(f"MD not found: {md_path}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    lines = md_path.read_text(encoding="utf-8").splitlines()

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    x = 72
    y = height - 72
    line_height = 14

    for line in lines:
        if y < 72:
            c.showPage()
            y = height - 72
        # trim very long lines to keep PDF generation simple
        c.drawString(x, y, line[:140])
        y -= line_height

    c.save()
    print(f"Wrote PDF: {pdf_path}")

if __name__ == "__main__":
    main()
