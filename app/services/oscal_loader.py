#!/usr/bin/env python3
"""
Parse NIST SP 800-53 Rev 5 OSCAL catalog JSON into a tidy controls dataset.

Outputs:
- data/oscal_parsed/controls_80053_rev5.jsonl
- data/oscal_parsed/controls_80053_rev5.parquet

Designed for Week 1: "good enough" structure for RAG + evaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


RE_WHITESPACE = re.compile(r"\s+")


def norm_text(s: str) -> str:
    """Normalize whitespace, preserve content."""
    return RE_WHITESPACE.sub(" ", s).strip()


def deep_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_parts_text(parts: List[Dict[str, Any]]) -> str:
    """
    OSCAL 'parts' are nested structures with prose typically in 'prose'.
    We flatten into readable text while keeping headings/labels when available.
    """
    lines: List[str] = []

    def walk(part: Dict[str, Any], depth: int = 0):
        name = part.get("name")
        label = part.get("label")
        prose = part.get("prose")

        prefix = "  " * depth
        header_bits = [b for b in [label, name] if b]
        if header_bits and prose:
            lines.append(prefix + f"[{'/'.join(header_bits)}] {norm_text(prose)}")
        elif prose:
            lines.append(prefix + norm_text(prose))
        elif header_bits:
            lines.append(prefix + f"[{'/'.join(header_bits)}]")

        for sp in part.get("parts", []) or []:
            walk(sp, depth + 1)

    for p in parts or []:
        walk(p, 0)

    return "\n".join(lines).strip()


def extract_parameters(parts: List[Dict[str, Any]]) -> List[str]:
    """
    Pull parameter-like placeholders from prose such as:
    - [Assignment: organization-defined ...]
    - [Selection: (one or more) ...]
    This is heuristic and fine for Week 1.
    """
    text = extract_parts_text(parts)
    params = re.findall(r"\[(Assignment|Selection):[^\]]+\]", text, flags=re.IGNORECASE)
    # Return unique markers (keep it simple)
    return sorted(set([p.lower() for p in params]))


@dataclass
class ControlRecord:
    control_id: str
    title: str
    family: str
    statement: str
    guidance: str
    enhancements: List[Dict[str, str]]
    parameters: List[str]
    source: str
    source_file: str
    oscal_path: str


def parse_control(control: Dict[str, Any], family: str, source_file: str, oscal_path: str) -> ControlRecord:
    control_id = (control.get("id") or "").upper()
    title = control.get("title") or ""

    parts = control.get("parts", []) or []
    # Heuristic: statements often live under part name "statement"
    statement_parts = [p for p in parts if (p.get("name") == "statement")]
    guidance_parts = [p for p in parts if (p.get("name") in {"guidance", "supplemental-guidance"} )]

    # If not found, fall back to all parts
    statement = extract_parts_text(statement_parts) if statement_parts else extract_parts_text(parts)
    guidance = extract_parts_text(guidance_parts) if guidance_parts else ""

    # Enhancements are child controls in "controls"
    enhancements = []
    for child in control.get("controls", []) or []:
        cid = (child.get("id") or "").upper()
        ctitle = child.get("title") or ""
        cparts = child.get("parts", []) or []
        cstatement_parts = [p for p in cparts if (p.get("name") == "statement")]
        cstatement = extract_parts_text(cstatement_parts) if cstatement_parts else extract_parts_text(cparts)
        enhancements.append({"control_id": cid, "title": ctitle, "statement": cstatement})

    params = extract_parameters(parts)

    return ControlRecord(
        control_id=control_id,
        title=title,
        family=family,
        statement=statement,
        guidance=guidance,
        enhancements=enhancements,
        parameters=params,
        source="oscal_nist_sp800-53_rev5",
        source_file=source_file,
        oscal_path=oscal_path,
    )


def main():
    repo_root = Path(__file__).resolve().parents[2]  # app/services/ -> repo root
    src = repo_root / "data/oscal_raw/oscal-content/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"
    out_dir = repo_root / "data/oscal_parsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"OSCAL catalog not found at: {src}")

    catalog = json.loads(src.read_text(encoding="utf-8"))
    groups = deep_get(catalog, ["catalog", "groups"], default=[]) or []

    records: List[ControlRecord] = []

    # In 800-53 OSCAL, top-level groups generally correspond to families (AC, AU, IR...)
    for gi, group in enumerate(groups):
        family = (group.get("id") or group.get("title") or f"group_{gi}").upper()
        controls = group.get("controls", []) or []
        for ci, control in enumerate(controls):
            oscal_path = f"catalog.groups[{gi}].controls[{ci}]"
            try:
                rec = parse_control(control, family=family, source_file=str(src), oscal_path=oscal_path)
                if rec.control_id:  # skip empty
                    records.append(rec)
            except Exception as e:
                # Keep going; log minimal context
                print(f"Failed parsing {family} control index {ci}: {e}")

    df = pd.DataFrame([asdict(r) for r in records])

    # Write outputs
    jsonl_path = out_dir / "controls_80053.jsonl"
    parquet_path = out_dir / "controls_80053.parquet"


    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    df.to_parquet(parquet_path, index=False)

    print(f"Parsed controls: {len(df)}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {parquet_path}")

    # Print a few samples
    sample_ids = ["AC-2", "AU-2", "IR-4"]
    print("\nSample controls:")
    for sid in sample_ids:
        row = df[df["control_id"] == sid]
        if not row.empty:
            print(f"\n== {sid} ==")
            print(row.iloc[0]["title"])
            print(row.iloc[0]["statement"][:600], "...")


if __name__ == "__main__":
    main()
