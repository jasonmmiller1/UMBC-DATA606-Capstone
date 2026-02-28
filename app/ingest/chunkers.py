from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
from typing import Dict, Iterable, List, Optional

import pandas as pd


CONTROL_ID_RE = re.compile(r"^[A-Z]{2,3}-\d+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class ChunkRecord:
    chunk_id: str
    chunk_text: str
    source_type: str
    control_id: Optional[str] = None
    control_part: Optional[str] = None
    enhancement_id: Optional[str] = None
    control_family: Optional[str] = None
    doc_id: Optional[str] = None
    doc_title: Optional[str] = None
    section_path: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


def _stable_chunk_id(prefix: str, parts: Iterable[Optional[str]]) -> str:
    payload = "::".join([(p or "") for p in parts])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}::{digest}"


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip())


def _to_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    # Parquet/object columns may materialize as tuples/arrays/series.
    if hasattr(value, "tolist"):
        as_list = value.tolist()
        if isinstance(as_list, list):
            return as_list
    if isinstance(value, tuple):
        return list(value)
    return []


def chunk_oscal_controls(oscal_df: pd.DataFrame) -> pd.DataFrame:
    records: List[ChunkRecord] = []

    for _, row in oscal_df.iterrows():
        control_id = str(row.get("control_id", "") or "").upper()
        title = str(row.get("title", "") or "")
        family = str(row.get("family", "") or "")
        doc_id = control_id or None
        doc_title = title or control_id or "OSCAL Control"

        statement = _clean_text(str(row.get("statement", "") or ""))
        guidance = _clean_text(str(row.get("guidance", "") or ""))
        enhancements = _to_list(row.get("enhancements", []))
        parameters = _to_list(row.get("parameters", []))

        if statement:
            records.append(
                ChunkRecord(
                    chunk_id=_stable_chunk_id("oscal", [control_id, "statement", statement[:120]]),
                    chunk_text=statement,
                    source_type="oscal_control",
                    control_id=control_id,
                    control_part="statement",
                    control_family=family,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_path=f"{control_id} Statement" if control_id else "Statement",
                )
            )

        if guidance:
            records.append(
                ChunkRecord(
                    chunk_id=_stable_chunk_id("oscal", [control_id, "guidance", guidance[:120]]),
                    chunk_text=guidance,
                    source_type="oscal_control",
                    control_id=control_id,
                    control_part="guidance",
                    control_family=family,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_path=f"{control_id} Guidance" if control_id else "Guidance",
                )
            )

        for enh in enhancements:
            enh_id = str(enh.get("control_id", "") or "").upper() or None
            enh_title = str(enh.get("title", "") or "").strip()
            enh_statement = _clean_text(str(enh.get("statement", "") or ""))
            if not enh_statement:
                continue
            section = f"Enhancement {enh_id}" if enh_id else "Enhancement"
            if enh_title:
                section = f"{section}: {enh_title}"
            records.append(
                ChunkRecord(
                    chunk_id=_stable_chunk_id("oscal", [control_id, "enhancement", enh_id, enh_statement[:120]]),
                    chunk_text=enh_statement,
                    source_type="oscal_control",
                    control_id=control_id,
                    control_part="enhancement",
                    enhancement_id=enh_id,
                    control_family=family,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_path=section,
                )
            )

        for idx, param in enumerate(parameters, start=1):
            param_text = _clean_text(str(param))
            if not param_text:
                continue
            records.append(
                ChunkRecord(
                    chunk_id=_stable_chunk_id("oscal", [control_id, "parameter", str(idx), param_text[:120]]),
                    chunk_text=param_text,
                    source_type="oscal_control",
                    control_id=control_id,
                    control_part="parameter",
                    control_family=family,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_path=f"{control_id} Parameter {idx}" if control_id else f"Parameter {idx}",
                )
            )

    return pd.DataFrame([r.__dict__ for r in records])


def _markdown_chunks(md_text: str) -> List[Dict[str, str]]:
    lines = md_text.splitlines()
    chunks: List[Dict[str, str]] = []
    heading_stack: Dict[int, str] = {}
    current_section = "Document"
    body_lines: List[str] = []

    def flush():
        nonlocal body_lines, current_section
        text = _clean_text("\n".join(body_lines))
        if text:
            chunks.append({"section_path": current_section, "chunk_text": text})
        body_lines = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack[level] = title
            for key in [k for k in heading_stack if k > level]:
                del heading_stack[key]
            current_section = " > ".join([heading_stack[k] for k in sorted(heading_stack)])
        else:
            body_lines.append(line)

    flush()
    return chunks


def chunk_policy_markdown_files(md_paths: Iterable[Path]) -> pd.DataFrame:
    records: List[ChunkRecord] = []
    for md_path in md_paths:
        text = md_path.read_text(encoding="utf-8")
        doc_id = md_path.stem
        for idx, chunk in enumerate(_markdown_chunks(text), start=1):
            chunk_text = chunk["chunk_text"]
            section_path = chunk["section_path"]
            records.append(
                ChunkRecord(
                    chunk_id=_stable_chunk_id("policy", [doc_id, str(idx), section_path, chunk_text[:120]]),
                    chunk_text=chunk_text,
                    source_type="policy_pdf",
                    doc_id=doc_id,
                    doc_title=doc_id.replace("_", " ").title(),
                    section_path=section_path,
                )
            )
    return pd.DataFrame([r.__dict__ for r in records])


def load_policy_markdown_paths(repo_root: Path) -> List[Path]:
    preferred_dirs = [
        repo_root / "data/policies_synth_md_v2",
        repo_root / "data/policies_synth_md_v0",
        repo_root / "data/policies_synth_md",
    ]
    for directory in preferred_dirs:
        if directory.exists():
            files = sorted([p for p in directory.glob("*.md") if not p.name.startswith("_")])
            if files:
                return files
    return []


def build_chunks_dataframe(
    repo_root: Path,
    oscal_parquet_path: Optional[Path] = None,
    policy_md_paths: Optional[List[Path]] = None,
) -> pd.DataFrame:
    if oscal_parquet_path is None:
        oscal_parquet_path = repo_root / "data/oscal_parsed/controls_80053.parquet"
    if policy_md_paths is None:
        policy_md_paths = load_policy_markdown_paths(repo_root)
    if not oscal_parquet_path.exists():
        raise FileNotFoundError(f"Missing OSCAL parquet file: {oscal_parquet_path}")
    if not policy_md_paths:
        raise FileNotFoundError("No policy markdown files found under data/policies_synth_md*")

    oscal_df = pd.read_parquet(oscal_parquet_path)
    oscal_chunks = chunk_oscal_controls(oscal_df)
    policy_chunks = chunk_policy_markdown_files(policy_md_paths)

    all_chunks = pd.concat([oscal_chunks, policy_chunks], ignore_index=True)
    all_chunks = all_chunks.dropna(subset=["chunk_id", "chunk_text"])
    all_chunks = all_chunks.drop_duplicates(subset=["chunk_id"]).reset_index(drop=True)
    return all_chunks


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    chunks = build_chunks_dataframe(repo_root)
    out_dir = repo_root / "data/index"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.parquet"
    chunks.to_parquet(out_path, index=False)
    summary = {
        "rows": int(len(chunks)),
        "source_counts": chunks["source_type"].value_counts().to_dict(),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
