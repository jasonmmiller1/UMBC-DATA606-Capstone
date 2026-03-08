from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


CONTROL_ID_RE = re.compile(r"^[A-Z]{2,3}-\d+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")

POLICY_MIN_CHUNK_CHARS = 400
POLICY_MAX_CHUNK_CHARS = 2800
SECTION_TYPES = {
    "purpose",
    "scope",
    "definitions",
    "exceptions",
    "policy",
    "procedures",
    "requirements",
    "retention",
    "roles",
    "controls",
    "other",
}


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
    heading: Optional[str] = None
    heading_level: Optional[int] = None
    section_type: Optional[str] = None
    chunk_len_chars: Optional[int] = None
    table_present: Optional[bool] = False
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class _PolicySection:
    heading: str
    heading_level: int
    section_path: str
    parent_path: str
    chunk_text: str


@dataclass
class _ChunkBlock:
    kind: str
    text: str


def _stable_chunk_id(prefix: str, parts: Iterable[Optional[str]]) -> str:
    payload = "::".join([(p or "") for p in parts])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}::{digest}"


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip())


def _normalize_title(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().strip("#").strip())
    return normalized


def _default_doc_title(doc_id: str) -> str:
    return doc_id.replace("_", " ").title()


def _infer_policy_section_type(heading: str) -> str:
    text = (heading or "").strip().lower()
    if not text:
        return "other"

    if "purpose" in text:
        return "purpose"
    if "scope" in text:
        return "scope"
    if "definition" in text or "glossary" in text or "terminology" in text or "terms" in text:
        return "definitions"
    if "exception" in text:
        return "exceptions"
    if "procedur" in text or "process" in text or "workflow" in text:
        return "procedures"
    if "retention" in text:
        return "retention"
    if "role" in text or "responsibilit" in text:
        return "roles"
    if "requirement" in text or "shall" in text:
        return "requirements"
    if "control" in text or "mapping" in text:
        return "controls"
    if "policy" in text or "statement" in text or "standard" in text:
        return "policy"
    return "other"


def _table_cells(line: str) -> List[str]:
    stripped = (line or "").strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_markdown_table_separator_row(line: str) -> bool:
    cells = _table_cells(line)
    if len(cells) < 2:
        return False
    return all(TABLE_SEPARATOR_CELL_RE.match(cell.replace(" ", "")) for cell in cells)


def _is_markdown_table_row(line: str) -> bool:
    if "|" not in (line or ""):
        return False
    cells = _table_cells(line)
    return len(cells) >= 2


def _is_markdown_table_start(lines: List[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = lines[index]
    separator = lines[index + 1]
    if not _is_markdown_table_row(header):
        return False
    return _is_markdown_table_separator_row(separator)


def _extract_chunk_blocks(chunk_text: str) -> List[_ChunkBlock]:
    lines = (chunk_text or "").splitlines()
    if not lines:
        return []

    blocks: List[_ChunkBlock] = []
    text_lines: List[str] = []

    def flush_text() -> None:
        nonlocal text_lines
        text = _clean_text("\n".join(text_lines))
        if text:
            blocks.append(_ChunkBlock(kind="text", text=text))
        text_lines = []

    idx = 0
    while idx < len(lines):
        if _is_markdown_table_start(lines, idx):
            flush_text()
            table_lines = [lines[idx], lines[idx + 1]]
            idx += 2
            while idx < len(lines) and _is_markdown_table_row(lines[idx]):
                table_lines.append(lines[idx])
                idx += 1
            blocks.append(_ChunkBlock(kind="table", text="\n".join(table_lines).rstrip()))
            continue

        text_lines.append(lines[idx])
        idx += 1

    flush_text()
    return blocks


def _chunk_contains_markdown_table(chunk_text: str) -> bool:
    for block in _extract_chunk_blocks(chunk_text):
        if block.kind == "table":
            return True
    return False


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
                    section_type="controls",
                    chunk_len_chars=len(statement),
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
                    section_type="controls",
                    chunk_len_chars=len(guidance),
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
                    section_type="controls",
                    chunk_len_chars=len(enh_statement),
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
                    section_type="controls",
                    chunk_len_chars=len(param_text),
                )
            )

    return pd.DataFrame([r.__dict__ for r in records])


def _infer_doc_title(lines: List[str], fallback: str) -> str:
    first_heading: Optional[str] = None
    for line in lines:
        m = HEADING_RE.match(line)
        if not m:
            continue
        title = _normalize_title(m.group(2))
        if not title:
            continue
        if first_heading is None:
            first_heading = title
        if len(m.group(1)) == 1:
            return title
    return first_heading or fallback


def _format_path(doc_title: str, heading_stack: Dict[int, str]) -> Tuple[str, str]:
    if not heading_stack:
        return doc_title, doc_title

    segments: List[str] = [doc_title]
    for level in sorted(heading_stack):
        title = heading_stack[level]
        if level == 1 and title == doc_title:
            continue
        segments.append(title)

    if not segments:
        segments = [doc_title]
    section_path = " > ".join(segments)
    parent_segments = segments[:-1] if len(segments) > 1 else segments
    parent_path = " > ".join(parent_segments)
    return section_path, parent_path


def _markdown_sections(md_text: str, *, doc_title: str) -> List[_PolicySection]:
    lines = md_text.splitlines()
    sections: List[_PolicySection] = []
    heading_stack: Dict[int, str] = {}
    current_heading = doc_title
    current_level = 1
    current_section = doc_title
    current_parent = doc_title
    body_lines: List[str] = []

    def flush() -> None:
        nonlocal body_lines
        text = _clean_text("\n".join(body_lines))
        if text:
            chunk_text = _clean_text(f"{current_heading}\n\n{text}")
            sections.append(
                _PolicySection(
                    heading=current_heading,
                    heading_level=current_level,
                    section_path=current_section,
                    parent_path=current_parent,
                    chunk_text=chunk_text,
                )
            )
        body_lines = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = _normalize_title(m.group(2))
            if not title:
                continue
            heading_stack[level] = title
            for key in [k for k in heading_stack if k > level]:
                del heading_stack[key]

            current_heading = title
            current_level = level
            current_section, current_parent = _format_path(doc_title, heading_stack)
        else:
            body_lines.append(line)

    flush()
    return sections


def _find_next_sibling_index(sections: List[_PolicySection], index: int) -> Optional[int]:
    current = sections[index]
    for idx in range(index + 1, len(sections)):
        candidate = sections[idx]
        if (
            candidate.heading_level == current.heading_level
            and candidate.parent_path == current.parent_path
        ):
            return idx
    return None


def _merge_tiny_policy_sections(
    sections: List[_PolicySection],
    *,
    min_chars: int,
) -> List[_PolicySection]:
    if not sections:
        return []
    working = list(sections)
    merged: List[_PolicySection] = []

    idx = 0
    while idx < len(working):
        current = working[idx]

        if len(current.chunk_text) < min_chars:
            sibling_idx = _find_next_sibling_index(working, idx)
            if sibling_idx is None and idx + 1 < len(working):
                # Fallback: if no formal sibling exists, merge into the next section
                # to avoid noisy standalone metadata/prelude chunks.
                sibling_idx = idx + 1
            if sibling_idx is not None:
                sibling = working.pop(sibling_idx)
                current.chunk_text = _clean_text(f"{current.chunk_text}\n\n{sibling.chunk_text}")
                continue
            if merged:
                merged[-1].chunk_text = _clean_text(f"{merged[-1].chunk_text}\n\n{current.chunk_text}")
                idx += 1
                continue

        merged.append(current)
        idx += 1

    return merged


def _split_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    paragraph = _clean_text(paragraph)
    if len(paragraph) <= max_chars:
        return [paragraph] if paragraph else []

    pieces: List[str] = []
    remaining = paragraph
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at < max_chars // 2:
            split_at = max_chars
        piece = remaining[:split_at].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _split_text_block(text: str, *, max_chars: int) -> List[str]:
    clean = _clean_text(text)
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(clean) if p.strip()]
    if not paragraphs:
        return _split_long_paragraph(clean, max_chars)

    out: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                out.append(current)
                current = ""
            out.extend(_split_long_paragraph(paragraph, max_chars))
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                out.append(current)
            current = paragraph

    if current:
        out.append(current)
    return [_clean_text(text) for text in out if _clean_text(text)]


def _split_table_block_by_rows(table_text: str, *, max_chars: int) -> List[str]:
    clean_table = _clean_text(table_text)
    if not clean_table:
        return []
    if len(clean_table) <= max_chars:
        return [clean_table]

    lines = clean_table.splitlines()
    if len(lines) < 2:
        return _split_long_paragraph(clean_table, max_chars)

    header = lines[0]
    separator = lines[1]
    rows = lines[2:]

    base_table = f"{header}\n{separator}"
    if len(base_table) >= max_chars:
        return [clean_table]
    if not rows:
        return [clean_table]

    segments: List[str] = []
    current_rows: List[str] = []

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate = "\n".join([header, separator] + candidate_rows)
        if len(candidate) <= max_chars or not current_rows:
            current_rows = candidate_rows
            continue

        segments.append("\n".join([header, separator] + current_rows))
        current_rows = [row]

    if current_rows:
        segments.append("\n".join([header, separator] + current_rows))

    return [_clean_text(segment) for segment in segments if _clean_text(segment)]


def _split_chunk_text_by_paragraphs(chunk_text: str, *, max_chars: int) -> List[str]:
    clean = _clean_text(chunk_text)
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    blocks = _extract_chunk_blocks(clean)
    if not blocks:
        return _split_text_block(clean, max_chars=max_chars)

    parts: List[str] = []
    for block in blocks:
        if block.kind == "table":
            parts.extend(_split_table_block_by_rows(block.text, max_chars=max_chars))
        else:
            parts.extend(_split_text_block(block.text, max_chars=max_chars))

    out: List[str] = []
    current = ""
    for part in parts:
        clean_part = _clean_text(part)
        if not clean_part:
            continue

        if not current:
            if len(clean_part) <= max_chars:
                current = clean_part
            else:
                out.append(clean_part)
            continue

        candidate = f"{current}\n\n{clean_part}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        out.append(_clean_text(current))
        if len(clean_part) <= max_chars:
            current = clean_part
        else:
            out.append(clean_part)
            current = ""

    if current:
        out.append(_clean_text(current))

    return out


def _split_large_policy_sections(
    sections: List[_PolicySection],
    *,
    max_chars: int,
) -> List[_PolicySection]:
    split_sections: List[_PolicySection] = []
    for section in sections:
        parts = _split_chunk_text_by_paragraphs(section.chunk_text, max_chars=max_chars)
        if len(parts) <= 1:
            split_sections.append(section)
            continue
        for part in parts:
            split_sections.append(
                _PolicySection(
                    heading=section.heading,
                    heading_level=section.heading_level,
                    section_path=section.section_path,
                    parent_path=section.parent_path,
                    chunk_text=part,
                )
            )
    return split_sections


def chunk_policy_markdown_files(md_paths: Iterable[Path]) -> pd.DataFrame:
    records: List[ChunkRecord] = []
    for md_path in md_paths:
        text = md_path.read_text(encoding="utf-8")
        doc_id = md_path.stem
        doc_title = _infer_doc_title(text.splitlines(), fallback=_default_doc_title(doc_id))
        sections = _markdown_sections(text, doc_title=doc_title)
        sections = _merge_tiny_policy_sections(sections, min_chars=POLICY_MIN_CHUNK_CHARS)
        sections = _split_large_policy_sections(sections, max_chars=POLICY_MAX_CHUNK_CHARS)

        for idx, section in enumerate(sections, start=1):
            chunk_text = section.chunk_text
            section_path = section.section_path
            table_present = _chunk_contains_markdown_table(chunk_text)
            section_type = _infer_policy_section_type(section.heading)
            records.append(
                ChunkRecord(
                    chunk_id=_stable_chunk_id("policy", [doc_id, str(idx), section_path, chunk_text[:120]]),
                    chunk_text=chunk_text,
                    source_type="policy_pdf",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_path=section_path,
                    heading=section.heading,
                    heading_level=section.heading_level,
                    section_type=section_type,
                    chunk_len_chars=len(chunk_text),
                    table_present=table_present,
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
    all_chunks["chunk_text"] = all_chunks["chunk_text"].astype(str).map(_clean_text)
    all_chunks["chunk_len_chars"] = all_chunks["chunk_text"].str.len()
    if "section_type" not in all_chunks.columns:
        all_chunks["section_type"] = "other"
    all_chunks["section_type"] = (
        all_chunks["section_type"]
        .fillna("other")
        .astype(str)
        .str.lower()
        .map(lambda value: value if value in SECTION_TYPES else "other")
    )
    if "table_present" not in all_chunks.columns:
        all_chunks["table_present"] = False
    all_chunks["table_present"] = all_chunks["table_present"].fillna(False).astype(bool)
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
