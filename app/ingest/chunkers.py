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
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|[A-Za-z][.)]\s+)")
THEMATIC_BREAK_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
PAGE_MARKER_RES = (
    re.compile(r"^\s*<!--\s*page\s*[:=-]?\s*(\d+)\s*-->\s*$", flags=re.IGNORECASE),
    re.compile(r"^\s*\[?\s*page\s+(\d+)(?:\s+of\s+\d+)?\s*\]?\s*$", flags=re.IGNORECASE),
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")

POLICY_MIN_CHUNK_CHARS = 400
POLICY_MAX_CHUNK_CHARS = 2800
POLICY_OVERLAP_TARGET_CHARS = 220
DIAGNOSTIC_SAMPLE_CHARS = 240
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
    source_document_id: Optional[str] = None
    title: Optional[str] = None
    source_file: Optional[str] = None
    section_path: Optional[str] = None
    heading: Optional[str] = None
    heading_level: Optional[int] = None
    section_type: Optional[str] = None
    chunk_type: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_len_chars: Optional[int] = None
    table_present: Optional[bool] = False
    table_markdown: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class _PolicySection:
    heading: str
    heading_level: int
    section_path: str
    parent_path: str
    body_text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class _ChunkBlock:
    kind: str
    text: str


@dataclass
class _TextUnit:
    kind: str
    text: str


@dataclass
class _SectionChunk:
    chunk_text: str
    chunk_type: str
    table_present: bool = False
    table_markdown: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


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


def _page_number_from_line(line: str) -> Optional[int]:
    for pattern in PAGE_MARKER_RES:
        match = pattern.match(line or "")
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return None
    return None


def _normalize_inline_text(lines: List[str]) -> str:
    return re.sub(r"\s+", " ", " ".join((line or "").strip() for line in lines if (line or "").strip())).strip()


def _extract_text_units(text: str) -> List[_TextUnit]:
    lines = (text or "").splitlines()
    if not lines:
        return []

    units: List[_TextUnit] = []
    paragraph_lines: List[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        paragraph = _normalize_inline_text(paragraph_lines)
        if paragraph:
            units.append(_TextUnit(kind="paragraph", text=paragraph))
        paragraph_lines = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = (line or "").strip()
        if not stripped:
            flush_paragraph()
            idx += 1
            continue

        if THEMATIC_BREAK_RE.match(stripped):
            flush_paragraph()
            units.append(_TextUnit(kind="divider", text=stripped))
            idx += 1
            continue

        if LIST_ITEM_RE.match(line):
            flush_paragraph()
            item_lines = [line]
            idx += 1
            while idx < len(lines):
                candidate = lines[idx]
                candidate_stripped = (candidate or "").strip()
                if not candidate_stripped:
                    break
                if LIST_ITEM_RE.match(candidate):
                    break
                item_lines.append(candidate)
                idx += 1
            item_text = _normalize_inline_text(item_lines)
            if item_text:
                units.append(_TextUnit(kind="list_item", text=item_text))
            continue

        paragraph_lines.append(line)
        idx += 1

    flush_paragraph()
    return units


def _split_long_text_unit(text: str, *, max_chars: int) -> List[str]:
    clean = _clean_text(text)
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(clean) if s.strip()]
    if len(sentences) <= 1:
        return _split_long_paragraph(clean, max_chars)

    pieces: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        if len(sentence) <= max_chars:
            current = sentence
            continue
        pieces.extend(_split_long_paragraph(sentence, max_chars))

    if current:
        pieces.append(current)
    return [_clean_text(piece) for piece in pieces if _clean_text(piece)]


def _join_text_units(units: List[_TextUnit]) -> str:
    out: List[str] = []
    prev_kind: Optional[str] = None
    for unit in units:
        text = _clean_text(unit.text)
        if not text:
            continue
        if not out:
            out.append(text)
        elif prev_kind == "list_item" and unit.kind == "list_item":
            out.append(f"\n{text}")
        else:
            out.append(f"\n\n{text}")
        prev_kind = unit.kind
    return "".join(out)


def _render_section_chunk_text(heading: str, body_text: str) -> str:
    clean_heading = _clean_text(heading)
    clean_body = _clean_text(body_text)
    if clean_heading and clean_body:
        return _clean_text(f"{clean_heading}\n\n{clean_body}")
    return clean_heading or clean_body


def _render_section_chunk_length(heading: str, units: List[_TextUnit]) -> int:
    return len(_render_section_chunk_text(heading, _join_text_units(units)))


def _select_overlap_units(units: List[_TextUnit], *, target_chars: int) -> List[_TextUnit]:
    if len(units) < 2:
        return []

    selected: List[_TextUnit] = []
    total = 0
    for unit in reversed(units):
        unit_len = len(_clean_text(unit.text))
        if unit_len <= 0:
            continue
        if unit_len > target_chars and not selected:
            return []
        if selected and (total + unit_len) > target_chars:
            break
        selected.insert(0, unit)
        total += unit_len
        if unit.kind != "list_item":
            break
    return selected


def _chunk_text_units(
    heading: str,
    units: List[_TextUnit],
    *,
    max_chars: int,
    overlap_target_chars: int,
) -> List[str]:
    if not units:
        return []

    available_chars = max(200, max_chars - len(_clean_text(heading)) - 2)
    working = list(units)
    chunks: List[str] = []
    current: List[_TextUnit] = []
    idx = 0

    while idx < len(working):
        unit = working[idx]
        if _render_section_chunk_length(heading, current + [unit]) <= max_chars:
            current.append(unit)
            idx += 1
            continue

        if current:
            chunk_text = _render_section_chunk_text(heading, _join_text_units(current))
            if chunk_text:
                chunks.append(chunk_text)
            overlap_units = _select_overlap_units(current, target_chars=overlap_target_chars)
            current = list(overlap_units)
            if current and _render_section_chunk_length(heading, current + [unit]) > max_chars:
                current = []
            continue

        split_units = [
            _TextUnit(kind=unit.kind, text=piece)
            for piece in _split_long_text_unit(unit.text, max_chars=available_chars)
        ]
        if len(split_units) <= 1:
            current = split_units
            idx += 1
            continue
        working[idx : idx + 1] = split_units

    if current:
        chunk_text = _render_section_chunk_text(heading, _join_text_units(current))
        if chunk_text:
            chunks.append(chunk_text)

    return [_clean_text(chunk_text) for chunk_text in chunks if _clean_text(chunk_text)]


def _markdown_table_to_structured_text(table_text: str) -> str:
    lines = [line.rstrip() for line in _clean_text(table_text).splitlines() if line.strip()]
    if len(lines) < 2:
        return _clean_text(table_text)

    columns = _table_cells(lines[0])
    rows = lines[2:]
    if not columns or not rows:
        return _clean_text(table_text)

    rendered_rows: List[str] = [f"Table columns: {', '.join(columns)}"]
    for idx, row in enumerate(rows, start=1):
        values = _table_cells(row)
        pairs: List[str] = []
        for col_index, column in enumerate(columns):
            value = values[col_index] if col_index < len(values) else ""
            value = re.sub(r"\s+", " ", value).strip()
            if value:
                pairs.append(f"{column}: {value}")
        if pairs:
            rendered_rows.append(f"- Row {idx}: " + "; ".join(pairs))

    return _clean_text("\n".join(rendered_rows))


def _split_table_block_for_heading(table_text: str, *, heading: str, max_chars: int) -> List[str]:
    clean_table = _clean_text(table_text)
    if not clean_table:
        return []

    lines = clean_table.splitlines()
    if len(lines) < 2:
        return [clean_table]

    header = lines[0]
    separator = lines[1]
    rows = lines[2:]
    if not rows:
        return [clean_table]

    def candidate_length(candidate_rows: List[str]) -> int:
        markdown = "\n".join([header, separator] + candidate_rows)
        structured = _markdown_table_to_structured_text(markdown)
        return len(_render_section_chunk_text(heading, structured))

    segments: List[str] = []
    current_rows: List[str] = []
    for row in rows:
        proposed = current_rows + [row]
        if candidate_length(proposed) <= max_chars or not current_rows:
            current_rows = proposed
            continue
        segments.append("\n".join([header, separator] + current_rows))
        current_rows = [row]

    if current_rows:
        segments.append("\n".join([header, separator] + current_rows))

    return [_clean_text(segment) for segment in segments if _clean_text(segment)]


def _build_policy_section_chunks(
    section: _PolicySection,
    *,
    max_chars: int,
    overlap_target_chars: int,
) -> List[_SectionChunk]:
    section_chunks: List[_SectionChunk] = []
    for block in _extract_chunk_blocks(section.body_text):
        if block.kind == "table":
            for table_segment in _split_table_block_for_heading(block.text, heading=section.heading, max_chars=max_chars):
                structured_table = _markdown_table_to_structured_text(table_segment)
                chunk_text = _render_section_chunk_text(section.heading, structured_table)
                if chunk_text:
                    section_chunks.append(
                        _SectionChunk(
                            chunk_text=chunk_text,
                            chunk_type="table",
                            table_present=True,
                            table_markdown=table_segment,
                            page_start=section.page_start,
                            page_end=section.page_end,
                        )
                    )
            continue

        text_units = _extract_text_units(block.text)
        for chunk_text in _chunk_text_units(
            section.heading,
            text_units,
            max_chars=max_chars,
            overlap_target_chars=overlap_target_chars,
        ):
            section_chunks.append(
                _SectionChunk(
                    chunk_text=chunk_text,
                    chunk_type="text",
                    table_present=False,
                    page_start=section.page_start,
                    page_end=section.page_end,
                )
            )

    return section_chunks


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
        source_file = str(row.get("source_file", "") or "").strip() or None
        chunk_index = 0

        def append_record(record: ChunkRecord) -> None:
            nonlocal chunk_index
            chunk_index += 1
            record.source_document_id = doc_id
            record.title = doc_title
            record.source_file = source_file
            record.chunk_index = chunk_index
            if not record.chunk_type:
                record.chunk_type = record.control_part or "text"
            records.append(record)

        statement = _clean_text(str(row.get("statement", "") or ""))
        guidance = _clean_text(str(row.get("guidance", "") or ""))
        enhancements = _to_list(row.get("enhancements", []))
        parameters = _to_list(row.get("parameters", []))

        if statement:
            append_record(
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
                    chunk_type="statement",
                    chunk_len_chars=len(statement),
                )
            )

        if guidance:
            append_record(
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
                    chunk_type="guidance",
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
            append_record(
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
                    chunk_type="enhancement",
                    chunk_len_chars=len(enh_statement),
                )
            )

        for idx, param in enumerate(parameters, start=1):
            param_text = _clean_text(str(param))
            if not param_text:
                continue
            append_record(
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
                    chunk_type="parameter",
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
    current_page: Optional[int] = None
    section_page_start: Optional[int] = None
    section_page_end: Optional[int] = None

    def flush() -> None:
        nonlocal body_lines, section_page_start, section_page_end
        text = _clean_text("\n".join(body_lines))
        if text:
            sections.append(
                _PolicySection(
                    heading=current_heading,
                    heading_level=current_level,
                    section_path=current_section,
                    parent_path=current_parent,
                    body_text=text,
                    page_start=section_page_start,
                    page_end=section_page_end,
                )
            )
        body_lines = []
        section_page_start = None
        section_page_end = None

    for line in lines:
        page_num = _page_number_from_line(line)
        if page_num is not None:
            current_page = page_num
            if section_page_start is None:
                section_page_start = page_num
            section_page_end = page_num
            continue

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
            if (line or "").strip():
                if section_page_start is None:
                    section_page_start = current_page
                section_page_end = current_page if current_page is not None else section_page_end

    flush()
    return sections


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


def chunk_policy_markdown_files(md_paths: Iterable[Path]) -> pd.DataFrame:
    records: List[ChunkRecord] = []
    for md_path in md_paths:
        text = md_path.read_text(encoding="utf-8")
        doc_id = md_path.stem
        doc_title = _infer_doc_title(text.splitlines(), fallback=_default_doc_title(doc_id))
        sections = _markdown_sections(text, doc_title=doc_title)

        chunk_index = 0
        for section in sections:
            section_type = _infer_policy_section_type(section.heading)
            section_chunks = _build_policy_section_chunks(
                section,
                max_chars=POLICY_MAX_CHUNK_CHARS,
                overlap_target_chars=POLICY_OVERLAP_TARGET_CHARS,
            )
            for section_chunk in section_chunks:
                chunk_index += 1
                chunk_text = section_chunk.chunk_text
                records.append(
                    ChunkRecord(
                        chunk_id=_stable_chunk_id(
                            "policy",
                            [
                                doc_id,
                                str(chunk_index),
                                section.section_path,
                                section_chunk.chunk_type,
                                chunk_text[:120],
                            ],
                        ),
                        chunk_text=chunk_text,
                        source_type="policy_pdf",
                        doc_id=doc_id,
                        doc_title=doc_title,
                        source_document_id=doc_id,
                        title=doc_title,
                        source_file=str(md_path),
                        section_path=section.section_path,
                        heading=section.heading,
                        heading_level=section.heading_level,
                        section_type=section_type,
                        chunk_type=section_chunk.chunk_type,
                        chunk_index=chunk_index,
                        chunk_len_chars=len(chunk_text),
                        table_present=section_chunk.table_present,
                        table_markdown=section_chunk.table_markdown,
                        page_start=section_chunk.page_start,
                        page_end=section_chunk.page_end,
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


def _length_distribution(lengths: pd.Series) -> Dict[str, int]:
    clean = lengths.dropna().astype(int)
    if clean.empty:
        return {
            "min": 0,
            "p25": 0,
            "median": 0,
            "p75": 0,
            "p90": 0,
            "max": 0,
        }
    quantiles = clean.quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
    return {
        "min": int(clean.min()),
        "p25": int(round(float(quantiles.get(0.25, clean.min())))),
        "median": int(round(float(quantiles.get(0.5, clean.median())))),
        "p75": int(round(float(quantiles.get(0.75, clean.max())))),
        "p90": int(round(float(quantiles.get(0.9, clean.max())))),
        "max": int(clean.max()),
    }


def _sample_chunk_preview(text: str, *, max_chars: int = DIAGNOSTIC_SAMPLE_CHARS) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def build_chunk_diagnostics(chunks_df: pd.DataFrame, *, sample_per_source: int = 2) -> Dict:
    if chunks_df.empty:
        return {
            "total_chunks": 0,
            "table_chunks": 0,
            "chunk_length_chars": _length_distribution(pd.Series(dtype=int)),
            "chunks_per_source_file": [],
            "sample_chunks": [],
        }

    working = chunks_df.copy()
    working["chunk_len_chars"] = working["chunk_len_chars"].fillna(
        working["chunk_text"].astype(str).str.len()
    ).astype(int)
    if "chunk_type" not in working.columns:
        working["chunk_type"] = "text"
    working["chunk_type"] = working["chunk_type"].fillna("text").astype(str)
    if "table_present" not in working.columns:
        working["table_present"] = False
    working["table_present"] = working["table_present"].fillna(False).astype(bool)
    working["_source_key"] = (
        working.get("source_file", pd.Series(index=working.index, dtype="object"))
        .fillna(working.get("doc_id", pd.Series(index=working.index, dtype="object")))
        .fillna("unknown")
        .astype(str)
    )

    per_source: List[Dict] = []
    sample_chunks: List[Dict] = []
    for source_key, group in working.groupby("_source_key", sort=True):
        source_table_mask = (group["chunk_type"] == "table") | group["table_present"]
        per_source.append(
            {
                "source_file": None if source_key == "unknown" else source_key,
                "doc_ids": sorted({str(v) for v in group["doc_id"].dropna().astype(str) if str(v)}),
                "titles": sorted({str(v) for v in group["doc_title"].dropna().astype(str) if str(v)}),
                "total_chunks": int(len(group)),
                "table_chunks": int(source_table_mask.sum()),
                "chunk_length_chars": _length_distribution(group["chunk_len_chars"]),
            }
        )
        for _, row in group.head(sample_per_source).iterrows():
            sample_chunks.append(
                {
                    "source_file": row.get("source_file"),
                    "doc_id": row.get("doc_id"),
                    "title": row.get("doc_title") or row.get("title"),
                    "section_path": row.get("section_path"),
                    "chunk_type": row.get("chunk_type"),
                    "chunk_index": row.get("chunk_index"),
                    "chunk_len_chars": int(row.get("chunk_len_chars", 0) or 0),
                    "preview": _sample_chunk_preview(str(row.get("chunk_text", ""))),
                }
            )

    table_mask = (working["chunk_type"] == "table") | working["table_present"]
    return {
        "total_chunks": int(len(working)),
        "table_chunks": int(table_mask.sum()),
        "chunk_length_chars": _length_distribution(working["chunk_len_chars"]),
        "chunks_per_source_file": per_source,
        "sample_chunks": sample_chunks,
    }


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
    if "source_document_id" not in all_chunks.columns:
        all_chunks["source_document_id"] = all_chunks.get("doc_id")
    all_chunks["source_document_id"] = all_chunks["source_document_id"].fillna(all_chunks.get("doc_id"))
    if "title" not in all_chunks.columns:
        all_chunks["title"] = all_chunks.get("doc_title")
    all_chunks["title"] = all_chunks["title"].fillna(all_chunks.get("doc_title"))
    if "chunk_type" not in all_chunks.columns:
        all_chunks["chunk_type"] = "text"
    all_chunks["chunk_type"] = all_chunks["chunk_type"].fillna("text").astype(str)
    if "chunk_index" not in all_chunks.columns:
        all_chunks["chunk_index"] = range(1, len(all_chunks) + 1)
    all_chunks["chunk_index"] = pd.to_numeric(all_chunks["chunk_index"], errors="coerce").fillna(0).astype(int)
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
    if "table_markdown" not in all_chunks.columns:
        all_chunks["table_markdown"] = None
    all_chunks = all_chunks.drop_duplicates(subset=["chunk_id"]).reset_index(drop=True)
    return all_chunks


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    chunks = build_chunks_dataframe(repo_root)
    diagnostics = build_chunk_diagnostics(chunks)
    out_dir = repo_root / "data/index"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.parquet"
    chunks.to_parquet(out_path, index=False)
    summary = {
        "rows": int(len(chunks)),
        "source_counts": chunks["source_type"].value_counts().to_dict(),
        "output": str(out_path),
        "diagnostics": diagnostics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
