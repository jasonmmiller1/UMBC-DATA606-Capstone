from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Mapping, Optional

from app.runtime import REPO_ROOT, UPLOAD_MD_DIR, UPLOAD_PDF_DIR


POLICY_SOURCE_TYPES = {"policy_pdf", "policy_md"}
TEXT_SOURCE_SUFFIXES = {".md", ".markdown", ".txt"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class PolicySourceView:
    doc_label: str
    doc_id: Optional[str]
    source_path: Optional[Path]
    source_kind: Optional[str]
    full_text: Optional[str]
    matched_section_label: Optional[str]
    matched_section_text: Optional[str]


def is_policy_chunk(chunk: Mapping[str, object]) -> bool:
    return str(chunk.get("source_type") or "").strip().lower() in POLICY_SOURCE_TYPES


def _normalize_heading(text: Optional[str]) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().strip("#").strip())
    return normalized.lower()


def _resolve_candidate_path(path_text: str, *, repo_root: Path) -> Optional[Path]:
    raw = str(path_text or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _add_candidate(candidates: list[Path], candidate: Optional[Path]) -> None:
    if candidate is None:
        return
    if candidate in candidates:
        return
    candidates.append(candidate)


def _source_path_candidates(
    chunk: Mapping[str, object],
    *,
    repo_root: Path,
    upload_md_dir: Path,
    upload_pdf_dir: Path,
) -> list[Path]:
    candidates: list[Path] = []
    source_file = _resolve_candidate_path(str(chunk.get("source_file") or ""), repo_root=repo_root)
    _add_candidate(candidates, source_file)

    doc_id = str(chunk.get("doc_id") or "").strip()
    if not doc_id:
        return candidates

    for base_dir in (
        upload_md_dir,
        repo_root / "data" / "policies_synth_md_v2",
        repo_root / "data" / "policies_synth_md_v0",
        repo_root / "data" / "policies_synth_md",
    ):
        _add_candidate(candidates, base_dir / f"{doc_id}.md")
        _add_candidate(candidates, base_dir / f"{doc_id}.txt")

    for base_dir in (
        upload_pdf_dir,
        repo_root / "data" / "policies_synth_pdf",
    ):
        _add_candidate(candidates, base_dir / f"{doc_id}.pdf")

    return candidates


@lru_cache(maxsize=64)
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_markdown_section(
    md_text: str,
    *,
    section_path: Optional[str] = None,
    heading: Optional[str] = None,
    chunk_text: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    lines = md_text.splitlines()
    heading_candidates: list[str] = []
    if heading:
        heading_candidates.append(str(heading))
    if section_path:
        parts = [part for part in str(section_path).split(" > ") if part]
        heading_candidates.extend(reversed(parts))

    normalized_candidates: list[str] = []
    seen_candidates: set[str] = set()
    for candidate in heading_candidates:
        normalized = _normalize_heading(candidate)
        if not normalized or normalized in seen_candidates:
            continue
        seen_candidates.add(normalized)
        normalized_candidates.append(normalized)

    match_start: Optional[int] = None
    match_level: Optional[int] = None
    match_label: Optional[str] = None

    for candidate in normalized_candidates:
        for idx, line in enumerate(lines):
            match = HEADING_RE.match(line)
            if not match:
                continue
            level = len(match.group(1))
            title = match.group(2).strip()
            if _normalize_heading(title) == candidate:
                match_start = idx
                match_level = level
                match_label = title
                break
        if match_start is not None:
            break

    if match_start is not None and match_level is not None:
        end = len(lines)
        for idx in range(match_start + 1, len(lines)):
            match = HEADING_RE.match(lines[idx])
            if not match:
                continue
            next_level = len(match.group(1))
            if next_level <= match_level:
                end = idx
                break
        section_text = "\n".join(lines[match_start:end]).strip()
        if section_text:
            return match_label or section_path or heading, section_text

    chunk_preview = str(chunk_text or "").strip()
    if chunk_preview and chunk_preview in md_text:
        return section_path or heading, chunk_preview
    return None, None


def resolve_policy_source_view(
    chunk: Mapping[str, object],
    *,
    repo_root: Path = REPO_ROOT,
    upload_md_dir: Path = UPLOAD_MD_DIR,
    upload_pdf_dir: Path = UPLOAD_PDF_DIR,
) -> Optional[PolicySourceView]:
    if not is_policy_chunk(chunk):
        return None

    doc_label = str(chunk.get("doc_title") or chunk.get("doc_id") or "Policy document")
    doc_id = str(chunk.get("doc_id") or "").strip() or None

    text_candidate: Optional[Path] = None
    pdf_candidate: Optional[Path] = None
    for candidate in _source_path_candidates(
        chunk,
        repo_root=repo_root,
        upload_md_dir=upload_md_dir,
        upload_pdf_dir=upload_pdf_dir,
    ):
        if not candidate.exists():
            continue
        if candidate.suffix.lower() in TEXT_SOURCE_SUFFIXES and text_candidate is None:
            text_candidate = candidate
            break
        if candidate.suffix.lower() == ".pdf" and pdf_candidate is None:
            pdf_candidate = candidate

    if text_candidate is not None:
        full_text = _read_text(text_candidate)
        matched_label, matched_text = extract_markdown_section(
            full_text,
            section_path=str(chunk.get("section_path") or "").strip() or None,
            heading=str(chunk.get("heading") or "").strip() or None,
            chunk_text=str(chunk.get("chunk_text") or "").strip() or None,
        )
        return PolicySourceView(
            doc_label=doc_label,
            doc_id=doc_id,
            source_path=text_candidate,
            source_kind="markdown" if text_candidate.suffix.lower() in {".md", ".markdown"} else "text",
            full_text=full_text,
            matched_section_label=matched_label,
            matched_section_text=matched_text,
        )

    return PolicySourceView(
        doc_label=doc_label,
        doc_id=doc_id,
        source_path=pdf_candidate,
        source_kind="pdf" if pdf_candidate is not None else None,
        full_text=None,
        matched_section_label=None,
        matched_section_text=None,
    )
