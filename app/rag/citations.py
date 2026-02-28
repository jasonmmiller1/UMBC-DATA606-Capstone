from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _as_payload(item: Dict) -> Dict:
    # Supports both retrieval result objects ({payload: {...}}) and flattened chunk dicts.
    payload = item.get("payload")
    if isinstance(payload, dict):
        merged = dict(payload)
        for k, v in item.items():
            if k not in merged:
                merged[k] = v
        return merged
    return dict(item)


def _norm_page(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _policy_dedupe_key(c: Dict) -> Tuple:
    return (
        c.get("source_type"),
        c.get("doc_id"),
        c.get("section_path"),
        c.get("page_start"),
        c.get("page_end"),
    )


def _oscal_dedupe_key(c: Dict) -> Tuple:
    return (
        c.get("source_type"),
        c.get("control_id"),
        c.get("control_part"),
    )


def normalize_citations(retrieved_chunks: List[Dict], max_items: Optional[int] = None) -> List[Dict]:
    """
    Normalize citations from retrieved chunks/results for two source types:
    - policy chunks: doc/section/pages
    - oscal controls: control_id/part/family

    Input is tolerant:
    - retrieval results with nested payload
    - flattened retrieved chunks
    """
    normalized: List[Dict] = []
    seen = set()

    for idx, item in enumerate(retrieved_chunks, start=1):
        payload = _as_payload(item)
        source_type = payload.get("source_type")
        citation_id = payload.get("citation_id") or item.get("citation_id") or f"C{idx}"
        chunk_id = payload.get("chunk_id") or item.get("chunk_id")

        if source_type == "oscal_control":
            citation = {
                "citation_id": citation_id,
                "source_type": "oscal_control",
                "chunk_id": chunk_id,
                "control_id": payload.get("control_id"),
                "control_part": payload.get("control_part"),
                "control_family": payload.get("control_family"),
            }
            dedupe_key = _oscal_dedupe_key(citation)
        else:
            citation = {
                "citation_id": citation_id,
                "source_type": source_type or "policy_pdf",
                "chunk_id": chunk_id,
                "doc_id": payload.get("doc_id"),
                "doc_title": payload.get("doc_title"),
                "section_path": payload.get("section_path"),
                "page_start": _norm_page(payload.get("page_start")),
                "page_end": _norm_page(payload.get("page_end")),
            }
            dedupe_key = _policy_dedupe_key(citation)

        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(citation)

        # Optional: stop early once we have enough citations
        if max_items is not None and len(normalized) >= max_items:
            break

    return normalized

def format_citations_markdown(citations: List[Dict]) -> str:
    """
    Render citations as a readable markdown bullet list.
    Safe with missing metadata fields.
    """
    if not citations:
        return "- (none)"

    lines: List[str] = []
    for c in citations:
        cid = c.get("citation_id") or "C?"
        source_type = c.get("source_type")

        if source_type == "oscal_control":
            control_id = c.get("control_id") or "unknown-control"
            part = c.get("control_part")
            family = c.get("control_family")
            detail = control_id
            if part:
                detail += f" ({part})"
            if family:
                detail += f" [{family}]"
            lines.append(f"- [{cid}] OSCAL: {detail}")
            continue

        doc_title = c.get("doc_title") or c.get("doc_id") or "unknown-doc"
        section = c.get("section_path") or "unknown-section"
        page_start = c.get("page_start")
        page_end = c.get("page_end")
        page_text = ""
        if page_start is not None and page_end is not None:
            page_text = f", pages {page_start}-{page_end}"
        elif page_start is not None:
            page_text = f", page {page_start}"
        lines.append(f"- [{cid}] Policy: {doc_title}, section `{section}`{page_text}")

    return "\n".join(lines)


def format_citations(citations: List[Dict]) -> List[str]:
    """
    Backward-compatible helper used by existing Streamlit UI.
    Returns list[str] lines instead of one markdown block.
    """
    md = format_citations_markdown(citations)
    return [line for line in md.splitlines() if line.strip()]

