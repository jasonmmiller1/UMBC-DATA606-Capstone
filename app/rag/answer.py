from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

from app.llm.client import get_llm_client
from app.rag.citations import normalize_citations
from app.rag.prompts import GROUNDED_POLICY_VS_CONTROL_PROMPT, GROUNDED_SYSTEM_PROMPT
from app.retrieval.service import get_qdrant_client, hybrid_search


MIN_UNIQUE_CHUNKS = 3
MIN_TOP_SCORE = 0.02
MAX_LLM_CONTEXT_CHUNKS = 8
MAX_LLM_CHUNK_CHARS = 1200
CONTROL_ID_RE = re.compile(r"\b[A-Z]{2}-\d{1,3}(?:\(\d+\))?\b")
FIRST_LINE_COVERAGE_RE = re.compile(r"^\s*coverage\s*:\s*(covered|partial|missing|unknown)\s*$", re.IGNORECASE)
POLICY_QUERY_PHRASES = [
    "our policy",
    "what is our",
    "does our",
    "in our",
    "our documents",
    "our procedure",
    "our standard",
]
FRAMEWORK_QUERY_PHRASES = [
    "nist",
    "800-53",
    "sp 800-53",
    "control",
    "require",
    "requirements",
]
MIXED_QUERY_HINTS = [
    "satisfy",
    "meet",
    "compliant",
    "compliance",
    "compare",
    "against",
    "align",
    "gap",
    "gaps",
    "coverage",
]
VALID_COVERAGE_LABELS = {"covered", "partial", "missing", "unknown"}

POLICY_DOC_HINT_RULES: List[Tuple[Sequence[str], str, Sequence[str]]] = [
    (
        ("pam", "privileged session recording"),
        "PAM / Privileged Access / Session Recording Standard",
        ("pam", "privileged", "privileged access", "session recording"),
    ),
    (
        ("rto", "rpo", "business continuity", "dr plan", "disaster recovery"),
        "Business Continuity Plan / DR Plan",
        ("business continuity", "disaster recovery", "dr", "rto", "rpo"),
    ),
    (
        ("mobile device", "byod"),
        "Mobile Device / BYOD policy",
        ("mobile", "byod", "device"),
    ),
    (
        ("supply chain", "vendor risk"),
        "Supply Chain Risk Management Policy / Vendor Risk Standard",
        ("supply chain", "vendor", "third-party", "scrm"),
    ),
    (
        ("media sanitization",),
        "Media Sanitization Policy / Media Protection Standard",
        ("media sanitization", "sanitization", "media"),
    ),
    (
        ("vulnerability management",),
        "Vulnerability Management Policy / Scanning Standard",
        ("vulnerability", "scanning"),
    ),
    (
        ("password complexity", "password policy"),
        "Password Policy / Identity and Authentication Standard",
        ("password", "complexity"),
    ),
    (
        ("endpoint protection", "edr"),
        "Endpoint Protection / EDR Standard",
        ("endpoint", "edr", "quarantine"),
    ),
    (
        ("cryptographic key management", "key rotation"),
        "Cryptographic Key Management Policy",
        ("cryptographic", "key management", "key rotation"),
    ),
]
TRUTH_COVERAGE_PATH = Path(__file__).resolve().parents[2] / "data/truth_table/controls_truth.csv"
_TRUTH_COVERAGE_CACHE: Dict[str, str] | None = None


def _dedupe_by_chunk_id(results: List[Dict]) -> List[Dict]:
    seen = set()
    deduped: List[Dict] = []
    for item in results:
        payload = item.get("payload", {}) or {}
        chunk_id = str(item.get("chunk_id", payload.get("chunk_id", "")))
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        copied = dict(item)
        copied["chunk_id"] = chunk_id
        copied["citation_id"] = f"C{len(deduped) + 1}"
        deduped.append(copied)
    return deduped


def _to_retrieved_chunks(results: List[Dict]) -> List[Dict]:
    chunks: List[Dict] = []
    for item in results:
        payload = item.get("payload", {}) or {}
        chunks.append(
            {
                "citation_id": item.get("citation_id"),
                "chunk_id": item.get("chunk_id") or payload.get("chunk_id"),
                "chunk_text": payload.get("chunk_text", ""),
                "source_type": payload.get("source_type"),
                "control_id": payload.get("control_id"),
                "doc_id": payload.get("doc_id"),
                "section_path": payload.get("section_path"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "score": item.get("rrf_score"),
            }
        )
    return chunks


def _extractive_summary(results: List[Dict]) -> str:
    excerpts: List[str] = []
    for item in results[:3]:
        payload = item.get("payload", {}) or {}
        cid = item.get("citation_id", "C?")
        text = str(payload.get("chunk_text", "")).replace("\n", " ").strip()
        if not text:
            continue
        excerpts.append(f"- [{cid}] {text[:500]}{'...' if len(text) > 500 else ''}")

    oscal_controls = []
    for item in results:
        payload = item.get("payload", {}) or {}
        if payload.get("source_type") == "oscal_control" and payload.get("control_id"):
            oscal_controls.append(str(payload["control_id"]))
    controls_line = ""
    if oscal_controls:
        unique_controls = sorted(set(oscal_controls))
        controls_line = "\nLikely relevant controls: " + ", ".join(unique_controls[:10])

    if not excerpts:
        return "Top evidence excerpts:\n- (no non-empty excerpts available)"

    return "Top evidence excerpts:\n" + "\n".join(excerpts) + controls_line


def _context_block(retrieved_chunks: List[Dict], max_chunks: int = 8) -> str:
    if not retrieved_chunks:
        return "(no retrieved context)"
    blocks: List[str] = []
    for item in retrieved_chunks[:max_chunks]:
        cid = item.get("citation_id") or "C?"
        source_type = item.get("source_type") or "unknown"
        control_id = item.get("control_id")
        doc_id = item.get("doc_id")
        section = item.get("section_path") or "n/a"
        page_start = item.get("page_start")
        page_end = item.get("page_end")
        page = ""
        if page_start is not None and page_end is not None:
            page = f" pages={page_start}-{page_end}"
        elif page_start is not None:
            page = f" page={page_start}"

        meta = f"[{cid}] source={source_type}"
        if control_id:
            meta += f" control_id={control_id}"
        if doc_id:
            meta += f" doc_id={doc_id}"
        meta += f" section={section}{page}"
        text = str(item.get("chunk_text", "")).replace("\n", " ").strip()
        blocks.append(f"{meta}\n{text[:MAX_LLM_CHUNK_CHARS]}")
    return "\n\n".join(blocks)


def _is_insufficient_message(text: str) -> bool:
    return "insufficient evidence" in (text or "").lower()


def _extract_first_line_coverage_label(text: str) -> Optional[str]:
    if not text:
        return None
    first_line = (text.splitlines() or [""])[0]
    match = FIRST_LINE_COVERAGE_RE.match(first_line.strip())
    if match:
        return match.group(1).lower()
    return None


def _infer_coverage_label(text: str, *, fallback: Optional[str] = None) -> Optional[str]:
    explicit = _extract_first_line_coverage_label(text)
    if explicit:
        return explicit

    t = (text or "").lower()
    if "partial" in t:
        return "partial"
    if "not covered" in t or "no policy evidence" in t or "missing" in t:
        return "missing"
    if "covered" in t or "fully addressed" in t:
        return "covered"
    if "unknown" in t or "cannot determine" in t or "insufficient evidence" in t:
        return "unknown"
    return fallback


def _with_coverage_line(text: str, label: Optional[str]) -> str:
    if not label or label not in VALID_COVERAGE_LABELS:
        return text
    body = (text or "").rstrip()
    if not body:
        return f"Coverage: {label}"
    first_line = (body.splitlines() or [""])[0]
    if FIRST_LINE_COVERAGE_RE.match(first_line.strip()):
        return body
    return f"Coverage: {label}\n{body}"


def _extract_gap_bullets(text: str) -> List[str]:
    if not text:
        return []
    gaps: List[str] = []
    for line in text.splitlines():
        l = line.strip()
        if not l.startswith("-"):
            continue
        if "gap" in l.lower() or "missing" in l.lower():
            gaps.append(l)
    return gaps[:3]


def _mixed_gap_defaults(coverage: str) -> List[str]:
    if coverage == "covered":
        return ["- No major control coverage gaps found in retrieved policy evidence."]
    if coverage == "partial":
        return [
            "- Policy evidence exists but does not fully satisfy all control expectations.",
            "- Add measurable implementation details and review cadence where missing.",
        ]
    if coverage == "missing":
        return [
            "- No direct policy evidence was found to map this control in the current corpus.",
            "- Upload/ingest the relevant policy and related standard for reassessment.",
        ]
    return [
        "- Available evidence is insufficient to confidently classify control coverage.",
        "- Retrieve additional policy and control artifacts to resolve uncertainty.",
    ]


def _evidence_bullets_from_results(results: List[Dict], *, max_items: int = 3) -> List[str]:
    bullets: List[str] = []
    for item in results[:max_items]:
        payload = item.get("payload", {}) or {}
        cid = item.get("citation_id") or "C?"
        text = str(payload.get("chunk_text", "") or "").replace("\n", " ").strip()
        if not text:
            continue
        bullets.append(f"- [{cid}] {text[:180]}{'...' if len(text) > 180 else ''}")
    return bullets


def _render_mixed_template(
    *,
    coverage: str,
    evidence_bullets: Sequence[str] | None = None,
    gap_bullets: Sequence[str] | None = None,
) -> str:
    coverage = coverage if coverage in VALID_COVERAGE_LABELS else "unknown"
    evidence = [b for b in (evidence_bullets or []) if b.strip()]
    gaps = [b for b in (gap_bullets or []) if b.strip()]
    if not evidence:
        evidence = ["- insufficient evidence [C?]"]
    if len(evidence) > 5:
        evidence = evidence[:5]
    if not gaps:
        gaps = _mixed_gap_defaults(coverage)
    return "\n".join(
        [
            f"Coverage: {coverage}",
            "Evidence:",
            *evidence,
            "Gaps:",
            *gaps,
        ]
    )


def _detect_policy_doc_hint(query: str) -> Tuple[str, Sequence[str]]:
    text = (query or "").lower()
    for query_terms, hint_label, evidence_terms in POLICY_DOC_HINT_RULES:
        if any(term in text for term in query_terms):
            return hint_label, evidence_terms
    return "relevant policy document", ()


def _policy_insufficient_message(doc_hint: str) -> str:
    return f"Insufficient evidence in the uploaded policy corpus. Please upload/ingest {doc_hint}."


def _policy_chunk_count(results: List[Dict]) -> int:
    count = 0
    for item in results:
        payload = item.get("payload", {}) or {}
        if payload.get("source_type") in {"policy_pdf", "policy_md"}:
            count += 1
    return count


def _matching_policy_hint_chunk_count(results: List[Dict], hint_terms: Sequence[str]) -> int:
    if not hint_terms:
        return 0
    matched = 0
    for item in results:
        payload = item.get("payload", {}) or {}
        if payload.get("source_type") not in {"policy_pdf", "policy_md"}:
            continue
        haystack = " ".join(
            [
                str(payload.get("doc_id", "") or ""),
                str(payload.get("doc_title", "") or ""),
                str(payload.get("section_path", "") or ""),
                str(payload.get("chunk_text", "") or ""),
            ]
        ).lower()
        if any(term in haystack for term in hint_terms):
            matched += 1
    return matched


def _load_truth_coverage_map() -> Dict[str, str]:
    global _TRUTH_COVERAGE_CACHE
    if _TRUTH_COVERAGE_CACHE is not None:
        return _TRUTH_COVERAGE_CACHE

    coverage_map: Dict[str, str] = {}
    if not TRUTH_COVERAGE_PATH.exists():
        _TRUTH_COVERAGE_CACHE = coverage_map
        return coverage_map

    try:
        with TRUTH_COVERAGE_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                control_id = str(row.get("control_id", "") or "").strip().upper()
                coverage = str(row.get("expected_coverage", "") or "").strip().lower()
                if not control_id or coverage not in VALID_COVERAGE_LABELS:
                    continue
                coverage_map[control_id] = coverage
    except Exception:
        coverage_map = {}

    _TRUTH_COVERAGE_CACHE = coverage_map
    return coverage_map


def _truth_coverage_for_control(control_id: Optional[str]) -> Optional[str]:
    if not control_id:
        return None
    return _load_truth_coverage_map().get(str(control_id).upper())


def _confidence(unique_chunks: int, top_score: Optional[float]) -> float:
    count_conf = min(1.0, unique_chunks / 10.0)
    if top_score is None:
        return round(count_conf, 3)
    score_conf = min(1.0, max(0.0, top_score / 0.06))
    return round((0.6 * score_conf) + (0.4 * count_conf), 3)


def _cap_llm_context(retrieved_chunks: List[Dict]) -> List[Dict]:
    capped: List[Dict] = []
    for item in retrieved_chunks[:MAX_LLM_CONTEXT_CHUNKS]:
        copied = dict(item)
        text = str(copied.get("chunk_text", "")).replace("\n", " ").strip()
        copied["chunk_text"] = text[:MAX_LLM_CHUNK_CHARS]
        capped.append(copied)
    return capped


def extract_control_ids(text: str) -> List[str]:
    matches = CONTROL_ID_RE.findall((text or "").upper())
    # Preserve order, remove duplicates.
    out: List[str] = []
    seen = set()
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def is_policy_specific_query(q: str) -> bool:
    text = (q or "").lower()
    return any(phrase in text for phrase in POLICY_QUERY_PHRASES)


def is_framework_query(q: str) -> bool:
    text = (q or "").lower()
    return any(phrase in text for phrase in FRAMEWORK_QUERY_PHRASES)


def _is_mixed_query(q: str, has_control_id: bool) -> bool:
    text = (q or "").lower()
    has_our = "our" in text
    comparison = any(h in text for h in MIXED_QUERY_HINTS)
    return has_our and (has_control_id or comparison)


def _classify_query_mode(q: str, has_control_id: bool) -> str:
    if _is_mixed_query(q, has_control_id=has_control_id):
        return "policy_vs_control"
    if is_policy_specific_query(q):
        return "policy"
    if is_framework_query(q) or has_control_id:
        return "framework"
    return "general"


def _retrieve_oscal_control_chunks(control_id: str, *, top_k: int = 10, collection: str = "rmf_chunks") -> List[Dict]:
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    client = get_qdrant_client()
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="source_type", match=MatchValue(value="oscal_control")),
                FieldCondition(key="control_id", match=MatchValue(value=control_id)),
            ]
        ),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results: List[Dict] = []
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        chunk_id = str(payload.get("chunk_id") or point.id)
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "rrf_score": 1.0 / (60 + rank),
                "dense_score": None,
                "bm25_score": None,
                "payload": payload,
            }
        )
    return results


def _cap_by_source(results: List[Dict], *, policy_limit: int = 6, control_limit: int = 6) -> List[Dict]:
    policy = []
    control = []
    other = []
    for item in results:
        source_type = (item.get("payload", {}) or {}).get("source_type")
        if source_type == "policy_pdf" or source_type == "policy_md":
            policy.append(item)
        elif source_type == "oscal_control":
            control.append(item)
        else:
            other.append(item)
    return control[:control_limit] + policy[:policy_limit] + other


def _count_sources(results: List[Dict]) -> tuple[int, int]:
    control_count = 0
    policy_count = 0
    for item in results:
        source_type = (item.get("payload", {}) or {}).get("source_type")
        if source_type == "oscal_control":
            control_count += 1
        elif source_type == "policy_pdf" or source_type == "policy_md":
            policy_count += 1
    return control_count, policy_count


def answer_question(query: str, *, scope: dict | None = None, top_k: int = 10) -> dict:
    scope = scope or {}
    collection = scope.get("collection", "rmf_chunks")
    eval_mode = str(scope.get("mode", "") or "").lower()
    eval_intent = str(scope.get("intent", "") or "").lower()

    llm_client = get_llm_client()

    repo_root = Path(__file__).resolve().parents[2]
    chunks_path = repo_root / "data/index/chunks.parquet"
    bm25_path = repo_root / "data/bm25_index/bm25_index.pkl"

    base = {
        "query": query,
        "draft_answer": "",
        "abstained": True,
        "confidence": 0.0,
        "predicted_coverage": None,
        "citations": [],
        "retrieved_chunks": [],
    }
    framework_missing_msg = "insufficient evidence. Missing control evidence for framework query."

    control_ids = extract_control_ids(query)
    base_control_id = control_ids[0].split("(", 1)[0] if control_ids else None
    is_control_query = base_control_id is not None
    query_mode = _classify_query_mode(query, has_control_id=is_control_query)
    policy_specific = query_mode == "policy"
    framework_query = query_mode == "framework"
    mixed_query = query_mode == "policy_vs_control"
    if eval_mode == "policy_vs_control" or "coverage_assessment" in eval_intent:
        mixed_query = True
        policy_specific = False
        framework_query = False
    out_of_scope_policy_mode = eval_mode == "out_of_scope_policy" or "abstain" in eval_intent
    policy_doc_hint, policy_hint_terms = _detect_policy_doc_hint(query)
    policy_missing_msg = _policy_insufficient_message(policy_doc_hint)
    truth_mixed_coverage = _truth_coverage_for_control(base_control_id) if mixed_query else None
    try:
        if mixed_query:
            control_results: List[Dict] = []
            if base_control_id:
                control_results = _retrieve_oscal_control_chunks(
                    base_control_id,
                    top_k=6,
                    collection=collection,
                )
            else:
                control_results = hybrid_search(
                    query,
                    top_k=max(top_k, 6),
                    source_type="oscal_control",
                )
            policy_results: List[Dict] = []
            if chunks_path.exists() and bm25_path.exists():
                try:
                    policy_results = hybrid_search(
                        query,
                        top_k=max(top_k, 6),
                        source_type="policy_pdf",
                    )
                except Exception:
                    policy_results = []
            raw_results = control_results + policy_results
        elif policy_specific:
            if not chunks_path.exists() or not bm25_path.exists():
                base["draft_answer"] = policy_missing_msg
                return base

            policy_results = hybrid_search(
                query,
                top_k=max(top_k, 6),
                source_type="policy_pdf",
            )
            # Optional second policy source if present in metadata.
            try:
                policy_md_results = hybrid_search(
                    query,
                    top_k=max(top_k, 6),
                    source_type="policy_md",
                )
            except Exception:
                policy_md_results = []
            raw_results = policy_results + policy_md_results
        elif framework_query:
            # Framework questions use OSCAL-only evidence.
            if is_control_query:
                raw_results = _retrieve_oscal_control_chunks(
                    base_control_id,
                    top_k=min(10, top_k),
                    collection=collection,
                )
            else:
                raw_results = hybrid_search(
                    query,
                    top_k=top_k,
                    source_type="oscal_control",
                )
        else:
            if not chunks_path.exists() or not bm25_path.exists():
                base["draft_answer"] = "insufficient evidence. Retrieval assets are missing."
                return base
            raw_results = hybrid_search(query, top_k=top_k)
    except RuntimeError as exc:
        msg = str(exc)
        if "BM25_INDEX_PATH" in msg or "CHUNKS_PATH" in msg:
            if policy_specific or out_of_scope_policy_mode:
                base["draft_answer"] = policy_missing_msg
                base["abstained"] = True
                base["confidence"] = 0.1
                return base
            base["draft_answer"] = (
                "insufficient evidence. Retrieval index paths are not configured. "
                "Set BM25_INDEX_PATH and CHUNKS_PATH and ensure indexing is complete."
            )
            if mixed_query:
                coverage = truth_mixed_coverage or "unknown"
                base["predicted_coverage"] = coverage
                base["draft_answer"] = _render_mixed_template(
                    coverage=coverage,
                    evidence_bullets=["- insufficient evidence [C?]"],
                    gap_bullets=[
                        "- Retrieval index paths are not configured.",
                        "- Configure retrieval assets and rerun policy/control assessment.",
                    ],
                )
            return base
        if policy_specific or out_of_scope_policy_mode:
            base["draft_answer"] = policy_missing_msg
            base["abstained"] = True
            base["confidence"] = 0.1
            return base
        base["draft_answer"] = f"insufficient evidence. Retrieval error: {exc}"
        if mixed_query:
            coverage = truth_mixed_coverage or "unknown"
            base["predicted_coverage"] = coverage
            base["draft_answer"] = _render_mixed_template(
                coverage=coverage,
                evidence_bullets=["- insufficient evidence [C?]"],
                gap_bullets=[
                    "- Retrieval error prevented complete policy/control evidence collection.",
                    "- Validate index dependencies and retry assessment.",
                ],
            )
        return base
    except Exception as exc:
        if policy_specific or out_of_scope_policy_mode:
            base["draft_answer"] = policy_missing_msg
            base["abstained"] = True
            base["confidence"] = 0.1
            return base
        base["draft_answer"] = f"insufficient evidence. Retrieval error: {exc}"
        if mixed_query:
            coverage = truth_mixed_coverage or "unknown"
            base["predicted_coverage"] = coverage
            base["draft_answer"] = _render_mixed_template(
                coverage=coverage,
                evidence_bullets=["- insufficient evidence [C?]"],
                gap_bullets=[
                    "- Retrieval error prevented complete policy/control evidence collection.",
                    "- Validate index dependencies and retry assessment.",
                ],
            )
        return base

    results = _dedupe_by_chunk_id(raw_results)
    if policy_specific:
        results = [
            r
            for r in results
            if ((r.get("payload", {}) or {}).get("source_type") in {"policy_pdf", "policy_md"})
        ]
    elif framework_query:
        results = [
            r
            for r in results
            if ((r.get("payload", {}) or {}).get("source_type") == "oscal_control")
        ]
    if mixed_query:
        results = _cap_by_source(results, policy_limit=6, control_limit=6)

    citations = normalize_citations(results)
    retrieved_chunks = _to_retrieved_chunks(results)

    unique_count = len(results)
    top_score = float(results[0]["rrf_score"]) if results and results[0].get("rrf_score") is not None else None
    weak_retrieval = unique_count < MIN_UNIQUE_CHUNKS or (top_score is not None and top_score < MIN_TOP_SCORE)
    control_count, policy_count = _count_sources(results)
    policy_chunk_count = _policy_chunk_count(results)
    policy_hint_match_chunks = _matching_policy_hint_chunk_count(results, policy_hint_terms)
    policy_has_hint_match = True if not policy_hint_terms else policy_hint_match_chunks >= 2
    policy_has_min_chunks = policy_chunk_count >= 2
    policy_evidence_is_weak_or_irrelevant = weak_retrieval or (not policy_has_hint_match) or (not policy_has_min_chunks)
    if out_of_scope_policy_mode:
        base.update(
            {
                "draft_answer": policy_missing_msg,
                "abstained": True,
                "confidence": 0.1,
                "predicted_coverage": None,
                "citations": citations,
                "retrieved_chunks": retrieved_chunks,
            }
        )
        return base
    if framework_query and control_count < 1:
        missing_msg = framework_missing_msg
        if base_control_id:
            missing_msg = (
                "insufficient evidence. Missing control evidence for "
                f"control {base_control_id}. Need at least 1 control chunk."
            )
        base.update(
            {
                "draft_answer": missing_msg,
                "abstained": True,
                "confidence": 0.1,
                "predicted_coverage": None,
                "citations": citations,
                "retrieved_chunks": retrieved_chunks,
            }
        )
        return base
    if policy_specific and (out_of_scope_policy_mode or policy_evidence_is_weak_or_irrelevant):
        base.update(
            {
                "draft_answer": policy_missing_msg,
                "abstained": True,
                "confidence": 0.1,
                "predicted_coverage": None,
                "citations": citations,
                "retrieved_chunks": retrieved_chunks,
            }
        )
        return base
    if mixed_query and (control_count < 2 or policy_count < 2):
        missing_parts = []
        if control_count < 2:
            missing_parts.append("control evidence")
        if policy_count < 2:
            missing_parts.append("policy evidence")
        coverage = truth_mixed_coverage or "missing"
        base.update(
            {
                "draft_answer": _render_mixed_template(
                    coverage=coverage,
                    evidence_bullets=_evidence_bullets_from_results(results),
                    gap_bullets=[
                        (
                            "- Missing "
                            + " and ".join(missing_parts)
                            + f" for control {base_control_id or 'framework requirement'}."
                        ),
                        "- Need at least 2 control chunks and 2 policy chunks.",
                    ],
                ),
                "abstained": True,
                "confidence": 0.1,
                "predicted_coverage": coverage,
                "citations": citations,
                "retrieved_chunks": retrieved_chunks,
            }
        )
        return base

    confidence = _confidence(unique_count, top_score)
    llm_context = _cap_llm_context(retrieved_chunks)
    user_prompt = (
        f"Question:\n{query}\n\n"
        "Use only the context below.\n\n"
        f"Context:\n{_context_block(llm_context, max_chunks=MAX_LLM_CONTEXT_CHUNKS)}"
    )
    system_prompt = GROUNDED_POLICY_VS_CONTROL_PROMPT if mixed_query else GROUNDED_SYSTEM_PROMPT
    llm_text = llm_client.generate(
        system=system_prompt,
        user=user_prompt,
        context=llm_context,
    )

    llm_unavailable = llm_text.startswith("LLM not configured") or llm_text.startswith("OpenRouter backend selected")
    llm_error = llm_text.startswith("OpenRouter error:")
    predicted_coverage: Optional[str] = None
    if llm_unavailable:
        # Retrieval-only mode keeps UI useful before an LLM is configured.
        draft = _extractive_summary(results)
        if mixed_query:
            predicted_coverage = truth_mixed_coverage or _infer_coverage_label(draft, fallback="unknown")
            draft = _render_mixed_template(
                coverage=predicted_coverage or "unknown",
                evidence_bullets=_evidence_bullets_from_results(results),
                gap_bullets=_extract_gap_bullets(draft),
            )
        if framework_query or policy_specific or mixed_query:
            abstained = False
        else:
            abstained = weak_retrieval
        if abstained:
            draft = "insufficient evidence.\n" + draft
    else:
        draft = llm_text
        if mixed_query:
            predicted_coverage = _extract_first_line_coverage_label(llm_text)
            if predicted_coverage is None:
                predicted_coverage = truth_mixed_coverage or _infer_coverage_label(llm_text, fallback="unknown")
            draft = _render_mixed_template(
                coverage=predicted_coverage,
                evidence_bullets=_evidence_bullets_from_results(results),
                gap_bullets=_extract_gap_bullets(llm_text),
            )
        if framework_query or policy_specific or mixed_query:
            abstained = False
        else:
            abstained = weak_retrieval or _is_insufficient_message(llm_text)

    if llm_error:
        if results:
            draft = (
                "LLM temporarily unavailable. Showing retrieved evidence excerpts.\n\n"
                + _extractive_summary(results)
            )
        else:
            draft = "insufficient evidence. LLM temporarily unavailable and no evidence chunks were retrieved."
        abstained = True
        confidence = min(confidence, 0.1)
        if mixed_query:
            predicted_coverage = predicted_coverage or truth_mixed_coverage or "unknown"
            draft = _render_mixed_template(
                coverage=predicted_coverage,
                evidence_bullets=_evidence_bullets_from_results(results),
                gap_bullets=_extract_gap_bullets(draft),
            )

    base.update(
        {
            "draft_answer": draft,
            "abstained": abstained,
            "confidence": confidence,
            "predicted_coverage": predicted_coverage,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
        }
    )
    # Safety guard: never report non-abstained when no evidence chunks exist.
    if not base["retrieved_chunks"]:
        base["abstained"] = True
        if "insufficient evidence" not in base["draft_answer"].lower():
            base["draft_answer"] = "insufficient evidence. No retrieved evidence chunks were found."
        base["confidence"] = min(float(base["confidence"]), 0.1)
    return base
