from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app.index.bm25_index import load_index
from app.index.qdrant_schema import DEFAULT_COLLECTION
from app.utils.embeddings import embed_texts

logger = logging.getLogger(__name__)


CONTROL_QUERY_RE = re.compile(r"^[A-Za-z]{2,3}-\d+([(.]\d+[)]?)?$")
CONTROL_ID_TOKEN_RE = re.compile(r"\b[A-Za-z]{2,3}-\d{1,3}(?:\(\d+\))?\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
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
LOW_SIGNAL_SECTION_TYPES = {"purpose", "scope", "definitions", "exceptions"}
HIGH_SIGNAL_SECTION_TYPES = {"procedures", "requirements", "retention", "roles", "policy"}
POLICY_SOURCE_TYPES = {"policy_pdf", "policy_md"}
SECTION_TYPE_QUERY_HINTS: Dict[str, Tuple[str, ...]] = {
    "purpose": ("purpose",),
    "scope": ("scope",),
    "definitions": ("definition", "definitions", "glossary", "terminology"),
    "exceptions": ("exception", "exceptions"),
}
RERANKER_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "for",
    "of",
    "in",
    "on",
    "with",
    "our",
    "what",
    "does",
    "is",
    "are",
}
DEFAULT_RERANK_INTENTS = {"policy"}
DEFAULT_TOP_K = 10
DEFAULT_DENSE_K = 20
DEFAULT_BM25_K = 20
DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_BM25_WEIGHT = 1.0


def build_qdrant_filter(
    source_type: Optional[str] = None,
    control_id: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> Optional[Filter]:
    must: List[FieldCondition] = []
    if source_type:
        must.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))
    if control_id:
        must.append(FieldCondition(key="control_id", match=MatchValue(value=control_id.upper())))
    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    if not must:
        return None
    return Filter(must=must)


def retrieve_dense(
    query: str,
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 20,
    source_type: Optional[str] = None,
    control_id: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> List[Dict]:
    qvec = embed_texts([query])[0].tolist()
    qfilter = build_qdrant_filter(source_type=source_type, control_id=control_id, doc_id=doc_id)
    response = client.query_points(
        collection_name=collection_name,
        query=qvec,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
    )
    hits = response.points
    return [
        {
            "chunk_id": str((hit.payload or {}).get("chunk_id") or hit.id),
            "score": float(hit.score),
            "payload": hit.payload or {},
        }
        for hit in hits
    ]


def retrieve_bm25(query: str, bm25_index_path: Path, top_k: int = 20) -> List[Tuple[str, float]]:
    bm25 = load_index(bm25_index_path)
    return bm25.query(query, top_k=top_k)


def rrf_fuse(
    dense_ranked_ids: Sequence[str],
    bm25_ranked_ids: Sequence[str],
    k: int = 60,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ranked_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (float(dense_weight) / (k + rank))
    for rank, chunk_id in enumerate(bm25_ranked_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (float(bm25_weight) / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _control_id_from_query(query: str) -> Optional[str]:
    q = query.strip().upper()
    if CONTROL_QUERY_RE.match(q):
        return q
    return None


def _chunks_lookup(chunks_path: Path) -> Dict[str, Dict]:
    df = pd.read_parquet(chunks_path)
    return {str(r["chunk_id"]): r for r in df.to_dict(orient="records")}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "")
    if not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "")
    if not value.strip():
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _resolve_hybrid_retrieval_config(
    *,
    top_k: int = DEFAULT_TOP_K,
    dense_k: Optional[int] = None,
    bm25_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
    dense_weight: Optional[float] = None,
    bm25_weight: Optional[float] = None,
) -> Dict[str, float | int]:
    resolved_dense_k = dense_k if dense_k is not None else _env_int("RETRIEVAL_DENSE_K", DEFAULT_DENSE_K)
    resolved_bm25_k = bm25_k if bm25_k is not None else _env_int("RETRIEVAL_BM25_K", DEFAULT_BM25_K)
    resolved_rrf_k = rrf_k if rrf_k is not None else _env_int("RETRIEVAL_RRF_K", DEFAULT_RRF_K)
    resolved_dense_weight = (
        dense_weight if dense_weight is not None else _env_float("RETRIEVAL_DENSE_WEIGHT", DEFAULT_DENSE_WEIGHT)
    )
    resolved_bm25_weight = (
        bm25_weight if bm25_weight is not None else _env_float("RETRIEVAL_BM25_WEIGHT", DEFAULT_BM25_WEIGHT)
    )
    return {
        "top_k": int(top_k),
        "dense_k": int(resolved_dense_k),
        "bm25_k": int(resolved_bm25_k),
        "rrf_k": int(resolved_rrf_k),
        "dense_weight": float(resolved_dense_weight),
        "bm25_weight": float(resolved_bm25_weight),
    }


def _normalize_intent(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _rerank_enabled_default() -> bool:
    return True


def _rerank_enabled() -> bool:
    default = _rerank_enabled_default()
    rerank_enabled_raw = os.getenv("RERANK_ENABLED", "")
    if rerank_enabled_raw.strip():
        return _env_bool("RERANK_ENABLED", default)

    # Backward compatibility with previous flag name.
    legacy_raw = os.getenv("ENABLE_RERANKER", "")
    if legacy_raw.strip():
        return _env_bool("ENABLE_RERANKER", default)

    return default


def _rerank_intents() -> set[str]:
    raw = os.getenv("RERANK_INTENTS", "policy")
    intents = {_normalize_intent(part) for part in raw.split(",") if _normalize_intent(part)}
    return intents or set(DEFAULT_RERANK_INTENTS)


def _should_apply_reranker(intent: Optional[str]) -> bool:
    if not _rerank_enabled():
        return False
    normalized_intent = _normalize_intent(intent)
    if not normalized_intent:
        return False
    return normalized_intent in _rerank_intents()


def get_retrieval_config_snapshot(
    *,
    top_k: int = DEFAULT_TOP_K,
    dense_k: Optional[int] = None,
    bm25_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
    dense_weight: Optional[float] = None,
    bm25_weight: Optional[float] = None,
) -> Dict[str, object]:
    resolved = _resolve_hybrid_retrieval_config(
        top_k=top_k,
        dense_k=dense_k,
        bm25_k=bm25_k,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    return {
        **resolved,
        "policy_section_weighting_enabled": _env_bool("RETRIEVAL_POLICY_SECTION_WEIGHTING_ENABLED", True),
        "policy_low_signal_multiplier": _env_float("RETRIEVAL_POLICY_LOW_SIGNAL_MULTIPLIER", 0.6),
        "policy_high_signal_multiplier": _env_float("RETRIEVAL_POLICY_HIGH_SIGNAL_MULTIPLIER", 1.1),
        "rerank_enabled": _rerank_enabled(),
        "rerank_intents": sorted(_rerank_intents()),
        "rerank_candidates": max(
            int(resolved["top_k"]),
            int(_env_float("RERANK_CANDIDATES", 20)),
        ),
        "rerank_control_match_boost": _env_float("RERANK_CONTROL_MATCH_BOOST", 0.25),
        "rerank_heading_overlap_boost": _env_float("RERANK_HEADING_OVERLAP_BOOST", 0.04),
        "rerank_low_signal_penalty": _env_float("RERANK_LOW_SIGNAL_PENALTY", 0.08),
    }


def _infer_section_type_from_heading(heading: str) -> str:
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


def _section_type_from_payload(payload: Dict) -> str:
    raw = str(payload.get("section_type", "") or "").strip().lower()
    if raw in SECTION_TYPES:
        return raw

    heading = str(payload.get("heading", "") or payload.get("section_path", "") or "")
    inferred = _infer_section_type_from_heading(heading)
    return inferred if inferred in SECTION_TYPES else "other"


def _query_requests_low_signal_section(query: str, section_type: str) -> bool:
    hints = SECTION_TYPE_QUERY_HINTS.get(section_type, ())
    if not hints:
        return False
    query_lower = (query or "").lower()
    return any(term in query_lower for term in hints)


def _policy_section_multiplier(query: str, payload: Dict) -> float:
    if not _env_bool("RETRIEVAL_POLICY_SECTION_WEIGHTING_ENABLED", True):
        return 1.0

    source_type = str(payload.get("source_type", "") or "").strip().lower()
    if source_type not in POLICY_SOURCE_TYPES:
        return 1.0

    section_type = _section_type_from_payload(payload)
    low_signal_multiplier = _env_float("RETRIEVAL_POLICY_LOW_SIGNAL_MULTIPLIER", 0.6)
    high_signal_multiplier = _env_float("RETRIEVAL_POLICY_HIGH_SIGNAL_MULTIPLIER", 1.1)

    if section_type in LOW_SIGNAL_SECTION_TYPES:
        if _query_requests_low_signal_section(query, section_type):
            return 1.0
        return low_signal_multiplier
    if section_type in HIGH_SIGNAL_SECTION_TYPES:
        return high_signal_multiplier
    return 1.0


def _base_control_id(value: str) -> str:
    match = CONTROL_ID_TOKEN_RE.search((value or "").upper())
    if not match:
        return ""
    text = match.group(0)
    return text.split("(", 1)[0]


def _query_control_ids(query: str) -> set[str]:
    return {_base_control_id(token) for token in CONTROL_ID_TOKEN_RE.findall(query or "") if _base_control_id(token)}


def _keyword_terms(text: str) -> set[str]:
    terms = {t.lower() for t in TOKEN_RE.findall(text or "")}
    return {t for t in terms if t not in RERANKER_STOPWORDS and len(t) > 1}


def _rerank_candidates(
    query: str,
    candidates: List[Dict],
    *,
    top_k: int,
) -> List[Dict]:
    if not candidates:
        return []

    control_match_boost = _env_float("RERANK_CONTROL_MATCH_BOOST", 0.25)
    heading_overlap_boost = _env_float("RERANK_HEADING_OVERLAP_BOOST", 0.04)
    low_signal_penalty = _env_float("RERANK_LOW_SIGNAL_PENALTY", 0.08)

    query_controls = _query_control_ids(query)
    query_terms = _keyword_terms(query)

    reranked: List[Dict] = []
    for item in candidates:
        payload = item.get("payload", {}) or {}
        delta = 0.0

        payload_control_id = _base_control_id(str(payload.get("control_id", "") or ""))
        if payload_control_id and payload_control_id in query_controls:
            delta += control_match_boost

        heading_text = " ".join(
            [
                str(payload.get("heading", "") or ""),
                str(payload.get("section_path", "") or ""),
            ]
        )
        heading_terms = _keyword_terms(heading_text)
        overlap = len(query_terms & heading_terms)
        if overlap > 0:
            delta += heading_overlap_boost * float(overlap)

        section_type = _section_type_from_payload(payload)
        if section_type in LOW_SIGNAL_SECTION_TYPES and not _query_requests_low_signal_section(query, section_type):
            delta -= low_signal_penalty

        rerank_score = float(item.get("rrf_score", 0.0)) + delta
        enriched = dict(item)
        enriched["rerank_delta"] = float(delta)
        enriched["rerank_score"] = float(rerank_score)
        enriched["rrf_score"] = float(rerank_score)
        reranked.append(enriched)

    reranked.sort(
        key=lambda item: (
            float(item.get("rerank_score", item.get("rrf_score", 0.0))),
            float(item.get("base_rrf_score", item.get("rrf_score", 0.0))),
            str(item.get("chunk_id", "")),
        ),
        reverse=True,
    )
    return reranked[:top_k]


def hybrid_retrieve(
    query: str,
    client: QdrantClient,
    bm25_index_path: Path,
    chunks_path: Path,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = DEFAULT_TOP_K,
    dense_k: Optional[int] = None,
    bm25_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
    dense_weight: Optional[float] = None,
    bm25_weight: Optional[float] = None,
    source_type: Optional[str] = None,
    control_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    intent: Optional[str] = None,
) -> List[Dict]:
    inferred_control = _control_id_from_query(query)
    if inferred_control and not control_id:
        control_id = inferred_control
        source_type = source_type or "oscal_control"

    resolved = _resolve_hybrid_retrieval_config(
        top_k=top_k,
        dense_k=dense_k,
        bm25_k=bm25_k,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    top_k = int(resolved["top_k"])
    dense_k = int(resolved["dense_k"])
    bm25_k = int(resolved["bm25_k"])
    rrf_k = int(resolved["rrf_k"])
    dense_weight = float(resolved["dense_weight"])
    bm25_weight = float(resolved["bm25_weight"])

    dense_hits = retrieve_dense(
        query=query,
        client=client,
        collection_name=collection_name,
        top_k=dense_k,
        source_type=source_type,
        control_id=control_id,
        doc_id=doc_id,
    )
    bm25_hits = retrieve_bm25(query=query, bm25_index_path=bm25_index_path, top_k=bm25_k)

    dense_ids = [h["chunk_id"] for h in dense_hits]
    bm25_ids = [cid for cid, _ in bm25_hits]
    fused = rrf_fuse(
        dense_ids,
        bm25_ids,
        k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )

    dense_payload_map = {h["chunk_id"]: (h.get("payload") or {}) for h in dense_hits}
    dense_score_map = {h["chunk_id"]: h["score"] for h in dense_hits}
    bm25_score_map = {cid: score for cid, score in bm25_hits}
    dense_rank_map = {h["chunk_id"]: rank for rank, h in enumerate(dense_hits, start=1)}
    bm25_rank_map = {cid: rank for rank, (cid, _) in enumerate(bm25_hits, start=1)}
    chunk_map = _chunks_lookup(chunks_path)

    weighted: List[Tuple[str, float, float, float, Optional[int], Optional[int]]] = []
    for chunk_id, fusion_score in fused:
        payload = dict(chunk_map.get(chunk_id, {}))
        payload.update(dense_payload_map.get(chunk_id, {}))
        multiplier = _policy_section_multiplier(query, payload)
        weighted_score = float(fusion_score) * float(multiplier)
        weighted.append(
            (
                chunk_id,
                weighted_score,
                float(multiplier),
                float(fusion_score),
                dense_rank_map.get(chunk_id),
                bm25_rank_map.get(chunk_id),
            )
        )
    weighted.sort(key=lambda item: (item[1], item[3], item[0]), reverse=True)

    scored: List[Dict] = []
    for chunk_id, weighted_score, section_multiplier, fusion_score, dense_rank, bm25_rank in weighted:
        payload = dict(chunk_map.get(chunk_id, {}))
        payload.update(dense_payload_map.get(chunk_id, {}))
        scored.append(
            {
                "chunk_id": chunk_id,
                "rrf_score": float(weighted_score),
                "base_rrf_score": float(fusion_score),
                "fusion_score": float(fusion_score),
                "policy_section_multiplier": float(section_multiplier),
                "dense_score": dense_score_map.get(chunk_id),
                "bm25_score": bm25_score_map.get(chunk_id),
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "payload": payload,
            }
        )

    scored_top_k = scored[:top_k]
    if _should_apply_reranker(intent):
        rerank_candidates = max(
            top_k,
            int(_env_float("RERANK_CANDIDATES", 20)),
        )
        candidate_pool = scored[:rerank_candidates]
        start = time.perf_counter()
        try:
            scored = _rerank_candidates(
                query=query,
                candidates=candidate_pool,
                top_k=top_k,
            )
            rerank_time_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "Reranker applied intent=%s candidate_k=%d final_top_k=%d rerank_time_ms=%.2f",
                _normalize_intent(intent) or "unknown",
                len(candidate_pool),
                top_k,
                rerank_time_ms,
            )
        except Exception:
            rerank_time_ms = (time.perf_counter() - start) * 1000.0
            logger.warning(
                "Reranker failed intent=%s candidate_k=%d final_top_k=%d rerank_time_ms=%.2f; using baseline ordering",
                _normalize_intent(intent) or "unknown",
                len(candidate_pool),
                top_k,
                rerank_time_ms,
                exc_info=True,
            )
            scored = scored_top_k
    else:
        scored = scored_top_k

    results: List[Dict] = []
    for rank, item in enumerate(scored, start=1):
        row = dict(item)
        row["rank"] = rank
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval over Qdrant + local BM25 with RRF.")
    parser.add_argument("query")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--dense-k", type=int, default=None)
    parser.add_argument("--bm25-k", type=int, default=None)
    parser.add_argument("--rrf-k", type=int, default=None)
    parser.add_argument("--dense-weight", type=float, default=None)
    parser.add_argument("--bm25-weight", type=float, default=None)
    parser.add_argument("--source-type", default=None)
    parser.add_argument("--control-id", default=None)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--intent", default=None)
    parser.add_argument("--chunks-path", default="data/index/chunks.parquet")
    parser.add_argument("--bm25-index-path", default="data/bm25_index/bm25_index.pkl")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    chunks_path = repo_root / args.chunks_path
    bm25_path = repo_root / args.bm25_index_path

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks parquet: {chunks_path}")
    if not bm25_path.exists():
        raise FileNotFoundError(f"Missing BM25 index: {bm25_path}")

    config_snapshot = get_retrieval_config_snapshot(
        top_k=args.top_k,
        dense_k=args.dense_k,
        bm25_k=args.bm25_k,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
    )

    client = QdrantClient(host=args.host, port=args.port)
    results = hybrid_retrieve(
        query=args.query,
        client=client,
        bm25_index_path=bm25_path,
        chunks_path=chunks_path,
        collection_name=args.collection,
        top_k=args.top_k,
        dense_k=args.dense_k,
        bm25_k=args.bm25_k,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        source_type=args.source_type,
        control_id=args.control_id,
        doc_id=args.doc_id,
        intent=args.intent,
    )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    if args.print_config:
        print("Retrieval config:")
        print(json.dumps(config_snapshot, indent=2))

    for item in results:
        payload = item["payload"]
        title = payload.get("doc_title") or payload.get("doc_id") or "unknown"
        section = payload.get("section_path") or "n/a"
        text = (payload.get("chunk_text") or "").replace("\n", " ").strip()
        preview = text[:180] + ("..." if len(text) > 180 else "")
        print(
            f"{item['rank']:2d}. chunk_id={item['chunk_id']} score={item['rrf_score']:.4f} "
            f"fusion={item.get('fusion_score')} dense={item['dense_score']} bm25={item['bm25_score']}\n"
            f"    {title} | {section}\n"
            f"    {preview}"
        )


if __name__ == "__main__":
    main()
