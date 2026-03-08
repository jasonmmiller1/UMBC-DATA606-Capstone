from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app.index.bm25_index import load_index
from app.index.qdrant_schema import DEFAULT_COLLECTION
from app.utils.embeddings import embed_texts


CONTROL_QUERY_RE = re.compile(r"^[A-Za-z]{2,3}-\d+([(.]\d+[)]?)?$")
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
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ranked_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, chunk_id in enumerate(bm25_ranked_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
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


def hybrid_retrieve(
    query: str,
    client: QdrantClient,
    bm25_index_path: Path,
    chunks_path: Path,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 10,
    dense_k: int = 20,
    bm25_k: int = 20,
    rrf_k: int = 60,
    source_type: Optional[str] = None,
    control_id: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> List[Dict]:
    inferred_control = _control_id_from_query(query)
    if inferred_control and not control_id:
        control_id = inferred_control
        source_type = source_type or "oscal_control"

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
    fused = rrf_fuse(dense_ids, bm25_ids, k=rrf_k)

    dense_payload_map = {h["chunk_id"]: (h.get("payload") or {}) for h in dense_hits}
    dense_score_map = {h["chunk_id"]: h["score"] for h in dense_hits}
    bm25_score_map = {cid: score for cid, score in bm25_hits}
    chunk_map = _chunks_lookup(chunks_path)

    weighted: List[Tuple[str, float, float, float]] = []
    for chunk_id, fused_score in fused:
        payload = dict(chunk_map.get(chunk_id, {}))
        payload.update(dense_payload_map.get(chunk_id, {}))
        multiplier = _policy_section_multiplier(query, payload)
        weighted_score = float(fused_score) * float(multiplier)
        weighted.append((chunk_id, weighted_score, float(multiplier), float(fused_score)))
    weighted.sort(key=lambda item: (item[1], item[3], item[0]), reverse=True)
    weighted = weighted[:top_k]

    results: List[Dict] = []
    for rank, (chunk_id, fused_score, section_multiplier, base_rrf_score) in enumerate(weighted, start=1):
        payload = dict(chunk_map.get(chunk_id, {}))
        payload.update(dense_payload_map.get(chunk_id, {}))
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "rrf_score": fused_score,
                "base_rrf_score": base_rrf_score,
                "policy_section_multiplier": section_multiplier,
                "dense_score": dense_score_map.get(chunk_id),
                "bm25_score": bm25_score_map.get(chunk_id),
                "payload": payload,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval over Qdrant + local BM25 with RRF.")
    parser.add_argument("query")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--bm25-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--source-type", default=None)
    parser.add_argument("--control-id", default=None)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--chunks-path", default="data/index/chunks.parquet")
    parser.add_argument("--bm25-index-path", default="data/bm25_index/bm25_index.pkl")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    chunks_path = repo_root / args.chunks_path
    bm25_path = repo_root / args.bm25_index_path

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks parquet: {chunks_path}")
    if not bm25_path.exists():
        raise FileNotFoundError(f"Missing BM25 index: {bm25_path}")

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
        source_type=args.source_type,
        control_id=args.control_id,
        doc_id=args.doc_id,
    )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for item in results:
        payload = item["payload"]
        title = payload.get("doc_title") or payload.get("doc_id") or "unknown"
        section = payload.get("section_path") or "n/a"
        text = (payload.get("chunk_text") or "").replace("\n", " ").strip()
        preview = text[:180] + ("..." if len(text) > 180 else "")
        print(
            f"{item['rank']:2d}. chunk_id={item['chunk_id']} rrf={item['rrf_score']:.4f} "
            f"dense={item['dense_score']} bm25={item['bm25_score']}\n"
            f"    {title} | {section}\n"
            f"    {preview}"
        )


if __name__ == "__main__":
    main()
