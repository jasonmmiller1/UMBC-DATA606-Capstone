from __future__ import annotations

import inspect
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_qdrant_client():
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant_client dependency is missing. Install requirements and retry."
        ) from exc

    host = os.getenv("QDRANT_HOST", "localhost").strip() or "localhost"
    port_raw = os.getenv("QDRANT_PORT", "6333").strip() or "6333"
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid QDRANT_PORT value: {port_raw}") from exc
    return QdrantClient(host=host, port=port)


def get_index_paths() -> tuple[Path, Path]:
    bm25_env = os.getenv("BM25_INDEX_PATH", "").strip()
    chunks_env = os.getenv("CHUNKS_PATH", "").strip()
    if not bm25_env or not chunks_env:
        raise RuntimeError(
            "Missing index env vars. Set BM25_INDEX_PATH and CHUNKS_PATH before retrieval."
        )

    repo_root = _repo_root()
    bm25_path = Path(bm25_env)
    chunks_path = Path(chunks_env)
    if not bm25_path.is_absolute():
        bm25_path = repo_root / bm25_path
    if not chunks_path.is_absolute():
        chunks_path = repo_root / chunks_path

    if not bm25_path.exists():
        raise RuntimeError(f"BM25 index file not found: {bm25_path}")
    if not chunks_path.exists():
        raise RuntimeError(f"Chunks parquet file not found: {chunks_path}")
    return bm25_path, chunks_path


def _build_hybrid_kwargs(
    hybrid_retrieve_fn,
    query: str,
    client,
    bm25_index_path: Path,
    chunks_path: Path,
    top_k: int,
    extra_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    sig = inspect.signature(hybrid_retrieve_fn)
    params = sig.parameters
    call_kwargs: Dict[str, Any] = {}

    alias_values = {
        "query": query,
        "client": client,
        "top_k": top_k,
    }
    bm25_aliases = ["bm25_index_path", "bm25_path"]
    chunks_aliases = ["chunks_path", "chunk_path", "chunks_parquet_path"]

    for name, value in alias_values.items():
        if name in params:
            call_kwargs[name] = value
    for name in bm25_aliases:
        if name in params:
            call_kwargs[name] = bm25_index_path
            break
    for name in chunks_aliases:
        if name in params:
            call_kwargs[name] = chunks_path
            break

    for k, v in extra_kwargs.items():
        if k in params:
            call_kwargs[k] = v
    return call_kwargs


def hybrid_search(query: str, top_k: int = 10, **kwargs):
    bm25_index_path, chunks_path = get_index_paths()
    client = get_qdrant_client()
    from app.retrieval.retrieve import hybrid_retrieve

    call_kwargs = _build_hybrid_kwargs(
        hybrid_retrieve_fn=hybrid_retrieve,
        query=query,
        client=client,
        bm25_index_path=bm25_index_path,
        chunks_path=chunks_path,
        top_k=top_k,
        extra_kwargs=kwargs,
    )
    return hybrid_retrieve(**call_kwargs)


if __name__ == "__main__":
    print(len(hybrid_search("AC-2", top_k=5)))
