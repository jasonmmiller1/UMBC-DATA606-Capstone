from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Dict

from app.runtime import BM25_INDEX_PATH, CHUNKS_PATH, create_qdrant_client


@lru_cache(maxsize=1)
def get_qdrant_client():
    return create_qdrant_client()


def get_index_paths():
    bm25_path = BM25_INDEX_PATH
    chunks_path = CHUNKS_PATH
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
