from __future__ import annotations

import time
from typing import Any

import pandas as pd

from app.index.bm25_index import build_index
from app.index.index_to_qdrant import index_chunks
from app.ingest.chunkers import build_chunk_diagnostics, build_chunks_dataframe
from app.runtime import (
    BM25_INDEX_PATH,
    CHUNKS_PATH,
    QDRANT_COLLECTION,
    REPO_ROOT,
    create_qdrant_client,
    qdrant_target_label,
)


def prepare_local_indexes(*, force: bool = False) -> pd.DataFrame:
    if force or not CHUNKS_PATH.exists():
        chunks_df = build_chunks_dataframe(REPO_ROOT)
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        chunks_df.to_parquet(CHUNKS_PATH, index=False)
    else:
        chunks_df = pd.read_parquet(CHUNKS_PATH)

    if force or not BM25_INDEX_PATH.exists():
        build_index(chunks_df, BM25_INDEX_PATH.parent)

    return chunks_df


def local_index_snapshot(chunks_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "chunks_path": str(CHUNKS_PATH),
        "bm25_index_path": str(BM25_INDEX_PATH),
        "rows": int(len(chunks_df)),
        "diagnostics": build_chunk_diagnostics(chunks_df),
    }


def wait_for_qdrant(
    *,
    timeout_seconds: float,
    require_collection: bool,
    collection_name: str = QDRANT_COLLECTION,
    poll_interval_seconds: float = 2.0,
) -> None:
    deadline = time.monotonic() + max(float(timeout_seconds), 1.0)
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            client = create_qdrant_client(timeout=min(max(timeout_seconds, 1.0), 5.0))
            if require_collection:
                client.get_collection(collection_name)
            else:
                client.get_collections()
            return
        except Exception as exc:  # pragma: no cover - integration behavior
            last_error = exc
            time.sleep(max(float(poll_interval_seconds), 0.1))

    requirement = f"collection={collection_name}" if require_collection else "service reachability"
    detail = str(last_error) if last_error else "unknown error"
    raise RuntimeError(f"Timed out waiting for Qdrant ({requirement}) at {qdrant_target_label()}: {detail}")


def seed_qdrant(
    *,
    force_local_indexes: bool = False,
    batch_size: int = 64,
    wait_timeout_seconds: float = 60.0,
    collection_name: str = QDRANT_COLLECTION,
) -> int:
    chunks_df = prepare_local_indexes(force=force_local_indexes)
    wait_for_qdrant(
        timeout_seconds=wait_timeout_seconds,
        require_collection=False,
        collection_name=collection_name,
    )
    client = create_qdrant_client()
    index_chunks(
        chunks_df=chunks_df,
        client=client,
        collection_name=collection_name,
        batch_size=batch_size,
    )
    return int(len(chunks_df))
