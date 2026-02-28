from __future__ import annotations

import argparse
import math
from pathlib import Path
import uuid
from typing import Dict, List

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.ingest.chunkers import build_chunks_dataframe
from app.index.qdrant_schema import DEFAULT_COLLECTION, ensure_collection
from app.utils.embeddings import embed_texts


CHUNK_ID_NAMESPACE = uuid.UUID("49b7b7f2-1291-4ed5-a4b6-b68de66f7b8e")


def _point_id_from_chunk_id(chunk_id: str) -> str:
    # Qdrant IDs must be uint64 or UUID. Use deterministic UUID5 so re-indexing is idempotent.
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, chunk_id))


def _payload_from_row(row: pd.Series) -> Dict:
    payload = {
        "source_type": row.get("source_type"),
        "control_id": row.get("control_id"),
        "control_part": row.get("control_part"),
        "enhancement_id": row.get("enhancement_id"),
        "control_family": row.get("control_family"),
        "doc_id": row.get("doc_id"),
        "doc_title": row.get("doc_title"),
        "section_path": row.get("section_path"),
        "page_start": row.get("page_start"),
        "page_end": row.get("page_end"),
        "chunk_id": row.get("chunk_id"),
        "chunk_text": row.get("chunk_text"),
    }
    return {k: v for k, v in payload.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}


def index_chunks(
    chunks_df: pd.DataFrame,
    client: QdrantClient,
    collection_name: str,
    batch_size: int = 64,
) -> None:
    if chunks_df.empty:
        raise ValueError("No chunks to index.")

    probe = embed_texts([chunks_df.iloc[0]["chunk_text"]])
    vector_size = int(probe.shape[1])
    ensure_collection(client, collection_name=collection_name, vector_size=vector_size)

    total = len(chunks_df)
    for start in range(0, total, batch_size):
        batch = chunks_df.iloc[start : start + batch_size]
        vectors = embed_texts(batch["chunk_text"].tolist())

        points: List[PointStruct] = []
        for i, (_, row) in enumerate(batch.iterrows()):
            chunk_id = str(row["chunk_id"])
            points.append(
                PointStruct(
                    id=_point_id_from_chunk_id(chunk_id),
                    vector=vectors[i].tolist(),
                    payload=_payload_from_row(row),
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)
        print(f"Upserted {min(start + batch_size, total)}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk OSCAL + policies and index into Qdrant.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    chunks = build_chunks_dataframe(repo_root)

    out_dir = repo_root / "data/index"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.parquet"
    chunks.to_parquet(chunks_path, index=False)
    print(f"Saved chunks dataframe: {chunks_path} rows={len(chunks)}")

    client = QdrantClient(host=args.host, port=args.port)
    index_chunks(
        chunks_df=chunks,
        client=client,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )
    print(f"Indexing complete: collection={args.collection}")


if __name__ == "__main__":
    main()
