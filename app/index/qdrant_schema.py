from __future__ import annotations

import argparse

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.utils.embeddings import embed_texts


DEFAULT_COLLECTION = "rmf_chunks"


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    collections = client.get_collections().collections
    existing = {c.name for c in collections}
    if collection_name in existing:
        details = client.get_collection(collection_name=collection_name)
        existing_size = int(details.config.params.vectors.size)
        if existing_size != vector_size:
            raise ValueError(
                f"Collection '{collection_name}' exists with vector size {existing_size}, "
                f"but requested size is {vector_size}."
            )
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create rmf_chunks collection in Qdrant if missing.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--vector-size", type=int, default=None)
    args = parser.parse_args()

    vector_size = args.vector_size or int(embed_texts(["dimension probe"]).shape[1])
    client = QdrantClient(host=args.host, port=args.port)
    ensure_collection(client, args.collection, vector_size)
    print(
        f"Collection ready: name={args.collection} "
        f"vector_size={vector_size} distance=cosine host={args.host}:{args.port}"
    )


if __name__ == "__main__":
    main()
