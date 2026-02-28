from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from app.index.qdrant_schema import DEFAULT_COLLECTION
from app.retrieval.service import hybrid_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 2 retrieval test queries and save JSONL results.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--queries-path", default="data/tests/week2_queries.json")
    parser.add_argument("--output-path", default="data/tests/week2_retrieval_results.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--bm25-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--chunks-path", default="data/index/chunks.parquet")
    parser.add_argument("--bm25-index-path", default="data/bm25_index/bm25_index.pkl")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    queries_path = repo_root / args.queries_path
    output_path = repo_root / args.output_path
    chunks_path = repo_root / args.chunks_path
    bm25_index_path = repo_root / args.bm25_index_path

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(queries, list):
        raise ValueError("week2_queries.json must contain a list of query objects.")

    os.environ["QDRANT_HOST"] = args.host
    os.environ["QDRANT_PORT"] = str(args.port)
    os.environ["CHUNKS_PATH"] = str(chunks_path)
    os.environ["BM25_INDEX_PATH"] = str(bm25_index_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for item in queries:
            query = item["query"]
            filters = item.get("filters") or {}
            results = hybrid_search(
                query,
                top_k=args.top_k,
                collection_name=args.collection,
                dense_k=args.dense_k,
                bm25_k=args.bm25_k,
                rrf_k=args.rrf_k,
                source_type=filters.get("source_type"),
                control_id=filters.get("control_id"),
                doc_id=filters.get("doc_id"),
            )
            out.write(
                json.dumps(
                    {
                        "query": query,
                        "filters": filters,
                        "results": results,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
            print(f"Processed {count}: {query}")

    print(f"Wrote retrieval results: {output_path}")


if __name__ == "__main__":
    main()
