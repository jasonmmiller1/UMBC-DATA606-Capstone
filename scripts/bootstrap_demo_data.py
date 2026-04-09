from __future__ import annotations

import argparse
import json

from app.index.index_to_qdrant import index_chunks
from app.runtime import QDRANT_COLLECTION, create_qdrant_client, qdrant_target_label
from app.runtime_bootstrap import local_index_snapshot, prepare_local_indexes, wait_for_qdrant


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local retrieval assets and optionally seed the configured Qdrant collection."
    )
    parser.add_argument("--force", action="store_true", help="Rebuild local chunk and BM25 assets even if present.")
    parser.add_argument("--seed-qdrant", action="store_true", help="Upsert the demo corpus into Qdrant.")
    parser.add_argument("--batch-size", type=int, default=64, help="Qdrant upsert batch size.")
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=60.0,
        help="How long to wait for Qdrant before failing when --seed-qdrant is used.",
    )
    args = parser.parse_args()

    chunks_df = prepare_local_indexes(force=args.force)
    print(json.dumps(local_index_snapshot(chunks_df), indent=2))

    if not args.seed_qdrant:
        return

    wait_for_qdrant(
        timeout_seconds=args.wait_timeout_seconds,
        require_collection=False,
        collection_name=QDRANT_COLLECTION,
    )
    client = create_qdrant_client()
    index_chunks(
        chunks_df=chunks_df,
        client=client,
        collection_name=QDRANT_COLLECTION,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "qdrant_target": qdrant_target_label(),
                "collection": QDRANT_COLLECTION,
                "rows_upserted": int(len(chunks_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
