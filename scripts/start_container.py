from __future__ import annotations

import os
import sys

from app.runtime import BM25_INDEX_PATH, CHUNKS_PATH, PORT, env_bool, env_float
from app.runtime_bootstrap import prepare_local_indexes, wait_for_qdrant


def _validate_local_indexes() -> None:
    missing = [str(path) for path in (CHUNKS_PATH, BM25_INDEX_PATH) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing local retrieval assets. Either enable PREPARE_LOCAL_INDEXES_ON_START=1 "
            f"or create these files ahead of time: {', '.join(missing)}"
        )


def main() -> None:
    prepare_local_indexes_on_start = env_bool("PREPARE_LOCAL_INDEXES_ON_START", True)
    wait_for_qdrant_on_start = env_bool("WAIT_FOR_QDRANT_ON_START", True)
    require_qdrant_collection = env_bool("REQUIRE_QDRANT_COLLECTION", True)
    qdrant_wait_seconds = env_float("STARTUP_QDRANT_TIMEOUT_SECONDS", 120.0)

    if prepare_local_indexes_on_start:
        chunks_df = prepare_local_indexes(force=False)
        print(
            "Prepared local retrieval assets "
            f"rows={len(chunks_df)} chunks_path={CHUNKS_PATH} bm25_index_path={BM25_INDEX_PATH}"
        )
    else:
        _validate_local_indexes()
        print(f"Using existing retrieval assets chunks_path={CHUNKS_PATH} bm25_index_path={BM25_INDEX_PATH}")

    if wait_for_qdrant_on_start:
        wait_for_qdrant(
            timeout_seconds=qdrant_wait_seconds,
            require_collection=require_qdrant_collection,
        )
        collection_msg = "with collection check" if require_qdrant_collection else "without collection check"
        print(f"Qdrant startup check passed {collection_msg}.")
    else:
        print("Skipping Qdrant startup check because WAIT_FOR_QDRANT_ON_START=0.")

    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            "--server.address=0.0.0.0",
            "--server.port",
            str(PORT),
            "--browser.gatherUsageStats=false",
        ],
    )


if __name__ == "__main__":
    main()
