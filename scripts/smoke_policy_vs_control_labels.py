#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag.answer import answer_question


TARGET_LABELS = {
    "gq024": "covered",
    "gq027": "partial",
    "gq028": "partial",
    "gq030": "missing",
}


def _load_questions(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("id", "") or "")
            if qid:
                rows[qid] = row
    return rows


def main() -> int:
    # Force retrieval-only mode.
    os.environ["LLM_BACKEND"] = "none"
    os.environ.setdefault("BM25_INDEX_PATH", "data/bm25_index/bm25_index.pkl")
    os.environ.setdefault("CHUNKS_PATH", "data/index/chunks.parquet")
    os.environ.setdefault("QDRANT_HOST", "localhost")
    os.environ.setdefault("QDRANT_PORT", "6333")

    questions_path = REPO_ROOT / "data/eval/golden_questions.jsonl"
    rows = _load_questions(questions_path)

    failures: list[str] = []
    for qid, expected_label in TARGET_LABELS.items():
        row = rows.get(qid)
        if row is None:
            failures.append(f"{qid}: missing from {questions_path}")
            continue

        result = answer_question(
            str(row.get("question", "") or ""),
            eval_mode="policy_vs_control",
            eval_intent=str(row.get("intent", "") or ""),
            expected=row.get("expected") or {},
            top_k=10,
        )
        predicted = str(result.get("predicted_coverage", "") or "").strip().lower()
        print(f"{qid}: predicted={predicted or '<empty>'} expected={expected_label}")
        if predicted != expected_label:
            failures.append(f"{qid}: predicted={predicted or '<empty>'} expected={expected_label}")

    if failures:
        print("\nFAIL")
        for line in failures:
            print(f"- {line}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
