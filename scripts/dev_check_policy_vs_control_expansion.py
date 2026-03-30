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


TARGET_IDS = ("gq024", "gq027", "gq028", "gq030")


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
    os.environ.setdefault("LLM_BACKEND", "none")
    questions = _load_questions(REPO_ROOT / "data/eval/golden_questions.jsonl")

    for qid in TARGET_IDS:
        row = questions.get(qid)
        if not row:
            print(f"{qid}: missing from golden questions")
            continue
        result = answer_question(
            str(row.get("question", "") or ""),
            eval_mode="policy_vs_control",
            eval_intent=str(row.get("intent", "") or ""),
            expected=row.get("expected") or {},
            top_k=10,
        )
        policy_doc_ids = [
            str((chunk.get("doc_id") or "")).strip()
            for chunk in (result.get("retrieved_chunks") or [])
            if (chunk.get("source_type") in {"policy_pdf", "policy_md"})
        ]
        debug = result.get("debug") if isinstance(result, dict) else None
        print(f"\n{qid} predicted_coverage={result.get('predicted_coverage')}")
        print(f"debug={debug}")
        print(f"top_policy_doc_ids={policy_doc_ids[:5]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
