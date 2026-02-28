#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag.answer import answer_question


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    input_path = REPO_ROOT / "data/tests/week3_chat_questions.jsonl"
    output_path = REPO_ROOT / "data/tests/week3_chat_outputs.jsonl"

    questions = _load_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    abstained_count = 0
    confidence_sum = 0.0

    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(questions, start=1):
            question = row.get("question", "")
            filters = row.get("filters", {}) or {}
            result = answer_question(
                question,
                scope={
                    "source_type": filters.get("source_type"),
                    "control_id": filters.get("control_id"),
                    "doc_id": filters.get("doc_id"),
                },
            )
            abstained = bool(result.get("abstained", True))
            confidence = float(result.get("confidence", 0.0) or 0.0)
            top_chunk_ids = [
                str(c.get("chunk_id"))
                for c in (result.get("retrieved_chunks", []) or [])[:5]
                if c.get("chunk_id")
            ]
            draft_answer = str(result.get("draft_answer", "") or "")
            record = {
                "question": question,
                "abstained": abstained,
                "confidence": confidence,
                "citations": result.get("citations", []),
                "top_chunk_ids": top_chunk_ids,
                "draft_answer_preview": draft_answer[:300],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(questions)}] processed: {question}")
            abstained_count += 1 if abstained else 0
            confidence_sum += confidence

    total = len(questions)
    avg_confidence = (confidence_sum / total) if total else 0.0
    print(f"Wrote: {output_path}")
    print(
        f"Summary: #questions={total}, #abstained={abstained_count}, "
        f"avg confidence={avg_confidence:.3f}"
    )


if __name__ == "__main__":
    main()
