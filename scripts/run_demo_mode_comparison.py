#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag.answer import answer_question
from app.rag.answer_state import derive_answer_view_state


INLINE_CITATION_RE = re.compile(r"\[C\d+\]")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 1)


def _median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return round(float(statistics.median(vals)), 1)


@contextmanager
def _temporary_llm_env(*, backend: str, model: Optional[str] = None) -> Iterator[None]:
    names = ["LLM_BACKEND", "OPENROUTER_MODEL"]
    saved = {name: os.environ.get(name) for name in names}
    os.environ["LLM_BACKEND"] = backend
    if model is not None:
        os.environ["OPENROUTER_MODEL"] = model
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _risk_flags(result: Dict[str, Any], state: str, answer_preview: str) -> List[str]:
    flags: List[str] = []
    citations = list(result.get("citations") or [])
    inline_citations = bool(INLINE_CITATION_RE.search(answer_preview))
    abstained = bool(result.get("abstained"))
    weak_retrieval = bool(result.get("weak_retrieval"))
    llm_status = str(result.get("llm_status") or "").strip().lower()

    if not abstained and not citations:
        flags.append("non_abstained_without_citations")
    if llm_status == "ok" and not inline_citations:
        flags.append("llm_answer_missing_inline_citations")
    if weak_retrieval and not abstained:
        flags.append("weak_evidence_non_abstained")
    if state == "conflicting_evidence" and not abstained:
        flags.append("conflict_not_abstained")
    if state == "backend_error":
        flags.append("backend_failure_visible")
    return flags


def _perceived_quality(state: str, result: Dict[str, Any], risk_flags: Sequence[str]) -> Dict[str, Any]:
    abstained = bool(result.get("abstained"))
    if state == "strong_answer" and not risk_flags:
        return {"score": 4, "label": "high"}
    if state in {"partial_answer", "retrieval_only"} and len(risk_flags) <= 1:
        return {"score": 3, "label": "medium"}
    if state in {"no_evidence", "conflicting_evidence"} and abstained:
        return {"score": 3, "label": "medium"}
    if state == "backend_error":
        return {"score": 1, "label": "poor"}
    return {"score": 2, "label": "low"}


def _run_mode(
    row: Dict[str, Any],
    *,
    backend: str,
    top_k: int,
    openrouter_model: Optional[str] = None,
) -> Dict[str, Any]:
    question = str(row.get("question", "") or "")
    eval_mode = str(row.get("mode", "") or "").strip() or None
    eval_intent = str(row.get("intent", "") or "").strip() or None
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else None
    started = time.monotonic()
    with _temporary_llm_env(backend=backend, model=openrouter_model if backend == "openrouter" else None):
        result = answer_question(
            question,
            top_k=top_k,
            eval_mode=eval_mode,
            eval_intent=eval_intent,
            expected=expected,
        )
    latency_ms = int(round((time.monotonic() - started) * 1000))
    view = derive_answer_view_state(result)
    answer_preview = str(result.get("draft_answer", "") or "").strip().replace("\n", " ")
    answer_preview = answer_preview[:280] + ("..." if len(answer_preview) > 280 else "")
    risk_flags = _risk_flags(result, view.state, answer_preview)
    quality = _perceived_quality(view.state, result, risk_flags)

    return {
        "backend": backend,
        "state": view.state,
        "state_title": view.title,
        "quality_label": quality["label"],
        "quality_score": quality["score"],
        "abstained": bool(result.get("abstained")),
        "citation_count": len(list(result.get("citations") or [])),
        "retrieved_chunk_count": len(list(result.get("retrieved_chunks") or [])),
        "weak_retrieval": bool(result.get("weak_retrieval")),
        "latency_ms": latency_ms,
        "llm_backend": result.get("llm_backend"),
        "llm_mode": result.get("llm_mode"),
        "llm_status": result.get("llm_status"),
        "llm_error_type": result.get("llm_error_type"),
        "llm_requested_model": result.get("llm_requested_model"),
        "llm_used_model": result.get("llm_used_model"),
        "llm_fallback_triggered": bool(result.get("llm_fallback_triggered")),
        "llm_retries": int(result.get("llm_retries") or 0),
        "llm_latency_ms": result.get("llm_latency_ms"),
        "risk_flags": list(risk_flags),
        "answer_preview": answer_preview,
    }


def _compare_modes(retrieval_only: Dict[str, Any], openrouter: Dict[str, Any]) -> str:
    if openrouter.get("skipped"):
        return "OpenRouter comparison skipped because the backend was not configured."
    retrieval_score = int(retrieval_only.get("quality_score") or 0)
    openrouter_score = int(openrouter.get("quality_score") or 0)
    retrieval_risks = len(list(retrieval_only.get("risk_flags") or []))
    openrouter_risks = len(list(openrouter.get("risk_flags") or []))
    retrieval_latency = int(retrieval_only.get("latency_ms") or 0)
    openrouter_latency = int(openrouter.get("latency_ms") or 0)

    if openrouter_score > retrieval_score and openrouter_risks <= retrieval_risks:
        return "OpenRouter improved the explanation while keeping demo risk at or below retrieval-only mode."
    if openrouter_score < retrieval_score or openrouter_risks > retrieval_risks:
        return "OpenRouter introduced more demo risk than retrieval-only mode for this question."
    if retrieval_score == openrouter_score and openrouter_latency > max(500, retrieval_latency * 2):
        return "OpenRouter added noticeable latency without a clear quality gain."
    return "OpenRouter and retrieval-only mode performed similarly for this question."


def _summarize_mode(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("skipped")]
    if not valid_rows:
        return {
            "n": 0,
            "avg_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "abstained_count": 0,
            "risk_case_count": 0,
            "states": {},
            "qualities": {},
        }

    return {
        "n": len(valid_rows),
        "avg_latency_ms": _mean(row.get("latency_ms", 0.0) for row in valid_rows),
        "median_latency_ms": _median(row.get("latency_ms", 0.0) for row in valid_rows),
        "abstained_count": sum(1 for row in valid_rows if row.get("abstained")),
        "risk_case_count": sum(1 for row in valid_rows if row.get("risk_flags")),
        "states": dict(Counter(str(row.get("state") or "unknown") for row in valid_rows)),
        "qualities": dict(Counter(str(row.get("quality_label") or "unknown") for row in valid_rows)),
    }


def _render_markdown(
    *,
    questions_path: Path,
    output_json: Path,
    retrieval_summary: Dict[str, Any],
    openrouter_summary: Dict[str, Any],
    comparisons: Sequence[Dict[str, Any]],
    openrouter_requested_model: Optional[str],
    openrouter_enabled: bool,
) -> str:
    lines: List[str] = []
    lines.append("# Demo Mode Comparison")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Input questions: `{questions_path}`")
    lines.append(f"- JSON output: `{output_json}`")
    lines.append(f"- OpenRouter configured: `{openrouter_enabled}`")
    lines.append(f"- OpenRouter requested model: `{openrouter_requested_model or '(not set)'}`")
    lines.append("")
    lines.append("## Mode Summary")
    lines.append("")
    lines.append("| mode | n | avg latency ms | median latency ms | abstained | risky cases | states | quality labels |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for label, summary in (("retrieval_only", retrieval_summary), ("openrouter", openrouter_summary)):
        lines.append(
            "| "
            + f"{label} | {summary['n']} | {summary['avg_latency_ms']:.1f} | {summary['median_latency_ms']:.1f} | "
            + f"{summary['abstained_count']} | {summary['risk_case_count']} | {summary['states']} | {summary['qualities']} |"
        )
    lines.append("")
    lines.append("## Per Question")
    lines.append("")
    lines.append("| id | retrieval-only | OpenRouter | comparison note |")
    lines.append("|---|---|---|---|")
    for row in comparisons:
        retrieval = row["retrieval_only"]
        openrouter = row["openrouter"]
        openrouter_cell = (
            "skipped"
            if openrouter.get("skipped")
            else f"{openrouter['state']} / quality={openrouter['quality_label']} / latency={openrouter['latency_ms']}ms"
        )
        lines.append(
            "| "
            + f"{row['id']} | {retrieval['state']} / quality={retrieval['quality_label']} / latency={retrieval['latency_ms']}ms | "
            + f"{openrouter_cell} | {row['comparison_note']} |"
        )
    lines.append("")
    for row in comparisons:
        lines.append(f"### {row['id']}: {row['question']}")
        goal = str(row.get("demo_goal") or "").strip()
        if goal:
            lines.append(f"- Demo goal: {goal}")
        lines.append(
            "- Retrieval-only: "
            + f"state=`{row['retrieval_only']['state']}` quality=`{row['retrieval_only']['quality_label']}` "
            + f"abstained=`{row['retrieval_only']['abstained']}` latency=`{row['retrieval_only']['latency_ms']} ms` "
            + f"citations=`{row['retrieval_only']['citation_count']}` risks=`{row['retrieval_only']['risk_flags']}`"
        )
        lines.append(f"  Preview: {row['retrieval_only']['answer_preview']}")
        if row["openrouter"].get("skipped"):
            lines.append("- OpenRouter: skipped because the backend was not configured.")
        else:
            lines.append(
                "- OpenRouter: "
                + f"state=`{row['openrouter']['state']}` quality=`{row['openrouter']['quality_label']}` "
                + f"abstained=`{row['openrouter']['abstained']}` latency=`{row['openrouter']['latency_ms']} ms` "
                + f"citations=`{row['openrouter']['citation_count']}` risks=`{row['openrouter']['risk_flags']}` "
                + f"llm_status=`{row['openrouter']['llm_status']}` fallback=`{row['openrouter']['llm_fallback_triggered']}`"
            )
            lines.append(f"  Preview: {row['openrouter']['answer_preview']}")
        lines.append(f"- Comparison note: {row['comparison_note']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval-only and OpenRouter-backed demo behavior.")
    parser.add_argument("--input-path", default="data/eval/demo_questions.jsonl")
    parser.add_argument("--output-json", default="data/eval/demo_mode_comparison.json")
    parser.add_argument("--output-md", default="data/eval/demo_mode_comparison.md")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--openrouter-model", default=os.getenv("OPENROUTER_MODEL", "").strip() or None)
    parser.add_argument("--skip-openrouter", action="store_true")
    args = parser.parse_args()

    questions_path = REPO_ROOT / args.input_path
    output_json = REPO_ROOT / args.output_json
    output_md = REPO_ROOT / args.output_md

    questions = _load_jsonl(questions_path)
    openrouter_enabled = bool(os.getenv("OPENROUTER_API_KEY", "").strip()) and not args.skip_openrouter

    comparisons: List[Dict[str, Any]] = []
    retrieval_rows: List[Dict[str, Any]] = []
    openrouter_rows: List[Dict[str, Any]] = []

    for index, row in enumerate(questions, start=1):
        retrieval_only = _run_mode(row, backend="none", top_k=args.top_k)
        retrieval_rows.append(retrieval_only)

        if openrouter_enabled:
            openrouter_row = _run_mode(
                row,
                backend="openrouter",
                top_k=args.top_k,
                openrouter_model=args.openrouter_model,
            )
        else:
            openrouter_row = {
                "backend": "openrouter",
                "skipped": True,
                "quality_label": "skipped",
                "quality_score": 0,
                "latency_ms": 0,
                "state": "skipped",
                "risk_flags": [],
            }
        openrouter_rows.append(openrouter_row)

        comparison = {
            "id": row.get("id") or f"demo_{index:02d}",
            "question": row.get("question"),
            "demo_goal": row.get("demo_goal"),
            "retrieval_only": retrieval_only,
            "openrouter": openrouter_row,
            "comparison_note": _compare_modes(retrieval_only, openrouter_row),
        }
        comparisons.append(comparison)
        print(
            f"[{index}/{len(questions)}] {comparison['id']} "
            f"retrieval_only={retrieval_only['state']} openrouter={openrouter_row.get('state')}"
        )

    retrieval_summary = _summarize_mode(retrieval_rows)
    openrouter_summary = _summarize_mode(openrouter_rows)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(questions_path),
        "openrouter_enabled": openrouter_enabled,
        "openrouter_requested_model": args.openrouter_model,
        "retrieval_only_summary": retrieval_summary,
        "openrouter_summary": openrouter_summary,
        "comparisons": comparisons,
    }
    _write_json(output_json, payload)

    markdown = _render_markdown(
        questions_path=questions_path,
        output_json=output_json,
        retrieval_summary=retrieval_summary,
        openrouter_summary=openrouter_summary,
        comparisons=comparisons,
        openrouter_requested_model=args.openrouter_model,
        openrouter_enabled=openrouter_enabled,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    print("")
    print(f"Wrote JSON comparison: {output_json}")
    print(f"Wrote Markdown comparison: {output_md}")
    print("")
    print("=== Demo Mode Summary ===")
    print(f"retrieval_only={retrieval_summary}")
    print(f"openrouter={openrouter_summary}")


if __name__ == "__main__":
    main()
