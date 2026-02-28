#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.eval.scoring import score_abstention, score_context_precision, score_coverage_accuracy
from app.rag.answer import answer_question


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_expected_control_id(expected: Dict[str, Any]) -> Optional[str]:
    control_ids = expected.get("expected_control_ids")
    if isinstance(control_ids, list) and control_ids:
        first = control_ids[0]
        if first is not None:
            return str(first)
    return None


def _try_load_assess_control() -> Tuple[Optional[Callable[..., Any]], Optional[str]]:
    try:
        from app.assess.engine import assess_control  # type: ignore

        if callable(assess_control):
            return assess_control, None
        return None, "app.assess.engine.assess_control is not callable"
    except Exception as exc:
        return None, str(exc)


def _invoke_assess_control(
    assess_control_fn: Callable[..., Any],
    question: str,
    row: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    expected = row.get("expected") or {}
    first_control_id = _first_expected_control_id(expected)

    candidate_kwargs: Dict[str, Any] = {
        "question": question,
        "query": question,
        "control_id": first_control_id,
        "control_ids": expected.get("expected_control_ids"),
        "expected": expected,
        "mode": row.get("mode"),
        "intent": row.get("intent"),
        "top_k": top_k,
    }

    signature = inspect.signature(assess_control_fn)
    params = signature.parameters
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kwargs:
        kwargs = {k: v for k, v in candidate_kwargs.items() if v is not None}
    else:
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in params and v is not None}

    if kwargs:
        try:
            output = assess_control_fn(**kwargs)
            if isinstance(output, dict):
                return output
            return {"raw_output": output}
        except TypeError:
            pass

    positional_attempts: List[Tuple[Any, ...]] = []
    if first_control_id:
        positional_attempts.append((question, first_control_id))
    positional_attempts.append((question,))
    if first_control_id:
        positional_attempts.append((first_control_id,))

    for args in positional_attempts:
        try:
            output = assess_control_fn(*args)
            if isinstance(output, dict):
                return output
            return {"raw_output": output}
        except TypeError:
            continue

    output = assess_control_fn(question)
    if isinstance(output, dict):
        return output
    return {"raw_output": output}


def _extract_citations(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    citations = response.get("citations")
    if isinstance(citations, list):
        return [c for c in citations if isinstance(c, dict)]

    evidence = response.get("evidence")
    if isinstance(evidence, list):
        return [e for e in evidence if isinstance(e, dict)]

    return []


def _extract_predicted_coverage(
    response: Dict[str, Any],
    expected_coverage: Any,
) -> Optional[Any]:
    for key in ("predicted_coverage", "coverage", "coverage_label", "label"):
        if response.get(key) is not None:
            return response.get(key)

    for parent_key in ("assessment", "classification", "result"):
        parent = response.get(parent_key)
        if isinstance(parent, dict):
            for key in ("predicted_coverage", "coverage", "coverage_label", "label"):
                if parent.get(key) is not None:
                    return parent.get(key)

    # If expected coverage is null, keep this null to avoid penalizing non-coverage tasks.
    if expected_coverage is None:
        return None

    # For coverage-bearing tasks, allow deterministic label extraction from free text.
    for key in ("draft_answer", "answer", "final_answer", "raw_output"):
        if response.get(key):
            return response.get(key)

    return None


def _extract_abstained_flag(response: Dict[str, Any], predicted_coverage: Any) -> bool:
    for key in ("abstained", "did_abstain", "abstain"):
        if key in response:
            return bool(response.get(key))

    if isinstance(predicted_coverage, str) and predicted_coverage.strip().lower() == "abstain":
        return True

    text = " ".join(
        str(response.get(k, "") or "")
        for k in ("draft_answer", "answer", "final_answer", "raw_output")
    ).lower()
    return (
        "insufficient evidence" in text
        or "out of scope" in text
        or "don't have sufficient evidence" in text
    )


def _pick_engine_for_row(
    requested_engine: str,
    row_mode: str,
    assess_control_fn: Optional[Callable[..., Any]],
) -> str:
    if requested_engine == "answer":
        return "answer_question"
    if requested_engine == "assess":
        return "assess_control"
    # auto
    if row_mode == "policy_vs_control" and assess_control_fn is not None:
        return "assess_control"
    return "answer_question"


def _run_one(
    row: Dict[str, Any],
    requested_engine: str,
    assess_control_fn: Optional[Callable[..., Any]],
    top_k: int,
) -> Dict[str, Any]:
    qid = row.get("id")
    mode = str(row.get("mode", "") or "")
    intent = str(row.get("intent", "") or "")
    question = str(row.get("question", "") or "")
    expected = row.get("expected") or {}
    expected_coverage = expected.get("expected_coverage")

    engine_used = _pick_engine_for_row(requested_engine, mode, assess_control_fn)

    error: Optional[str] = None
    response: Dict[str, Any]
    try:
        if engine_used == "assess_control":
            if assess_control_fn is None:
                raise RuntimeError("assess_control requested but app.assess.engine is unavailable")
            response = _invoke_assess_control(assess_control_fn, question, row, top_k=top_k)
        else:
            response = answer_question(question, top_k=top_k, scope={})
    except Exception as exc:
        error = str(exc)
        response = {
            "draft_answer": f"insufficient evidence. Evaluation execution error: {exc}",
            "abstained": True,
            "citations": [],
        }

    citations = _extract_citations(response)
    predicted_coverage = _extract_predicted_coverage(response, expected_coverage=expected_coverage)
    abstained = _extract_abstained_flag(response, predicted_coverage=predicted_coverage)

    context_precision = score_context_precision(expected, citations, intent)
    coverage_accuracy = score_coverage_accuracy(expected_coverage, predicted_coverage)
    abstention_score = score_abstention(intent, expected_coverage, abstained, citations)
    overall = round((context_precision + coverage_accuracy + abstention_score) / 3.0, 4)

    answer_text = (
        response.get("draft_answer")
        or response.get("answer")
        or response.get("final_answer")
        or response.get("raw_output")
        or ""
    )

    return {
        "id": qid,
        "mode": mode,
        "intent": intent,
        "question": question,
        "engine_used": engine_used,
        "expected": expected,
        "prediction": {
            "abstained": abstained,
            "predicted_coverage": predicted_coverage,
            "citation_count": len(citations),
            "citations": citations,
            "answer_preview": str(answer_text)[:400],
        },
        "scores": {
            "context_precision": float(context_precision),
            "coverage_accuracy": float(coverage_accuracy),
            "abstention": float(abstention_score),
            "overall": float(overall),
        },
        "error": error,
    }


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 4)


def _build_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    score_rows = [r.get("scores", {}) or {} for r in results]
    mode_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        mode_buckets[str(r.get("mode", "") or "")].append(r)

    by_mode: Dict[str, Dict[str, Any]] = {}
    for mode, rows in mode_buckets.items():
        by_mode[mode] = {
            "n": len(rows),
            "avg_context_precision": _mean((x.get("scores", {}).get("context_precision", 0.0) for x in rows)),
            "avg_coverage_accuracy": _mean((x.get("scores", {}).get("coverage_accuracy", 0.0) for x in rows)),
            "avg_abstention": _mean((x.get("scores", {}).get("abstention", 0.0) for x in rows)),
            "avg_overall": _mean((x.get("scores", {}).get("overall", 0.0) for x in rows)),
        }

    engine_counts = Counter(str(r.get("engine_used", "unknown")) for r in results)
    error_count = sum(1 for r in results if r.get("error"))
    abstained_count = sum(1 for r in results if bool((r.get("prediction") or {}).get("abstained")))

    return {
        "total_questions": total,
        "error_count": error_count,
        "abstained_count": abstained_count,
        "engine_counts": dict(engine_counts),
        "avg_context_precision": _mean((s.get("context_precision", 0.0) for s in score_rows)),
        "avg_coverage_accuracy": _mean((s.get("coverage_accuracy", 0.0) for s in score_rows)),
        "avg_abstention": _mean((s.get("abstention", 0.0) for s in score_rows)),
        "avg_overall": _mean((s.get("overall", 0.0) for s in score_rows)),
        "by_mode": by_mode,
    }


def _render_summary_markdown(
    summary: Dict[str, Any],
    *,
    input_path: Path,
    output_jsonl: Path,
    requested_engine: str,
    assess_available: bool,
    assess_error: Optional[str],
) -> str:
    lines: List[str] = []
    lines.append("# Week 5 Baseline Evaluation Summary")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Results JSONL: `{output_jsonl}`")
    lines.append(f"- Requested engine mode: `{requested_engine}`")
    lines.append(f"- `assess_control` available: `{assess_available}`")
    if assess_error:
        lines.append(f"- `assess_control` import note: `{assess_error}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Questions: {summary['total_questions']}")
    lines.append(f"- Errors: {summary['error_count']}")
    lines.append(f"- Abstained: {summary['abstained_count']}")
    lines.append(f"- Engine usage: {summary['engine_counts']}")
    lines.append(f"- Avg context precision: {summary['avg_context_precision']:.4f}")
    lines.append(f"- Avg coverage accuracy: {summary['avg_coverage_accuracy']:.4f}")
    lines.append(f"- Avg abstention score: {summary['avg_abstention']:.4f}")
    lines.append(f"- Avg overall score: {summary['avg_overall']:.4f}")
    lines.append("")
    lines.append("## By Mode")
    lines.append("")
    lines.append("| mode | n | avg_context_precision | avg_coverage_accuracy | avg_abstention | avg_overall |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode in sorted(summary.get("by_mode", {}).keys()):
        row = summary["by_mode"][mode]
        lines.append(
            "| "
            + f"{mode} | {row['n']} | {row['avg_context_precision']:.4f} | "
            + f"{row['avg_coverage_accuracy']:.4f} | {row['avg_abstention']:.4f} | {row['avg_overall']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 5 baseline evaluation over golden questions.")
    parser.add_argument("--input-path", default="data/eval/golden_questions.jsonl")
    parser.add_argument("--output-jsonl", default="data/eval/week5_baseline_results.jsonl")
    parser.add_argument("--summary-path", default="data/eval/week5_baseline_summary.md")
    parser.add_argument("--engine", choices=["auto", "answer", "assess"], default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of questions to run.")
    args = parser.parse_args()

    input_path = REPO_ROOT / args.input_path
    output_jsonl = REPO_ROOT / args.output_jsonl
    summary_path = REPO_ROOT / args.summary_path

    rows = _load_jsonl(input_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    assess_control_fn, assess_error = _try_load_assess_control()
    assess_available = assess_control_fn is not None
    if args.engine == "assess" and not assess_available:
        raise RuntimeError(
            "Engine mode 'assess' requested, but app.assess.engine.assess_control is unavailable: "
            f"{assess_error}"
        )

    results: List[Dict[str, Any]] = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        result = _run_one(
            row,
            requested_engine=args.engine,
            assess_control_fn=assess_control_fn,
            top_k=args.top_k,
        )
        results.append(result)
        print(
            f"[{idx}/{total}] {result.get('id', '?')} mode={result.get('mode')} "
            f"engine={result.get('engine_used')} overall={result.get('scores', {}).get('overall', 0.0):.4f}"
        )

    _write_jsonl(output_jsonl, results)

    summary = _build_summary(results)
    summary_md = _render_summary_markdown(
        summary,
        input_path=input_path,
        output_jsonl=output_jsonl,
        requested_engine=args.engine,
        assess_available=assess_available,
        assess_error=assess_error,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_md, encoding="utf-8")

    print("")
    print(f"Wrote results: {output_jsonl}")
    print(f"Wrote summary: {summary_path}")
    print("")
    print("=== Week 5 Baseline Summary ===")
    print(f"questions={summary['total_questions']} errors={summary['error_count']} abstained={summary['abstained_count']}")
    print(
        "avg_context_precision={:.4f} avg_coverage_accuracy={:.4f} avg_abstention={:.4f} avg_overall={:.4f}".format(
            summary["avg_context_precision"],
            summary["avg_coverage_accuracy"],
            summary["avg_abstention"],
            summary["avg_overall"],
        )
    )
    print(f"engine_counts={summary['engine_counts']}")
    for mode in sorted(summary.get("by_mode", {}).keys()):
        row = summary["by_mode"][mode]
        print(
            f"mode={mode} n={row['n']} context={row['avg_context_precision']:.4f} "
            f"coverage={row['avg_coverage_accuracy']:.4f} abstention={row['avg_abstention']:.4f} "
            f"overall={row['avg_overall']:.4f}"
        )


if __name__ == "__main__":
    main()
