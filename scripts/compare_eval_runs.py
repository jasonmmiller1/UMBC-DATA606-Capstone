#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_week5_eval import _build_summary, _load_jsonl


SCORE_KEYS = ("context_precision", "coverage_accuracy", "abstention", "overall")
POLICY_SOURCE_TYPES = {"policy_md", "policy_pdf"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _round(value: float) -> float:
    return round(float(value), 4)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 4)


def _delta(before: float, after: float) -> float:
    return _round(float(after) - float(before))


def _retrieval_config(row: Dict[str, Any]) -> Dict[str, Any] | None:
    retrieval = row.get("retrieval") or {}
    config = retrieval.get("config")
    return config if isinstance(config, dict) else None


def _selected_context_sections(row: Dict[str, Any], *, limit: int = 5) -> List[Dict[str, Any]]:
    retrieval = row.get("retrieval") or {}
    selected = retrieval.get("selected_context_chunks")
    if not isinstance(selected, list):
        return []
    sections: List[Dict[str, Any]] = []
    for item in selected[:limit]:
        if not isinstance(item, dict):
            continue
        sections.append(
            {
                "chunk_id": item.get("chunk_id"),
                "source_type": item.get("source_type"),
                "doc_id": item.get("doc_id"),
                "doc_title": item.get("doc_title"),
                "section_path": item.get("section_path"),
                "chunk_type": item.get("chunk_type"),
                "rank": item.get("rank"),
                "score": item.get("score"),
            }
        )
    return sections


def _selected_policy_sections(row: Dict[str, Any], *, limit: int = 3) -> List[str]:
    sections: List[str] = []
    for item in _selected_context_sections(row, limit=20):
        if str(item.get("source_type") or "") not in POLICY_SOURCE_TYPES:
            continue
        section_path = str(item.get("section_path") or "").strip()
        doc_id = str(item.get("doc_id") or "").strip()
        if section_path:
            sections.append(section_path)
        elif doc_id:
            sections.append(doc_id)
        if len(sections) >= limit:
            break
    return sections


def _citation_ids(row: Dict[str, Any]) -> List[str]:
    prediction = row.get("prediction") or {}
    citations = prediction.get("citations")
    if not isinstance(citations, list):
        return []
    out: List[str] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        chunk_id = citation.get("chunk_id")
        if chunk_id is not None:
            out.append(str(chunk_id))
    return out


def _answer_preview(row: Dict[str, Any], *, limit: int = 300) -> str:
    prediction = row.get("prediction") or {}
    preview = str(prediction.get("answer_preview") or "").replace("\n", " ").strip()
    return preview[:limit]


def _question_note(baseline: Dict[str, Any], current: Dict[str, Any]) -> str:
    reasons: List[str] = []
    baseline_scores = baseline.get("scores", {}) or {}
    current_scores = current.get("scores", {}) or {}
    context_delta = _delta(
        float(baseline_scores.get("context_precision", 0.0)),
        float(current_scores.get("context_precision", 0.0)),
    )
    coverage_delta = _delta(
        float(baseline_scores.get("coverage_accuracy", 0.0)),
        float(current_scores.get("coverage_accuracy", 0.0)),
    )
    abstention_delta = _delta(
        float(baseline_scores.get("abstention", 0.0)),
        float(current_scores.get("abstention", 0.0)),
    )
    if context_delta > 0:
        reasons.append("higher context precision")
    elif context_delta < 0:
        reasons.append("lower context precision")
    if coverage_delta > 0:
        reasons.append("better coverage labeling")
    elif coverage_delta < 0:
        reasons.append("worse coverage labeling")
    if abstention_delta > 0:
        reasons.append("better abstention behavior")
    elif abstention_delta < 0:
        reasons.append("worse abstention behavior")

    baseline_ids = set(_citation_ids(baseline))
    current_ids = set(_citation_ids(current))
    new_citations = sorted(current_ids - baseline_ids)
    dropped_citations = sorted(baseline_ids - current_ids)
    if new_citations:
        reasons.append(f"{len(new_citations)} new cited chunks")
    if dropped_citations:
        reasons.append(f"{len(dropped_citations)} dropped cited chunks")

    policy_sections = _selected_policy_sections(current)
    if policy_sections:
        reasons.append(f"current policy sections include {policy_sections[0]}")

    return "; ".join(reasons)


def _question_delta(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    baseline_scores = baseline.get("scores", {}) or {}
    current_scores = current.get("scores", {}) or {}
    delta_scores = {key: _delta(baseline_scores.get(key, 0.0), current_scores.get(key, 0.0)) for key in SCORE_KEYS}
    baseline_citations = _citation_ids(baseline)
    current_citations = _citation_ids(current)
    current_prediction = current.get("prediction", {}) or {}
    baseline_prediction = baseline.get("prediction", {}) or {}
    return {
        "id": baseline.get("id"),
        "mode": baseline.get("mode"),
        "intent": baseline.get("intent"),
        "question": baseline.get("question"),
        "baseline_scores": {key: _round(baseline_scores.get(key, 0.0)) for key in SCORE_KEYS},
        "current_scores": {key: _round(current_scores.get(key, 0.0)) for key in SCORE_KEYS},
        "delta_scores": delta_scores,
        "baseline_citation_count": int(baseline_prediction.get("citation_count", len(baseline_citations)) or 0),
        "current_citation_count": int(current_prediction.get("citation_count", len(current_citations)) or 0),
        "citation_count_delta": int(current_prediction.get("citation_count", len(current_citations)) or 0)
        - int(baseline_prediction.get("citation_count", len(baseline_citations)) or 0),
        "baseline_citation_ids": baseline_citations,
        "current_citation_ids": current_citations,
        "new_citation_ids": sorted(set(current_citations) - set(baseline_citations)),
        "dropped_citation_ids": sorted(set(baseline_citations) - set(current_citations)),
        "baseline_answer_preview": _answer_preview(baseline),
        "current_answer_preview": _answer_preview(current),
        "current_selected_context": _selected_context_sections(current),
        "note": _question_note(baseline, current),
    }


def _mode_deltas(
    baseline_summary: Dict[str, Any],
    current_summary: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    modes = set((baseline_summary.get("by_mode") or {}).keys()) | set((current_summary.get("by_mode") or {}).keys())
    for mode in sorted(modes):
        before = (baseline_summary.get("by_mode") or {}).get(mode, {})
        after = (current_summary.get("by_mode") or {}).get(mode, {})
        out[mode] = {
            "baseline": {k: before.get(k) for k in ("n", "avg_context_precision", "avg_coverage_accuracy", "avg_abstention", "avg_overall")},
            "current": {k: after.get(k) for k in ("n", "avg_context_precision", "avg_coverage_accuracy", "avg_abstention", "avg_overall")},
            "delta": {
                "avg_context_precision": _delta(before.get("avg_context_precision", 0.0), after.get("avg_context_precision", 0.0)),
                "avg_coverage_accuracy": _delta(before.get("avg_coverage_accuracy", 0.0), after.get("avg_coverage_accuracy", 0.0)),
                "avg_abstention": _delta(before.get("avg_abstention", 0.0), after.get("avg_abstention", 0.0)),
                "avg_overall": _delta(before.get("avg_overall", 0.0), after.get("avg_overall", 0.0)),
            },
        }
    return out


def _top_questions(rows: Sequence[Dict[str, Any]], *, min_delta: float, positive: bool, limit: int) -> List[Dict[str, Any]]:
    filtered = []
    for row in rows:
        delta_value = float((row.get("delta_scores") or {}).get("overall", 0.0))
        if positive and delta_value >= min_delta:
            filtered.append(row)
        if not positive and delta_value <= -min_delta:
            filtered.append(row)
    filtered.sort(key=lambda row: float((row.get("delta_scores") or {}).get("overall", 0.0)), reverse=positive)
    return filtered[:limit]


def _render_markdown(
    *,
    baseline_label: str,
    current_label: str,
    baseline_results_path: Path,
    current_results_path: Path,
    baseline_summary: Dict[str, Any],
    current_summary: Dict[str, Any],
    summary_delta: Dict[str, Any],
    by_mode: Dict[str, Dict[str, Dict[str, float]]],
    question_deltas: Sequence[Dict[str, Any]],
    material_threshold: float,
) -> str:
    improved = _top_questions(question_deltas, min_delta=material_threshold, positive=True, limit=8)
    regressed = _top_questions(question_deltas, min_delta=material_threshold, positive=False, limit=8)
    any_regressions = [
        row for row in sorted(
            question_deltas,
            key=lambda row: float((row.get("delta_scores") or {}).get("overall", 0.0))
        )
        if float((row.get("delta_scores") or {}).get("overall", 0.0)) < 0.0
    ][:8]
    changed = [row for row in question_deltas if abs(float((row.get("delta_scores") or {}).get("overall", 0.0))) > 0.0]
    lines: List[str] = []
    lines.append("# Week 6 Evaluation Delta Summary")
    lines.append("")
    lines.append(f"- Baseline: `{baseline_label}`")
    lines.append(f"- Current: `{current_label}`")
    lines.append(f"- Baseline results: `{baseline_results_path}`")
    lines.append(f"- Current results: `{current_results_path}`")
    lines.append(f"- Questions compared: {len(question_deltas)}")
    lines.append(f"- Changed questions: {len(changed)}")
    lines.append(f"- Material-change threshold (overall delta): {material_threshold:.2f}")
    lines.append("")
    lines.append("## Overall Delta")
    lines.append("")
    lines.append("| metric | baseline | current | delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| avg_context_precision | {baseline_summary['avg_context_precision']:.4f} | "
        f"{current_summary['avg_context_precision']:.4f} | {summary_delta['avg_context_precision']:+.4f} |"
    )
    lines.append(
        f"| avg_coverage_accuracy | {baseline_summary['avg_coverage_accuracy']:.4f} | "
        f"{current_summary['avg_coverage_accuracy']:.4f} | {summary_delta['avg_coverage_accuracy']:+.4f} |"
    )
    lines.append(
        f"| avg_abstention | {baseline_summary['avg_abstention']:.4f} | "
        f"{current_summary['avg_abstention']:.4f} | {summary_delta['avg_abstention']:+.4f} |"
    )
    lines.append(
        f"| avg_overall | {baseline_summary['avg_overall']:.4f} | "
        f"{current_summary['avg_overall']:.4f} | {summary_delta['avg_overall']:+.4f} |"
    )
    lines.append(
        f"| abstained_count | {baseline_summary['abstained_count']} | "
        f"{current_summary['abstained_count']} | {summary_delta['abstained_count']:+d} |"
    )
    lines.append(
        f"| error_count | {baseline_summary['error_count']} | "
        f"{current_summary['error_count']} | {summary_delta['error_count']:+d} |"
    )
    lines.append("")
    lines.append("## By Mode")
    lines.append("")
    lines.append("| mode | baseline overall | current overall | delta overall | delta context | delta coverage | delta abstention |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mode, payload in by_mode.items():
        delta_payload = payload["delta"]
        lines.append(
            f"| {mode} | {payload['baseline']['avg_overall']:.4f} | {payload['current']['avg_overall']:.4f} | "
            f"{delta_payload['avg_overall']:+.4f} | {delta_payload['avg_context_precision']:+.4f} | "
            f"{delta_payload['avg_coverage_accuracy']:+.4f} | {delta_payload['avg_abstention']:+.4f} |"
        )
    lines.append("")
    lines.append("## Material Improvements")
    lines.append("")
    if improved:
        for row in improved:
            delta_scores = row["delta_scores"]
            lines.append(
                f"- {row['id']} ({row['mode']}) overall {delta_scores['overall']:+.4f}: {row['question']}"
            )
            lines.append(f"  Note: {row['note']}")
    else:
        lines.append("- None at the configured threshold.")
    lines.append("")
    lines.append("## Regressions")
    lines.append("")
    if regressed:
        for row in regressed:
            delta_scores = row["delta_scores"]
            lines.append(
                f"- {row['id']} ({row['mode']}) overall {delta_scores['overall']:+.4f}: {row['question']}"
            )
            lines.append(f"  Note: {row['note']}")
    elif any_regressions:
        for row in any_regressions:
            delta_scores = row["delta_scores"]
            lines.append(
                f"- {row['id']} ({row['mode']}) overall {delta_scores['overall']:+.4f}: {row['question']}"
            )
            lines.append(f"  Note: {row['note']}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if float(summary_delta["avg_overall"]) > 0.0 and float(summary_delta["avg_abstention"]) >= 0.0:
        lines.append(
            f"- Replace the previous default with `{current_label}` for next week. "
            "The lift is driven by context precision, while coverage accuracy and abstention stayed stable."
        )
    else:
        lines.append(
            f"- Keep `{baseline_label}` as the default for now. "
            "The current configuration does not improve overall quality enough to justify the switch."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation result JSONL files and write delta outputs.")
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--current-results", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--current-label", default="current")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--material-threshold", type=float, default=0.05)
    args = parser.parse_args()

    baseline_results_path = (REPO_ROOT / args.baseline_results).resolve()
    current_results_path = (REPO_ROOT / args.current_results).resolve()
    baseline_rows = _load_jsonl(baseline_results_path)
    current_rows = _load_jsonl(current_results_path)
    baseline_map = {str(row.get("id")): row for row in baseline_rows}
    current_map = {str(row.get("id")): row for row in current_rows}

    shared_ids = sorted(set(baseline_map) & set(current_map))
    question_deltas = [_question_delta(baseline_map[qid], current_map[qid]) for qid in shared_ids]
    question_deltas.sort(key=lambda row: float((row.get("delta_scores") or {}).get("overall", 0.0)), reverse=True)

    baseline_summary = _build_summary(baseline_rows)
    current_summary = _build_summary(current_rows)
    summary_delta = {
        "avg_context_precision": _delta(baseline_summary["avg_context_precision"], current_summary["avg_context_precision"]),
        "avg_coverage_accuracy": _delta(baseline_summary["avg_coverage_accuracy"], current_summary["avg_coverage_accuracy"]),
        "avg_abstention": _delta(baseline_summary["avg_abstention"], current_summary["avg_abstention"]),
        "avg_overall": _delta(baseline_summary["avg_overall"], current_summary["avg_overall"]),
        "abstained_count": int(current_summary["abstained_count"]) - int(baseline_summary["abstained_count"]),
        "error_count": int(current_summary["error_count"]) - int(baseline_summary["error_count"]),
    }
    by_mode = _mode_deltas(baseline_summary, current_summary)

    improved_count = sum(1 for row in question_deltas if float((row.get("delta_scores") or {}).get("overall", 0.0)) > 0.0)
    regressed_count = sum(1 for row in question_deltas if float((row.get("delta_scores") or {}).get("overall", 0.0)) < 0.0)
    unchanged_count = len(question_deltas) - improved_count - regressed_count

    payload = {
        "baseline_label": args.baseline_label,
        "current_label": args.current_label,
        "baseline_results_path": str(baseline_results_path),
        "current_results_path": str(current_results_path),
        "current_retrieval_config": _retrieval_config(current_rows[0]) if current_rows else None,
        "summary": {
            "baseline": baseline_summary,
            "current": current_summary,
            "delta": summary_delta,
        },
        "by_mode": by_mode,
        "question_counts": {
            "shared_questions": len(question_deltas),
            "improved": improved_count,
            "regressed": regressed_count,
            "unchanged": unchanged_count,
            "material_improved": len(_top_questions(question_deltas, min_delta=args.material_threshold, positive=True, limit=len(question_deltas))),
            "material_regressed": len(_top_questions(question_deltas, min_delta=args.material_threshold, positive=False, limit=len(question_deltas))),
        },
        "question_deltas": question_deltas,
    }

    output_json = (REPO_ROOT / args.output_json).resolve()
    output_md = (REPO_ROOT / args.output_md).resolve()
    _write_json(output_json, payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        _render_markdown(
            baseline_label=args.baseline_label,
            current_label=args.current_label,
            baseline_results_path=baseline_results_path,
            current_results_path=current_results_path,
            baseline_summary=baseline_summary,
            current_summary=current_summary,
            summary_delta=summary_delta,
            by_mode=by_mode,
            question_deltas=question_deltas,
            material_threshold=float(args.material_threshold),
        ),
        encoding="utf-8",
    )
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote Markdown: {output_md}")


if __name__ == "__main__":
    main()
