#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_week5_eval import _build_summary, _load_jsonl


DEFAULT_MATRIX: List[Dict[str, Any]] = [
    {
        "name": "equal_no_rerank",
        "top_k": 10,
        "dense_k": 20,
        "bm25_k": 20,
        "rrf_k": 60,
        "dense_weight": 1.0,
        "bm25_weight": 1.0,
        "final_context_k": 8,
        "rerank_enabled": False,
        "rerank_candidates": 20,
    },
    {
        "name": "equal_rerank20",
        "top_k": 10,
        "dense_k": 20,
        "bm25_k": 20,
        "rrf_k": 60,
        "dense_weight": 1.0,
        "bm25_weight": 1.0,
        "final_context_k": 8,
        "rerank_enabled": True,
        "rerank_candidates": 20,
    },
    {
        "name": "bm25_lean_rerank20",
        "top_k": 10,
        "dense_k": 20,
        "bm25_k": 20,
        "rrf_k": 60,
        "dense_weight": 0.9,
        "bm25_weight": 1.1,
        "final_context_k": 8,
        "rerank_enabled": True,
        "rerank_candidates": 20,
    },
    {
        "name": "dense_lean_rerank20",
        "top_k": 10,
        "dense_k": 20,
        "bm25_k": 20,
        "rrf_k": 60,
        "dense_weight": 1.1,
        "bm25_weight": 0.9,
        "final_context_k": 8,
        "rerank_enabled": True,
        "rerank_candidates": 20,
    },
    {
        "name": "deeper_rerank30_ctx10",
        "top_k": 10,
        "dense_k": 30,
        "bm25_k": 30,
        "rrf_k": 60,
        "dense_weight": 1.0,
        "bm25_weight": 1.0,
        "final_context_k": 10,
        "rerank_enabled": True,
        "rerank_candidates": 30,
    },
    {
        "name": "shallow_no_rerank_ctx6",
        "top_k": 8,
        "dense_k": 12,
        "bm25_k": 12,
        "rrf_k": 60,
        "dense_weight": 1.0,
        "bm25_weight": 1.0,
        "final_context_k": 6,
        "rerank_enabled": False,
        "rerank_candidates": 12,
    },
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bool_env(value: bool) -> str:
    return "true" if value else "false"


def _env_overrides(config: Dict[str, Any], *, llm_backend: str) -> Dict[str, str]:
    return {
        "LLM_BACKEND": llm_backend,
        "RETRIEVAL_DENSE_K": str(config["dense_k"]),
        "RETRIEVAL_BM25_K": str(config["bm25_k"]),
        "RETRIEVAL_RRF_K": str(config["rrf_k"]),
        "RETRIEVAL_DENSE_WEIGHT": str(config["dense_weight"]),
        "RETRIEVAL_BM25_WEIGHT": str(config["bm25_weight"]),
        "RETRIEVAL_FINAL_CONTEXT_K": str(config["final_context_k"]),
        "RERANK_ENABLED": _bool_env(bool(config["rerank_enabled"])),
        "RERANK_CANDIDATES": str(config["rerank_candidates"]),
    }


def _complexity_score(config: Dict[str, Any]) -> float:
    score = 0.0
    score += 1.0 if bool(config["rerank_enabled"]) else 0.0
    score += abs(float(config["dense_weight"]) - 1.0)
    score += abs(float(config["bm25_weight"]) - 1.0)
    score += max(0.0, (float(config["dense_k"]) - 20.0) / 20.0)
    score += max(0.0, (float(config["bm25_k"]) - 20.0) / 20.0)
    score += max(0.0, (float(config["final_context_k"]) - 8.0) / 8.0)
    score += (float(config["rerank_candidates"]) / 100.0) if bool(config["rerank_enabled"]) else 0.0
    return round(score, 4)


def _recommend_run(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranked = sorted(runs, key=lambda run: float(run["summary"]["avg_overall"]), reverse=True)
    best_overall = float(ranked[0]["summary"]["avg_overall"])
    contenders = [run for run in ranked if float(run["summary"]["avg_overall"]) >= (best_overall - 0.01)]
    contenders.sort(
        key=lambda run: (
            _complexity_score(run["config"]),
            -float(run["summary"]["avg_context_precision"]),
            -float(run["summary"]["avg_abstention"]),
            -float(run["summary"]["avg_overall"]),
        )
    )
    return contenders[0]


def _render_markdown(runs: List[Dict[str, Any]], recommended: Dict[str, Any], *, input_path: Path) -> str:
    lines: List[str] = []
    lines.append("# Week 6 Retrieval Tuning Summary")
    lines.append("")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Experiments: {len(runs)}")
    lines.append(f"- Recommended default: `{recommended['name']}`")
    lines.append("")
    lines.append("| experiment | overall | context | coverage | abstention | rerank | top_k | dense_k | bm25_k | dense_w | bm25_w | ctx_k | cand_k |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for run in sorted(runs, key=lambda item: float(item["summary"]["avg_overall"]), reverse=True):
        summary = run["summary"]
        config = run["config"]
        lines.append(
            "| "
            + f"{run['name']} | "
            + f"{summary['avg_overall']:.4f} | "
            + f"{summary['avg_context_precision']:.4f} | "
            + f"{summary['avg_coverage_accuracy']:.4f} | "
            + f"{summary['avg_abstention']:.4f} | "
            + f"{'on' if config['rerank_enabled'] else 'off'} | "
            + f"{config['top_k']} | {config['dense_k']} | {config['bm25_k']} | "
            + f"{config['dense_weight']:.2f} | {config['bm25_weight']:.2f} | "
            + f"{config['final_context_k']} | {config['rerank_candidates']} |"
        )
    lines.append("")
    lines.append("## Recommended Config")
    lines.append("")
    lines.append(f"- Name: `{recommended['name']}`")
    lines.append(f"- Avg overall: {recommended['summary']['avg_overall']:.4f}")
    lines.append(f"- Avg context precision: {recommended['summary']['avg_context_precision']:.4f}")
    lines.append(f"- Avg coverage accuracy: {recommended['summary']['avg_coverage_accuracy']:.4f}")
    lines.append(f"- Avg abstention: {recommended['summary']['avg_abstention']:.4f}")
    lines.append(f"- Complexity score: {_complexity_score(recommended['config']):.4f}")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(recommended["config"], indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small Week 6 retrieval tuning matrix over the golden set.")
    parser.add_argument("--input-path", default="data/eval/golden_questions.jsonl")
    parser.add_argument("--output-dir", default="data/eval/week6_retrieval_tuning")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of questions to run.")
    parser.add_argument("--llm-backend", default=os.getenv("LLM_BACKEND", "none").strip().lower() or "none")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    input_path = (REPO_ROOT / args.input_path).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_path": str(input_path),
        "llm_backend": args.llm_backend,
        "matrix": DEFAULT_MATRIX,
    }
    _write_json(output_dir / "experiment_manifest.json", manifest)

    runs: List[Dict[str, Any]] = []
    for config in DEFAULT_MATRIX:
        name = str(config["name"])
        result_path = output_dir / f"{name}_results.jsonl"
        summary_path = output_dir / f"{name}_summary.md"
        env = os.environ.copy()
        env.update(_env_overrides(config, llm_backend=args.llm_backend))
        cmd = [
            args.python_bin,
            str(REPO_ROOT / "scripts/run_week5_eval.py"),
            "--engine",
            "answer",
            "--input-path",
            str(input_path),
            "--output-jsonl",
            str(result_path),
            "--summary-path",
            str(summary_path),
            "--top-k",
            str(config["top_k"]),
            "--record-retrieval-details",
        ]
        if args.limit and args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])

        print(f"Running {name} ...")
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
        results = _load_jsonl(result_path)
        summary = _build_summary(results)
        runs.append(
            {
                "name": name,
                "config": config,
                "summary": summary,
                "result_path": str(result_path),
                "summary_path": str(summary_path),
            }
        )

    recommended = _recommend_run(runs)
    aggregate = {
        "input_path": str(input_path),
        "llm_backend": args.llm_backend,
        "runs": runs,
        "recommended": {
            "name": recommended["name"],
            "config": recommended["config"],
            "summary": recommended["summary"],
            "complexity_score": _complexity_score(recommended["config"]),
        },
    }
    _write_json(output_dir / "comparison.json", aggregate)
    summary_md = _render_markdown(runs, recommended, input_path=input_path)
    (output_dir / "comparison.md").write_text(summary_md, encoding="utf-8")

    print("")
    print("=== Week 6 Retrieval Tuning ===")
    for run in sorted(runs, key=lambda item: float(item["summary"]["avg_overall"]), reverse=True):
        summary = run["summary"]
        print(
            f"{run['name']}: overall={summary['avg_overall']:.4f} "
            f"context={summary['avg_context_precision']:.4f} "
            f"coverage={summary['avg_coverage_accuracy']:.4f} "
            f"abstention={summary['avg_abstention']:.4f}"
        )
    print("")
    print(f"Recommended default: {recommended['name']}")
    print(f"Comparison JSON: {output_dir / 'comparison.json'}")
    print(f"Comparison Markdown: {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
