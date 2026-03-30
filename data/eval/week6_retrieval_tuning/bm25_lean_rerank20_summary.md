# Week 5 Baseline Evaluation Summary

- Generated (UTC): 2026-03-30 01:33:30
- Input: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/golden_questions.jsonl`
- Results JSONL: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/week6_retrieval_tuning/bm25_lean_rerank20_results.jsonl`
- Requested engine mode: `answer`
- LLM backend: `none`
- OpenRouter model: `nvidia/nemotron-nano-9b-v2:free`
- `assess_control` available: `False`
- `assess_control` import note: `No module named 'app.assess'`

## Overall

- Questions: 40
- Errors: 0
- Abstained: 20
- Engine usage: {'answer_question': 40}
- Avg context precision: 0.7474
- Avg coverage accuracy: 0.9750
- Avg abstention score: 0.7500
- Avg overall score: 0.8241

## By Mode

| mode | n | avg_context_precision | avg_coverage_accuracy | avg_abstention | avg_overall |
|---|---:|---:|---:|---:|---:|
| framework | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| out_of_scope_policy | 10 | 0.9000 | 1.0000 | 1.0000 | 0.9667 |
| policy | 10 | 0.4300 | 1.0000 | 1.0000 | 0.8100 |
| policy_vs_control | 10 | 0.6595 | 0.9000 | 0.0000 | 0.5199 |
