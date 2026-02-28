# Week 5 Baseline Evaluation Summary

- Generated (UTC): 2026-02-28 20:41:11
- Input: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/golden_questions.jsonl`
- Results JSONL: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/week5_baseline_results.jsonl`
- Requested engine mode: `auto`
- `assess_control` available: `False`
- `assess_control` import note: `No module named 'app.assess'`

## Overall

- Questions: 40
- Errors: 0
- Abstained: 22
- Engine usage: {'answer_question': 40}
- Avg context precision: 0.4190
- Avg coverage accuracy: 0.7250
- Avg abstention score: 0.6050
- Avg overall score: 0.5830

## By Mode

| mode | n | avg_context_precision | avg_coverage_accuracy | avg_abstention | avg_overall |
|---|---:|---:|---:|---:|---:|
| framework | 10 | 0.8619 | 1.0000 | 0.2000 | 0.6873 |
| out_of_scope_policy | 10 | 0.0000 | 0.9000 | 0.7200 | 0.5400 |
| policy | 10 | 0.3808 | 1.0000 | 0.7000 | 0.6936 |
| policy_vs_control | 10 | 0.4333 | 0.0000 | 0.8000 | 0.4111 |
