# Week 5 Baseline Evaluation Summary

- Generated (UTC): 2026-03-08 19:12:35
- Input: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/golden_questions.jsonl`
- Results JSONL: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/week5_baseline_results.jsonl`
- Requested engine mode: `auto`
- `assess_control` available: `False`
- `assess_control` import note: `No module named 'app.assess'`

## Overall

- Questions: 40
- Errors: 0
- Abstained: 13
- Engine usage: {'answer_question': 40}
- Avg context precision: 0.6432
- Avg coverage accuracy: 0.9000
- Avg abstention score: 0.9250
- Avg overall score: 0.8227

## By Mode

| mode | n | avg_context_precision | avg_coverage_accuracy | avg_abstention | avg_overall |
|---|---:|---:|---:|---:|---:|
| framework | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| out_of_scope_policy | 10 | 0.8000 | 1.0000 | 1.0000 | 0.9333 |
| policy | 10 | 0.3455 | 1.0000 | 0.7000 | 0.6818 |
| policy_vs_control | 10 | 0.4274 | 0.6000 | 1.0000 | 0.6758 |
