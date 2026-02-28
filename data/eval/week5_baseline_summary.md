# Week 5 Baseline Evaluation Summary

- Generated (UTC): 2026-02-28 21:22:24
- Input: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/golden_questions.jsonl`
- Results JSONL: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/week5_baseline_results.jsonl`
- Requested engine mode: `auto`
- `assess_control` available: `False`
- `assess_control` import note: `No module named 'app.assess'`

## Overall

- Questions: 40
- Errors: 0
- Abstained: 2
- Engine usage: {'answer_question': 40}
- Avg context precision: 0.4758
- Avg coverage accuracy: 0.7750
- Avg abstention score: 0.7450
- Avg overall score: 0.6653

## By Mode

| mode | n | avg_context_precision | avg_coverage_accuracy | avg_abstention | avg_overall |
|---|---:|---:|---:|---:|---:|
| framework | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| out_of_scope_policy | 10 | 0.0000 | 0.9000 | 0.0800 | 0.3266 |
| policy | 10 | 0.3808 | 1.0000 | 0.9000 | 0.7603 |
| policy_vs_control | 10 | 0.5225 | 0.2000 | 1.0000 | 0.5742 |
