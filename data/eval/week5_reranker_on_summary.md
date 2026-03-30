# Week 5 Baseline Evaluation Summary

- Generated (UTC): 2026-03-09 22:26:25
- Input: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/golden_questions.jsonl`
- Results JSONL: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/week5_reranker_on_results.jsonl`
- Requested engine mode: `answer`
- `assess_control` available: `False`
- `assess_control` import note: `No module named 'app.assess'`

## Overall

- Questions: 40
- Errors: 0
- Abstained: 10
- Engine usage: {'answer_question': 40}
- Avg context precision: 0.6708
- Avg coverage accuracy: 1.0000
- Avg abstention score: 1.0000
- Avg overall score: 0.8902

## By Mode

| mode | n | avg_context_precision | avg_coverage_accuracy | avg_abstention | avg_overall |
|---|---:|---:|---:|---:|---:|
| framework | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| out_of_scope_policy | 10 | 0.8000 | 1.0000 | 1.0000 | 0.9333 |
| policy | 10 | 0.4244 | 1.0000 | 1.0000 | 0.8081 |
| policy_vs_control | 10 | 0.4586 | 1.0000 | 1.0000 | 0.8195 |
