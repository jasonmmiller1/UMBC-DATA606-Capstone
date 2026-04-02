# Week 5 Baseline Evaluation Summary

- Generated (UTC): 2026-03-10 01:26:19
- Input: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/golden_questions.jsonl`
- Results JSONL: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/week5_policy_vs_control_expanded_results.jsonl`
- Requested engine mode: `answer`
- LLM backend: `none`
- OpenRouter model: `nvidia/nemotron-nano-9b-v2:free`
- `assess_control` available: `False`
- `assess_control` import note: `No module named 'app.assess'`

## Overall

- Questions: 40
- Errors: 0
- Abstained: 10
- Engine usage: {'answer_question': 40}
- Avg context precision: 0.7159
- Avg coverage accuracy: 0.9750
- Avg abstention score: 1.0000
- Avg overall score: 0.8970

## By Mode

| mode | n | avg_context_precision | avg_coverage_accuracy | avg_abstention | avg_overall |
|---|---:|---:|---:|---:|---:|
| framework | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| out_of_scope_policy | 10 | 0.8000 | 1.0000 | 1.0000 | 0.9333 |
| policy | 10 | 0.4033 | 1.0000 | 1.0000 | 0.8011 |
| policy_vs_control | 10 | 0.6603 | 0.9000 | 1.0000 | 0.8534 |
