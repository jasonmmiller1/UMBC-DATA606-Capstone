# Week 6 Retrieval Tuning Summary

- Input: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/golden_questions.jsonl`
- Experiments: 6
- Recommended default: `equal_rerank20`

| experiment | overall | context | coverage | abstention | rerank | top_k | dense_k | bm25_k | dense_w | bm25_w | ctx_k | cand_k |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| equal_rerank20 | 0.8989 | 0.7218 | 0.9750 | 1.0000 | on | 10 | 20 | 20 | 1.00 | 1.00 | 8 | 20 |
| deeper_rerank30_ctx10 | 0.8971 | 0.7162 | 0.9750 | 1.0000 | on | 10 | 30 | 30 | 1.00 | 1.00 | 10 | 30 |
| shallow_no_rerank_ctx6 | 0.8886 | 0.7406 | 0.9750 | 0.9500 | off | 8 | 12 | 12 | 1.00 | 1.00 | 6 | 12 |
| equal_no_rerank | 0.8864 | 0.7093 | 0.9750 | 0.9750 | off | 10 | 20 | 20 | 1.00 | 1.00 | 8 | 20 |
| dense_lean_rerank20 | 0.8852 | 0.6806 | 0.9750 | 1.0000 | on | 10 | 20 | 20 | 1.10 | 0.90 | 8 | 20 |
| bm25_lean_rerank20 | 0.8241 | 0.7474 | 0.9750 | 0.7500 | on | 10 | 20 | 20 | 0.90 | 1.10 | 8 | 20 |

## Recommended Config

- Name: `equal_rerank20`
- Avg overall: 0.8989
- Avg context precision: 0.7218
- Avg coverage accuracy: 0.9750
- Avg abstention: 1.0000
- Complexity score: 1.2000

```json
{
  "name": "equal_rerank20",
  "top_k": 10,
  "dense_k": 20,
  "bm25_k": 20,
  "rrf_k": 60,
  "dense_weight": 1.0,
  "bm25_weight": 1.0,
  "final_context_k": 8,
  "rerank_enabled": true,
  "rerank_candidates": 20
}
```
