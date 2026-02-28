# Golden Evaluation Dataset (Week 5)

This folder contains the initial golden question set for evaluating RAG behavior across framework Q&A, policy Q&A, control coverage assessment, and abstention behavior.

## Files

- `golden_questions.jsonl`: 40 labeled evaluation questions (JSONL, one object per line)
- `golden_readme.md`: dataset schema and label guidance

## Split

- `framework`: 10
- `policy`: 10
- `policy_vs_control`: 10
- `out_of_scope_policy`: 10

Total: 40

## Record schema

Each JSONL object has:

- `id` (string): unique stable question ID (`gq001` ...)
- `mode` (string): one of `framework`, `policy`, `policy_vs_control`, `out_of_scope_policy`
- `intent` (string): evaluation intent label for analysis slices
- `question` (string): user-style prompt
- `expected` (object): expected grounding target
  - `expected_control_ids` (array[string]): control IDs expected in answer grounding
  - `expected_policy_doc_ids` (array[string]): policy document IDs (markdown stem names)
  - `expected_coverage` (string|null): expected coverage label

## Coverage label semantics

- `covered`: policy evidence materially addresses the control intent
- `partial`: policy evidence exists but has meaningful gaps
- `missing`: no meaningful policy evidence in current corpus
- `unknown`: unresolved / intentionally not assessed
- `abstain`: question is policy-specific but outside the available policy corpus; assistant should decline and request relevant policy evidence
- `null`: coverage not applicable (framework-only or policy-only content lookup)

## Source alignment notes

- `policy_vs_control` labels align with `data/truth_table/controls_truth.csv` for controls included in that table.
- Policy document IDs use markdown stem names from `data/policies_synth_md_v2`:
  - `01_mini_ssp`
  - `02_access_control_policy`
  - `03_incident_response_plan`
  - `04_logging_monitoring_standard`

## Intended use

Use this dataset to score:

- retrieval grounding correctness (control and policy doc targeting)
- control coverage classification accuracy
- abstention precision for out-of-scope policy questions

