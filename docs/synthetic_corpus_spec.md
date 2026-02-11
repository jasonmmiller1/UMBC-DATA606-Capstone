# Synthetic Policy Corpus Specification (Academic Demo)

## Purpose
This project uses a synthetic policy corpus to simulate an RMF-relevant policy repository for a fictional organization and system. The corpus supports:
- Evidence-grounded policy Q&A (RAG) with citations
- Control gap analysis (covered/partial/missing/unknown)
- Reproducible evaluation via a truth table and golden question set

## Disclaimer
All documents in this corpus are:
- **SYNTHETIC DEMO ARTIFACTS — ACADEMIC USE ONLY**
- Not real DoD program artifacts
- Not marked with official DoD letterhead, distribution statements, or classification markings
- Not tied to any real organization, system, contract, or vendor

## Document Set (Policy Pack)
Target: 10–12 PDFs (initially authored in Markdown; exported to PDF later)

1. Mini-SSP / System Overview
2. Access Control Policy
3. Audit & Accountability (Logging/Monitoring) Standard
4. Incident Response Plan
5. Configuration Management Policy
6. Change / Release Management SOP
7. Vulnerability Management SOP
8. Media Protection & Sanitization Policy
9. Privacy & Data Handling Policy
10. Supplier / C-SCRM Policy (ties to SR family)
11. Security Awareness & Training Policy (optional)
12. POA&M-style “Known Gaps” Tracker (optional)

## Ground Truth Artifacts
- `data/truth_table/controls_truth.csv` — control → expected coverage → evidence location
- `data/golden_questions/golden_questions.jsonl` — question set with expected references

## Style Guide
- Use “shall” for requirements
- Include: Purpose, Scope, Roles, Policy Statements, Procedures, Exceptions, Review cadence
- Number sections (1.0, 1.1, 2.0…) to support chunking-by-header
- Every doc begins with the synthetic disclaimer banner