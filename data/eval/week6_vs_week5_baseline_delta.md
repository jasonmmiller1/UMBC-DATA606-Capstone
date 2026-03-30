# Week 6 Evaluation Delta Summary

- Baseline: `week5_baseline`
- Current: `week6_default`
- Baseline results: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/week5_baseline_results.jsonl`
- Current results: `/home/jaz3n/Repository/UMBC-DATA606-Capstone/data/eval/week6_default_eval_results.jsonl`
- Questions compared: 40
- Changed questions: 13
- Material-change threshold (overall delta): 0.05

## Overall Delta

| metric | baseline | current | delta |
|---|---:|---:|---:|
| avg_context_precision | 0.6577 | 0.7218 | +0.0641 |
| avg_coverage_accuracy | 0.9750 | 0.9750 | +0.0000 |
| avg_abstention | 1.0000 | 1.0000 | +0.0000 |
| avg_overall | 0.8776 | 0.8989 | +0.0213 |
| abstained_count | 10 | 10 | +0 |
| error_count | 0 | 0 | +0 |

## By Mode

| mode | baseline overall | current overall | delta overall | delta context | delta coverage | delta abstention |
|---|---:|---:|---:|---:|---:|---:|
| framework | 1.0000 | 1.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| out_of_scope_policy | 0.9333 | 0.9333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| policy | 0.8011 | 0.8139 | +0.0128 | +0.0383 | +0.0000 | +0.0000 |
| policy_vs_control | 0.7758 | 0.8485 | +0.0727 | +0.2182 | +0.0000 | +0.0000 |

## Material Improvements

- gq023 (policy_vs_control) overall +0.1458: What is our coverage for AC-6 using current policy artifacts?
  Note: higher context precision; 4 new cited chunks; 4 dropped cited chunks; current policy sections include Access Control Policy — Rivermark Operations Portal (ROP) (Fictional) > 4.0 Policy Statements
- gq025 (policy_vs_control) overall +0.1429: Is AU-6 fully addressed by our logging standard?
  Note: higher context precision; 2 new cited chunks; 4 dropped cited chunks; current policy sections include Logging & Monitoring Standard — Rivermark Operations Portal (ROP) (Fictional)
- gq024 (policy_vs_control) overall +0.1111: How well do our policies cover AU-2?
  Note: higher context precision; 7 new cited chunks; 7 dropped cited chunks; current policy sections include Logging & Monitoring Standard — Rivermark Operations Portal (ROP) (Fictional) > 4.0 Logging Requirements
- gq022 (policy_vs_control) overall +0.0889: Do our policies cover AC-3?
  Note: higher context precision; 5 new cited chunks; 4 dropped cited chunks; current policy sections include Access Control Policy — Rivermark Operations Portal (ROP) (Fictional) > 4.0 Policy Statements
- gq026 (policy_vs_control) overall +0.0879: What is the expected coverage label for AU-11?
  Note: higher context precision; 5 new cited chunks; 6 dropped cited chunks; current policy sections include Logging & Monitoring Standard — Rivermark Operations Portal (ROP) (Fictional) > 6.0 Retention & Protection
- gq030 (policy_vs_control) overall +0.0715: Is CM-2 covered by our current policy corpus?
  Note: higher context precision; 4 new cited chunks; 5 dropped cited chunks; current policy sections include Incident Response Plan — Rivermark Operations Portal (ROP) (Fictional) > 4.0 Policy Statements
- gq013 (policy) overall +0.0667: What are the main steps in our incident response procedure?
  Note: higher context precision; 3 new cited chunks; 3 dropped cited chunks; current policy sections include Incident Response Plan — Rivermark Operations Portal (ROP) (Fictional) > 5.0 Procedures

## Regressions

- gq029 (policy_vs_control) overall -0.0151: What coverage status should PL-2 receive based on the mini SSP?
  Note: lower context precision; 7 new cited chunks; 8 dropped cited chunks; current policy sections include Mini-SSP / System Overview — Rivermark Operations Portal (ROP) (Fictional) > 5.0 Data Types

## Recommendation

- Replace the previous default with `week6_default` for next week. The lift is driven by context precision, while coverage accuracy and abstention stayed stable.
