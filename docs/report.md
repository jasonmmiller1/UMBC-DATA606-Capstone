# RMF Assistant: Evidence-Grounded LLM for DoD RMF Policy Q&A and Control Gap Analysis

**Jason Miller**<br>
**Semester: Spring 2026**

- [Jason Miller - Capstone Project Spring 2026.pptx](./Jason%20Miller%20-%20Capstone%20Project%20Spring%202026.pptx)
- [GitHub Repository](https://github.com/jasonmmiller1/UMBC-DATA606-Capstone)

## Background

The Department of War (Defense) Risk Management Framework (RMF) is a structured approach for managing security and privacy risk in information systems. In practice, RMF work requires analysts to move repeatedly between control requirements, policy documents, implementation evidence, and gap assessments. The challenge is not only finding relevant text, but also determining whether a policy actually addresses the control intent and whether the available evidence is strong enough to support a conclusion.

This project was motivated by that evidence-tracking problem. Policy and control artifacts are long, heterogeneous, and often difficult to cross-reference quickly during assessment work. A conventional question-answering interface can make this problem worse if it produces fluent but weakly grounded answers. For that reason, the project focused on an evidence-first assistant rather than a free-form chatbot.

The resulting system is an RMF/policy assistant implemented with a Streamlit user interface, a hybrid retrieval pipeline, an external Qdrant vector store, and an optional OpenRouter-backed generation step. At runtime, the system retrieves policy and control evidence first, then either returns a retrieval-backed summary or a grounded generated explanation. A separate answer-state layer classifies the result as strong evidence, limited evidence, conflicting evidence, no evidence, retrieval-only fallback, or backend failure so that the interface communicates not just an answer, but the system's confidence and evidence posture.

## Description of Data Sources

### Runtime Data Sources

The runtime system relies on two primary content families.

The first is authoritative control content from NIST SP 800-53 Rev. 5, ingested through OSCAL. In this repository, the parsed control artifacts are committed in `data/oscal_parsed/`, which allows the system to operate reproducibly without requiring a fresh raw OSCAL clone during normal bring-up.

The second is a synthetic policy corpus created for academic demonstration. These documents are stored primarily in `data/policies_synth_md_v2/` and represent fictional but structured policy-like artifacts such as an access control policy, incident response plan, logging and monitoring standard, and related documents. The corpus is intentionally synthetic so the project can demonstrate RMF-style reasoning without relying on real organizational documentation.

The application also supports optional local uploads of PDF, Markdown, and text policy files through the Streamlit UI. These uploads are useful for local testing, but they are not the primary basis for the committed evaluation results in the repository.

In deployment terms, dense vectors and payload metadata live in Qdrant, while local chunk metadata and BM25 artifacts are rebuilt from committed source data into `data/index/chunks.parquet` and `data/bm25_index/bm25_index.pkl`.

### Evaluation and Truth Data Sources

The project also includes separate evaluation artifacts that are not the same as the runtime corpus.

- `data/eval/golden_questions.jsonl` contains a labeled 40-question evaluation set.
- `data/eval/golden_readme.md` documents its schema and split.
- `data/truth_table/controls_truth.csv` contains control-level coverage labels used for policy-versus-control evaluation slices.
- Week-specific summary files in `data/eval/` record baseline, tuned, and comparison results.

These artifacts are important because they distinguish system operation from system measurement. The runtime application answers questions, while the golden question set and truth table provide a way to score retrieval grounding, coverage labeling, and abstention behavior.

## Data Elements

The assistant reasons over chunked text rather than over whole documents. Each chunk is paired with metadata that allows the system to filter, rank, cite, and display evidence appropriately.

| Data element | Example | Role in the system |
|---|---|---|
| `control_id` | `AC-2`, `AU-6` | Links OSCAL chunks to canonical RMF control identifiers. |
| `doc_id` | `02_access_control_policy` | Stable identifier for policy documents and uploaded artifacts. |
| `doc_title` | `Access Control Policy — Rivermark Operations Portal (ROP)` | Human-readable source label shown in UI and citations. |
| `source_type` | `oscal_control`, `policy_pdf`, `policy_md` | Distinguishes framework evidence from policy evidence and supports filtered retrieval. |
| `chunk_id` | `oscal::...`, `policy::...` | Unique identifier for deduplication, ranking, and citation generation. |
| `chunk_text` | control statement text or policy section text | The actual evidence content retrieved and shown to the user. |
| `section_path` / `heading` | `... > 4.0 Policy Statements` | Preserves document structure and supports source-context display. |
| `chunk_type` | `statement`, `guidance`, `text` | Helps distinguish different kinds of control and policy content. |
| Retrieval scores | `dense_score`, `bm25_score`, `rrf_score` | Support hybrid ranking and diagnostics. |
| Evaluation labels | `expected_coverage`, expected control IDs, expected policy doc IDs | Provide scoring targets for the golden evaluation set. |

These data elements allow the system to do more than keyword lookup. They support intent-aware retrieval, evidence fusion across policy and control content, grounded citations, and UI features such as inline policy-source viewing.

## Results of EDA

This project was primarily a systems and applied-ML capstone rather than a traditional tabular modeling project, so exploratory data analysis (EDA) was lightweight and focused on corpus profiling, chunk diagnostics, and evaluation-set inspection rather than on conventional statistical visualization.

The most relevant data inspection findings are summarized below.

| Inspection result | Value | Interpretation |
|---|---:|---|
| Parsed OSCAL control records | 324 | The committed OSCAL artifact represents a one-record-per-control view of the control corpus used for framework retrieval. |
| Synthetic policy markdown documents | 12 | The demo corpus is intentionally compact and curated rather than large-scale. |
| Current local chunk corpus | 1,432 chunks | The runtime corpus is chunked enough to support section-level retrieval rather than document-level matching only. |
| `oscal_control` chunks | 1,316 | The chunk corpus is heavily weighted toward control text. |
| `policy_pdf` chunks | 116 | Policy evidence is much smaller than the control corpus, which explains why policy-aware retrieval tuning matters. |
| Truth-table labeled controls | 12 | Coverage evaluation is available, but on a limited control subset. |

Two EDA-oriented findings affected system design directly.

First, the chunk corpus is imbalanced toward control content. In the current local index, 1,316 of 1,432 chunks are `oscal_control` chunks, while only 116 are policy chunks. This imbalance helps explain why policy questions and policy-versus-control questions needed source-type filtering, section-aware weighting, and reranking rather than a naive mixed-corpus search.

Second, the truth labels are intentionally small and task-specific. The truth table contains 12 labeled controls, distributed as 4 `covered`, 5 `partial`, 2 `missing`, and 1 `unknown`. This was appropriate for focused evaluation of coverage reasoning, but it also limits how broadly coverage results can be generalized.

In short, the repository supports honest data profiling and diagnostics, but not a full conventional EDA narrative with broad inferential claims. The data work here was used mainly to verify source integrity, chunk structure, label coverage, and retrieval-readiness.

## Results of ML

In this project, the "ML" component is broader than a single predictive model. It includes dense embedding retrieval through Qdrant, BM25 sparse retrieval, reciprocal-rank fusion, light reranking, optional LLM-based answer generation through OpenRouter, and evaluation of grounded answer quality.

### Evaluation Setup

The primary quantitative evaluation set contains 40 labeled questions:

- 10 framework questions
- 10 policy questions
- 10 policy-versus-control questions
- 10 out-of-scope policy questions

The main scoring dimensions recorded in the repository are:

- context precision
- coverage accuracy
- abstention score
- overall score

This evaluation design is important because it rewards not only answering correctly, but also retrieving the right evidence and abstaining when the corpus does not support a confident answer.

### Week 5 Baseline and Week 6 Improvement

The clearest committed quantitative story in the repository is the progression from the Week 5 baseline to the Week 6 default retrieval configuration.

| Metric | Week 5 baseline | Week 6 default | Delta |
|---|---:|---:|---:|
| Avg context precision | 0.6577 | 0.7218 | +0.0641 |
| Avg coverage accuracy | 0.9750 | 0.9750 | +0.0000 |
| Avg abstention | 1.0000 | 1.0000 | +0.0000 |
| Avg overall | 0.8776 | 0.8989 | +0.0213 |
| Errors | 0 | 0 | +0 |
| Abstained questions | 10 | 10 | +0 |

These results show that the main improvement came from better retrieval quality, especially better context precision, while preserving already strong coverage accuracy and abstention behavior. This is a meaningful outcome for an evidence-grounded assistant, because improving answer fluency without improving evidence quality would have been less useful.

The strongest mode-specific improvement appeared in policy-versus-control reasoning:

- policy-versus-control overall score improved from 0.7758 to 0.8485
- policy-versus-control context precision improved from 0.4274 to 0.6456

That matters because policy-versus-control questions are the most distinctive part of the capstone. They are more demanding than pure framework lookup or pure policy lookup because the system must retrieve from both corpora and align policy evidence against control intent.

### Retrieval Tuning Results

Week 6 retrieval tuning compared six retrieval configurations. The repository summary recommends the `equal_rerank20` configuration as the best default. Its committed metrics are:

- average overall score: 0.8989
- average context precision: 0.7218
- average coverage accuracy: 0.9750
- average abstention: 1.0000

The tuning results are useful for understanding what worked:

- balanced dense and BM25 weights performed better than leaning too heavily toward one side
- reranking helped compared with no-rerank variants
- a BM25-leaning configuration produced the weakest abstention score at 0.7500, showing that better lexical matching alone was not enough

This supports the final architecture choice of hybrid retrieval with reranking rather than a single-mode search strategy.

### OpenRouter and Retrieval-Only Operation

The repository supports both retrieval-only mode and retrieval-plus-OpenRouter mode. From a system-design perspective, this is important because the project was never meant to depend on live generation in order to remain useful. Retrieval-only mode is treated as a valid operating mode, not merely a degraded error path.

However, the current repository does **not** contain a committed retrieval-only versus OpenRouter comparison artifact that would justify a quantitative claim about LLM uplift in final answer quality. As a result, this report does not claim a measured performance increase from generation. Instead, the repo-backed conclusion is narrower and more accurate:

- OpenRouter integration exists and is tested for configuration errors, retries, and fallback behavior.
- Retrieval-only mode remains a reliable baseline mode for grounded operation.
- The best-supported quantitative gains in the repository come from retrieval tuning, not from a committed LLM-vs-no-LLM benchmark.

### Week 7 System Validation

By the final project state, the system had moved beyond retrieval experiments into integration, trust signaling, and deployment preparation. The repository now includes:

- explicit answer-state rendering in the UI
- retrieval-only fallback behavior when the LLM is unavailable
- inline policy-source viewing for retrieved policy evidence
- containerized startup and deployment guidance with external Qdrant

During preparation of this final report, the repository test suite completed successfully with 45 passing unit tests on the current branch. These tests cover retrieval reranking, answer-state classification, OpenRouter failure handling, policy chunking behavior, and source-view resolution. That does not replace end-to-end human evaluation, but it does strengthen confidence that the implemented system behavior is internally consistent.

## Conclusion

This capstone produced a working RMF/policy assistant that is grounded in retrieved evidence rather than in unconstrained generation. The final system combines a Streamlit interface, hybrid retrieval over OSCAL controls and synthetic policy documents, optional OpenRouter generation, answer-state classification, and a cleaner containerized deployment path with external Qdrant.

The most important technical achievement was not simply adding an LLM to policy documents. It was building a system that can retrieve policy and control evidence, cite that evidence, reason across the two corpora for coverage-style questions, and abstain or degrade gracefully when reliable support is not present. The evaluation artifacts in the repository show that retrieval quality improved meaningfully between the Week 5 baseline and the Week 6 tuned default, especially for the policy-versus-control task that most closely represents the capstone's core research objective.

By the final Week 7 state, the project delivered a coherent, demo-ready prototype with documented runtime assumptions, tested fallback behavior, and a clearer deployment story. That is a meaningful outcome for a graduate capstone focused on applied AI for evidence-grounded security analysis.

## Limitations

Several important limitations remain.

- The policy corpus is synthetic. It is useful for safe, reproducible demonstration, but it cannot fully represent the messiness and ambiguity of real organizational policy sets.
- Coverage truth data are limited. The truth table contains only 12 labeled controls, so coverage results should be interpreted as focused task validation rather than broad generalization.
- The project includes lightweight contradiction detection and trust signaling, but conflict detection is still partly heuristic.
- A committed quantitative comparison between retrieval-only mode and retrieval-plus-OpenRouter mode is not yet present in the repository, so LLM-specific performance claims remain limited.
- Deployment guidance is materially clearer than before, but the repository does not claim a completed end-to-end production validation on DigitalOcean App Platform plus external persistent Qdrant.
- Uploaded files are convenient for local demos, but they are not durable in the hosted deployment path without external storage.
- The evaluation emphasis is on grounded retrieval behavior, coverage reasoning, and abstention rather than on broad human-user studies or operational adoption outcomes.

## Future Research Direction

The most useful next steps are concrete rather than speculative.

- Expand the policy corpus beyond synthetic artifacts to include larger and more heterogeneous real-world policy sets, if an approved dataset becomes available.
- Grow the truth-table and golden-question artifacts so coverage evaluation spans more controls, more document types, and more ambiguous edge cases.
- Add formal human evaluation, especially for policy-versus-control reasoning quality, trust calibration, and usability of answer-state signals.
- Improve contradiction detection with stronger cross-document consistency analysis rather than relying mainly on lexical conflict markers.
- Measure retrieval-only versus retrieval-plus-LLM behavior on a committed benchmark so the role of generation can be assessed quantitatively rather than anecdotally.
- Explore domain-adapted reranking or embedding strategies to further improve policy evidence selection in mixed queries.
- Extend the hosted architecture with durable upload storage and a fully validated cloud deployment path suitable for longer-running demonstrations or collaborative use.

Taken together, these directions would move the project from a strong academic prototype toward a more comprehensive evidence-assistance platform for RMF-style security documentation analysis.
