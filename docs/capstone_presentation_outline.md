# Capstone Presentation Outline

Presentation target: 8 to 12 minutes, plus a separate live demo.

Deck structure:

- 1 cover slide
- 10 main content slides
- 1 separate demo slide, clearly excluded from the main count

This repo does not currently include a committed slide-generation workflow such as Marp or a PowerPoint build script, so this file is written as PowerPoint-ready slide content with suggested visuals and short speaker notes.

## Final Proposed Slide List

Cover

1. Why this project exists
2. What RMF is and why it matters
3. Project objective and scope
4. System architecture overview
5. Data and retrieval pipeline
6. Runtime answer flow
7. Trustworthiness and answer-state logic
8. Evaluation approach
9. Results and progress highlights
10. Final system state and deliverables

Demo slide, not counted

---

## Cover Slide

### On-slide content

**RMF Assistant: Evidence-Grounded Policy and Control Reasoning for RMF**

- UMBC DATA 606 Capstone
- Streamlit + Hybrid Retrieval + Optional OpenRouter
- Jason Miller
- April 2026

### Suggested visual

- Clean title slide with a small architecture icon set:
  - policy document
  - control text
  - vector database
  - chat UI

### Speaker notes

- This project is a demo-quality RMF assistant, not a generic chatbot.
- The core idea is grounded reasoning over policy and control evidence, with citations and disciplined abstention.

---

## Slide 1 of 10: Why This Project Exists

### Key bullets

- RMF work is document-heavy and evidence-heavy.
- Teams must connect policy text to framework controls, not just retrieve isolated passages.
- Manual review is slow, repetitive, and easy to lose track of.
- The project goal is to reduce search effort while keeping answers auditable.

### Suggested visual

- Simple left-to-right problem graphic:
  - many documents -> manual analyst effort -> slow compliance reasoning

### Speaker notes

- The motivation is not “chat with documents.”
- The real problem is finding relevant policy evidence, connecting it to RMF controls, and avoiding overconfident answers when evidence is weak.

---

## Slide 2 of 10: What RMF Is and Why It Matters

### Key bullets

- RMF is the Risk Management Framework used to guide security and privacy risk management for information systems.
- It depends on mapping controls, policies, and implementation evidence.
- That makes traceability essential: what requirement, what evidence, what gap?
- This project focuses on helping that reasoning step, not replacing assessors.

### Suggested visual

- One simple diagram:
  - RMF control -> policy evidence -> analyst decision

### Speaker notes

- Keep RMF simple here.
- The audience does not need a compliance history lesson; they need to understand why policy-to-control reasoning matters.

---

## Slide 3 of 10: Project Objective and Scope

### Key bullets

- Build an RMF/policy assistant with grounded answers and citations.
- Support both policy Q&A and policy-versus-control coverage reasoning.
- Run in retrieval-only mode or retrieval plus OpenRouter mode.
- Stay honest when evidence is missing, weak, or conflicting.

### Suggested visual

- Scope box with two lanes:
  - policy lookup / policy Q&A
  - policy vs control coverage reasoning

### Speaker notes

- Emphasize what was built:
  - Streamlit UI
  - hybrid retrieval
  - optional LLM generation
  - answer-state logic
- Emphasize what was intentionally not claimed:
  - no autonomous compliance decision-making
  - no enterprise-scale orchestration

---

## Slide 4 of 10: System Architecture Overview

### Key bullets

- Offline: committed source data is chunked and indexed.
- Runtime: Streamlit sends questions into hybrid retrieval over local artifacts plus Qdrant.
- Optional OpenRouter generation sits after retrieval, not before it.
- Answer-state logic turns raw outputs into user-facing trust states.

### Suggested visual

- Reuse a simplified version of the architecture flow from [`docs/app_overview.md`](./app_overview.md):
  - source/bootstrap/indexing
  - runtime retrieval/generation
  - answer-state/UI presentation
  - deployment shape

### Speaker notes

- The important separation is:
  - preparation path
  - serving path
  - answer rendering
  - deployment path
- Qdrant should be explicit on this slide as the external dense retrieval store.

---

## Slide 5 of 10: Data and Retrieval Pipeline

### Key bullets

- Committed corpus includes parsed OSCAL controls and synthetic policy markdown for reproducible demos.
- Current repo artifacts include 324 parsed OSCAL control records and 12 synthetic policy markdown documents.
- Bootstrap rebuilds `chunks.parquet` and the local BM25 index, then seeds Qdrant.
- Retrieval combines dense search in Qdrant with BM25 locally, then fuses and reranks results.

### Suggested visual

- Pipeline graphic:
  - OSCAL + synthetic policies + optional uploads
  - chunking
  - BM25 index
  - embeddings to Qdrant
  - fused retrieval

### Speaker notes

- This is where the project becomes reproducible.
- The committed source data means the demo does not depend on a raw OSCAL clone at presentation time.
- Optional uploads exist, but the main academic demo path is the committed corpus.

---

## Slide 6 of 10: Runtime Answer Flow

### Key bullets

- The app classifies each question as policy, framework, mixed policy-vs-control, or abstain-oriented.
- Retrieval produces cited evidence chunks before any answer text is shown.
- `LLM_BACKEND=none` gives retrieval-only behavior; `LLM_BACKEND=openrouter` adds grounded generation.
- If OpenRouter is unavailable, the app falls back to retrieval-backed output instead of failing silently.

### Suggested visual

- Branching runtime flow:
  - user question -> retrieval -> retrieval-only path or OpenRouter path -> answer-state rendering

### Speaker notes

- This is the slide to explain retrieval-only versus retrieval-plus-LLM mode.
- Be explicit that the LLM is downstream of retrieval and does not replace evidence selection.

---

## Slide 7 of 10: Trustworthiness and Answer-State Logic

### Key bullets

- The UI classifies answers into strong evidence, limited evidence, conflicting evidence, no evidence, retrieval-only fallback, or backend failure.
- Citations and retrieved evidence are always shown as the source of truth.
- Policy hits now support inline full-policy viewing inside the app UI.
- The system is designed to abstain or hedge when support is weak.

### Suggested visual

- Six-state answer card grid with short labels and colors:
  - strong
  - limited
  - conflicting
  - no evidence
  - retrieval-only fallback
  - backend error

### Speaker notes

- This is a key differentiator for the project.
- The UI is not only “answering”; it is signaling how much trust the user should place in that answer.

---

## Slide 8 of 10: Evaluation Approach

### Key bullets

- The project uses a 40-question labeled golden set in four modes: framework, policy, policy-vs-control, and out-of-scope policy.
- Each mode has 10 questions.
- Reported metrics include context precision, coverage accuracy, abstention, and overall score.
- This separates runtime behavior from evaluation artifacts: the app serves answers, while the eval set scores grounding quality.

### Suggested visual

- Small 4-column evaluation dataset graphic:
  - framework 10
  - policy 10
  - policy_vs_control 10
  - out_of_scope_policy 10

### Speaker notes

- This is the evidence basis for the results slides.
- The evaluation framing is not only “did it answer,” but “did it answer with the right evidence and abstain when appropriate.”

---

## Slide 9 of 10: Results and Progress Highlights

### Key bullets

- Week 5 baseline on 40 questions: overall `0.8776`, context precision `0.6577`, coverage accuracy `0.9750`, abstention `1.0000`.
- Week 6 default retrieval config: overall `0.8989`, context precision `0.7218`, coverage accuracy `0.9750`, abstention `1.0000`.
- Largest supported gain was policy-vs-control overall: `0.7758 -> 0.8485` with context precision `0.4274 -> 0.6456`.
- Retrieval tuning chose `equal_rerank20` as the recommended default across 6 experiments.

### Suggested visual

- One simple comparison chart:
  - Week 5 baseline vs Week 6 default for overall and context precision
- Small callout box:
  - `equal_rerank20`
  - overall `0.8989`
  - coverage `0.9750`
  - abstention `1.0000`

### Speaker notes

- The safest story is that Week 6 improved retrieval quality, especially context precision, without degrading coverage accuracy or abstention.
- Avoid claiming a measured OpenRouter uplift here because there is no committed retrieval-vs-OpenRouter comparison artifact in the repo today.

---

## Slide 10 of 10: Final System State and Deliverables

### Key bullets

- Delivered system: Streamlit UI, hybrid retrieval, optional OpenRouter mode, answer-state rendering, inline policy-source viewing, and containerized deployment path.
- Deployment shape is now clearer: app container plus external Qdrant.
- Current branch validation: 45 unit tests passing, plus documented local build/bootstrap/runbook validation.
- Final demo shows grounded policy lookup, framework lookup, and policy-vs-control reasoning.

### Suggested visual

- “What was delivered” checklist with four icons:
  - UI
  - retrieval pipeline
  - trust/answer states
  - deployment/demo readiness

### Speaker notes

- This slide is the landing point before the live demo.
- Frame the project as a technically grounded academic prototype that now has a cleaner demo and deployment story.

---

## Demo Slide: Not Included in Main Slide Count

### Key bullets

- Demo 1: `What does our access control policy say about least privilege?`
- Demo 2: `Summarize the requirement in AC-2.`
- Demo 3: `How well do our policies cover PL-2 based on the mini SSP?`
- Watch for citations, retrieved evidence, answer-state banner, and full-policy source viewing.

### Suggested visual

- Minimal “Live Demo” slide with three example questions and a short “what to notice” box.

### Speaker notes

- Start with the policy question to show source-grounded evidence and the full-policy viewer.
- Then show a framework query.
- End with policy-vs-control reasoning to demonstrate disciplined coverage framing.
- If OpenRouter is unstable, switch to retrieval-only mode and keep the demo focused on evidence quality.

---

## Results Provenance

Use these repo artifacts when building the final slides:

- Architecture and deployment shape:
  - [`docs/app_overview.md`](./app_overview.md)
  - [`docs/demo_runbook.md`](./demo_runbook.md)
  - [`docs/deployment_digitalocean.md`](./deployment_digitalocean.md)
- Evaluation dataset:
  - [`data/eval/golden_readme.md`](../data/eval/golden_readme.md)
- Baseline metrics:
  - [`data/eval/week5_baseline_summary.md`](../data/eval/week5_baseline_summary.md)
- Week 6 default metrics:
  - [`data/eval/week6_default_eval_summary.md`](../data/eval/week6_default_eval_summary.md)
- Week-over-week delta:
  - [`data/eval/week6_vs_week5_baseline_delta.md`](../data/eval/week6_vs_week5_baseline_delta.md)
- Retrieval tuning recommendation:
  - [`data/eval/week6_retrieval_tuning/comparison.md`](../data/eval/week6_retrieval_tuning/comparison.md)
- Demo flow:
  - [`data/eval/demo_questions.jsonl`](../data/eval/demo_questions.jsonl)

Current branch validation used for Slide 10:

- `.venv/bin/python -m unittest discover -s tests -v` -> `45` tests passed on this branch during deck preparation

## Simplifications and Incomplete Data

- No committed `data/eval/demo_mode_comparison.md` artifact exists in the repo right now, so the deck does not claim a measured retrieval-only versus OpenRouter performance delta.
- The deck uses Week 5 and Week 6 evaluation artifacts for the strongest quantitative story because those are the clearest committed summaries.
- Week 7 is presented primarily as final system integration, trust/UI hardening, containerization, and deployment preparation, not as a new committed benchmark week.
