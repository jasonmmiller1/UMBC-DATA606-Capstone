---
title: "RMF Assistant: Evidence-Grounded LLM for DoD RMF Policy Q&A and Control Gap Analysis (OSCAL + Docling + Hybrid RAG)"
author: "Jason Miller"
course: "UMBC Data Science Master Degree Capstone"
instructor: "Dr Chaojie (Jay) Wang"
status: "draft"
timeline_weeks: 8-10
---

# Final Report Template (Living Document)

> This document is a guide for developing the project proposal and final report. It will be updated continuously using an agile approach.

---

## 1. Title and Author

- **Project Title**  
  RMF Assistant: Evidence-Grounded LLM for DoD RMF Policy Q&A and Control Gap Analysis (OSCAL + Docling + Hybrid RAG)

- **Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang**

- **Author Name**  
  Jason Miller

- **Link to the author's GitHub repo of the project**  
  https://github.com/jasonmmiller1/UMBC-DATA606-Capstone

- **Link to the author's LinkedIn profile**  
[(LinkedIn)](https://www.linkedin.com/in/jasonmmiller/)

- **Link to presentation**  
  [(RMF ASSISTANT — PROPOSAL PRESENTATION)](https://docs.google.com/presentation/d/1UsXvum-EVhE8DZbuMtrhUTdHkmSdaP5Gj-ZI-DUJN-c/edit?usp=sharing)

- **Link to your YouTube video**  
  *(TODO: add YouTube link)*

---

## 2. Background

### What is it about?
This capstone builds a demo-quality AI assistant to support the DoD Risk Management Framework (RMF) by:

1) enabling **policy Q&A** over uploaded PDF documents with **source-cited answers**, and  
2) producing a structured **gap analysis** that compares policy text against a selected subset of NIST SP 800-53 Rev. 5 controls.

The system is designed as a **hybrid architecture** where:
- Retrieval-Augmented Generation (RAG) provides grounded, auditable responses, and
- optional lightweight supervised tuning (LoRA SFT) may be used later to improve output consistency and rubric adherence.

### Why does it matter?
RMF compliance work is complex, hierarchical, and documentation-heavy. Teams spend significant effort locating relevant language across artifacts, mapping evidence to controls, and identifying gaps early enough to avoid delays. This project aims to reduce cognitive load by producing **traceable** answers and **repeatable** control-to-evidence assessments—without claiming to replace human assessors.

### Research questions
1) **RAG grounding and traceability:** How accurately can the system retrieve the right control text and policy evidence and generate answers faithful to retrieved sources?  
2) **Assessment usefulness:** Can the assistant classify control coverage (covered/partial/missing/unknown) and generate actionable remediation guidance with citations?  
3) **Feasibility:** Can an OSCAL + Docling + hybrid retrieval architecture deliver a reliable prototype within an 8-week compressed schedule?

---

## 3. Data

This project uses document-based data rather than a traditional row/column dataset.

### 3.1 Data sources

**A) Authoritative standards and references (public)**
- NIST SP 800-37 Rev. 2 (RMF lifecycle)
- NIST SP 800-53 Rev. 5 (security + privacy controls)
- NIST OSCAL content (machine-readable control catalogs/baselines)
- DoDI 8510.01 (DoD RMF policy anchor)
- Optional: DoD AI Cybersecurity RM Tailoring Guide (for AI-specific discussion)

**B) User policy corpus (synthetic for demo)**
- A curated set of **synthetic “policy-like” PDF artifacts**, clearly labeled as synthetic, used to simulate a program’s policy repository.

> **Why synthetic?** Real DoD program artifacts may be restricted (CUI/controlled). The synthetic corpus enables reproducible evaluation and safe external demonstration. All synthetic documents are labeled “Synthetic demo artifact — for academic use only.”

---

### 3.2 Synthetic Policy Corpus Generation (first-class dataset)

#### Synthetic corpus goals
- Provide realistic, policy-like content and structure
- Preserve headers, roles, frequencies, and tables so parsing/retrieval are meaningful
- Intentionally embed known gaps so assessment outputs can be evaluated

#### Synthetic document set (Policy Pack)
Target: **10–12 PDFs**, **3–15 pages each**:

1. Mini-SSP / System Overview (synthetic)
2. Access Control Policy
3. Audit & Accountability (Logging/Monitoring) Standard
4. Incident Response Plan
5. Configuration Management Policy
6. Change / Release Management SOP
7. Vulnerability Management SOP
8. Media Protection & Sanitization Policy
9. Privacy & Data Handling Policy
10. Supplier / C-SCRM Policy (ties to SR family)
11. Security Awareness & Training Policy *(optional)*
12. POA&M-style “Known Gaps” Tracker *(optional, useful for demo/eval)*

#### Scaffolding approach (to keep it grounded)
Synthetic policies are generated using:
- a consistent “organization + system profile” (fictional)
- policy templates (Purpose, Scope, Roles, Requirements, Procedures, Exceptions, Review cadence)
- control concepts derived from NIST 800-53 (via OSCAL) as *requirements scaffolding*  
- deliberate, documented omissions to create **covered/partial/missing** cases

#### Ground truth artifacts (critical for evaluation)
Alongside the PDFs, we create:

1) **Truth Table** (`controls_truth.csv` or `.json`)  
Columns:
- `control_id`
- `expected_coverage` ∈ {`covered`,`partial`,`missing`,`unknown`}
- `evidence_doc`
- `evidence_location` (page range or section path)
- `gap_notes` (what is intentionally missing)

2) **Golden Question Set** (`golden_questions.jsonl`)  
Fields:
- `question`
- `expected_controls` (list)
- `expected_evidence_refs`
- `expected_answer_type` (qa|assessment)
- `expected_coverage` (when applicable)

---

### 3.3 Data size (MB, GB, etc.)
- Synthetic Policy Pack: expected **50MB–500MB** depending on page counts and embedded tables
- OSCAL content: varies; typically tens to hundreds of MB depending on scope

*(TODO: record exact sizes after data creation)*

### 3.4 Data shape (# rows / # columns)
Not naturally tabular. After ingestion, documents are converted into a “tidy” chunk dataset:

- Rows ≈ number of chunks (estimated **5,000–50,000**)
- Columns = chunk text + metadata + retrieval features (embeddings, BM25 terms)

### 3.5 Time period
Not time-series. Optional metadata includes doc version and effective date.

### 3.6 What does each row represent?
Each row represents a single **semantic chunk**:
- either a user policy chunk (from Docling → Markdown → header-based chunking)
- or a control record chunk (from OSCAL)

### 3.7 Data dictionary (post-ingestion chunk table)

| Column name | Type | Definition | Values |
|---|---|---|---|
| `source_type` | categorical | Source of text | `policy_pdf`, `oscal_control` |
| `doc_id` | string | Unique document identifier | filename/UUID |
| `doc_title` | string | Document title | text |
| `page_start` | int | Starting page (policy PDFs) | 1..N |
| `page_end` | int | Ending page (policy PDFs) | 1..N |
| `section_path` | string | Heading path | e.g., `2.1 Access Control` |
| `control_id` | string | Control ID (OSCAL) | e.g., `AC-2`, `IR-4` |
| `control_part` | categorical | Control field type | `statement`, `guidance`, `enhancement`, `parameter` |
| `chunk_text` | string | Chunk content | text |
| `embedding` | vector | Dense embedding vector | float[] |
| `bm25_terms` | text/list | Keyword search terms | tokens |
| `ingest_ts` | datetime | Ingestion time | datetime |

### 3.8 Target/label in ML model
- MVP RAG system: **no supervised target required**
- Evaluation uses labels from the Truth Table / Golden Question Set:
  - `expected_controls`
  - `expected_coverage`
  - `expected_evidence_refs`

### 3.9 Candidate features/predictors
If optional scoring/classification is added:
- embedding similarity scores
- BM25 scores
- metadata features (section headings, doc type, control family)

---

## 7. Conclusion

### Summary and potential application
This project delivers a prototype RMF assistant that:
- retrieves authoritative control text from OSCAL
- ingests policy PDFs using semantic parsing (Docling)
- supports evidence-cited Q&A
- generates structured, repeatable control gap analysis reports

### Limitations
- Synthetic policy corpus may not fully represent real-world artifact variability
- Not intended to automate authorization decisions; it is decision support
- Performance depends on document parsing quality and retrieval configuration

---

## 8. References

- NIST SP 800-37 Rev. 2 (RMF lifecycle): https://csrc.nist.gov/publications/detail/sp/800-37/rev-2/final
- NIST SP 800-53 Rev. 5 (controls): https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final
- OSCAL reference models: https://pages.nist.gov/OSCAL-Reference/
- OSCAL content repo: https://github.com/usnistgov/oscal-content
- DoDI 8510.01 (DoD RMF policy): https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodi/851001p.pdf
- Optional: DoD AI Cybersecurity RM Tailoring Guide: https://dodcio.defense.gov/Portals/0/Documents/Library/AI-CybersecurityRMTailoringGuide.pdf

---

# 8-Week Agile Roadmap (Compressed Schedule)

## Week 1 — Data foundations + Synthetic corpus design
- [ ] Clone OSCAL content and parse 800-53 controls into structured records
- [ ] Implement Docling pipeline (PDF → semantic markdown)
- [ ] Define synthetic org/system profile + policy templates
- [ ] Generate first 3–4 synthetic docs (Mini-SSP, AC policy, IR plan, Logging standard)

## Week 2 — Synthetic corpus completion + indexing
- [ ] Generate remaining synthetic policy PDFs (full Policy Pack)
- [ ] Create Truth Table v1 (`controls_truth.csv`)
- [ ] Build chunking by headers + metadata schema
- [ ] Stand up vector DB + BM25 and index OSCAL + policy chunks

## Week 3 — RAG Chat MVP
- [ ] Implement hybrid retrieval (BM25 + embeddings)
- [ ] Chat Q&A with citations and abstention when evidence is insufficient
- [ ] Basic UI in Streamlit

## Week 4 — Assessment Mode v1
- [ ] Control subset selection UI
- [ ] Per-control retrieval + coverage label + evidence citations
- [ ] Export report (markdown/JSON)

## Week 5 — Golden dataset + evaluation harness
- [ ] Create Golden Questions from Truth Table (30–50)
- [ ] Implement scoring rubric (context precision, faithfulness, coverage accuracy)
- [ ] Run baseline evaluation and record results

## Week 6 — Retrieval + parsing tuning
- [ ] Improve chunking strategy (header boundaries, table handling)
- [ ] Tune hybrid retrieval weights and optional reranking
- [ ] Re-run golden evaluation and compare deltas

## Week 7 — Demo hardening + deployment
- [ ] UI polish + error handling (no evidence, conflicting evidence)
- [ ] Docker Compose packaging
- [ ] External demo exposure (tunnel + access control)
- [ ] Demo runbook (how to reproduce)

## Week 8 — Final results + deliverables
- [ ] Final evaluation on golden set
- [ ] Final report write-up + charts/tables
- [ ] Slide deck
- [ ] Recorded demo video

