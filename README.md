# RMF Assistant — Evidence-Grounded LLM for DoD RMF Policy Q&A + Control Gap Analysis  
**(OSCAL + Docling + Hybrid RAG | UMBC DATA 606 Capstone)**


## **NOTE --- WORK IN PROGRESS SPRING 2026 COMPLETION --- NOTE**

This repository contains a demo-quality prototype that supports the Department of Defense (DoD) Risk Management Framework (RMF) workflow by enabling:

1) **Policy Q&A** over uploaded policy documents with **source-cited answers**, and  
2) **Control-to-evidence gap analysis** comparing policy text to a selected subset of **NIST SP 800-53 Rev. 5** controls.

> **Note:** This project includes a **synthetic policy corpus** for safe, reproducible academic demonstration. No real DoD program artifacts are used.

---

## Project Overview

### What it does
- Ingests **authoritative control text** from **NIST OSCAL** (machine-readable 800-53 Rev.5)
- Converts **policy PDFs** into structured Markdown using **Docling** (preserves headings/tables where possible)
- Prepares data for a **Hybrid Retrieval-Augmented Generation (RAG)** workflow:
  - Retrieve relevant **controls + policy evidence**
  - Generate responses with **traceability** and (later) **coverage classification**: `covered | partial | missing | unknown`

### Why it matters
RMF compliance work is complex and documentation-heavy. Teams spend significant effort locating relevant language across artifacts, mapping evidence to controls, and identifying gaps early enough to avoid delays. This prototype aims to reduce cognitive load with **repeatable, auditable** analysis—without claiming to replace human assessors.

---

## Repository Structure

```
app/                # application code (services, loaders, generators)
data/
  oscal_raw/        # OSCAL source location (raw repo clone NOT committed)
  oscal_parsed/     # parsed control records (committed for reproducible demo)
  policies_synth_md/        # system_profile.yaml + docling test output
  policies_synth_md_v2/     # synthetic policy pack (markdown)
  truth_table/      # evaluation truth table schema / starter rows
docs/
  templates/        # policy templates used to generate synthetic corpus
  references.md     # pinned OSCAL commit hash + data sources
notebooks/          # validation + experiments (Week 1 smoketest, etc.)
```

---

## Data Sources

### Authoritative (public)
- **NIST SP 800-53 Rev. 5** via **OSCAL JSON** (preferred ground truth)
- Optional supporting references (not committed as PDFs):
  - NIST SP 800-37 Rev. 2
  - NIST SP 800-161 Rev. 1
  - DoDI 8510.01

### Synthetic policy corpus (demo dataset)
The repo includes a synthetic, policy-like set of documents generated from templates and a consistent fictional system profile:

- `data/policies_synth_md/system_profile.yaml`  
- `docs/templates/*.md` (templates)
- `data/policies_synth_md_v2/*.md` (generated corpus)

All synthetic documents are labeled:

> **SYNTHETIC DEMO ARTIFACT — ACADEMIC USE ONLY**

---

## Setup

### 1) Create the environment
If you have `environment.yml`:

```bash
conda env create -f environment.yml
conda activate rmf-assistant
```

Or using pip (`requirements.txt`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`environment.yml` was exported from a working Linux `aarch64` environment and pins `python=3.10.19`. For the most portable restore path on a fresh machine, prefer the `requirements.txt` + `.venv` flow above.

### 1.5) Current developer bring-up
This is the workflow validated in this repository on **March 16, 2026**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env

docker compose up -d
./scripts/index_offline.sh
python -m app.retrieval.run_week2_tests

LLM_BACKEND=none python scripts/run_week5_eval.py \
  --input-path data/eval/golden_questions.jsonl \
  --output-jsonl /tmp/rmf_week5_eval_results.jsonl \
  --summary-path /tmp/rmf_week5_eval_summary.md

python -m unittest discover -s tests -v
python -m streamlit run app.py
```

Notes:
- The repo already includes parsed OSCAL data, the synthetic policy corpus, and local BM25/chunk artifacts, so you do **not** need the raw OSCAL clone for normal local bring-up.
- `docker compose up -d` starts Qdrant, but the live `rmf_chunks` collection still needs `./scripts/index_offline.sh` to populate it.
- `LLM_BACKEND=none` is the safest first-run mode. Switch to `openrouter` only after adding a real `OPENROUTER_API_KEY` and model to `.env`.

### 2) (VS Code) Register the Jupyter kernel
```bash
python -m ipykernel install --user --name rmf-assistant --display-name "Python (rmf-assistant)"
```

---

## Quickstart: Week 1 Proof (Smoke Test)

Run the notebook:

- `notebooks/week1_smoketest.ipynb`

It verifies:
- `data/oscal_parsed/controls_80053.parquet` loads correctly
- synthetic policy pack exists
- Docling conversion output exists
- prints sample control records and file summaries

---

## Key Capabilities (Current)

### ✅ OSCAL parsing (NIST 800-53 Rev.5)
Outputs:
- `data/oscal_parsed/controls_80053.parquet`
- `data/oscal_parsed/controls_80053.jsonl`

### ✅ Synthetic policy pack generation
Templates:
- `docs/templates/*_template.md`

Generator:
- `app/services/generate_synth_policies.py`

Outputs:
- `data/policies_synth_md_v2/01_mini_ssp.md`
- `data/policies_synth_md_v2/02_access_control_policy.md`
- `data/policies_synth_md_v2/03_incident_response_plan.md`
- `data/policies_synth_md_v2/04_logging_monitoring_standard.md`

### ✅ PDF → Markdown ingestion (Docling)
- `app/services/pdf_to_md.py`
- Test output:
  - `data/policies_synth_md/_docling_test.md`

### ✅ Week 2 retrieval pipeline (Qdrant + local BM25 + RRF)
Modules:
- `app/ingest/chunkers.py`
- `app/index/qdrant_schema.py`
- `app/index/index_to_qdrant.py`
- `app/index/bm25_index.py`
- `app/retrieval/retrieve.py`
- `app/retrieval/run_week2_tests.py`

Quick run:
```bash
# 1) Start vector DB
docker compose up -d

# 2) Build chunks + index into Qdrant (offline model cache mode)
./scripts/index_offline.sh

# 3) Build local BM25 index
python -m app.index.bm25_index --build

# 4) Test hybrid retrieval
python -m app.retrieval.retrieve "account management requirements"

# 5) Run week 2 query set and save reproducible outputs
python -m app.retrieval.run_week2_tests
```

---

## Evaluation Artifacts (In Progress)

### Truth Table (ground truth for coverage classification)
- `data/truth_table/controls_truth_schema.csv`
  - `control_id`
  - `expected_coverage` ∈ {covered, partial, missing, unknown}
  - `evidence_doc`
  - `evidence_location`
  - `gap_notes`

Planned:
- Golden question set (`golden_questions.jsonl`) for Q&A + assessment evaluation

### Week 5 Eval Run Modes (Reproducible)

For stable/reproducible runs, explicitly set backend/model in the shell before running eval.

Run eval with LLM disabled (retrieval-only mode):

```bash
LLM_BACKEND=none python scripts/run_week5_eval.py \
  --input-path data/eval/golden_questions.jsonl \
  --summary-path data/eval/week5_none_summary.md
```

Run eval with OpenRouter paid model:

```bash
LLM_BACKEND=openrouter \
OPENROUTER_MODEL="<paid-model-id>" \
OPENROUTER_API_KEY="<your-key>" \
python scripts/run_week5_eval.py \
  --input-path data/eval/golden_questions.jsonl \
  --summary-path data/eval/week5_paid_model_summary.md
```

Why paid models for eval reliability:
- `:free` models are more likely to hit provider limits (429), transient upstream failures (502), or malformed responses.
- paid models generally provide steadier throughput and fewer fallback/retry events, which improves eval consistency.

---

## Roadmap (8-week compressed plan)

- **In Progress - Weeks 1–2:** data pipeline (OSCAL parse + Docling ingest), synthetic corpus + truth table
- **TO-DO Weeks 3–5:** indexing + retrieval (hybrid search: keyword + embeddings), citations/traceability
- **TO-DO Weeks 6–7:** gap analysis agent + evaluation (faithfulness, relevance, retrieval accuracy)
- **TO-DO Week 8:** Streamlit demo app + packaging for external demonstration

---

## Notes on Reproducibility

This repo intentionally **does not** commit:
- the raw OSCAL repo clone (`data/oscal_raw/oscal-content/`)
- downloaded standards PDFs (`data/standards_raw/`)
- vector DB storage volumes / embeddings
- model weights and checkpoints

Instead, it commits:
- the small parsed OSCAL dataset used in the demo (`controls_80053.parquet`)
- synthetic policy corpus (Markdown)
- templates + generator scripts
- notebooks + evaluation scaffolding

See `docs/references.md` for pinned source info.

---

## Author

**Jason Miller**  
UMBC Data Science Master’s — Capstone (DATA 606)  
GitHub: https://github.com/jasonmmiller1/UMBC-DATA606-Capstone  
LinkedIn: [(LinkedIn)](https://www.linkedin.com/in/jasonmmiller/)

---

## License / Disclaimer

This is an academic project. The synthetic corpus is fictional and intended only for demonstration and evaluation. Nothing in this repository should be interpreted as official compliance guidance.
