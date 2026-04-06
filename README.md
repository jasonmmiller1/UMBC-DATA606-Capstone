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
This is the quickest local container path for a reproducible demo:

```bash
docker compose up --build
```

What that does:
- builds the app image
- starts Qdrant locally
- seeds the `rmf_chunks` collection with the committed demo corpus
- starts the Streamlit app on `http://localhost:8501`

Deployment-oriented notes:
- The app image rebuilds local chunk/BM25 artifacts on startup from committed source data.
- Qdrant remains a separate service dependency and is not embedded into the app container.
- Uploaded files remain local to the running container unless you add external storage later.

Host-Python bring-up is still available for development. This was the workflow validated in this repository on **March 16, 2026**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env

docker compose up -d
python scripts/bootstrap_demo_data.py --force --seed-qdrant
python -m app.retrieval.run_week2_tests

LLM_BACKEND=none python scripts/run_week5_eval.py \
  --input-path data/eval/golden_questions.jsonl \
  --output-jsonl /tmp/rmf_week5_eval_results.jsonl \
  --summary-path /tmp/rmf_week5_eval_summary.md

python -m unittest discover -s tests -v
python -m streamlit run app.py
```

Notes:
- The repo already includes parsed OSCAL data and the synthetic policy corpus, and the local BM25/chunk artifacts can now be rebuilt from that committed source data, so you do **not** need the raw OSCAL clone for normal local bring-up.
- `docker compose up --build` is now the preferred containerized demo path.
- For host-Python runs, use `python scripts/bootstrap_demo_data.py --force --seed-qdrant` to populate the live `rmf_chunks` collection.
- `LLM_BACKEND=none` is the safest first-run mode. Switch to `openrouter` only after adding a real `OPENROUTER_API_KEY` and model to `.env`.
- DigitalOcean deployment guidance lives in [`docs/deployment_digitalocean.md`](docs/deployment_digitalocean.md).

### 1.6) LLM mode switching

Retrieval-only mode:

```bash
LLM_BACKEND=none python -m streamlit run app.py
```

Retrieval + OpenRouter mode:

```bash
LLM_BACKEND=openrouter \
OPENROUTER_MODEL="<paid-model-id>" \
OPENROUTER_API_KEY="<your-key>" \
python -m streamlit run app.py
```

Optional OpenRouter tuning env vars:
- `OPENROUTER_FALLBACK_MODELS` for model fallback order
- `OPENROUTER_TIMEOUT_SECONDS` for request timeout
- `OPENROUTER_RETRY_COUNT` for bounded retries
- `OPENROUTER_MAX_TOKENS` and `OPENROUTER_TEMPERATURE` for generation controls

The app keeps retrieval grounding in front of the LLM and degrades to retrieval-only fallback when OpenRouter is unavailable, misconfigured, rate-limited, or otherwise fails.

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
# 1) Start local Qdrant
docker compose up -d qdrant

# 2) Build chunks + BM25 and index into Qdrant
python scripts/bootstrap_demo_data.py --force --seed-qdrant

# 3) Test hybrid retrieval
python -m app.retrieval.retrieve "account management requirements"

# 4) Run week 2 query set and save reproducible outputs
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

Focused demo comparison workflow:

```bash
LLM_BACKEND=openrouter \
OPENROUTER_MODEL="<paid-model-id>" \
OPENROUTER_API_KEY="<your-key>" \
python scripts/run_demo_mode_comparison.py \
  --input-path data/eval/demo_questions.jsonl \
  --output-json data/eval/demo_mode_comparison.json \
  --output-md data/eval/demo_mode_comparison.md
```

This comparison runner always executes the same curated demo questions in:
- retrieval-only mode
- retrieval + OpenRouter mode

It records:
- answer state and abstention behavior
- citation counts and weak-grounding risk flags
- backend/model/fallback metadata
- end-to-end latency
- per-question notes on whether OpenRouter improved or hurt the demo experience

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
