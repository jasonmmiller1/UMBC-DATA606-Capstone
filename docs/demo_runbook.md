# Demo Runbook

Current Week 7 runbook as of April 9, 2026.

This document is intentionally operational and conservative. It reflects the local demo paths that exist in this repo today. It does not claim end-to-end DigitalOcean validation, and it does not assume a live OpenRouter key is currently configured.

## What has been validated on this branch

Validated locally on `feat/week7-demo-hardening`:

- Python files compile.
- `.venv/bin/python -m unittest discover -s tests -v` passes.
- `docker build --build-arg PRELOAD_EMBEDDING_MODEL=0 -t rmf-assistant:test .` succeeds.
- `docker run --rm --entrypoint python rmf-assistant:test scripts/bootstrap_demo_data.py --force` succeeds.

Not claimed as already validated here:

- a fresh end-to-end OpenRouter demo with a live paid model/key
- a full DigitalOcean App Platform deployment with external Qdrant

## Prerequisites

- Python 3.10 with project dependencies installed in `.venv`, or Docker with Compose
- a reachable Qdrant instance
- the demo corpus seeded into Qdrant
- for a remote demo, Qdrant should run on a separate persistent host such as a DigitalOcean Droplet + Block Storage or Qdrant Cloud
- `.env` created from [`.env.example`](../.env.example) for local runs
- optional: a valid `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` for LLM-backed runs

## Required environment variables

Always required:

- `LLM_BACKEND`
- `QDRANT_COLLECTION`
- either `QDRANT_URL` or `QDRANT_HOST` plus `QDRANT_PORT`

Required for local host-Python startup:

- `BM25_INDEX_PATH`
- `CHUNKS_PATH`

Required only for OpenRouter-backed runs:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`

Useful startup controls:

- `PREPARE_LOCAL_INDEXES_ON_START=1`
- `WAIT_FOR_QDRANT_ON_START=1`
- `REQUIRE_QDRANT_COLLECTION=1`
- `STARTUP_QDRANT_TIMEOUT_SECONDS=120`

## Pre-demo checklist

- `.env` is present and contains the intended demo values
- Qdrant is reachable
- `curl -fsS "$QDRANT_URL/readyz" -H "api-key: $QDRANT_API_KEY"` succeeds when using a remote Qdrant with API key
- Qdrant is seeded into `QDRANT_COLLECTION`
- the `QDRANT_COLLECTION` exists on that same Qdrant endpoint
- the app healthcheck passes for the containerized path you plan to show
- the app home page loads
- one retrieval-only smoke question succeeds
- one OpenRouter smoke question succeeds if LLM mode will be shown
- the public or shared endpoint is reachable from outside your laptop if using a remote deployment

## Local startup path

Preferred local container path:

```bash
docker compose up --build
```

Expected result:

- `qdrant` starts locally
- `qdrant-bootstrap` rebuilds local retrieval assets and seeds Qdrant
- `app` starts on `http://localhost:8501`

Host-Python path:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d qdrant
python scripts/bootstrap_demo_data.py --force --seed-qdrant --wait-timeout-seconds 120
LLM_BACKEND=none python -m streamlit run app.py
```

## OpenRouter-enabled startup path

Only use this path after confirming a working API key and model id outside the repo.

Host-Python:

```bash
source .venv/bin/activate
export LLM_BACKEND=openrouter
export OPENROUTER_MODEL="<paid-model-id>"
export OPENROUTER_API_KEY="<not-in-repo>"
python -m streamlit run app.py
```

Containerized local demo with Compose:

- put the OpenRouter variables in `.env`
- run `docker compose up --build`
- verify the app caption changes from retrieval-only to retrieval + OpenRouter

## Container build/run path

Build:

```bash
docker build --build-arg PRELOAD_EMBEDDING_MODEL=0 -t rmf-assistant:test .
```

Local asset bootstrap inside the image:

```bash
docker run --rm --entrypoint python rmf-assistant:test scripts/bootstrap_demo_data.py --force
```

Direct app run against an already reachable external Qdrant:

```bash
docker run --rm -p 8501:8501 \
  --env-file .env \
  -e QDRANT_URL="http://<qdrant-host>:6333" \
  -e QDRANT_COLLECTION="rmf_chunks" \
  rmf-assistant:test
```

Use Compose instead when Qdrant is also local.

## Health / smoke checks

Qdrant seed check:

```bash
python scripts/bootstrap_demo_data.py --force --seed-qdrant --wait-timeout-seconds 120
```

Retrieval-only smoke check:

```bash
LLM_BACKEND=none python -m app.retrieval.retrieve "account management requirements"
```

Container healthcheck script:

```bash
docker run --rm --entrypoint python rmf-assistant:test scripts/container_healthcheck.py
```

Remote Qdrant collection check:

```bash
curl -fsS "$QDRANT_URL/collections/$QDRANT_COLLECTION/exists" \
  -H "api-key: $QDRANT_API_KEY"
```

App smoke check in browser:

- load `http://localhost:8501`
- ask `What does our access control policy say about least privilege?`
- confirm a grounded answer appears with citations

OpenRouter smoke check in browser:

- run with `LLM_BACKEND=openrouter`
- ask `What does our incident response plan say about escalation?`
- confirm the response still includes citations and the mode caption indicates OpenRouter

## Demo question flow

Suggested happy-path sequence from [`data/eval/demo_questions.jsonl`](../data/eval/demo_questions.jsonl):

1. `What does our access control policy say about least privilege?`
2. `Summarize the requirement in AC-2.`
3. `How well do our policies cover PL-2 based on the mini SSP?`

What each one shows:

- policy answer with direct evidence
- framework/control retrieval without needing a policy upload
- policy-vs-control comparison with disciplined coverage framing

## Failure / edge-case demo flow

Use these when you want to show guarded behavior instead of just the happy path:

1. `Does our access control policy define how often access recertification must occur?`
2. `Where is privileged session recording mandated in our PAM standard?`

What to look for:

- the first question should hedge because the synthetic policy intentionally leaves that frequency undefined
- the second question should abstain or respond cautiously because the demo corpus does not include a PAM standard

If OpenRouter is unstable during the demo:

- switch `LLM_BACKEND=none`
- restart the app if needed
- continue with the retrieval-only path instead of improvising around provider failures

## Troubleshooting

Symptom: app loads but answers fail immediately

- verify Qdrant is reachable
- verify the remote Qdrant host is the separate persistent service you intended, not an ephemeral app container filesystem
- rerun `python scripts/bootstrap_demo_data.py --force --seed-qdrant --wait-timeout-seconds 120`

Symptom: local retrieval smoke test returns no useful chunks

- confirm `data/index/chunks.parquet` and `data/bm25_index/bm25_index.pkl` were rebuilt
- confirm the target Qdrant collection is `rmf_chunks` unless intentionally changed

Symptom: OpenRouter mode is selected but behaves like fallback

- check `OPENROUTER_API_KEY`
- check `OPENROUTER_MODEL`
- expect provider-side timeout, rate-limit, quota, or model-availability issues to fall back cleanly rather than fail silently

Symptom: upload demo goes badly on stage

- stop using live uploads and return to the committed demo corpus
- the current UI can surface raw ingest exception text

Symptom: container starts slowly

- first start may spend time rebuilding local retrieval assets
- image build time increases if embedding-model preload is enabled

## Current known gaps / assumptions

- Retrieval-vs-LLM comparison is not fully validated unless the current environment can run both retrieval-only and OpenRouter paths without backend-visible failures.
- The primary documented remote path is App Platform for the app plus a separate persistent Qdrant host. Qdrant Cloud remains a valid secondary option, but is not claimed as already validated in this branch.
- Uploaded files are ephemeral on DigitalOcean App Platform unless external storage is added.
- The current Streamlit upload/ingest flow can show raw exception text, which is a demo-risk UI issue for live uploads.
