# DigitalOcean Deployment

Current deployment notes for Week 7 as of April 9, 2026.

This document describes the deployment path that the repo is prepared for today. It does not claim that a full App Platform deployment has already been executed and signed off on in this branch.

## Recommended deployment shape

- deploy the app as one App Platform web service built from [`Dockerfile`](../Dockerfile)
- keep Qdrant outside the app container
- seed Qdrant before putting the app in front of demo users

Why this shape:

- App Platform service storage is ephemeral
- the app container can rebuild its local BM25/chunk artifacts from committed source data
- Qdrant data should survive app restarts and redeploys

## Prerequisites

- a reachable Qdrant endpoint
- a decided `QDRANT_COLLECTION` name, currently `rmf_chunks`
- a plan for how Qdrant will be hosted for the demo
- optional OpenRouter credentials if LLM mode will be shown
- a prepared App Platform spec based on [`.do/app.yaml`](../.do/app.yaml)

## Required environment variables

Required:

- `APP_ENV=production`
- `PORT=8501`
- `QDRANT_COLLECTION=rmf_chunks`
- either `QDRANT_URL` or `QDRANT_HOST` plus `QDRANT_PORT`
- `PREPARE_LOCAL_INDEXES_ON_START=1`
- `WAIT_FOR_QDRANT_ON_START=1`
- `REQUIRE_QDRANT_COLLECTION=1`
- `STARTUP_QDRANT_TIMEOUT_SECONDS=180`

Optional but expected when needed:

- `QDRANT_API_KEY`
- `LLM_BACKEND=openrouter`
- `OPENROUTER_MODEL`
- `OPENROUTER_API_KEY`

## What the image assumes

Included in the image:

- application code
- Streamlit app entrypoint
- committed OSCAL parquet
- committed synthetic markdown corpus

Not included in the image:

- `.env`
- any prior uploaded files
- local Qdrant storage
- a pre-seeded Qdrant collection
- raw standards PDFs or the raw OSCAL clone

At startup the app container:

1. rebuilds `data/index/chunks.parquet` if needed
2. rebuilds `data/bm25_index/bm25_index.pkl` if needed
3. waits for Qdrant, and by default waits for the configured collection
4. starts Streamlit on `0.0.0.0:$PORT`

## Deployment path

1. Choose where Qdrant will live.
2. Make the Qdrant endpoint reachable from App Platform.
3. Seed the demo corpus into Qdrant.
4. Create the App Platform service from [`.do/app.yaml`](../.do/app.yaml) or equivalent UI settings.
5. Add runtime env vars and secrets in DigitalOcean.
6. Deploy the app service.
7. Run smoke checks against the external endpoint.

## Seeding Qdrant

Use the same container image or a local Python environment to seed the demo corpus:

```bash
python scripts/bootstrap_demo_data.py --force --seed-qdrant --wait-timeout-seconds 180
```

Expected result:

- local retrieval assets rebuild
- Qdrant becomes reachable
- the configured collection receives the demo corpus

## Container build and local rehearsal

Build the image:

```bash
docker build --build-arg PRELOAD_EMBEDDING_MODEL=0 -t rmf-assistant:test .
```

Rehearse the app against a reachable Qdrant endpoint:

```bash
docker run --rm -p 8501:8501 \
  --env-file .env \
  -e QDRANT_URL="http://<qdrant-host>:6333" \
  -e QDRANT_COLLECTION="rmf_chunks" \
  rmf-assistant:test
```

For a full local container demo, prefer:

```bash
docker compose up --build
```

## Health and smoke checks

Before sharing the external URL:

- verify the app root page loads
- verify one retrieval-only question succeeds
- verify one OpenRouter-backed question succeeds if LLM mode is enabled
- verify the Qdrant seed command completed successfully

Useful checks:

```bash
python scripts/bootstrap_demo_data.py --force --seed-qdrant --wait-timeout-seconds 180
```

```bash
LLM_BACKEND=none python -m app.retrieval.retrieve "account management requirements"
```

In the browser:

- ask `What does our access control policy say about least privilege?`
- if OpenRouter is enabled, ask `What does our incident response plan say about escalation?`

## Troubleshooting

Symptom: App Platform service starts slowly or restarts

- confirm Qdrant is reachable from the service
- confirm the collection already exists if `REQUIRE_QDRANT_COLLECTION=1`
- increase `STARTUP_QDRANT_TIMEOUT_SECONDS` if needed

Symptom: app loads but answers fail

- verify the Qdrant endpoint and collection name
- rerun the seed step against the same endpoint

Symptom: OpenRouter mode looks degraded

- verify key, model, and provider availability
- if the provider is unstable, run the demo in retrieval-only mode instead of forcing LLM mode

Symptom: uploaded demo files disappear after restart

- this is expected on App Platform without external storage

## Current known gaps / assumptions

- Retrieval-vs-LLM comparison is not fully validated unless the current environment can run both retrieval-only and OpenRouter paths without backend-visible failures.
- Qdrant hosting choice is still pending if you have not already selected a durable external host for the demo.
- Uploaded files are ephemeral on DigitalOcean App Platform unless external storage is added.
- The current UI can surface raw upload/ingest exception text, which remains a demo risk if you perform live uploads.
