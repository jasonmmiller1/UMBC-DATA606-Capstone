# DigitalOcean Demo Deployment

This repo now supports a deployment-oriented container flow for a controlled demo. The app container is responsible for serving Streamlit and building its local BM25/chunk artifacts at startup. Qdrant is treated as a separate dependency that must already be reachable and seeded before the web service is considered ready.

## Recommended deployment shape

- Deploy the Streamlit app as a single DigitalOcean App Platform web service built from [`Dockerfile`](../Dockerfile).
- Keep Qdrant outside the app container.
- Seed Qdrant with the demo corpus before exposing the app.

Why this split:

- App Platform web-service filesystem is ephemeral, so Qdrant storage should not live inside the app container.
- The repo includes the source corpus needed to rebuild `chunks.parquet` and the BM25 index on container start.
- The vector collection should live in a durable service and survive app restarts/redeploys.

## Runtime assumptions

The container image includes:

- app code under `app/`
- Streamlit entrypoint [`app.py`](../app.py)
- committed source data:
  - [`data/oscal_parsed/controls_80053.parquet`](../data/oscal_parsed/controls_80053.parquet)
  - synthetic markdown corpus under [`data/policies_synth_md_v2`](../data/policies_synth_md_v2)

The container does not include:

- `.env`
- local Qdrant storage
- uploaded files from prior runs
- raw standards PDFs or the raw OSCAL clone
- a pre-seeded Qdrant collection

At startup, [`scripts/start_container.py`](../scripts/start_container.py):

1. builds `data/index/chunks.parquet` if missing
2. builds `data/bm25_index/bm25_index.pkl` if missing
3. waits for Qdrant reachability, and by default waits for the configured collection to exist
4. starts Streamlit on `0.0.0.0:$PORT`

## Required environment variables

Required for all container deployments:

- `QDRANT_URL` or `QDRANT_HOST` + `QDRANT_PORT`
- `QDRANT_COLLECTION`

Required only when using a secured Qdrant endpoint:

- `QDRANT_API_KEY`

Required only when enabling LLM responses:

- `LLM_BACKEND=openrouter`
- `OPENROUTER_MODEL`
- `OPENROUTER_API_KEY`

Recommended runtime values for the demo:

- `APP_ENV=production`
- `PORT=8501`
- `PREPARE_LOCAL_INDEXES_ON_START=1`
- `WAIT_FOR_QDRANT_ON_START=1`
- `REQUIRE_QDRANT_COLLECTION=1`
- `STARTUP_QDRANT_TIMEOUT_SECONDS=180`

## Seeding Qdrant

Use the same image to seed the demo corpus into Qdrant:

```bash
python scripts/bootstrap_demo_data.py --force --seed-qdrant --wait-timeout-seconds 180
```

That command:

- rebuilds local chunk/BM25 artifacts from committed source data
- waits for the configured Qdrant endpoint
- upserts the demo corpus into `QDRANT_COLLECTION`

For App Platform, the simplest controlled-demo approach is:

1. deploy or expose Qdrant first
2. run the bootstrap command once against that Qdrant target
3. deploy the web service

You can keep this as a manual one-off step, or convert it into a deploy-time job later if you want a more automated rollout.

## Storage expectations

- Streamlit uploads are stored in the container filesystem under `data/uploads_*`.
- Those uploads are ephemeral in App Platform and disappear on restart/redeploy.
- The committed synthetic corpus is always present in the image.
- Local BM25/chunk artifacts are reproducible and regenerated as needed.
- Qdrant should be considered the only durable retrieval store for deployed demos.

## DigitalOcean app spec template

Use [`.do/app.yaml`](../.do/app.yaml) as a starting point.

Update these placeholders before use:

- `github.repo`
- `github.branch`
- `QDRANT_URL` or `QDRANT_HOST`/`QDRANT_PORT`
- optional OpenRouter settings

Add secrets in the DigitalOcean UI or encrypt them before committing an app spec update:

- `QDRANT_API_KEY`
- `OPENROUTER_API_KEY`

## Local container demo path

For local reproducibility:

```bash
docker compose up --build
```

This starts:

- `qdrant`
- `qdrant-bootstrap` to seed the demo collection
- `app`

Then open `http://localhost:8501`.
