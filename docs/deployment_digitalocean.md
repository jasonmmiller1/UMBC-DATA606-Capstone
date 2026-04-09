# DigitalOcean Deployment

Current deployment notes for Week 7 as of April 9, 2026.

This document describes the deployment path that the repo is prepared for today. It does not claim that a full fresh DigitalOcean deployment has already been executed and signed off on in this branch.

## What runs where

| Component | Recommended host | Persistence | Notes |
| --- | --- | --- | --- |
| Streamlit app | DigitalOcean App Platform web service | Ephemeral filesystem | Built from [`Dockerfile`](../Dockerfile). Rebuilds local chunk and BM25 artifacts on startup when configured to do so. |
| Qdrant vector store | Separate DigitalOcean Droplet with attached Block Storage | Persistent | Primary documented path for the demo. |
| Optional vector-store alternative | Qdrant Cloud | Managed persistence | Simpler operationally, but secondary in this repo. |
| Advanced vector-store alternative | DOKS with persistent volume | Persistent | Valid, but more infrastructure than this demo needs. |

Do not treat App Platform local storage as durable Qdrant storage. App Platform containers have ephemeral filesystems, so the persistent vector store must live elsewhere.

## Recommended deployment shape

- deploy the app as one App Platform web service built from [`Dockerfile`](../Dockerfile)
- deploy Qdrant separately on a Droplet with a mounted Block Storage volume
- seed Qdrant before putting the app endpoint in front of demo users
- point the app at Qdrant with `QDRANT_URL` and `QDRANT_API_KEY`

## Runtime configuration the repo already supports

The current runtime behavior comes from [`app/runtime.py`](../app/runtime.py) and [`scripts/start_container.py`](../scripts/start_container.py).

- `QDRANT_URL` takes precedence over `QDRANT_HOST` and `QDRANT_PORT`
- `QDRANT_API_KEY` is passed through to the Qdrant Python client
- `PREPARE_LOCAL_INDEXES_ON_START=1` rebuilds local chunk and BM25 assets inside the app container if needed
- `WAIT_FOR_QDRANT_ON_START=1` blocks startup until Qdrant is reachable
- `REQUIRE_QDRANT_COLLECTION=1` blocks startup until the configured collection exists
- `STARTUP_QDRANT_TIMEOUT_SECONDS` controls that startup wait window

For App Platform, prefer `QDRANT_URL`. The `QDRANT_HOST` and `QDRANT_PORT` pair remains useful for local Compose only.

## Prerequisites

- a DigitalOcean App Platform app or app spec based on [`.do/app.yaml`](../.do/app.yaml)
- a DigitalOcean Droplet for Qdrant in the same region as the app if practical
- a DigitalOcean Block Storage volume attached to that Droplet
- Docker installed on the Droplet
- a chosen collection name, currently `rmf_chunks`
- optional OpenRouter credentials if LLM mode will be shown

## Required app environment variables

Required on App Platform:

- `APP_ENV=production`
- `PORT=8501`
- `QDRANT_URL=http://<qdrant-host-or-private-ip>:6333`
- `QDRANT_COLLECTION=rmf_chunks`
- `PREPARE_LOCAL_INDEXES_ON_START=1`
- `WAIT_FOR_QDRANT_ON_START=1`
- `REQUIRE_QDRANT_COLLECTION=1`
- `STARTUP_QDRANT_TIMEOUT_SECONDS=180`

Expected secrets:

- `QDRANT_API_KEY`
- `OPENROUTER_API_KEY` if `LLM_BACKEND=openrouter`

Optional:

- `LLM_BACKEND=openrouter`
- `OPENROUTER_MODEL`

## Primary Qdrant path: Droplet + Block Storage

This is the preferred path for the capstone demo because it keeps Qdrant persistent without introducing Kubernetes.

### 1. Create the Droplet and volume

- Create a small Ubuntu Droplet for Qdrant.
- Create a Block Storage volume in the same region.
- Attach the volume to the Droplet.
- If you can use private networking between App Platform and the Droplet, prefer the Droplet private IP for `QDRANT_URL`.

DigitalOcean volume docs:

- https://docs.digitalocean.com/products/volumes/how-to/
- https://docs.digitalocean.com/products/volumes/support/

### 2. Mount the Block Storage volume

If the volume is not already formatted and mounted by your provisioning flow, mount it on the Droplet and keep the mount stable across reboots.

Example commands:

```bash
lsblk
sudo mkdir -p /mnt/qdrant
sudo mkfs.ext4 /dev/disk/by-id/scsi-0DO_Volume_<volume-name>
sudo mount -o defaults,nofail,discard,noatime /dev/disk/by-id/scsi-0DO_Volume_<volume-name> /mnt/qdrant
echo '/dev/disk/by-id/scsi-0DO_Volume_<volume-name> /mnt/qdrant ext4 defaults,nofail,discard,noatime 0 2' | sudo tee -a /etc/fstab
sudo mkdir -p /mnt/qdrant/storage
```

Only run `mkfs.ext4` the first time on a new empty volume. Skip that step if the volume already contains data.

### 3. Run Qdrant in Docker with persistent storage

Create a host-only env file for the Qdrant API key:

```bash
sudo install -d -m 700 /etc/qdrant
sudo sh -c 'printf "QDRANT__SERVICE__API_KEY=%s\n" "<generate-a-real-secret>" > /etc/qdrant/qdrant.env'
sudo chmod 600 /etc/qdrant/qdrant.env
```

Run the Qdrant container:

```bash
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  --env-file /etc/qdrant/qdrant.env \
  -p 6333:6333 \
  -v /mnt/qdrant/storage:/qdrant/storage \
  qdrant/qdrant:latest
```

Notes:

- publish `6333/tcp` because this repo uses Qdrant over HTTP
- do not publish `6334/tcp` unless you explicitly add a gRPC client later
- do not store the Qdrant API key in the repo or in [`.env.example`](../.env.example)

Qdrant configuration docs:

- https://qdrant.tech/documentation/guides/configuration/

### 4. Secure the Qdrant service

For a controlled demo, use at least these controls:

- keep SSH restricted to your admin IPs
- require a Qdrant API key
- keep `6333/tcp` private if you can use VPC/private networking
- if you must expose `6333/tcp` publicly, restrict it as narrowly as you can with a DigitalOcean Cloud Firewall or host firewall

Example host-level firewall commands with `ufw`:

```bash
sudo ufw allow from <admin-ip>/32 to any port 22 proto tcp
sudo ufw allow from <trusted-cidr> to any port 6333 proto tcp
sudo ufw enable
```

Do not assume a perfect public-source allowlist for App Platform unless you have validated it in your own account. If you cannot use private networking for the demo, the safe minimum is a strong API key plus the narrowest firewall rules you can maintain operationally.

### 5. Verify Qdrant health

Health check:

```bash
curl -fsS http://<qdrant-host>:6333/readyz \
  -H "api-key: <qdrant-api-key>"
```

Expected response: `healthz check passed`

List collections:

```bash
curl -fsS http://<qdrant-host>:6333/collections \
  -H "api-key: <qdrant-api-key>"
```

Check whether the demo collection exists:

```bash
curl -fsS http://<qdrant-host>:6333/collections/rmf_chunks/exists \
  -H "api-key: <qdrant-api-key>"
```

Get collection details after seeding:

```bash
curl -fsS http://<qdrant-host>:6333/collections/rmf_chunks \
  -H "api-key: <qdrant-api-key>"
```

Useful Qdrant API docs:

- https://api.qdrant.tech/v-1-12-x/api-reference/service/readyz
- https://api.qdrant.tech/v-1-10-x/api-reference/collections/get-collections

### 6. Seed the collection with the repo's current bootstrap path

This repo currently seeds Qdrant with [`scripts/bootstrap_demo_data.py`](../scripts/bootstrap_demo_data.py). That script:

1. rebuilds local chunk and BM25 assets if needed
2. waits for Qdrant
3. upserts the current demo corpus into `QDRANT_COLLECTION`

From a local Python environment:

```bash
source .venv/bin/activate
export QDRANT_URL="http://<qdrant-host>:6333"
export QDRANT_API_KEY="<not-in-repo>"
export QDRANT_COLLECTION="rmf_chunks"
python scripts/bootstrap_demo_data.py --force --seed-qdrant --wait-timeout-seconds 180
```

From the container image:

```bash
docker run --rm \
  --env-file .env \
  -e QDRANT_URL="http://<qdrant-host>:6333" \
  -e QDRANT_API_KEY="<not-in-repo>" \
  -e QDRANT_COLLECTION="rmf_chunks" \
  --entrypoint python \
  rmf-assistant:test \
  scripts/bootstrap_demo_data.py --force --seed-qdrant --wait-timeout-seconds 180
```

Expected output includes JSON with:

- `qdrant_target`
- `collection`
- `rows_upserted`

If the collection is missing, this script will create and populate it through the current indexing path.

### 7. Operate and recover the Qdrant service

Check status and logs:

```bash
docker ps --filter name=qdrant
docker logs --tail 100 qdrant
df -h /mnt/qdrant
```

Restart the service:

```bash
docker restart qdrant
```

Recover after container loss:

```bash
docker rm -f qdrant
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  --env-file /etc/qdrant/qdrant.env \
  -p 6333:6333 \
  -v /mnt/qdrant/storage:/qdrant/storage \
  qdrant/qdrant:latest
```

As long as the Block Storage volume is still mounted at `/mnt/qdrant/storage`, the Qdrant data survives container recreation.

## Deploy the app on App Platform

1. Build and rehearse the image locally if you have not already:

```bash
docker build --build-arg PRELOAD_EMBEDDING_MODEL=0 -t rmf-assistant:test .
```

2. Create the App Platform service from [`.do/app.yaml`](../.do/app.yaml) or the control panel.
3. Set runtime env vars:

```text
APP_ENV=production
PORT=8501
QDRANT_URL=http://<qdrant-host-or-private-ip>:6333
QDRANT_COLLECTION=rmf_chunks
PREPARE_LOCAL_INDEXES_ON_START=1
WAIT_FOR_QDRANT_ON_START=1
REQUIRE_QDRANT_COLLECTION=1
STARTUP_QDRANT_TIMEOUT_SECONDS=180
```

4. Add secrets in the App Platform UI:

```text
QDRANT_API_KEY=<not-in-repo>
OPENROUTER_API_KEY=<optional-not-in-repo>
```

5. Deploy only after the Qdrant `readyz` check passes and the collection has been seeded.

App Platform note: container files are ephemeral, so do not place Qdrant storage or long-lived uploaded policy data there.

Relevant App Platform docs:

- https://docs.digitalocean.com/products/app-platform/how-to/console/
- https://docs.digitalocean.com/products/app-platform/reference/app-spec/

## Health and smoke checks

Before sharing the app URL:

- verify Qdrant `readyz` succeeds
- verify `rmf_chunks` exists on Qdrant
- verify the seed script completed successfully against that same endpoint
- verify the App Platform root page loads
- verify one retrieval-only question succeeds
- verify one OpenRouter-backed question succeeds if LLM mode is enabled

Useful repo-level retrieval smoke check:

```bash
LLM_BACKEND=none python -m app.retrieval.retrieve "account management requirements"
```

Useful browser questions:

- `What does our access control policy say about least privilege?`
- `What does our incident response plan say about escalation?`

## Secondary alternative: Qdrant Cloud

If you do not want to manage a Droplet for the demo, Qdrant Cloud is the simpler secondary path.

- create the cluster in Qdrant Cloud
- copy its HTTPS endpoint into `QDRANT_URL`
- copy its API key into `QDRANT_API_KEY`
- run the same [`scripts/bootstrap_demo_data.py`](../scripts/bootstrap_demo_data.py) seed step
- keep the App Platform app configuration the same except for the Qdrant endpoint value

This branch has runtime support for that path through `QDRANT_URL` and `QDRANT_API_KEY`, but this repo does not add any Qdrant-Cloud-specific automation beyond those env vars.

## Advanced alternative: DOKS with persistent volume

This is a valid path if you already have Kubernetes in scope, but it is intentionally not the primary recommendation for the capstone demo.

- run Qdrant as a StatefulSet or equivalent
- back it with a persistent volume
- expose the Qdrant HTTP endpoint on `6333`
- set `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` in App Platform the same way
- seed with [`scripts/bootstrap_demo_data.py`](../scripts/bootstrap_demo_data.py)

## Troubleshooting

Symptom: App Platform service starts slowly or restarts

- confirm `QDRANT_URL` resolves from the app environment
- confirm `rmf_chunks` already exists if `REQUIRE_QDRANT_COLLECTION=1`
- increase `STARTUP_QDRANT_TIMEOUT_SECONDS` if Qdrant is healthy but slow to respond

Symptom: Qdrant `readyz` fails

- check `docker ps` and `docker logs --tail 100 qdrant` on the Droplet
- confirm the Block Storage volume is still mounted at `/mnt/qdrant/storage`
- confirm firewall rules still allow the intended traffic path

Symptom: app loads but answers fail

- verify the app is pointing at the same Qdrant endpoint you seeded
- rerun [`scripts/bootstrap_demo_data.py`](../scripts/bootstrap_demo_data.py) against that endpoint
- verify `QDRANT_COLLECTION` matches the collection you actually seeded

Symptom: uploaded demo files disappear after restart

- this is expected on App Platform without external storage
- the current deployment path assumes the committed demo corpus, not durable user uploads

## Current known gaps / assumptions

- This branch documents a concrete Qdrant path, but it does not claim a fresh end-to-end App Platform plus Droplet deployment has already been executed in this exact branch state.
- Retrieval-vs-LLM comparison is not fully validated unless the current environment can run both retrieval-only and OpenRouter paths without backend-visible failures.
- Uploaded files are ephemeral on DigitalOcean App Platform unless external storage is added.
- The current UI can surface raw upload/ingest exception text, which remains a demo risk if you perform live uploads.
