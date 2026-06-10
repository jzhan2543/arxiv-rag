# GCP setup runbook (Stage 3 prerequisite)

One-time, human-only setup (interactive console, billing, OAuth). Work through it top to bottom; at the end, the commented deploy steps in `.github/workflows/deploy.yml` can be uncommented and every push to `main` deploys to Cloud Run.

Everything below assumes the `gcloud` CLI (`brew install google-cloud-sdk`, then `gcloud auth login`). Console-only steps are marked.

## 0. Naming used throughout

| Thing | Value |
|---|---|
| Project id | `arxiv-rag-prod` (pick your own; must be globally unique) |
| Region | `us-central1` (Tier-1 pricing) |
| Deploy SA | `gh-deployer@arxiv-rag-prod.iam.gserviceaccount.com` |
| WIF pool / provider | `github-pool` / `github-provider` |
| GitHub repo | `jzhan2543/arxiv-rag` |

## 1. Project + billing (console)

1. https://console.cloud.google.com → New Project → `arxiv-rag-prod`.
2. Link a billing account (required even inside the free tier, per the Feb 2026 GCP change). 
3. **Budget alert**: Billing → Budgets & alerts → Create budget → scope to this project, amount **$1/month**, alert at 50/90/100% → email. This is the tripwire for "scale-to-zero stopped being $0".

## 2. Enable APIs

```sh
gcloud config set project arxiv-rag-prod
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com
```

## 3. Artifact Registry remote repo (GHCR proxy)

**Why:** Cloud Run can only deploy images stored in Artifact Registry — it cannot pull `ghcr.io/...` directly. Instead of double-pushing, create a *remote repository* that proxies GHCR; Cloud Run pulls through it, images keep living in GHCR.

```sh
gcloud artifacts repositories create ghcr-remote \
  --repository-format=docker \
  --location=us-central1 \
  --mode=remote-repository \
  --remote-docker-repo=https://ghcr.io \
  --description="pull-through proxy for GitHub Container Registry"
```

Deployable image path becomes:
`us-central1-docker.pkg.dev/arxiv-rag-prod/ghcr-remote/jzhan2543/arxiv-rag:<sha>`

(Public GHCR images need no upstream credentials. If the GH repo ever goes private, add an upstream auth secret to the remote repo.)

## 4. Least-privilege deploy service account

```sh
gcloud iam service-accounts create gh-deployer --display-name="GitHub Actions deployer"

gcloud projects add-iam-policy-binding arxiv-rag-prod \
  --member="serviceAccount:gh-deployer@arxiv-rag-prod.iam.gserviceaccount.com" \
  --role="roles/run.admin"

# Lets the deployer "act as" the Cloud Run runtime service account:
gcloud iam service-accounts add-iam-policy-binding \
  "$(gcloud projects describe arxiv-rag-prod --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
  --member="serviceAccount:gh-deployer@arxiv-rag-prod.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Read access through the AR proxy for the Cloud Run service agent is
# project-default; the deployer itself needs none (no docker push to AR).
```

## 5. Workload Identity Federation (keyless auth)

```sh
PROJECT_NUMBER=$(gcloud projects describe arxiv-rag-prod --format='value(projectNumber)')

gcloud iam workload-identity-pools create github-pool \
  --location=global --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global --workload-identity-pool=github-pool \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == 'jzhan2543/arxiv-rag'"

gcloud iam service-accounts add-iam-policy-binding \
  gh-deployer@arxiv-rag-prod.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/jzhan2543/arxiv-rag"
```

The attribute condition pins the trust to this one repo — tokens minted by any other repo's workflow are rejected.

## 6. GitHub repo configuration

```sh
gh secret set WIF_PROVIDER \
  --body "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
gh secret set WIF_SERVICE_ACCOUNT \
  --body "gh-deployer@arxiv-rag-prod.iam.gserviceaccount.com"
gh variable set GCP_PROJECT --body "arxiv-rag-prod"
```

The runtime env vars (`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`) must also reach the Cloud Run service — set them once at first deploy:

```sh
gcloud run services update arxiv-rag --region us-central1 \
  --set-env-vars "ANTHROPIC_API_KEY=...,VOYAGE_API_KEY=..."
```

(Cleaner v0.1: Secret Manager + `--set-secrets`.)

## 7. Flip the switch

1. Uncomment the `google-github-actions/auth` + `deploy-cloudrun` steps in `.github/workflows/deploy.yml`.
2. Push to `main`. Watch Actions: build → GHCR push → Cloud Run deploy.
3. Verify:
   ```sh
   URL=$(gcloud run services describe arxiv-rag --region us-central1 --format='value(status.url)')
   curl "$URL/healthz"                       # {"status":"ok"}
   curl -X POST "$URL/ask" -H 'content-type: application/json' \
     -d '{"question":"What four agentic design patterns does the Singh et al. survey identify?"}'
   ```
4. Cold start: check the `container/startup_latencies` metric (expect 3–5s; `--cpu-boost` is already in the deploy flags).
5. Confirm the budget alert is armed.

## v0.1 (deferred): decouple the index from the image

GCS bucket + gcsfuse volume mount (gen2) for the read-only index; ingestion becomes a Cloud Run Job (same image, command `python -m app.ingest`) triggered by `ingest.yml` via `gcloud run jobs execute`; the API stops baking `index.db` into the image. Cap: do NOT let the service write to a gcsfuse-mounted sqlite file — FUSE + sqlite locking is fragile; all writes stay in the job.
