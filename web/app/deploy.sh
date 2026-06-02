#!/usr/bin/env bash
# Deploy the predicators browser POC to a public GCS bucket.
#
# One-time bucket setup is idempotent (create + public-read IAM +
# website config). Per-deploy, we stage only what the browser needs
# (app/, wheels/, predicators_assets, and the two node_modules
# packages the importmap resolves), with symlinks resolved, then
# rsync the stage tree into the bucket.
#
# After deploy, open:
#   https://storage.googleapis.com/$BUCKET/app/index.html
#
# Env overrides:
#   PROJECT  (default: mara-452721)
#   BUCKET   (default: mara-predicators-web)  -- positional arg also accepted
#   LOCATION (default: us-central1)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WEB_DIR="$REPO_ROOT/web"

PROJECT="${PROJECT:-mara-452721}"
BUCKET="${1:-${BUCKET:-mara-predicators-web}}"
LOCATION="${LOCATION:-us-central1}"

echo "Project : $PROJECT"
echo "Bucket  : gs://$BUCKET"
echo "Location: $LOCATION"
echo

# ----------------------------------------------------------------------------
# 1) Bucket bootstrap (idempotent)
# ----------------------------------------------------------------------------
if ! gcloud storage buckets describe "gs://$BUCKET" --project="$PROJECT" >/dev/null 2>&1; then
  echo "[init] Creating bucket gs://$BUCKET …"
  gcloud storage buckets create "gs://$BUCKET" \
    --project="$PROJECT" \
    --location="$LOCATION" \
    --uniform-bucket-level-access

  echo "[init] Granting public read (allUsers : storage.objectViewer) …"
  gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member=allUsers --role=roles/storage.objectViewer >/dev/null

  echo "[init] Setting website config (index.html) …"
  gcloud storage buckets update "gs://$BUCKET" \
    --web-main-page-suffix=index.html >/dev/null
else
  echo "[init] Bucket already exists, reusing."
fi

# ----------------------------------------------------------------------------
# 2) Stage what the browser actually needs (resolve symlinks, prune cruft)
# ----------------------------------------------------------------------------
STAGE="$(mktemp -d -t predicators-web-XXXXXX)"
trap "rm -rf $STAGE" EXIT
echo
echo "[stage] $STAGE"

# app/, wheels/: small, copy as-is.
rsync -aL \
  --exclude="__pycache__" --exclude="*.pyc" \
  "$WEB_DIR/app" "$WEB_DIR/wheels" \
  "$STAGE/"

# predicators_assets is a symlink into the repo (~140 MB of meshes/URDFs).
# -L resolves it; Three.js fetches these URLs at runtime.
rsync -aL "$WEB_DIR/predicators_assets" "$STAGE/"

# Only the two node_modules packages the importmap actually resolves
# (puppeteer + chromium-bidi etc. are dev-only).
mkdir -p "$STAGE/node_modules"
rsync -aL "$WEB_DIR/node_modules/three"        "$STAGE/node_modules/"
rsync -aL "$WEB_DIR/node_modules/urdf-loader"  "$STAGE/node_modules/"

echo "[stage] tree size: $(du -sh "$STAGE" | cut -f1)"

# ----------------------------------------------------------------------------
# 3) Upload (rsync semantics, gzip text-y files in flight)
# ----------------------------------------------------------------------------
echo
echo "[upload] gs://$BUCKET/ (this can take a few minutes the first time) …"
gcloud storage rsync -r "$STAGE/" "gs://$BUCKET/" \
  --project="$PROJECT" \
  --delete-unmatched-destination-objects \
  --gzip-in-flight=js,mjs,html,css,xml,urdf,dae,obj,mtl,svg,txt,json

echo
echo "[done] Open:"
echo "  https://storage.googleapis.com/$BUCKET/app/index.html"
