#!/usr/bin/env bash
# Deploy the worker image and update the API's FLY_WORKER_IMAGE secret
# Usage: ./scripts/deploy-worker.sh [--no-cache]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTRA_FLAGS="${1:-}"

echo "==> Deploying seechords-worker..."
OUTPUT=$(fly deploy --config "$REPO_ROOT/fly.worker.toml" --remote-only $EXTRA_FLAGS 2>&1)
echo "$OUTPUT"

# Extract the image tag from deploy output
IMAGE=$(echo "$OUTPUT" | grep -oE 'registry\.fly\.io/seechords-worker:deployment-[A-Za-z0-9]+' | head -1)

if [ -z "$IMAGE" ]; then
  echo "ERROR: Could not extract image tag from deploy output"
  exit 1
fi

echo ""
echo "==> Updating FLY_WORKER_IMAGE on seechords API to: $IMAGE"
fly secrets set "FLY_WORKER_IMAGE=$IMAGE" -a seechords

echo ""
echo "==> Done. Worker and API both updated."
