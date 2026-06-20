#!/usr/bin/env bash
# Export fresh YouTube cookies from your browser and upload to Fly.
#
# Usage:
#   ./scripts/refresh-cookies.sh                  # default: Chrome
#   ./scripts/refresh-cookies.sh firefox           # use Firefox
#   ./scripts/refresh-cookies.sh brave             # use Brave
#
# What it does:
#   1. Exports all cookies from your browser via yt-dlp
#   2. Filters to YouTube/Google rows only (keeps the secret small)
#   3. Uploads as YTDLP_COOKIES_B64 to the seechords API on Fly.io
#
# Prerequisites:
#   - yt-dlp installed locally (brew install yt-dlp)
#   - flyctl installed and logged in (fly auth login)
#   - Logged into YouTube in the browser you specify

set -euo pipefail

BROWSER="${1:-chrome}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP_RAW=$(mktemp /tmp/seechords-cookies-raw.XXXXXX)
TMP_FILTERED=$(mktemp /tmp/seechords-cookies-filtered.XXXXXX)
# yt-dlp refuses to overwrite existing non-Netscape files — remove the empty file mktemp created
rm -f "$TMP_RAW"

cleanup() { rm -f "$TMP_RAW" "$TMP_FILTERED"; }
trap cleanup EXIT

# Check prerequisites
for cmd in yt-dlp fly python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' not found. Install it first." >&2
    exit 1
  fi
done

echo "==> Exporting cookies from $BROWSER..."
# yt-dlp may exit non-zero on format/signature errors but still writes the cookie jar —
# ignore its exit code and verify the file contents afterwards
yt-dlp --cookies-from-browser "$BROWSER" \
  --cookies "$TMP_RAW" \
  --skip-download \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" >/dev/null 2>&1 || true

if [[ ! -s "$TMP_RAW" ]]; then
  echo "Error: cookie export produced an empty file. Is $BROWSER running and logged into YouTube?" >&2
  exit 1
fi

RAW_LINES=$(grep -c $'\t' "$TMP_RAW" || true)
echo "    Exported $RAW_LINES cookie rows"

echo "==> Filtering to YouTube/Google domains..."
python3 "$REPO_ROOT/scripts/filter-netscape-cookies-youtube.py" "$TMP_RAW" "$TMP_FILTERED"

echo "==> Uploading to Fly (app: seechords)..."
"$REPO_ROOT/scripts/set-ytdlp-cookies-secret.sh" "$TMP_FILTERED" --force

echo ""
echo "Done! Fresh cookies are now live. Test with a previously-failing video."
