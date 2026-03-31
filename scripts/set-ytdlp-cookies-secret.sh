#!/usr/bin/env bash
# Upload Netscape cookies.txt to Fly as YTDLP_COOKIES_B64 on the SeeChords API app.
#
# Usage:
#   ./scripts/set-ytdlp-cookies-secret.sh [path/to/cookies.txt] [--app APP] [--force]
#
# Prerequisites:
#   - flyctl installed and logged in (`fly auth login`)
#   - cookies.txt in Netscape format (e.g. from yt-dlp --cookies-from-browser or a browser extension)
#
# Example (run from repo root; create cookies.txt first):
#   yt-dlp --cookies-from-browser chrome --cookies cookies.txt "https://www.youtube.com/watch?v=..." --skip-download
#   ./scripts/set-ytdlp-cookies-secret.sh cookies.txt

set -euo pipefail

APP="${FLY_APP:-seechords}"
COOKIES_FILE=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)
      APP="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      sed -n '2,20p' "$0" | head -n 18
      exit 0
      ;;
    *)
      if [[ -z "$COOKIES_FILE" ]]; then
        COOKIES_FILE="$1"
      else
        echo "Unexpected argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

COOKIES_FILE="${COOKIES_FILE:-cookies.txt}"

if ! command -v fly >/dev/null 2>&1; then
  echo "Error: 'fly' not found. Install: https://fly.io/docs/hands-on/install-flyctl/" >&2
  exit 1
fi

if [[ ! -f "$COOKIES_FILE" ]]; then
  echo "Error: file not found: $COOKIES_FILE" >&2
  echo "Create Netscape cookies first, e.g.:" >&2
  echo "  yt-dlp --cookies-from-browser chrome --cookies cookies.txt 'https://www.youtube.com/watch?v=...' --skip-download" >&2
  exit 1
fi

if [[ "$FORCE" -eq 0 ]] && ! head -n 1 "$COOKIES_FILE" | grep -qiE '^(# HTTP Cookie File|# Netscape HTTP Cookie File)'; then
  echo "Warning: $COOKIES_FILE may not be Netscape format (expected '# Netscape HTTP Cookie File' or similar)." >&2
  echo "If you are sure, run again with --force" >&2
  exit 1
fi

# Single-line base64; works on macOS and Linux (no base64 -i / -w0 differences)
B64="$(openssl base64 -A -in "$COOKIES_FILE")"

echo "Setting YTDLP_COOKIES_B64 on Fly app '$APP' (${#B64} base64 chars)..."
fly secrets set "YTDLP_COOKIES_B64=$B64" -a "$APP"

echo "Done. Machines will restart; test analysis after a minute."
echo "Verify: fly secrets list -a $APP | grep YTDLP"
