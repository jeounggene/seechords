# SeeChords — deployment guide

This document describes how to deploy **SeeChords** to [Fly.io](https://fly.io): the **API** app (`seechords`), the **worker** app (`seechords-worker`), required **secrets**, and how the **Chrome extension** ties in.

## Architecture (production)

| Piece | Fly app | Role |
|--------|---------|------|
| **API** | `seechords` | Flask + Gunicorn: REST API, marketing site, spawns analysis jobs, talks to Turso. |
| **Worker** | `seechords-worker` | Heavy image: PyTorch, chord models, **yt-dlp** download + analysis. Machines are created **on demand** via the Fly Machines API and **`auto_destroy`** after the job. |
| **Database** | (Turso) | `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN` — chord cache, jobs, versions. |

The API does **not** run full ML inference for YouTube flows; it starts an ephemeral machine on `seechords-worker` using `FLY_API_TOKEN`.

**Important:** Deploy **both** images when you change server code paths used by either app. API-only deploys are not enough if `worker.py`, `analyze_chords.py`, models, or yt-dlp behavior changed.

---

## Prerequisites

1. **Fly.io account** and [flyctl](https://fly.io/docs/hands-on/install-flyctl/) installed and logged in:

   ```bash
   fly auth login
   ```

2. **This repository** cloned locally.

3. **Turso** (or compatible libSQL) database with URL and auth token for production.

4. For **YouTube analysis** from Fly datacenter IPs, you almost always need **Netscape-format cookies** (see [YouTube cookies](#youtube-cookies-optional-but-recommended)).

---

## Repository layout (deploy-related)

| Path | Purpose |
|------|---------|
| `server/fly.toml` | API app config; build uses `Dockerfile.api` with context **parent of `server/`** (repo root). |
| `fly.worker.toml` | Worker app config; build uses `Dockerfile.worker` from **repo root**. |
| `Dockerfile.api` | Lightweight API image. |
| `Dockerfile.worker` | ML + yt-dlp worker image (~3 GB). |
| `.dockerignore` | Shrinks build context; keep large training data out. |

---

## Secrets and environment variables

Set with Fly **per app** where noted. Sensitive values use `fly secrets set`; non-secret defaults can live in `fly.toml` `[env]` (already set for the API).

### API app (`seechords`) — required

| Secret / env | Description |
|--------------|-------------|
| `TURSO_DATABASE_URL` | Turso database URL. |
| `TURSO_AUTH_TOKEN` | Turso auth token. |
| `FLY_API_TOKEN` | Fly API token with permission to **create machines** on `seechords-worker` (used to spawn workers). Create under Fly dashboard → Access Tokens, or `fly tokens create`. |

### API app (`seechords`) — optional

| Secret / env | Description |
|--------------|-------------|
| `YTDLP_COOKIES_B64` | Base64-encoded **Netscape** `cookies.txt` while logged into YouTube. Forwarded to each worker machine so **yt-dlp** can authenticate. Recommended for production. Use a **throwaway** account — automated use can get an account flagged. Refresh with `./scripts/refresh-cookies.sh`. |
| `YTDLP_COOKIEFILE` | Alternative: path **on the API VM** to a readable `cookies.txt`; the API reads the file and sends it to workers as base64. Only works if the file exists in the running image or volume (uncommon). |
| `YTDLP_PROXY` | Optional proxy URL (e.g. `http://user:pass@host:port` or `socks5://…`) for **yt-dlp**, forwarded to each worker. Use a **residential proxy** to bypass a persistent datacenter-IP block on the Fly egress IP — the durable fix when cookies alone aren't enough. |
| `YTDLP_YOUTUBE_PLAYER_CLIENT` | Optional override to pin a single yt-dlp player client (e.g. `android_vr`) instead of the rotating ladder. Forwarded to workers. |
| `FLY_WORKER_APP` | Default `seechords-worker` (set in `server/fly.toml`). Override if you rename the worker app. |
| `FLY_WORKER_IMAGE` | Docker image tag for spawned workers (e.g. `registry.fly.io/seechords-worker:deployment-XXXXX`). **Must be updated after each worker deploy** — `./scripts/deploy-worker.sh` does this automatically. |
| `SEECHORDS_CHROME_STORE_URL` | Marketing site Chrome Web Store link (optional). |

### Worker app (`seechords-worker`)

Usually **no extra secrets** are required for cookies if you set `YTDLP_COOKIES_B64` on **`seechords`** — the API injects cookies into each spawned machine’s environment.

You *may* set `YTDLP_COOKIES_B64` on `seechords-worker` as well if you rely on Fly merging app secrets into machines (documented behavior can vary; **prefer setting cookies on the API** so `_spawn_worker` logs show forwarding).

---

## Deploy commands (read this carefully)

**Always run `fly deploy` from the repository root** (the folder that contains `Dockerfile.api` and `fly.worker.toml`), not from `server/`, so Docker build context paths (`COPY server/...`, `COPY training/...`) resolve correctly. Replace placeholder paths like `/path/to/seechords` with your real clone path (e.g. `~/projects/seechords` on your machine).

### 1. Deploy the worker image (`seechords-worker`)

**Use the deploy script** — it builds the worker image, then automatically updates the API's `FLY_WORKER_IMAGE` secret so spawned workers use the new image:

```bash
cd /path/to/seechords
./scripts/deploy-worker.sh            # normal deploy
./scripts/deploy-worker.sh --no-cache  # force fresh build (use after yt-dlp updates)
```

- App name: **`seechords-worker`** (see `fly.worker.toml`).
- First deploy creates the app if it does not exist (`fly apps create seechords-worker` may be required once).

> **Why the script matters:** The API spawns workers using the `FLY_WORKER_IMAGE` secret. If you deploy the worker with plain `fly deploy` and forget to update that secret, the API keeps spawning the old image. The script handles this automatically.

Manual alternative (not recommended):

```bash
fly deploy --config fly.worker.toml --remote-only
# Then manually update the API secret with the image tag from the deploy output:
fly secrets set FLY_WORKER_IMAGE="registry.fly.io/seechords-worker:deployment-XXXXX" -a seechords
```

### 2. Deploy the API (`seechords`)

```bash
cd /path/to/seechords
fly deploy --config server/fly.toml
```

- App name: **`seechords`** (see `server/fly.toml`).
- Internal HTTP port **8080** (see `server/fly.toml` `[http_service]`).

### Typical release order

1. Deploy **worker** (via `./scripts/deploy-worker.sh`) when `worker.py`, `analyze_chords.py`, models under the worker Dockerfile, or yt-dlp usage changed.
2. Deploy **API** when `app.py`, routes, spawn logic, or API-only code changed.
3. For a full release touching both, deploy **worker first**, then **API**.

---

## YouTube cookies (optional but recommended)

Datacenter IPs are often blocked by YouTube with “Sign in to confirm you’re not a bot”. The worker uses **yt-dlp** with `--cookies` and `player_client=web` when cookies are present.

1. On your **local machine**, while logged into YouTube in the browser, export **Netscape** `cookies.txt` (e.g. `yt-dlp --cookies-from-browser chrome --cookies cookies.txt "https://www.youtube.com/watch?v=..."` or a browser extension).

2. **Shrink the file (recommended):** Exporting all of Chrome can produce **hundreds of thousands** of characters and Fly secret rollouts may **time out**. Keep only YouTube/Google-related rows:

   ```bash
   python3 scripts/filter-netscape-cookies-youtube.py cookies.txt cookies-youtube.txt
   ```

   Use `cookies-youtube.txt` in the next step instead of `cookies.txt`.

3. Set the secret on the **API** app — either manually:

   ```bash
   fly secrets set YTDLP_COOKIES_B64="$(base64 -i cookies-youtube.txt | tr -d '\n')" -a seechords
   ```

   Or use the repo script (portable base64 via `openssl`; defaults to app `seechords`):

   ```bash
   ./scripts/set-ytdlp-cookies-secret.sh cookies-youtube.txt
   # Other app: ./scripts/set-ytdlp-cookies-secret.sh cookies-youtube.txt --app my-staging-api
   ```

4. Redeploying the API is not always required for secret-only changes (Fly restarts machines), but confirm logs after a test job.

**Security:** Do not commit `cookies.txt` or secrets to git. Rotate cookies if they leak.

---

## Post-deploy verification

### Health

```bash
curl -sS https://seechords.fly.dev/api/health
```

### Logs

```bash
fly logs -a seechords
fly logs -a seechords-worker
```

After triggering **Analyze this video** (or `POST /api/analyze-youtube`):

- **API:** Look for `Worker spawn: forwarding YTDLP_COOKIES_B64 (N base64 chars)` with **N > 0** if cookies are configured.
- **Worker:** Ephemeral logs may show `env YTDLP_COOKIES_B64 present: yes` and `yt-dlp extractor: youtube:player_client=web` when cookies are used.

---

## Chrome extension

The extension calls the production API at **`https://seechords.fly.dev`** (see `extension/background.js` → `API_BASE`).

1. Update `API_BASE` if you use a custom domain or staging URL.
2. Load unpacked in `chrome://extensions` for development, or zip the `extension/` folder (exclude junk) for **Chrome Web Store** submission. See `misc/STORE_SUBMISSION.md` for listing/privacy notes.

---

## Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| `COPY server/... not found` during Docker build | Deploy from **repo root** with `--config server/fly.toml` or `--config fly.worker.toml`, not from `server/`. |
| Worker spawn fails | `FLY_API_TOKEN` valid; token can create machines on `FLY_WORKER_APP`; `seechords-worker` image exists in registry. |
| YouTube “Sign in / bot” errors | Set `YTDLP_COOKIES_B64` on **`seechords`** (fresh cookies via `./scripts/refresh-cookies.sh`); worker image redeployed; yt-dlp updated in `Dockerfile.worker`. If the datacenter IP is *persistently* blocked (all clients fail / HTTP 429), cookies may not be enough — set `YTDLP_PROXY` to a residential proxy. |
| Old worker code running | Redeploy **`fly.worker.toml`** after changing worker code; API uses `registry.fly.io/seechords-worker:latest` by default. |

---

## Local development (optional)

- API: run Flask/gunicorn from `server/` with `TURSO_*` (or `DB_PATH` for SQLite) set in the environment.
- Worker: run `python worker.py` with `JOB_ID`, `VIDEO_ID`, `TURSO_*` set; optional `YTDLP_COOKIEFILE` pointing to a local `cookies.txt`.

Production behavior (Fly Machines, registry images) differs from local; use this guide for production deploys.

---

## Quick reference — copy/paste

```bash
# From repository root
cd /path/to/seechords

# Secrets (once per environment; replace values)
fly secrets set TURSO_DATABASE_URL="libsql://..." -a seechords
fly secrets set TURSO_AUTH_TOKEN="..." -a seechords
fly secrets set FLY_API_TOKEN="..." -a seechords
python3 scripts/filter-netscape-cookies-youtube.py cookies.txt cookies-youtube.txt
fly secrets set YTDLP_COOKIES_B64="$(base64 -i cookies-youtube.txt | tr -d '\n')" -a seechords
# or: ./scripts/set-ytdlp-cookies-secret.sh cookies-youtube.txt

# Deploy
fly deploy --config fly.worker.toml
fly deploy --config server/fly.toml
```
