# Public re-analyze — design

**Date:** 2026-07-22
**Goal:** Let any web-player user re-analyze a song's chords (e.g. after the beat-timing
fix, or when chords look wrong) without ingest authentication — with guardrails that
protect cost and the YouTube datacenter-IP reputation.

## Background

The web player (`server/templates/play.html`) already has a "↻ Re-analyze" button, but it
calls `POST /api/ingest/reanalyze/<video_id>`, which is gated by `@_require_ingest_auth`
(shows "Auth required — log in via ingest page first"). We want it to work for anyone.

Relevant existing behavior:
- `_cache_get` picks the displayed version as `verified > current model > legacy`.
- Re-analysis replaces only the `CURRENT_MODEL_SOURCE` (chordmini) version; it never touches
  a `verified` row. So re-analyzing a verified song cannot harm the curated "Made by Jin"
  chords — verified stays top priority and is preserved. No verified-specific block needed.
- The worker fetches a cached WAV instead of hitting YouTube when spawned with
  `skip_download=True` (WAV served from `wav_cache_backups` / disk via the internal cache).

## Requirements

1. Anonymous users can trigger re-analysis of a song from the web player.
2. **Per-song cooldown: 10 minutes.** A song can be re-analyzed at most once per 10 min.
   An in-flight job for that song is always blocked regardless of the cooldown.
3. **Prefer cached WAV.** If a WAV is cached for the video, re-analyze from it
   (`skip_download=True`, no YouTube). Otherwise re-download from YouTube (cookies handle auth).
4. Do not change data safety for verified songs (already safe — see Background).
5. Keep the existing authed force-path (`/api/ingest/reanalyze/<id>`) unchanged so the owner
   can bypass the cooldown.

## Backend — new endpoint `POST /api/reanalyze/<video_id>` (public, no auth)

Flow:
1. Validate `video_id` against `^[a-zA-Z0-9_-]{11}$` → 400 on bad input.
2. **In-flight check:** if `jobs` has a row for this `video_id` with status in
   (`pending`, `processing`) updated within the last 15 min → `429`
   `{ "error": "Re-analysis already in progress." }`. (15-min window so a stuck/abandoned
   job can't block the song forever.)
3. **Cooldown check:** look up the most recent `analyzed_at` for this video's
   `CURRENT_MODEL_SOURCE` row. If `now - analyzed_at < 600s` → `429`
   `{ "error": "Re-analyzed recently. Try again in N min.", "retryAfterSec": ... }`.
4. **skip_download decision:** `skip_download = _has_cached_wav(video_id)` — true if a
   `wav_cache_backups` row (or on-disk `WAV_CACHE_DIR/<id>.wav`) exists.
5. Look up title; delete the existing `CURRENT_MODEL_SOURCE` version (mirrors the ingest
   force-path so the new result is authoritative).
6. `_create_job(...)` + `_spawn_worker(job_id, video_id, title, skip_download=skip_download)`.
7. Return `{ "job_id": ..., "skipDownload": bool }` (frontend polls `/api/status/<job_id>`).

Notes:
- No global "one worker at a time" lock (unlike the ingest endpoint) — the per-song
  in-flight + cooldown checks bound abuse per song; different songs may run concurrently,
  matching the existing public `/api/analyze-youtube` behavior.
- Residual risk (accepted): a user could re-analyze many *different* non-cached songs to
  drive YouTube traffic. Cooldown + cached-WAV preference limit this; a global YouTube
  rate-limit is out of scope for now.

## Helpers

- `_has_cached_wav(video_id) -> bool` — disk (`WAV_CACHE_DIR/<id>.wav`) OR
  `SELECT 1 FROM wav_cache_backups WHERE video_id=?`.
- `_reanalyze_cooldown_remaining(video_id) -> int` — seconds remaining (0 if allowed),
  from `MAX(analyzed_at)` of the `CURRENT_MODEL_SOURCE` version vs a 600s window.
- `_song_job_in_flight(video_id) -> bool` — any recent `jobs` row pending/processing.

## Frontend — `server/templates/play.html`

- Change the re-analyze button `fetch` from `/api/ingest/reanalyze/` to `/api/reanalyze/`.
- Remove the `401` "Auth required" branch (no longer reachable).
- Keep the `429` branch; show `error` message (now the cooldown message) to the user.

## Testing

Unit tests (stdlib `unittest`, importing `app` with a temp/local SQLite DB or by testing the
pure helper functions):
- `_reanalyze_cooldown_remaining`: returns >0 within 10 min of `analyzed_at`, 0 after.
- `_has_cached_wav`: true when a backup row/disk file exists, false otherwise.
- skip_download decision: cached → True, not cached → False.

(The endpoint's spawn/HTTP layer is exercised manually end-to-end against the deployed API,
consistent with how the other worker flows are verified.)

## Out of scope

- Global/per-IP rate limiting beyond the per-song cooldown.
- Any change to verified-song data handling or the ingest force-path.
- New UI beyond re-pointing the existing button.

## Deploy

Backend change in `server/app.py` → redeploy API (`fly deploy --config server/fly.toml`).
No worker/image change (worker already supports `skip_download`). Frontend is served by the
API, so the same API deploy ships the `play.html` change.
