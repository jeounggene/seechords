# Ingest Site Redesign — Design Spec

**Date:** 2026-04-03
**Approach:** Incremental refactor of existing ingest page and Flask routes

---

## Goals

1. Add server-side authentication to the ingest site
2. Add a home view with YouTube link input and saved versions list
3. Split save flow into "Save" (draft) and "Save & Upload" (live)
4. Use YouTube streaming for playback (no local audio storage for editing)
5. Attribution: "Made by Jin" for human-curated, "Made by SeeChords" for model-generated

## Non-goals

- Multi-user accounts (single password gate for now)
- Renaming the `verified/` directory
- Schema changes to `chord_versions`

---

## 1. Authentication

### Mechanism

- Server-side password validation only. Password never exposed to client JS.
- Password stored in env var `INGEST_PASSWORD` (Fly secret in production).
- Flask `session` (signed cookie via `SECRET_KEY`) tracks auth state.

### Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ingest` | Checks session cookie. Renders login form if unauthenticated, ingest UI if authenticated. |
| `POST` | `/ingest/login` | Validates password against `INGEST_PASSWORD`. Sets `session['ingest_auth'] = True` on success, redirects to `/ingest`. Returns error on failure. |
| `POST` | `/ingest/logout` | Clears session, redirects to `/ingest`. |

### API protection

All `/api/ingest/*` routes check `session.get('ingest_auth')`. Return 401 JSON if not authenticated.

### Config

- `INGEST_PASSWORD` — env var / Fly secret (required for ingest access)
- `SECRET_KEY` — Flask secret key for session signing (required; set as Fly secret or env var)

---

## 2. Ingest Home View

The current ingest page goes straight to the editor. The redesign adds a home view as the default landing.

### Layout

1. **YouTube link input** at the top — text field + "Analyze" button
2. **Saved versions list** below — all versions from `chord_versions` with `source` in (`ingest-edit`, `verified`)

### Saved versions list

Each row displays:
- Song title
- Key and BPM
- Source label: "Made by Jin" (verified) or "Draft" (ingest-edit)
- Date saved
- Segment count (number of chord changes)

Clicking a row loads that version into the editor with YouTube streaming audio.

### New song flow

1. User pastes a YouTube URL and clicks "Analyze"
2. Client POSTs to existing `POST /api/ingest/youtube`
3. Server downloads audio, runs BTC model + Beat This! beat tracker
4. Results load into the editor pre-filled with model chords
5. User edits, then saves

### New API endpoint

`GET /api/ingest/versions` — returns all versions where `source` in (`ingest-edit`, `verified`), ordered by most recent. Each entry includes `version_id`, `video_id`, `title`, `key`, `bpm`, `source`, `analyzed_at`, `segment_count`.

---

## 3. Editor View

Mostly unchanged from current ingest editor. Key differences:

### Removed

- Chord lines feature (removed per user request)

### Kept

- Chord timeline with beat-aligned segments
- Chord bank (click to assign chords)
- Playalong beat line
- Playback controls

### Playback

Audio playback uses YouTube streaming via `video_id`, same mechanism as the web player (`/play` page). No local audio download needed for editing sessions.

When loading a previously saved version, the client fetches `GET /api/version/<version_id>` (already exists) to get chords, beat times, key, BPM, and `video_id`. The `video_id` is used to stream audio.

### Save buttons

Two buttons replace the current single save:

| Button | Action |
|--------|--------|
| **Save** | Inserts into `chord_versions` with `source='ingest-edit'`, `is_active=1`. Deactivates previous versions for same `video_id`. Not promoted to `verified/`. Not served to end users. |
| **Save & Upload** | Inserts into `chord_versions` with `source='verified'`, `is_active=1`. Writes `.lab` to `server/verified/labels/`. Updates `video_map.json`. Chords go live for extension and web player users. |

### Back button

Returns to home view (versions list + YouTube input).

---

## 4. Data Flow & Version Attribution

### Version sources and what they mean

| `source` value | Label shown to users | Served to extension/web player | Created by |
|----------------|---------------------|-------------------------------|------------|
| `verified` | "Made by Jin" | Yes | Save & Upload on ingest |
| `model` / `btc` | "Made by SeeChords" | Yes | Automatic analysis pipeline |
| `ingest-edit` | "Draft" (ingest only) | No | Save on ingest |

### Version history

- Saving never overwrites — each save creates a new row in `chord_versions`
- Previous versions for the same `video_id` are deactivated (`is_active=0`) but kept
- The home view shows the most recent version per video

### No schema changes

`chord_versions` already has all needed columns: `video_id`, `title`, `key`, `bpm`, `chords`, `beat_times`, `source`, `analyzed_at`, `is_active`.

---

## 5. Files Changed

### Server (app.py)

- Add auth check decorator/helper for ingest routes
- Add `POST /ingest/login`, `POST /ingest/logout`
- Modify `GET /ingest` to serve login form or ingest UI based on session
- Add `GET /api/ingest/versions` for home page version list
- Existing `POST /api/ingest/<job_id>/save-version` becomes the "Save" backend
- Existing `POST /api/ingest/<job_id>/promote-verified` becomes the "Save & Upload" backend

### Frontend (ingest.html)

- Add login form (shown when not authenticated — server-rendered)
- Add home view: YouTube input + saved versions list
- Modify editor view: remove chord lines, add Save / Save & Upload buttons
- Switch playback to YouTube streaming (embed or audio extraction matching web player approach)
- Back button returns to home view

### Config / Deployment

- New Fly secret: `INGEST_PASSWORD`
- Ensure `SECRET_KEY` is set for Flask session signing

### No changes to

- `chord_versions` schema
- `verified/` directory naming
- Extension or web player code
- Worker pipeline
