# Live Analyze: Streaming Chord Analysis via Tab Audio Capture

## Problem
YouTube blocks yt-dlp on cloud servers (bot detection). The extension needs to send audio directly to the server instead of relying on server-side YouTube downloads.

## Solution
Capture audio from the browser tab in real-time using `chrome.tabCapture`, send 15-second chunks to the server for fast analysis (librosa chromagram path), display chords progressively with a looping playback UI.

## Architecture

### Extension Side

**Manifest changes:**
- Add `tabCapture` permission

**content.js changes:**
- Replace "Analyze & Generate Chords" button with "Live Analyze" button
- New states: listening (recording first chunk), analyzing, live (chords arriving), complete
- Looping display: when playhead passes last known chord, loop the display index back to start of analyzed range
- Stitch incoming chunk results into the chord timeline at correct time offsets
- Track `captureStartTime` (video.currentTime when capture began) to map chunk times to video times
- "Stop" button ends capture; "Enhance with AI" button (optional) triggers full BTC re-analysis
- Status bar shows: recording indicator, analyzed time range, chunk progress

**background.js changes:**
- New message type: `START_LIVE_CAPTURE` — calls `chrome.tabCapture.capture({ audio: true, video: false })`, creates MediaRecorder with `timeslice: 15000`
- New message type: `STOP_LIVE_CAPTURE` — stops MediaRecorder, releases stream
- On `ondataavailable`: upload chunk blob to `POST /api/analyze-chunk` with metadata (videoId, chunkIndex, startTime offset)
- Forward chunk results back to content script via `chrome.tabs.sendMessage`
- Accumulate all chunk blobs for optional full-song upload later

### Server Side

**New endpoint: `POST /api/analyze-chunk`**
- Accepts: multipart form with `file` (WebM/Opus audio blob), `videoId`, `chunkIndex`, `startTime`
- Converts WebM to WAV via ffmpeg
- Runs `_detect_chords_librosa()` (fast chromagram path, ~2-3s for 15s audio)
- Returns: `{ chords, bpm, key, chunkIndex, startTime, endTime }`
- Runs synchronously on the API server (no worker needed — lightweight enough)

**No changes to existing endpoints.** The `/api/analyze` upload path remains for the optional "Enhance with AI" full-song re-analysis.

## UI States

1. **No chords** → Show "Live Analyze" button with hint to play video first
2. **Listening** → Progress bar filling over 15s, "Capturing first 15 seconds..."
3. **Analyzing** → "Analyzing chunk 1..." while chunk 2 records in background
4. **Live** → Timeline + 3-card chord display. Status: "⏺ Live | 0:00-0:15 analyzed". Looping indicator when playhead passes analyzed range
5. **Complete** → Normal chord display. Optional "Enhance with AI" button
6. **Cached** → Standard instant-load from Turso (existing behavior)

## Looping Behavior
When video.currentTime exceeds the last analyzed chord's end time, the overlay keeps cycling through the known chords (display index wraps around). A subtle "🔄 Looping" indicator shows. As new chunks arrive, the loop boundary extends forward and looping stops once the analyzed range catches up to playback.

## Chunk Stitching
Each chunk returns chords with times relative to chunk start (0-15s). The extension adds `captureStartTime + chunkIndex * 15` to map them to video time. Overlapping chord boundaries are merged (if chunk N ends with "Am" at 14.8s and chunk N+1 starts with "Am" at 0.1s, merge into one segment).

## Timing Budget (per chunk)
- MediaRecorder capture: 15s (real-time)
- Upload to server: ~0.5-1s (15s of Opus ≈ 200KB)
- ffmpeg convert: ~0.5s
- Librosa analysis: ~2-3s
- Response: ~0.1s
- **Total analysis latency: ~3-5s per chunk**
- Buffer margin: ~10-12s ahead of analysis

## File Changes Summary
- `extension/manifest.json` — add `tabCapture` permission
- `extension/content.js` — live analyze UI states, chunk stitching, looping display
- `extension/background.js` — tabCapture + MediaRecorder management, chunk upload
- `extension/overlay.css` — live indicator styles, looping indicator
- `server/app.py` — new `/api/analyze-chunk` endpoint
