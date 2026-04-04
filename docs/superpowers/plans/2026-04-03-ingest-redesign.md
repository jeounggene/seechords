# Ingest Site Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auth, a home view with saved versions list, and split save into Save (draft) vs Save & Upload (live) to the ingest site.

**Architecture:** Incremental refactor of the existing Flask routes in `server/app.py` and the single-file `server/templates/ingest.html`. Auth uses Flask `session` with a server-side password check. The home view replaces the current upload form as the default landing. The editor switches from `<audio>` element to YouTube IFrame API for playback (matching `play.html`).

**Tech Stack:** Flask sessions (`itsdangerous` via `SECRET_KEY`), existing Turso/libsql `chord_versions` table, YouTube IFrame API.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `server/app.py` | Modify (lines ~1794-2520) | Add login/logout routes, auth decorator, `GET /api/ingest/versions` endpoint |
| `server/templates/ingest.html` | Modify (2573 lines) | Add login form, home view with versions list, swap audio playback to YT IFrame, split save buttons, remove chord lines |

No new files. No schema changes.

---

### Task 1: Server-side auth — login/logout routes and decorator

**Files:**
- Modify: `server/app.py:1-44` (app config, imports)
- Modify: `server/app.py:1794-1800` (ingest_page route)

- [ ] **Step 1: Add `SECRET_KEY` and `INGEST_PASSWORD` config**

Near the top of `server/app.py`, after `CORS(app, ...)` (line 44), add:

```python
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
INGEST_PASSWORD = os.environ.get('INGEST_PASSWORD', '')
```

Also add `session` to the Flask import on line 27:

```python
from flask import Flask, request, jsonify, send_file, render_template, redirect, Response, stream_with_context, session
```

- [ ] **Step 2: Add `_require_ingest_auth` decorator**

Add this right before the `@app.route('/ingest')` block (~line 1793):

```python
from functools import wraps

def _require_ingest_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('ingest_auth'):
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect('/ingest')
        return f(*args, **kwargs)
    return decorated
```

- [ ] **Step 3: Add login and logout routes**

Add these right after the decorator:

```python
@app.route('/ingest/login', methods=['POST'])
def ingest_login():
    password = request.form.get('password', '')
    if not INGEST_PASSWORD:
        return redirect('/ingest')  # no password configured = open access
    if password == INGEST_PASSWORD:
        session['ingest_auth'] = True
        return redirect('/ingest')
    return render_template('ingest.html', login_error='Incorrect password', show_login=True)


@app.route('/ingest/logout', methods=['POST'])
def ingest_logout():
    session.pop('ingest_auth', None)
    return redirect('/ingest')
```

- [ ] **Step 4: Update `ingest_page` to gate on auth**

Replace the current `ingest_page` function:

```python
@app.route('/ingest')
def ingest_page():
    """Serve the chord sheet ingest + review UI (auth-gated)."""
    if INGEST_PASSWORD and not session.get('ingest_auth'):
        return render_template('ingest.html', show_login=True)
    return render_template('ingest.html', show_login=False)
```

Note: This changes from `open(html_path)` to `render_template()`. The template will use a Jinja2 variable `show_login` to decide which view to render.

- [ ] **Step 5: Add `@_require_ingest_auth` to all API ingest routes**

Add the decorator to each of these existing route functions:
- `list_saved_labs` (line ~1803)
- `get_saved_lab` (line ~1841)
- `ingest_silver_audio` (line ~1931)
- `save_saved_lab` (line ~1942)
- `ingest_upload` (line ~2053)
- `ingest_youtube` (line ~2105)
- `serve_ingest_audio` (line ~2176)
- `ingest_status` (line ~2200)
- `ingest_save` (line ~2210)
- `ingest_save_version` (line ~2323)
- `ingest_list_versions` (line ~2376)
- `promote_to_verified` (line ~2434)

Example pattern for each:
```python
@app.route('/api/ingest/saved-labs')
@_require_ingest_auth
def list_saved_labs():
    ...
```

- [ ] **Step 6: Test auth locally**

Set env vars and start the server:
```bash
cd /Users/genej/projects/seechords
SECRET_KEY=test-secret INGEST_PASSWORD=test123 python server/app.py
```

Verify:
1. `curl -s http://127.0.0.1:5005/ingest` — should show login form HTML
2. `curl -s http://127.0.0.1:5005/api/ingest/saved-labs` — should return `{"error": "Unauthorized"}` with status 401
3. `curl -s -X POST -d 'password=test123' -c cookies.txt -L http://127.0.0.1:5005/ingest/login` — should redirect to ingest page
4. `curl -s -b cookies.txt http://127.0.0.1:5005/api/ingest/saved-labs` — should return JSON array

- [ ] **Step 7: Commit**

```bash
git add server/app.py
git commit -m "feat(ingest): add server-side auth with session cookie"
```

---

### Task 2: Login form in ingest.html

**Files:**
- Modify: `server/templates/ingest.html:1-10` (doctype/head area)
- Modify: `server/templates/ingest.html:393-434` (upload-section area)

- [ ] **Step 1: Add Jinja2 login form block**

At the very top of `<body>` (after the opening `<body>` tag but before any existing content), add a login form that is conditionally shown via Jinja2:

```html
{% if show_login %}
<div style="max-width:400px;margin:120px auto;padding:0 20px;">
  <h1 style="font-size:1.6em;margin-bottom:24px;color:#e94560;text-align:center;">SeeChords Ingest</h1>
  <form method="POST" action="/ingest/login" style="background:#16213e;padding:32px;border-radius:12px;">
    {% if login_error %}
    <div style="color:#e94560;font-size:0.9em;margin-bottom:12px;">{{ login_error }}</div>
    {% endif %}
    <label style="display:block;margin-bottom:6px;font-size:0.9em;color:#888;">Password</label>
    <input type="password" name="password" autofocus
           style="width:100%;padding:10px;margin-bottom:16px;background:#0f3460;border:1px solid #333;border-radius:4px;color:#eee;font-family:inherit;font-size:0.95em;">
    <button type="submit" style="width:100%;padding:12px;background:#e94560;color:#fff;border:none;border-radius:4px;font-family:inherit;font-size:0.95em;font-weight:600;cursor:pointer;">
      Log in
    </button>
  </form>
</div>
{% else %}
```

At the very end of the file, before `</body>`, add the closing:

```html
{% endif %}
```

- [ ] **Step 2: Escape any Jinja2 conflicts**

The existing ingest.html uses `{{ }}` and `{% %}` in JavaScript template literals. Search for these patterns and wrap any JavaScript blocks that contain them with `{% raw %}...{% endraw %}` tags. Alternatively, since the current file is served via `open().read()` (not `render_template`), check if any such patterns exist:

Search for `{{` and `{%` in the JS sections. If found, wrap the `<script>` block with:
```html
{% raw %}
<script>
... existing JS ...
</script>
{% endraw %}
```

- [ ] **Step 3: Add logout button to the ingest UI**

Add a logout button in the header area (near the `<h1>` tag, line ~393 area):

```html
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
  <h1>SeeChords – Ingest</h1>
  <form method="POST" action="/ingest/logout" style="margin:0;">
    <button type="submit" style="padding:6px 14px;background:#333;color:#888;border:1px solid #555;border-radius:4px;cursor:pointer;font-size:0.8em;">Logout</button>
  </form>
</div>
```

Remove the existing standalone `<h1>` tag.

- [ ] **Step 4: Test login flow in browser**

1. Open `http://127.0.0.1:5005/ingest` — should see login form
2. Enter wrong password — should see "Incorrect password" error
3. Enter correct password — should redirect to ingest UI
4. Click "Logout" — should return to login form

- [ ] **Step 5: Commit**

```bash
git add server/templates/ingest.html
git commit -m "feat(ingest): add login form and logout button"
```

---

### Task 3: Home view — YouTube input + saved versions list

**Files:**
- Modify: `server/app.py` (add `GET /api/ingest/versions`)
- Modify: `server/templates/ingest.html` (restructure upload-section into home view)

- [ ] **Step 1: Add `/api/ingest/versions` endpoint in app.py**

Add this after the existing `_list_versions` helper (~line 2413):

```python
@app.route('/api/ingest/versions')
@_require_ingest_auth
def ingest_all_versions():
    """List all ingest-created and verified versions for the home page."""
    con = _get_db()
    rows = con.execute('''
        SELECT cv.version_id, cv.video_id, cv.title, cv.key, cv.bpm,
               cv.source, cv.analyzed_at, cv.is_active, cv.chords
        FROM chord_versions cv
        WHERE cv.source IN ('ingest-edit', 'verified')
        ORDER BY cv.analyzed_at DESC
    ''').fetchall()
    con.close()

    versions = []
    seen_videos = set()
    for r in rows:
        d = {
            'versionId': r[0], 'videoId': r[1], 'title': r[2],
            'key': r[3], 'bpm': r[4], 'source': r[5],
            'analyzedAt': r[6], 'isActive': bool(r[7]),
        }
        chords = json.loads(r[8]) if isinstance(r[8], str) else (r[8] or [])
        d['segmentCount'] = len(chords)
        # Show only the most recent version per video
        vid = r[1]
        if vid in seen_videos:
            continue
        seen_videos.add(vid)
        versions.append(d)
    return jsonify({'versions': versions})
```

- [ ] **Step 2: Restructure the upload-section into a home view**

Replace the current `upload-section` div (lines ~393-424) with a home view layout. Keep the YouTube URL input (remove the file upload tab since new songs always come from YouTube). Add a versions list container below:

```html
<div id="home-section">
  <div class="upload-area">
    <label>Paste a YouTube link</label>
    <div style="display:flex;gap:8px;">
      <input type="text" id="youtube-url" placeholder="https://www.youtube.com/watch?v=..." style="flex:1;">
      <button class="btn-primary" id="btn-analyze" onclick="startYouTubeIngest()">Analyze</button>
    </div>
    <div id="status" style="margin-top:8px;"></div>
  </div>

  <div class="upload-area" style="margin-top:12px;">
    <h2>Saved Versions</h2>
    <div id="versions-list" style="font-size:0.85em;">Loading...</div>
  </div>
</div>
```

- [ ] **Step 3: Add JS to fetch and render saved versions list**

Add a function that fetches `/api/ingest/versions` and renders the list. Each row is clickable to load that version into the editor:

```javascript
async function loadVersionsList() {
  const el = document.getElementById('versions-list');
  if (!el) return;
  try {
    const resp = await fetch(API + '/api/ingest/versions');
    if (resp.status === 401) { location.reload(); return; }
    const data = await resp.json();
    const versions = data.versions || [];
    if (!versions.length) {
      el.textContent = 'No saved versions yet.';
      return;
    }
    el.innerHTML = '';
    const table = document.createElement('table');
    table.className = 'review-table';
    table.innerHTML = `<thead><tr>
      <th>Title</th><th>Key</th><th>BPM</th><th>Source</th><th>Chords</th><th>Date</th>
    </tr></thead>`;
    const tbody = document.createElement('tbody');
    for (const v of versions) {
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
      tr.onclick = () => loadSavedVersion(v.versionId, v.videoId);
      const sourceLabel = v.source === 'verified' ? 'Made by Jin' : 'Draft';
      const date = new Date(v.analyzedAt * 1000).toLocaleDateString();
      tr.innerHTML = `
        <td>${v.title || '—'}</td>
        <td>${v.key || '—'}</td>
        <td>${v.bpm || '—'}</td>
        <td>${sourceLabel}</td>
        <td>${v.segmentCount}</td>
        <td>${date}</td>`;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    el.appendChild(table);
  } catch (e) {
    el.textContent = 'Error loading versions.';
  }
}
```

Call `loadVersionsList()` on page load (in the existing `DOMContentLoaded` or inline at the end of the script).

- [ ] **Step 4: Add `loadSavedVersion` function**

This loads a previously saved version into the editor. It fetches the version data, sets up YouTube playback, and opens the editor:

```javascript
async function loadSavedVersion(versionId, videoId) {
  document.getElementById('home-section').classList.add('hidden');
  document.getElementById('review-section').classList.remove('hidden');
  document.getElementById('playalong-section').classList.remove('hidden');

  try {
    const resp = await fetch(API + '/api/version/' + versionId);
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }

    // Store state for save operations
    _currentIngestVideoId = videoId;
    paCurrentVideoId = videoId;

    // Parse chords into segments
    const chords = typeof data.chords === 'string' ? JSON.parse(data.chords) : data.chords;
    currentSegments = chords.map(c => ({
      start: c.start, end: c.end, chord: c.chord,
    }));

    // Init the play-along view with YouTube streaming
    initPlayAlongFromVersion(data, videoId);
  } catch (e) {
    alert('Error loading version: ' + e.message);
  }
}
```

- [ ] **Step 5: Add `startYouTubeIngest` function**

Replace or rename the existing YouTube ingest trigger to work with the new home view:

```javascript
async function startYouTubeIngest() {
  const urlInput = document.getElementById('youtube-url');
  const raw = (urlInput.value || '').trim();
  const m = raw.match(/(?:v=|youtu\.be\/|\/shorts\/)([a-zA-Z0-9_-]{11})/);
  if (!m) { alert('Please paste a valid YouTube URL.'); return; }
  const videoId = m[1];

  document.getElementById('btn-analyze').disabled = true;
  document.getElementById('status').textContent = 'Analyzing...';

  const form = new FormData();
  form.append('videoId', videoId);

  try {
    const resp = await fetch(API + '/api/ingest/youtube', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.error) { throw new Error(data.error); }
    currentJobId = data.jobId;
    _currentIngestVideoId = videoId;
    pollIngestStatus(videoId);
  } catch (e) {
    document.getElementById('status').innerHTML = `<span class="status-error">${e.message}</span>`;
    document.getElementById('btn-analyze').disabled = false;
  }
}
```

- [ ] **Step 6: Show home view and hide editor when clicking Back**

Update the back button handler to return to the home view and refresh the versions list:

```javascript
function backToHome() {
  document.getElementById('review-section').classList.add('hidden');
  document.getElementById('playalong-section').classList.add('hidden');
  document.getElementById('home-section').classList.remove('hidden');
  // Stop playback
  if (typeof paStopPlay === 'function') paStopPlay();
  // Refresh the versions list
  loadVersionsList();
}
```

- [ ] **Step 7: Test home view**

1. Open `http://127.0.0.1:5005/ingest` (after logging in)
2. Should see YouTube input + saved versions list (may be empty)
3. Paste a YouTube URL and click Analyze — should start analysis, then open editor
4. Click Back — should return to home view

- [ ] **Step 8: Commit**

```bash
git add server/app.py server/templates/ingest.html
git commit -m "feat(ingest): add home view with YouTube input and saved versions list"
```

---

### Task 4: YouTube IFrame playback in editor

**Files:**
- Modify: `server/templates/ingest.html` (replace `<audio>` with YT IFrame player)

- [ ] **Step 1: Add YouTube IFrame API script tag**

In the `<head>` of ingest.html, add:

```html
<script src="https://www.youtube.com/iframe_api"></script>
```

- [ ] **Step 2: Add hidden YT player element**

In the editor section (near the existing `<audio>` element area), add a hidden div for the YT player and replace the audio element:

```html
<div id="ytPlayerEl" style="position:absolute;width:1px;height:1px;overflow:hidden;"></div>
```

Remove or hide the existing `<audio id="audio-player" controls>` element. Keep the existing play/pause button UI.

- [ ] **Step 3: Port YT player logic from play.html**

Add these variables and functions to the `<script>` block, adapted from `play.html`:

```javascript
let ytPlayer = null;

function _isYTPlaying() {
  return ytPlayer && typeof ytPlayer.getPlayerState === 'function' &&
         ytPlayer.getPlayerState() === YT.PlayerState.PLAYING;
}

function _getYTCurrentTime() {
  return ytPlayer && typeof ytPlayer.getCurrentTime === 'function'
    ? ytPlayer.getCurrentTime() : 0;
}

function _getYTDuration() {
  return ytPlayer && typeof ytPlayer.getDuration === 'function'
    ? ytPlayer.getDuration() : 0;
}

window.onYouTubeIframeAPIReady = function() {
  if (window._ytPendingVideoId) {
    _createYTPlayer(window._ytPendingVideoId);
    window._ytPendingVideoId = null;
  }
};

function _createYTPlayer(videoId) {
  if (ytPlayer) { ytPlayer.destroy(); ytPlayer = null; }

  const old = document.getElementById('ytPlayerEl');
  if (!old) return;
  const fresh = document.createElement('div');
  fresh.id = 'ytPlayerEl';
  old.replaceWith(fresh);

  ytPlayer = new YT.Player('ytPlayerEl', {
    width: 1, height: 1,
    videoId,
    playerVars: { autoplay: 0, controls: 0, rel: 0, fs: 0, disablekb: 1, playsinline: 1 },
    events: {
      onReady() { /* player ready */ },
      onStateChange(e) {
        if (e.data === YT.PlayerState.PLAYING) startTracking();
        else if (e.data === YT.PlayerState.PAUSED || e.data === YT.PlayerState.ENDED) stopTracking();
      },
    },
  });
}

function initYTPlayerForVideo(videoId) {
  if (typeof YT !== 'undefined' && YT.Player) {
    _createYTPlayer(videoId);
  } else {
    window._ytPendingVideoId = videoId;
  }
}
```

- [ ] **Step 4: Wire up play/pause/seek to YT player**

Update the existing `paTogglePlay` to use the YT player:

```javascript
function paTogglePlay() {
  if (!ytPlayer) return;
  if (_isYTPlaying()) {
    ytPlayer.pauseVideo();
  } else {
    ytPlayer.playVideo();
  }
}

function paSeekTo(time) {
  if (ytPlayer && typeof ytPlayer.seekTo === 'function') {
    ytPlayer.seekTo(time, true);
  }
}
```

Update the tracking loop (`paTrackLoop` or equivalent) to use `_getYTCurrentTime()` instead of `audioEl.currentTime`.

- [ ] **Step 5: Initialize YT player when entering editor**

In both `loadSavedVersion` and the analysis-complete handler, call `initYTPlayerForVideo(videoId)` to set up playback.

For the analysis flow, after the job completes and results are loaded, instead of setting `audioEl.src`, call:
```javascript
initYTPlayerForVideo(_currentIngestVideoId);
```

- [ ] **Step 6: Test playback**

1. Load a saved version — YT player should initialize, play/pause should work
2. Analyze a new YouTube link — after analysis completes, playback should work via YT player
3. Seek by clicking the progress bar — should seek correctly

- [ ] **Step 7: Commit**

```bash
git add server/templates/ingest.html
git commit -m "feat(ingest): switch playback to YouTube IFrame API"
```

---

### Task 5: Split Save into Save (draft) and Save & Upload (live)

**Files:**
- Modify: `server/templates/ingest.html` (buttons + JS handlers)

- [ ] **Step 1: Replace save buttons in the editor**

Find the existing save button area. Replace with two buttons:

```html
<button class="btn-save" id="btn-save-draft" onclick="saveDraft()">Save</button>
<button class="btn-primary" id="btn-save-upload" onclick="saveAndUpload()">Save &amp; Upload</button>
<span id="save-status"></span>
```

Also update the play-along header — replace the single `pa-save-badge` with two badges:

```html
<span class="pa-badge" id="pa-save-draft-badge" onclick="saveDraft()" style="background:#1a1a36;cursor:pointer;border-color:rgba(78,204,163,.25);color:rgba(78,204,163,.75);" title="Save as draft (not visible to users)">Save</span>
<span class="pa-badge" id="pa-save-upload-badge" onclick="saveAndUpload()" style="background:#2a6e2a;cursor:pointer;" title="Save and make live for all users">Save &amp; Upload</span>
```

- [ ] **Step 2: Implement `saveDraft` function**

This calls the existing `save-version` endpoint with `source='ingest-edit'`:

```javascript
async function saveDraft() {
  const segments = getCurrentSegments();
  if (!segments.length) { alert('No chords to save.'); return; }

  const videoId = _currentIngestVideoId || paCurrentVideoId;
  if (!videoId || videoId.length !== 11) {
    alert('A valid YouTube video ID is required.');
    return;
  }

  const songName = _currentSongName || 'unknown';
  const bpm = parseFloat(document.getElementById('pa-bpm-badge')?.textContent?.replace('BPM: ', '')) || 120;
  const badge = document.getElementById('pa-save-draft-badge');

  try {
    const resp = await fetch(API + '/api/ingest/' + (currentJobId || videoId) + '/save-version', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        segments, songName, videoId,
        key: paBaseKey || paCurrentKey, bpm,
        beatTimes: paBeatTimes,
      }),
    });
    const data = await resp.json();
    if (resp.ok && data.saved) {
      if (badge) {
        badge.textContent = 'Saved!';
        setTimeout(() => { badge.textContent = 'Save'; }, 2000);
      }
    } else {
      alert('Save failed: ' + (data.error || 'Unknown error'));
    }
  } catch (e) {
    alert('Save error: ' + e.message);
  }
}
```

- [ ] **Step 3: Implement `saveAndUpload` function**

This calls `promote-verified` (existing endpoint) which saves with `source='verified'` and writes .lab files:

```javascript
async function saveAndUpload() {
  const segments = getCurrentSegments();
  if (!segments.length) { alert('No chords to save.'); return; }

  const videoId = _currentIngestVideoId || paCurrentVideoId;
  if (!videoId || videoId.length !== 11) {
    alert('A valid YouTube video ID is required.');
    return;
  }

  const songName = _currentSongName || 'unknown';
  const bpm = parseFloat(document.getElementById('pa-bpm-badge')?.textContent?.replace('BPM: ', '')) || 120;
  const badge = document.getElementById('pa-save-upload-badge');

  if (!confirm('This will make these chords live for all SeeChords users. Continue?')) return;

  try {
    const resp = await fetch(API + '/api/ingest/' + (currentJobId || videoId) + '/promote-verified', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        segments, songName, videoId,
        key: paBaseKey || paCurrentKey, bpm,
        beatTimes: paBeatTimes,
      }),
    });
    const data = await resp.json();
    if (resp.ok && data.saved) {
      paCurrentVideoId = data.videoId;
      if (badge) {
        badge.textContent = 'Uploaded!';
        badge.style.background = '#1a8c1a';
        setTimeout(() => { badge.textContent = 'Save & Upload'; badge.style.background = '#2a6e2a'; }, 3000);
      }
    } else {
      alert('Upload failed: ' + (data.error || 'Unknown error'));
    }
  } catch (e) {
    alert('Upload error: ' + e.message);
  }
}
```

- [ ] **Step 4: Add `getCurrentSegments` helper**

This extracts the current chord segments from the play-along state. It should reuse the existing segment extraction logic from `paSaveAndVerify`:

```javascript
function getCurrentSegments() {
  return (currentSegments || []).map(s => ({
    chord: s.sheetChord || s.chord || 'N',
    start: s.start,
    end: s.end,
  }));
}
```

- [ ] **Step 5: Remove old `paSaveAndVerify` and single save badge references**

Remove the old `paSaveAndVerify` function and the old `pa-save-badge` element. Update any references to use the new split buttons.

- [ ] **Step 6: Test save flows**

1. Analyze a song, edit chords, click "Save" — should save as draft
2. Check the home view — draft should appear with "Draft" label
3. Load the draft, click "Save & Upload" — should show confirmation, then save as verified
4. Check the home view — should now show "Made by Jin"

- [ ] **Step 7: Commit**

```bash
git add server/templates/ingest.html
git commit -m "feat(ingest): split save into Save (draft) and Save & Upload (live)"
```

---

### Task 6: Remove chord lines feature

**Files:**
- Modify: `server/templates/ingest.html`

- [ ] **Step 1: Remove chord lines CSS**

Remove the following CSS rules (~lines 210-276):
- `.pa-lines-wrap` and related styles
- `.pa-line`, `.pa-line-num`, `.pa-block`, `.pa-block-chord`, `.pa-block-time`, `.pa-block-beats`, `.pa-block-word` styles
- `.pa-chord-bank` and `.pa-bank-chip` styles

- [ ] **Step 2: Remove chord lines HTML**

Remove the chord lines section from the play-along HTML (~lines 545-551):
```html
<div class="pa-lines-wrap collapsed">
  ...
  <div id="pa-lines"></div>
</div>
```

Also remove the chord bank HTML.

- [ ] **Step 3: Remove chord lines JavaScript**

Remove these functions and their references:
- `paToggleChordLines`
- `paRenderChordLines` / the chord line rendering code (~line 1461+)
- Line highlighting logic in the tracking loop that references `#pa-lines .pa-line`
- Chord bank rendering and click handlers
- Any `pa-block` click/edit handlers for the line grid

Keep the chord trio cards (prev/active/next) and the beat timeline — those stay.

- [ ] **Step 4: Test that play-along still works**

1. Open the editor with a song
2. Chord trio cards should display correctly
3. Beat timeline should work
4. Play/pause, seek should work
5. No JS errors in console

- [ ] **Step 5: Commit**

```bash
git add server/templates/ingest.html
git commit -m "refactor(ingest): remove chord lines feature from editor"
```

---

### Task 7: Wire up `initPlayAlongFromVersion` for loading saved versions

**Files:**
- Modify: `server/templates/ingest.html`

- [ ] **Step 1: Implement `initPlayAlongFromVersion`**

This function initializes the play-along view from a saved version's data (fetched from `/api/version/<id>`):

```javascript
function initPlayAlongFromVersion(data, videoId) {
  const chords = typeof data.chords === 'string' ? JSON.parse(data.chords) : (data.chords || []);
  const beats = typeof data.beat_times === 'string' ? JSON.parse(data.beat_times) : (data.beat_times || []);

  currentSegments = chords.map(c => ({
    start: c.start, end: c.end, chord: c.chord,
  }));

  _currentSongName = data.title || 'unknown';
  paBaseKey = data.key || '';
  paCurrentKey = paBaseKey;
  paBeatTimes = beats;
  paTransposeSteps = 0;

  // Update header badges
  const keyBadge = document.getElementById('pa-key-badge');
  const bpmBadge = document.getElementById('pa-bpm-badge');
  if (keyBadge) keyBadge.textContent = 'Key: ' + (data.key || '—');
  if (bpmBadge) bpmBadge.textContent = 'BPM: ' + (data.bpm || '—');

  // Set up the play-along display (chord trio, timeline, etc.)
  // Reuse existing initPlayAlong logic
  initPlayAlong();

  // Start YouTube player
  initYTPlayerForVideo(videoId);
}
```

- [ ] **Step 2: Update analysis-complete handler to use same flow**

When the YouTube analysis job completes, instead of using `<audio>` for playback, call `initYTPlayerForVideo` and initialize the play-along the same way.

In the existing polling success handler (where `job.status === 'done'`), update to:

```javascript
_currentIngestVideoId = videoId;
paCurrentVideoId = videoId;
currentSegments = job.segments; // however they're currently assigned
_currentSongName = job.songName;
// ... existing play-along init ...
initYTPlayerForVideo(videoId);
```

- [ ] **Step 3: Show Back button in editor, hide home section**

Ensure the Back button is visible when in the editor:

```javascript
document.getElementById('pa-back-badge').style.display = '';
document.getElementById('pa-back-badge').onclick = backToHome;
```

- [ ] **Step 4: Test full flows**

1. **New song flow:** Paste YouTube URL → Analyze → edit chords → Save → Back → see in list
2. **Load saved flow:** Click a saved version → chords load → YT plays → edit → Save & Upload
3. **Back navigation:** Click Back from editor → home view with updated list

- [ ] **Step 5: Commit**

```bash
git add server/templates/ingest.html
git commit -m "feat(ingest): wire up saved version loading and YT playback in editor"
```

---

### Task 8: Attribution labels in extension/web player

**Files:**
- Modify: `server/templates/play.html` (display "Made by Jin" / "Made by SeeChords")
- Modify: `extension/content.js` (display attribution in overlay)

- [ ] **Step 1: Check how source is returned to clients**

The `/api/chords/<videoId>` endpoint returns chord data. Check if it includes `source`. If not, add it to the response.

In `app.py`, find the chords endpoint response and ensure it includes:
```python
'source': version_row['source'] if version_row else 'model'
```

- [ ] **Step 2: Add attribution label to play.html**

In the player header area, add a badge that shows the source:

```javascript
const sourceLabel = data.source === 'verified' ? 'Made by Jin' : 'Made by SeeChords';
// Add to the header or info area
```

- [ ] **Step 3: Add attribution label to extension overlay**

In `content.js`, when displaying chords, show the source attribution:

```javascript
const sourceLabel = chordsData.source === 'verified' ? 'Made by Jin' : 'Made by SeeChords';
// Add small text to the overlay header
```

- [ ] **Step 4: Test attribution display**

1. View a song with verified chords in web player — should show "Made by Jin"
2. View a song with model chords — should show "Made by SeeChords"
3. Same in extension overlay

- [ ] **Step 5: Commit**

```bash
git add server/app.py server/templates/play.html extension/content.js
git commit -m "feat: show attribution labels (Made by Jin / Made by SeeChords)"
```

---

### Task 9: Deploy config and final cleanup

**Files:**
- Modify: `server/fly.toml` (add `SECRET_KEY` env note)

- [ ] **Step 1: Set Fly secrets**

```bash
fly secrets set INGEST_PASSWORD="<your-password>" -a seechords
fly secrets set SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" -a seechords
```

- [ ] **Step 2: Remove stale file-upload-only code**

In the home view, the old file upload tab (`audio-file-area`) and its handler (`startIngest` for file uploads) can be removed since new songs always come from YouTube. Remove:
- The file upload HTML (`<input type="file" id="audio-file">`)
- The tab choice buttons for Audio File / YouTube
- The `startIngest` function that handles file uploads (if the YouTube flow has its own function now)

Keep the `startIngest` function if it's still referenced as a fallback, or rename to clarify.

- [ ] **Step 3: Remove old saved-labs section**

The old `saved-labs-section` (line ~426-432) that listed `.lab` files is replaced by the new versions list. Remove:
- `<div id="saved-labs-section">`
- The `loadSavedLabs()` function
- The `openLabEditor()` function (if lab-file editing is fully replaced by version-based editing)

- [ ] **Step 4: End-to-end test**

Full flow test:
1. Deploy locally with `INGEST_PASSWORD` and `SECRET_KEY` set
2. Visit `/ingest` — see login form
3. Log in with correct password
4. See home view with YouTube input and versions list
5. Paste a YouTube URL, click Analyze
6. Wait for analysis, see editor with pre-filled chords
7. Edit some chords using the chord bank
8. Click "Save" — version saved as draft
9. Click Back — see draft in versions list
10. Click the draft — loads into editor with YT playback
11. Click "Save & Upload" — confirm dialog, then saved as verified
12. Back — version now shows "Made by Jin"
13. Check web player for same video — shows "Made by Jin" attribution
14. Click Logout — returns to login form

- [ ] **Step 5: Commit**

```bash
git add server/app.py server/templates/ingest.html
git commit -m "chore(ingest): remove old file upload and lab editor, clean up"
```

---

## Task Dependency Order

```
Task 1 (auth routes) → Task 2 (login form) → Task 3 (home view) → Task 4 (YT playback)
                                                                  → Task 5 (split save buttons)
                                                                  → Task 6 (remove chord lines)
                                                                  → Task 7 (wire up version loading)
                                                                  → Task 8 (attribution labels)
                                                                  → Task 9 (cleanup + deploy)
```

Tasks 5 and 6 can be done in parallel after Task 3. Task 4 should come after Task 3. Task 7 depends on Tasks 3, 4, and 5 (it wires up `loadSavedVersion` from Task 3 with YT playback from Task 4 and save functions from Task 5). Task 8 is independent of 4-7 but should come after Task 3. Task 9 is last.
