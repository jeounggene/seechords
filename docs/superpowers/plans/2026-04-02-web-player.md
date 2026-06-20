# SeeChords Web Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/play` page where users upload an audio file, see it analyzed, then play it with a beat-synced chord timeline — no video, no YouTube, works on mobile.

**Architecture:** Single Flask route renders a self-contained `play.html` template (inline CSS + JS, matching site pattern). Audio plays in-browser via `URL.createObjectURL()` from the user's uploaded File — no server-side audio storage. Chord logic (timeline, cards, transpose, SVG diagrams) is ported directly from `extension/content.js` into the template's `<script>` block. Analysis uses the existing `POST /api/analyze` + `GET /api/status/<job_id>` endpoints unchanged.

**Tech Stack:** Flask/Jinja2, vanilla JS (ES2020), HTML5 Audio API, requestAnimationFrame

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `server/app.py` | Add `GET /play` route |
| Create | `server/templates/play.html` | Full player page: upload → progress → player |

No new dependencies. No new API endpoints.

---

## Task 1: Flask Route + Page Skeleton

**Files:**
- Modify: `server/app.py` (after the `/report-bug` route, around line 1677)
- Create: `server/templates/play.html`

- [ ] **Step 1: Add the `/play` route to `server/app.py`**

Find the `/report-bug` route (line ~1671) and add after it:

```python
@app.route('/play')
def play():
    """Web-based chord player: upload audio, get beat-synced chord display."""
    return render_template('play.html')
```

- [ ] **Step 2: Create the skeleton `server/templates/play.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <meta name="description" content="SeeChords Player — upload audio and see beat-synced chords">
  <title>SeeChords Player</title>
  <style>
    :root {
      --bg:           #080812;
      --bg-card:      #12122a;
      --bg-input:     #1a1a36;
      --border:       #2a2a50;
      --accent:       #6c47ff;
      --accent-hover: #8264ff;
      --gold:         #ffd700;
      --text:         #f0f0f8;
      --muted:        #8888aa;
      --radius:       12px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      min-height: 100dvh;
    }
    a { color: var(--accent-hover); text-decoration: none; }

    /* ── Header ── */
    .header {
      padding: 16px 20px;
      background: linear-gradient(135deg, #110d2e 0%, #1a0938 45%, #080812 100%);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.43em;
      font-size: clamp(1.4rem, 3.5vw, 1.8rem);
      text-decoration: none;
    }
    .brand-logo {
      width: 1.5556em;
      height: 1.5556em;
      border-radius: 0.214em;
      object-fit: contain;
    }
    .brand-name { font-weight: 800; color: #fff; letter-spacing: -0.5px; }
    .header-sub { color: var(--muted); font-size: 0.95rem; margin-left: auto; }

    /* ── Upload Phase ── */
    #uploadPhase {
      max-width: 560px;
      margin: 48px auto;
      padding: 0 20px;
    }
    .upload-zone {
      border: 2px dashed var(--border);
      border-radius: var(--radius);
      padding: 48px 24px;
      text-align: center;
      cursor: pointer;
      transition: border-color .2s, background .2s;
    }
    .upload-zone.drag-over {
      border-color: var(--accent);
      background: rgba(108,71,255,.08);
    }
    .upload-icon { font-size: 3rem; margin-bottom: 16px; }
    .upload-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 8px; }
    .upload-sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; }
    .upload-btn {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 12px 28px;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      transition: background .2s;
    }
    .upload-btn:hover { background: var(--accent-hover); }
    #fileInput { display: none; }

    /* ── Progress Phase ── */
    #progressPhase { display: none; max-width: 560px; margin: 48px auto; padding: 0 20px; }
    .prog-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 6px; }
    .prog-file { color: var(--muted); font-size: 0.9rem; margin-bottom: 20px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .prog-track {
      height: 8px; background: var(--bg-input); border-radius: 4px; overflow: hidden; margin-bottom: 10px;
    }
    .prog-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width .4s ease; width: 0%; }
    .prog-msg { color: var(--muted); font-size: 0.9rem; }
    .prog-err { color: #ff6b6b; font-size: 0.9rem; margin-top: 8px; display: none; }

    /* ── Player Phase ── */
    #playerPhase { display: none; }
  </style>
</head>
<body>

<header class="header">
  <a href="/" class="brand">
    <img class="brand-logo" src="/static/icon128.png" alt="SeeChords" width="128" height="128">
    <span class="brand-name">SeeChords</span>
  </a>
  <span class="header-sub">Web Player</span>
</header>

<!-- Phase 1: Upload -->
<div id="uploadPhase">
  <div class="upload-zone" id="dropZone">
    <div class="upload-icon">🎵</div>
    <div class="upload-title">Drop an audio file</div>
    <div class="upload-sub">MP3, WAV, M4A, FLAC, AAC, OGG supported</div>
    <button class="upload-btn" id="pickBtn">Choose File</button>
    <input type="file" id="fileInput" accept=".mp3,.wav,.m4a,.flac,.aac,.ogg,.webm">
  </div>
</div>

<!-- Phase 2: Progress -->
<div id="progressPhase">
  <div class="prog-title">Analyzing chords…</div>
  <div class="prog-file" id="progFile"></div>
  <div class="prog-track"><div class="prog-fill" id="progFill"></div></div>
  <div class="prog-msg" id="progMsg">Uploading…</div>
  <div class="prog-err" id="progErr"></div>
</div>

<!-- Phase 3: Player (populated by JS) -->
<div id="playerPhase"></div>

<script>
'use strict';
// State populated after analysis
let audioFile = null;
let audioEl   = null;
let chords    = [];
let beatTimes = [];
let beatChords = [];
let baseKey   = '';
let bpm       = 120;
let transposeSteps = 0;
let currentBeatIdx  = -999;
let currentChordIdx = -999;
let rafId = null;

// ── Phase switching ──────────────────────────────────────────
function showPhase(name) {
  document.getElementById('uploadPhase').style.display   = name === 'upload'   ? '' : 'none';
  document.getElementById('progressPhase').style.display = name === 'progress' ? '' : 'none';
  document.getElementById('playerPhase').style.display   = name === 'player'   ? '' : 'none';
}
</script>
</body>
</html>
```

- [ ] **Step 3: Verify the route works**

Start the dev server locally and visit `http://localhost:8080/play`. You should see the header and the upload drop zone. No JS errors in the console.

```bash
cd server && python app.py
```

- [ ] **Step 4: Commit**

```bash
git add server/app.py server/templates/play.html
git commit -m "feat: add /play route and skeleton template"
```

---

## Task 2: Upload Zone + Analysis + Progress Polling

**Files:**
- Modify: `server/templates/play.html` — replace the `<script>` block with upload + polling logic

- [ ] **Step 1: Add drag-and-drop and file-pick handlers**

Replace the `<script>` block content (after the state variables and `showPhase`) with:

```javascript
// ── Random 11-char videoId (must match ^[a-zA-Z0-9_-]{11}$) ──
function randomVideoId() {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-';
  return Array.from({length: 11}, () => chars[Math.floor(Math.random() * chars.length)]).join('');
}

// ── Upload zone wiring ────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

document.getElementById('pickBtn').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) startUpload(fileInput.files[0]);
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) startUpload(f);
});
```

- [ ] **Step 2: Add the `startUpload` function**

```javascript
async function startUpload(file) {
  audioFile = file;
  showPhase('progress');
  document.getElementById('progFile').textContent = file.name;
  document.getElementById('progMsg').textContent = 'Uploading…';
  setProg(5);

  const videoId = randomVideoId();
  const title = file.name.replace(/\.[^.]+$/, '').slice(0, 120) || 'Uploaded Audio';

  const fd = new FormData();
  fd.append('file', file);
  fd.append('videoId', videoId);
  fd.append('title', title);

  let jobId;
  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Upload failed');
    jobId = data.job_id;
  } catch (err) {
    showError(err.message);
    return;
  }

  pollStatus(jobId);
}

function setProg(pct) {
  document.getElementById('progFill').style.width = pct + '%';
}

function showError(msg) {
  document.getElementById('progErr').textContent = msg;
  document.getElementById('progErr').style.display = '';
  document.getElementById('progMsg').textContent = 'Analysis failed.';
}
```

- [ ] **Step 3: Add the `pollStatus` function**

```javascript
function pollStatus(jobId) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/status/${jobId}`);
      const data = await res.json();

      setProg(data.progress || 0);
      document.getElementById('progMsg').textContent = data.message || '';

      if (data.status === 'done') {
        clearInterval(interval);
        initPlayer(data);
      } else if (data.status === 'error') {
        clearInterval(interval);
        showError(data.message || 'Analysis failed.');
      }
    } catch (err) {
      // network blip — keep polling
    }
  }, 2000);
}
```

- [ ] **Step 4: Test upload + polling manually**

1. Start the server
2. Go to `/play`
3. Upload an MP3
4. Watch the progress bar advance and message update
5. After ~1-2 minutes, status should reach `done` (console.log the data for now)

Add a temporary `console.log('done', data)` inside `if (data.status === 'done')` to verify the response shape:
```javascript
// Expected shape when done:
// { status: 'done', progress: 100, chords: [{chord, start, end},...],
//   beat_times: [0.42, 0.91, ...], key: 'C', bpm: 120.3, title: '...' }
```

- [ ] **Step 5: Commit**

```bash
git add server/templates/play.html
git commit -m "feat: web player upload zone and analysis progress polling"
```

---

## Task 3: Port Chord Logic from Extension

**Files:**
- Modify: `server/templates/play.html` — add chord functions to the `<script>` block

- [ ] **Step 1: Add CHORD_DIAGRAMS constant**

Copy the `CHORD_DIAGRAMS` object verbatim from `extension/content.js` lines 70–342 into the `<script>` block. It's a large literal — paste it as-is.

```javascript
const CHORD_DIAGRAMS = {
  'C':   { f:[-1,3,2,0,1,0],     b:1 },
  // ... (full object from content.js lines 70-342)
};
```

- [ ] **Step 2: Add transposition and formatting functions**

Copy verbatim from `extension/content.js` lines 344–388:

```javascript
const CHROMATIC = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'];
const NOTE_IDX = {};
CHROMATIC.forEach((n, i) => { NOTE_IDX[n] = i; });
NOTE_IDX['Db'] = 1; NOTE_IDX['D#'] = 3; NOTE_IDX['Gb'] = 6;
NOTE_IDX['G#'] = 8; NOTE_IDX['A#'] = 10;

function transposeNote(note, steps) {
  const idx = NOTE_IDX[note];
  if (idx === undefined) return note;
  const shifted = ((idx + steps) % 12 + 12) % 12;
  const sharp = CHROMATIC[shifted];
  const flatMap = {1:'Db',3:'Eb',6:'Gb',8:'Ab',10:'Bb'};
  if (flatMap[shifted]) return flatMap[shifted];
  return sharp;
}

function transposeChord(chord, steps) {
  if (steps === 0) return chord;
  const slashIdx = chord.indexOf('/');
  if (slashIdx !== -1) {
    const base = chord.slice(0, slashIdx);
    const bass = chord.slice(slashIdx + 1);
    return transposeChord(base, steps) + '/' + transposeNote(bass, steps);
  }
  const m = chord.match(/^([A-G][#b]?)(.*)$/);
  if (!m) return chord;
  return transposeNote(m[1], steps) + m[2];
}

function formatChord(raw) {
  return raw.replace(/([A-G])#/, '$1♯').replace(/([A-G])b/, '$1♭');
}

function formatChordHTML(raw) {
  const display = formatChord(raw);
  const m = display.match(/^([A-G][♯♭]?)(.*?)$/);
  if (!m) return display.replace(/♭/g, '<span class="sc-flat">♭</span>');
  const root = m[1].replace(/♭/g, '<span class="sc-flat">♭</span>');
  if (!m[2]) return root;
  return `${root}<span class="sc-chord-suffix">${m[2]}</span>`;
}
```

- [ ] **Step 3: Add SVG diagram builder**

Copy verbatim from `extension/content.js` lines 391–455:

```javascript
function buildChordSVG(chordName) {
  const key = chordName.replace('♯','#').replace('♭','b');
  let data = CHORD_DIAGRAMS[key];
  if (!data) {
    const enharmonic = { 'D#':'Eb', 'G#':'Ab', 'A#':'Bb' };
    const re = /^([A-G]#)/;
    const m = key.match(re);
    if (m && enharmonic[m[1]]) {
      data = CHORD_DIAGRAMS[enharmonic[m[1]] + key.slice(m[1].length)];
    }
  }
  if (!data && key.includes('/')) data = CHORD_DIAGRAMS[key.split('/')[0]];
  if (!data) return '';
  const { f: strings, b: baseFret } = data;

  const SX = [6, 17, 28, 39, 50, 61];
  const NUT_Y = 16, FRET_H = 12, N_FRETS = 4, DOT_R = 4.5;
  const bottom = NUT_Y + N_FRETS * FRET_H;
  const p = [];

  if (baseFret === 1) {
    p.push(`<rect x="${SX[0]}" y="${NUT_Y-3}" width="${SX[5]-SX[0]}" height="3" fill="currentColor" opacity=".6"/>`);
  } else {
    p.push(`<line x1="${SX[0]}" y1="${NUT_Y}" x2="${SX[5]}" y2="${NUT_Y}" stroke="currentColor" stroke-width="1.2" opacity=".3"/>`);
    const lblY = NUT_Y + FRET_H * 0.75;
    p.push(`<text x="-2" y="${lblY}" text-anchor="middle" font-size="8" font-weight="700" font-family="sans-serif" fill="currentColor" opacity=".85">${baseFret}</text>`);
  }
  for (let i = 1; i <= N_FRETS; i++) {
    const y = NUT_Y + i * FRET_H;
    p.push(`<line x1="${SX[0]}" y1="${y}" x2="${SX[5]}" y2="${y}" stroke="currentColor" stroke-width=".8" opacity=".18"/>`);
  }
  for (const x of SX) {
    p.push(`<line x1="${x}" y1="${NUT_Y}" x2="${x}" y2="${bottom}" stroke="currentColor" stroke-width="1" opacity=".28"/>`);
  }
  for (let i = 0; i < 6; i++) {
    const x = SX[i], fret = strings[i];
    if (fret === -1) {
      p.push(`<text x="${x}" y="12" text-anchor="middle" font-size="10" font-family="sans-serif" fill="currentColor" opacity=".5">×</text>`);
    } else if (fret === 0) {
      p.push(`<circle cx="${x}" cy="8" r="3.5" fill="none" stroke="currentColor" stroke-width="1" opacity=".45"/>`);
    }
  }
  for (let i = 0; i < 6; i++) {
    const fret = strings[i];
    if (fret > 0) {
      const x = SX[i], y = NUT_Y + (fret - 0.5) * FRET_H;
      p.push(`<circle cx="${x}" cy="${y}" r="${DOT_R}" fill="currentColor" opacity=".88"/>`);
    }
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-8 0 84 72" width="140" height="149" aria-hidden="true">${p.join('')}</svg>`;
}
```

- [ ] **Step 4: Add beat-building and lookup functions**

Copy verbatim from `extension/content.js` lines 458–505:

```javascript
function buildBeatChords() {
  if (!beatTimes.length || !chords.length) return [];
  const result = [];
  let bi = 0, lastSegIdx = -1;
  while (bi < beatTimes.length) {
    const t = beatTimes[bi];
    const nextBt = beatTimes[bi + 1] ?? (t + (t - (beatTimes[bi - 1] ?? t - 0.5)));
    const segIdxByBt = chords.findIndex(c => c.start <= t && c.end > t);
    const mid = (t + nextBt) / 2;
    const segIdxByMid = chords.findIndex(c => c.start <= mid && c.end > mid);
    const segIdx = segIdxByBt >= 0 ? segIdxByBt : segIdxByMid;
    const c = segIdx >= 0 ? chords[segIdx] : null;
    const actualIdx = segIdx >= 0 ? segIdx : -1;
    const chordName = c ? c.chord : 'N';
    if (result.length && result[result.length - 1].chord === chordName && actualIdx === lastSegIdx) {
      result[result.length - 1].beatCount++;
    } else {
      result.push({ chord: chordName, beatStart: bi, beatCount: 1 });
      lastSegIdx = actualIdx;
    }
    bi++;
  }
  return result;
}

function findBeatAt(t) {
  if (!beatTimes.length) return -1;
  if (t < beatTimes[0]) return -1;
  let bi = 0;
  for (let i = 0; i < beatTimes.length; i++) {
    if (beatTimes[i] <= t) bi = i; else break;
  }
  return bi;
}

function findBeatChordAt(t) {
  const bi = findBeatAt(t);
  if (bi < 0) return -1;
  for (let i = beatChords.length - 1; i >= 0; i--) {
    if (beatChords[i].beatStart <= bi) return i;
  }
  return -1;
}
```

- [ ] **Step 5: Verify no errors**

Open browser console on `/play`. The script block should parse without errors even though `initPlayer` isn't wired yet. Run in the console:

```javascript
transposeChord('Am7', 2)  // → 'Bm7'
buildChordSVG('G').slice(0, 20)  // → '<svg xmlns="http://w'
```

- [ ] **Step 6: Commit**

```bash
git add server/templates/play.html
git commit -m "feat: port chord logic (diagrams, transpose, beat builders) into web player"
```

---

## Task 4: Player HTML Structure + CSS

**Files:**
- Modify: `server/templates/play.html` — add player HTML into `#playerPhase` and CSS styles

- [ ] **Step 1: Add player CSS to the `<style>` block**

Append to the existing `<style>` block:

```css
/* ── Player wrapper ── */
.player {
  display: flex;
  flex-direction: column;
  height: calc(100dvh - 57px); /* subtract header */
  overflow: hidden;
}

/* ── Song header bar ── */
.player-header {
  padding: 10px 16px;
  background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}
.song-title {
  font-weight: 700;
  font-size: 1rem;
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.badge {
  font-size: 0.8rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 99px;
  border: 1px solid var(--border);
  color: var(--muted);
  white-space: nowrap;
  flex-shrink: 0;
}
.badge-key { border-color: rgba(255,215,0,.25); color: rgba(255,215,0,.75); }
.badge-bpm { border-color: rgba(108,71,255,.25); color: rgba(180,123,255,.75); }

/* ── Transpose controls ── */
.transpose-ctrl {
  display: flex;
  align-items: center;
  gap: 4px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 99px;
  padding: 2px 8px;
}
.ctrl-btn {
  background: none;
  border: none;
  color: var(--muted);
  font-size: 1rem;
  cursor: pointer;
  padding: 6px 8px;
  border-radius: 4px;
  min-width: 36px;
  min-height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color .15s, background .15s;
  -webkit-tap-highlight-color: transparent;
}
.ctrl-btn:hover, .ctrl-btn:active { color: var(--text); background: var(--border); }
.ctrl-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--muted);
  min-width: 60px;
  text-align: center;
  white-space: nowrap;
}

/* ── Timeline ── */
.timeline-wrap {
  flex-shrink: 0;
  height: 68px;
  overflow-x: auto;
  overflow-y: hidden;
  border-bottom: 1px solid rgba(255,255,255,.07);
  scrollbar-width: none;
  background: #0a0a1a;
}
.timeline-wrap::-webkit-scrollbar { display: none; }
.tl-row {
  display: flex;
  align-items: stretch;
  height: 100%;
}
.tl-pad { flex-shrink: 0; width: 68px; height: 100%; }
.tl-beat {
  flex-shrink: 0;
  position: relative;
  width: 68px;
  height: 100%;
  border-right: 1px solid rgba(255,255,255,.09);
  display: flex;
  align-items: flex-end;
  overflow: visible;
  cursor: pointer;
  background: transparent;
  transition: background .15s;
  -webkit-tap-highlight-color: transparent;
}
.tl-beat:hover, .tl-beat:active { background: rgba(255,255,255,.06); }
.tl-beat.active { background: rgba(255,255,255,.18); }
.tl-beat.measure-start { border-left: 2px solid rgba(255,255,255,.32); }
.tl-beat.empty { cursor: default; pointer-events: none; border-right-color: rgba(255,255,255,.05); }
.tl-beat.empty:hover { background: transparent; }
.beat-name {
  position: absolute;
  bottom: 4px;
  left: 5px;
  font-weight: 700;
  font-size: 1.1rem;
  line-height: 1;
  color: rgba(255,255,255,.75);
  white-space: nowrap;
  pointer-events: none;
}
.tl-beat.active .beat-name { color: #fff; }
.beat-name.beat-n { font-size: 0.75rem; font-weight: 600; color: var(--muted); }
.sc-chord-suffix { font-size: 0.9em; font-weight: 400; opacity: .8; }
.sc-flat { font-size: 0.9em; }

/* ── Chord cards ── */
.chord-trio {
  display: flex;
  flex: 1;
  overflow: hidden;
}
.chord-card {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 12px 8px;
  gap: 8px;
  border-right: 1px solid var(--border);
  overflow: hidden;
}
.chord-card:last-child { border-right: none; }
.chord-card.card-prev, .chord-card.card-next {
  opacity: 0.4;
  flex: 0.6;
}
.chord-card.card-active { background: rgba(108,71,255,.06); }
.chord-name {
  font-size: clamp(1.8rem, 6vw, 3rem);
  font-weight: 800;
  line-height: 1;
  text-align: center;
  color: var(--text);
}
.chord-card.card-active .chord-name { color: #fff; }
.chord-diagram { color: var(--text); flex-shrink: 0; }
.chord-diagram svg { display: block; }

/* ── Audio controls ── */
.audio-controls {
  flex-shrink: 0;
  padding: 12px 16px;
  padding-bottom: max(12px, env(safe-area-inset-bottom));
  background: var(--bg-card);
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.seek-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.time-display {
  font-size: 0.8rem;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  min-width: 80px;
  text-align: right;
}
.seek-bar {
  flex: 1;
  -webkit-appearance: none;
  appearance: none;
  height: 4px;
  background: var(--bg-input);
  border-radius: 2px;
  outline: none;
  cursor: pointer;
  position: relative;
}
.seek-bar::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
}
.seek-bar::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent);
  border: none;
  cursor: pointer;
}
.btn-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
}
.play-btn {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: var(--accent);
  border: none;
  color: #fff;
  font-size: 1.4rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background .15s, transform .1s;
  -webkit-tap-highlight-color: transparent;
}
.play-btn:hover { background: var(--accent-hover); }
.play-btn:active { transform: scale(0.93); }

/* ── Mobile tweaks ── */
@media (max-width: 480px) {
  .player-header { padding: 8px 12px; }
  .badge { display: none; }
  .chord-card.card-prev, .chord-card.card-next { flex: 0.5; }
  .chord-diagram svg { width: 80px; height: auto; }
}
@media (max-height: 600px) and (orientation: landscape) {
  .chord-trio { min-height: 80px; }
  .chord-diagram { display: none; }
  .play-btn { width: 44px; height: 44px; font-size: 1.1rem; }
}
```

- [ ] **Step 2: Add player HTML template (built by `buildPlayerHTML` JS function)**

Add this JS function before the closing `</script>` tag:

```javascript
function buildPlayerHTML(title) {
  return `
    <div class="player">
      <div class="player-header">
        <span class="song-title" id="songTitle">${title}</span>
        <span class="badge badge-key" id="keyBadge">Key: —</span>
        <span class="badge badge-bpm" id="bpmBadge">BPM: —</span>
        <div class="transpose-ctrl">
          <button class="ctrl-btn" id="transpDown" aria-label="Transpose down">▼</button>
          <span class="ctrl-label" id="transpLabel">Original</span>
          <button class="ctrl-btn" id="transpUp" aria-label="Transpose up">▲</button>
        </div>
      </div>
      <div class="timeline-wrap" id="timelineWrap">
        <div class="tl-row" id="tlRow"></div>
      </div>
      <div class="chord-trio">
        <div class="chord-card card-prev" id="cardPrev">
          <span class="chord-name" id="prevName"></span>
          <span class="chord-diagram" id="prevDiag"></span>
        </div>
        <div class="chord-card card-active" id="cardActive">
          <span class="chord-name" id="activeName"></span>
          <span class="chord-diagram" id="activeDiag"></span>
        </div>
        <div class="chord-card card-next" id="cardNext">
          <span class="chord-name" id="nextName"></span>
          <span class="chord-diagram" id="nextDiag"></span>
        </div>
      </div>
      <div class="audio-controls">
        <div class="seek-row">
          <input type="range" class="seek-bar" id="seekBar" min="0" max="100" step="0.1" value="0">
          <span class="time-display" id="timeDisplay">0:00 / 0:00</span>
        </div>
        <div class="btn-row">
          <button class="play-btn" id="playBtn" aria-label="Play">▶</button>
        </div>
      </div>
    </div>
  `;
}
```

- [ ] **Step 3: Verify structure renders (stub `initPlayer`)**

Temporarily add at the end of `<script>`:

```javascript
// TEMP: call initPlayer with stub data to check layout
window._testPlayer = () => {
  document.getElementById('playerPhase').innerHTML = buildPlayerHTML('Test Song');
  showPhase('player');
};
```

Open browser console on `/play`, run `_testPlayer()`. Verify the layout: header, timeline strip, three chord cards, and audio controls are visible. Remove the stub after verifying.

- [ ] **Step 4: Commit**

```bash
git add server/templates/play.html
git commit -m "feat: web player layout — timeline, chord cards, audio controls CSS + HTML"
```

---

## Task 5: Audio Controls + Seek Bar

**Files:**
- Modify: `server/templates/play.html` — add `initPlayer` and audio wiring

- [ ] **Step 1: Add `formatTime` helper**

```javascript
function formatTime(sec) {
  const s = Math.floor(sec), m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, '0')}`;
}
```

- [ ] **Step 2: Add `initPlayer` function**

This is called by `pollStatus` when the job is done:

```javascript
function initPlayer(data) {
  chords    = data.chords    || [];
  beatTimes = data.beat_times || [];
  baseKey   = data.key       || '';
  bpm       = data.bpm       || 120;
  beatChords = buildBeatChords();

  const title = data.title || audioFile.name;
  document.getElementById('playerPhase').innerHTML = buildPlayerHTML(escapeHTML(title));
  showPhase('player');

  // Badges
  document.getElementById('keyBadge').textContent = `Key: ${baseKey || '—'}`;
  document.getElementById('bpmBadge').textContent = `BPM: ${bpm}`;

  // Audio element — play from browser memory, no server round-trip
  audioEl = new Audio();
  audioEl.src = URL.createObjectURL(audioFile);
  audioEl.preload = 'auto';

  // Seek bar sync
  const seekBar  = document.getElementById('seekBar');
  const timeDisp = document.getElementById('timeDisplay');

  audioEl.addEventListener('loadedmetadata', () => {
    seekBar.max = audioEl.duration;
    timeDisp.textContent = `0:00 / ${formatTime(audioEl.duration)}`;
  });

  let userSeeking = false;
  seekBar.addEventListener('input', () => { userSeeking = true; });
  seekBar.addEventListener('change', () => {
    audioEl.currentTime = parseFloat(seekBar.value);
    userSeeking = false;
  });

  // Play/pause button
  const playBtn = document.getElementById('playBtn');
  playBtn.addEventListener('click', () => {
    if (audioEl.paused) { audioEl.play(); } else { audioEl.pause(); }
  });
  audioEl.addEventListener('play',  () => { playBtn.textContent = '⏸'; startTracking(); });
  audioEl.addEventListener('pause', () => { playBtn.textContent = '▶'; stopTracking(); });
  audioEl.addEventListener('ended', () => { playBtn.textContent = '▶'; stopTracking(); });

  // Transpose buttons
  document.getElementById('transpDown').addEventListener('click', () => {
    transposeSteps = Math.max(-11, transposeSteps - 1);
    refreshDisplay();
  });
  document.getElementById('transpUp').addEventListener('click', () => {
    transposeSteps = Math.min(11, transposeSteps + 1);
    refreshDisplay();
  });

  // Render timeline + initial card state
  renderTimeline();
  refreshDisplay();

  // rAF loop for seek bar + chord sync while playing
  function loop() {
    if (!audioEl.paused && !userSeeking) {
      seekBar.value = audioEl.currentTime;
      timeDisp.textContent = `${formatTime(audioEl.currentTime)} / ${formatTime(audioEl.duration || 0)}`;
    }
    rafId = requestAnimationFrame(loop);
  }
  rafId = requestAnimationFrame(loop);
}

function escapeHTML(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
```

- [ ] **Step 3: Test audio controls**

1. Upload a short MP3 (~2 min)
2. After analysis, the player should appear
3. Click ▶ — audio plays
4. Click ⏸ — audio pauses
5. Drag seek bar — audio jumps to new position
6. Time display updates in real time

- [ ] **Step 4: Commit**

```bash
git add server/templates/play.html
git commit -m "feat: web player audio controls — play/pause, seek bar, time display"
```

---

## Task 6: Timeline Rendering + Beat-Sync Loop

**Files:**
- Modify: `server/templates/play.html` — add `renderTimeline`, `applyFromAudioTime`, `startTracking`, `stopTracking`

- [ ] **Step 1: Add `renderTimeline`**

```javascript
const PX_PER_BEAT = 68;
const EDGE_BEATS  = 14;

function renderTimeline() {
  const row  = document.getElementById('tlRow');
  const wrap = document.getElementById('timelineWrap');
  if (!row || !beatTimes.length) return;
  row.innerHTML = '';

  const tlHalf = Math.ceil((wrap ? wrap.offsetWidth : 400) / 2 / PX_PER_BEAT);
  const beatToGroup = new Array(beatTimes.length).fill(-1);
  beatChords.forEach((bc, gi) => {
    for (let b = bc.beatStart; b < bc.beatStart + bc.beatCount; b++) beatToGroup[b] = gi;
  });

  const makePad = () => { const d = document.createElement('div'); d.className = 'tl-pad'; return d; };
  const makeEmpty = () => {
    const d = document.createElement('div');
    d.className = 'tl-beat empty';
    d.setAttribute('aria-hidden', 'true');
    return d;
  };

  for (let p = 0; p < tlHalf; p++) row.appendChild(makePad());
  for (let e = 0; e < EDGE_BEATS; e++) row.appendChild(makeEmpty());

  beatTimes.forEach((_, bi) => {
    const div = document.createElement('div');
    div.className = 'tl-beat';
    if (bi % 4 === 0) div.classList.add('measure-start');
    div.dataset.bi = bi;

    const gi = beatToGroup[bi];
    const bc = beatChords[gi];
    if (bc && bc.beatStart === bi) {
      const name = document.createElement('span');
      if (bc.chord === 'N') {
        name.className = 'beat-name beat-n';
        name.textContent = 'N';
      } else {
        name.className = 'beat-name';
        name.innerHTML = formatChordHTML(transposeChord(bc.chord, transposeSteps));
      }
      div.appendChild(name);
    }

    div.addEventListener('click', () => {
      if (audioEl) {
        audioEl.currentTime = beatTimes[bi];
        audioEl.play();
      }
    });
    row.appendChild(div);
  });

  for (let e = 0; e < EDGE_BEATS; e++) row.appendChild(makeEmpty());
  for (let p = 0; p < tlHalf; p++) row.appendChild(makePad());
}
```

- [ ] **Step 2: Add `applyFromAudioTime` and tracking functions**

```javascript
function applyFromAudioTime(t) {
  const bi  = findBeatAt(t);
  const bci = findBeatChordAt(t);

  if (bi !== currentBeatIdx) {
    currentBeatIdx = bi;
    document.querySelectorAll('#tlRow .tl-beat:not(.empty)').forEach((el, idx) => {
      el.classList.toggle('active', bi >= 0 && idx === bi);
    });
    const active = document.querySelector('#tlRow .tl-beat.active');
    if (active) {
      const wrap = document.getElementById('timelineWrap');
      if (wrap) {
        const wRect = wrap.getBoundingClientRect();
        const eRect = active.getBoundingClientRect();
        const target = wrap.scrollLeft + (eRect.left - wRect.left) - (wRect.width / 2) + (eRect.width / 2);
        wrap.scrollTo({ left: target, behavior: 'smooth' });
      }
    }
  }

  if (bci !== currentChordIdx) {
    currentChordIdx = bci;
    setCardContent(bci);
  }
}

function setCardContent(bci) {
  [
    { nameId: 'prevName',   diagId: 'prevDiag',   i: bci - 1 },
    { nameId: 'activeName', diagId: 'activeDiag', i: bci     },
    { nameId: 'nextName',   diagId: 'nextDiag',   i: bci + 1 },
  ].forEach(({ nameId, diagId, i }) => {
    const nameEl = document.getElementById(nameId);
    const diagEl = document.getElementById(diagId);
    if (!nameEl || !diagEl) return;
    const bc = beatChords[i];
    if (!bc || bc.chord === 'N') {
      nameEl.innerHTML = bc ? 'N' : '';
      diagEl.innerHTML = '';
    } else {
      const ch = transposeChord(bc.chord, transposeSteps);
      nameEl.innerHTML = formatChordHTML(ch);
      diagEl.innerHTML = buildChordSVG(ch);
    }
  });
}

function startTracking() {
  stopTracking();
  currentBeatIdx  = -999;
  currentChordIdx = -999;
  function loop() {
    if (audioEl) applyFromAudioTime(audioEl.currentTime);
    rafId = requestAnimationFrame(loop);
  }
  rafId = requestAnimationFrame(loop);
}

function stopTracking() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
}
```

- [ ] **Step 3: Add `refreshDisplay`**

```javascript
function refreshDisplay() {
  const capo = transposeSteps < 0 ? Math.abs(transposeSteps) : 0;
  const lbl  = transposeSteps === 0 ? 'Original'
    : (transposeSteps > 0 ? `▲${transposeSteps}` : `▼${Math.abs(transposeSteps)} (Capo ${capo})`);
  const lblEl = document.getElementById('transpLabel');
  if (lblEl) lblEl.textContent = lbl;

  const keyEl = document.getElementById('keyBadge');
  if (keyEl && baseKey) keyEl.textContent = `Key: ${transposeChord(baseKey, transposeSteps)}`;

  // Re-label beat blocks
  document.querySelectorAll('#tlRow .beat-name').forEach(nameEl => {
    const bi = +nameEl.closest('.tl-beat').dataset.bi;
    const gi = beatChords.findIndex(bc => bc.beatStart === bi);
    if (gi < 0) return;
    const ch = beatChords[gi].chord;
    if (ch === 'N') {
      nameEl.classList.add('beat-n');
      nameEl.textContent = 'N';
    } else {
      nameEl.classList.remove('beat-n');
      nameEl.innerHTML = formatChordHTML(transposeChord(ch, transposeSteps));
    }
  });

  if (currentChordIdx >= 0) setCardContent(currentChordIdx);
}
```

- [ ] **Step 4: Test beat sync**

1. Upload a song, wait for analysis
2. Press play — timeline should scroll to highlight the current beat
3. Click a beat in the timeline — audio should jump to that time and continue playing
4. Press ▲ transpose — key badge and chord labels update, chord diagrams update

- [ ] **Step 5: Commit**

```bash
git add server/templates/play.html
git commit -m "feat: web player beat-sync loop, timeline click-to-seek, chord cards, transpose"
```

---

## Task 7: Mobile Polish + Static Asset + Navigation Link

**Files:**
- Modify: `server/templates/play.html` — touch improvements
- Modify: `server/app.py` — static icon route if needed
- Modify: `server/templates/site_home.html` — add link to player

- [ ] **Step 1: Fix the brand logo path**

The header uses `/static/icon128.png`. Check if Flask has a static folder:

```bash
ls server/static/ 2>/dev/null || echo "no static dir"
```

If no `server/static/` exists, add a route to serve the icon from the extension folder. Add to `server/app.py` after the `/play` route:

```python
@app.route('/static/icon128.png')
def static_icon():
    icon_path = os.path.join(os.path.dirname(__file__), '..', 'extension', 'icons', 'icon128.png')
    return send_file(os.path.realpath(icon_path), mimetype='image/png')
```

- [ ] **Step 2: Add touch event improvement for timeline on iOS**

iOS Safari doesn't always fire `click` on non-interactive elements. Add `touch-action: manipulation` to the timeline beat CSS:

```css
.tl-beat {
  /* existing rules … */
  touch-action: manipulation;
}
```

And add `touch-action: manipulation` to `.play-btn` and `.ctrl-btn`.

- [ ] **Step 3: Add `pointer: coarse` chord card sizing for mobile**

Append to `<style>`:

```css
@media (pointer: coarse) {
  .tl-beat { width: 76px; }
  .tl-pad  { width: 76px; }
  .chord-name { font-size: clamp(2rem, 7vw, 3.5rem); }
}
```

- [ ] **Step 4: Add "Try Web Player" link to home page**

In `server/templates/site_home.html`, find the footer or CTA section and add a link:

```html
<a href="/play" class="...">Try Web Player →</a>
```

Match the existing button/link styles already in that template.

- [ ] **Step 5: Test on mobile**

Open `/play` on iPhone or iPad (or Chrome DevTools device mode at 390px width):
- Drop zone is tappable
- After analysis, play button is easy to tap (≥44px)
- Timeline scrolls with finger swipe; tapping a beat seeks audio
- Chord cards and diagrams are legible at 390px
- Safe area inset at bottom doesn't overlap play controls on notched phones

- [ ] **Step 6: Commit**

```bash
git add server/app.py server/templates/play.html server/templates/site_home.html
git commit -m "feat: web player mobile polish, static icon route, home page link"
```

---

## Self-Review Against Spec

| Requirement | Task |
|---|---|
| Upload audio file | Task 2 |
| Chord analysis (uses existing API) | Task 2 |
| Progress display | Task 2 |
| Beat-synced chord timeline | Task 6 |
| Click timeline to seek | Task 6 |
| Play / Pause button | Task 5 |
| Audio playback (no video) | Task 5 |
| Transpose controls | Task 6 (`refreshDisplay`) |
| Key + BPM badges | Task 5 (`initPlayer`) |
| Mobile-friendly (iPad + phone) | Task 4 CSS + Task 7 |
| Safe area insets | Task 4 CSS (`env(safe-area-inset-bottom)`) |
| Touch-friendly tap targets | Task 4 CSS + Task 7 |
| No YouTube / no video | ✓ — audio-only from File API |

No gaps found.
