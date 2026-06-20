# Chord Sheet Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only chord sheet panel to the web player (desktop only) that displays lyrics with chords positioned above the words where chord changes occur, synced to playback.

**Architecture:** Fetch synced lyrics from LRCLIB (a free, open lyrics API with word-level timestamps) client-side. Render a scrollable chord sheet below the audio controls where each line shows chords above lyrics. The chord names respect the current transposition. During playback, the active line auto-scrolls into view and highlights. Desktop-only — hidden on mobile and not used in the extension.

**Tech Stack:** Vanilla JS (matches existing play.html), LRCLIB API (https://lrclib.net/api), existing chord/beat data structures.

---

## Key Design Decisions

1. **Lyrics source:** LRCLIB.net — free, no API key, provides synced lyrics with timestamps. Fetched client-side via `fetch()`. Falls back to "no lyrics available" message if not found.

2. **Chord placement:** Each chord from `beatChords` is mapped to the closest lyric word by timestamp. Chords are rendered on a line above the lyric line, positioned to align with the word where the chord starts.

3. **Desktop only:** The chord sheet `<div>` is only injected when `!/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)`. The extension (`content.js`) is untouched.

4. **Transposition:** Chord names in the sheet use `transposeChord(chord, transposeSteps)` — same function the chord cards use. When the user changes transposition, the sheet re-renders.

5. **No interactive elements:** Pure display. No editing, no clicking chords, no lyric editing.

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `server/templates/play.html` | Modify | All changes — HTML, CSS, JS are inline in this single-file player |

This is a single-file change. The web player is entirely self-contained in `play.html` (HTML template + inline CSS + inline JS). No new files needed.

---

### Task 1: Add Chord Sheet CSS

**Files:**
- Modify: `server/templates/play.html` (CSS section, around line 607)

- [ ] **Step 1: Add chord sheet styles**

Insert after the existing `.a2hs-prompt` CSS block (around line 870). These styles define the chord sheet panel that sits below the audio controls.

```css
/* ── Chord Sheet (desktop only) ── */
.chord-sheet-wrap {
  max-height: 45vh;
  overflow-y: auto;
  padding: 16px 20px;
  margin-top: 8px;
  background: rgba(255,255,255,.03);
  border-radius: 8px;
  scroll-behavior: smooth;
}
.chord-sheet-wrap::-webkit-scrollbar { width: 4px; }
.chord-sheet-wrap::-webkit-scrollbar-thumb { background: rgba(255,255,255,.15); border-radius: 2px; }
.cs-line {
  position: relative;
  margin-bottom: 4px;
  font-family: 'Courier New', monospace;
  white-space: pre;
  line-height: 1.3;
}
.cs-chord-line {
  color: #f0c040;
  font-size: 0.85em;
  font-weight: 600;
  min-height: 1.2em;
  user-select: text;
}
.cs-lyric-line {
  color: rgba(255,255,255,.85);
  font-size: 0.9em;
  user-select: text;
}
.cs-line-group.active .cs-lyric-line {
  color: #fff;
}
.cs-line-group.active .cs-chord-line {
  color: #ffd700;
}
.cs-section-label {
  color: rgba(255,255,255,.4);
  font-size: 0.8em;
  font-style: italic;
  margin-top: 12px;
  margin-bottom: 2px;
}
.cs-no-lyrics {
  color: rgba(255,255,255,.3);
  font-style: italic;
  padding: 12px 0;
  text-align: center;
}
@media (orientation: portrait) and (max-width: 900px) {
  .chord-sheet-wrap { display: none; }
}
```

- [ ] **Step 2: Verify CSS parses correctly**

Open the browser dev tools and confirm no CSS parse errors in the console. The chord sheet wrap should not be visible yet (no HTML rendered).

- [ ] **Step 3: Commit**

```bash
git add server/templates/play.html
git commit -m "feat(play): add chord sheet CSS styles"
```

---

### Task 2: Add Chord Sheet HTML to Player

**Files:**
- Modify: `server/templates/play.html` — `buildPlayerHTML()` function (line ~1652)

- [ ] **Step 1: Insert chord sheet container into player HTML**

In the `buildPlayerHTML(title)` function, add the chord sheet div after the `.audio-ctrl` div and before the A2HS prompt. Only render on desktop.

Find this block (around line 1652):

```javascript
      </div>
      ${/iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ...
```

Insert before it:

```javascript
      ${!/iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ? '<div class="chord-sheet-wrap" id="chordSheetWrap"><div class="cs-no-lyrics">Loading lyrics…</div></div>' : ''}
```

The full context after the edit — the end of `buildPlayerHTML` should look like:

```javascript
      </div>
      ${!/iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ? '<div class="chord-sheet-wrap" id="chordSheetWrap"><div class="cs-no-lyrics">Loading lyrics…</div></div>' : ''}
      ${/iPhone|iPad|iPod|Android/i.test(navigator.userAgent) && !window.navigator.standalone && !window.matchMedia('(display-mode: standalone)').matches ? '<div class="a2hs-prompt" id="a2hsPrompt">...' : ''}
    </div>
  `;
```

- [ ] **Step 2: Verify the container appears on desktop**

Load the web player in a desktop browser. Confirm "Loading lyrics…" appears below the audio controls. Load on mobile — confirm the sheet is absent.

- [ ] **Step 3: Commit**

```bash
git add server/templates/play.html
git commit -m "feat(play): add chord sheet container to player HTML (desktop only)"
```

---

### Task 3: Fetch Synced Lyrics from LRCLIB

**Files:**
- Modify: `server/templates/play.html` — JS section (after `initPlayer` function, around line 2090)

- [ ] **Step 1: Add lyrics fetching function**

Add this function in the JS section, near the other data-loading functions:

```javascript
// ── Lyrics fetching (LRCLIB) ───────────────────────────────
let syncedLyrics = null; // [{time: float, text: string}, ...] or null

async function fetchLyrics(title) {
  syncedLyrics = null;
  const el = document.getElementById('chordSheetWrap');
  if (!el) return; // mobile — no sheet

  // Clean title: remove "(Official Video)", "[Official Audio]", etc.
  const clean = title
    .replace(/\s*[\(\[](official|lyric|music|audio|visualizer|live|hd|hq|4k|video|mv)[\s\w]*[\)\]]/gi, '')
    .replace(/\s*\|.*$/, '')
    .trim();

  // Try to split into artist - title
  let artist = '', track = clean;
  const dash = clean.match(/^(.+?)\s*[-–—]\s*(.+)$/);
  if (dash) { artist = dash[1].trim(); track = dash[2].trim(); }

  const params = new URLSearchParams({ track_name: track });
  if (artist) params.set('artist_name', artist);

  try {
    const res = await fetch('https://lrclib.net/api/search?' + params.toString());
    if (!res.ok) { _showNoLyrics(); return; }
    const results = await res.json();
    // Find first result with synced lyrics
    const match = results.find(r => r.syncedLyrics);
    if (!match) { _showNoLyrics(); return; }

    // Parse LRC format: [mm:ss.xx] text
    syncedLyrics = [];
    for (const line of match.syncedLyrics.split('\n')) {
      const m = line.match(/^\[(\d+):(\d+\.\d+)\]\s*(.*)$/);
      if (m) {
        const time = parseInt(m[1]) * 60 + parseFloat(m[2]);
        syncedLyrics.push({ time, text: m[3] });
      }
    }
    if (!syncedLyrics.length) { syncedLyrics = null; _showNoLyrics(); return; }
    renderChordSheet();
  } catch (e) {
    console.warn('[SeeChords] Lyrics fetch failed:', e);
    _showNoLyrics();
  }
}

function _showNoLyrics() {
  const el = document.getElementById('chordSheetWrap');
  if (el) el.innerHTML = '<div class="cs-no-lyrics">No lyrics available for this song</div>';
}
```

- [ ] **Step 2: Call `fetchLyrics` from `initPlayer`**

In the `initPlayer(data)` function, after the line that sets up the key badge (around line 2046), add:

```javascript
  // Fetch lyrics for chord sheet (desktop only)
  fetchLyrics(title);
```

- [ ] **Step 3: Verify lyrics fetch works**

Open the web player, load a well-known song (e.g., "Keane - Somewhere Only We Know"). Open Network tab — confirm a request to `lrclib.net/api/search` is made and returns results. Check console for the parsed `syncedLyrics` array.

- [ ] **Step 4: Commit**

```bash
git add server/templates/play.html
git commit -m "feat(play): fetch synced lyrics from LRCLIB API"
```

---

### Task 4: Render Chord Sheet with Chords Above Lyrics

**Files:**
- Modify: `server/templates/play.html` — JS section

- [ ] **Step 1: Add the chord sheet rendering function**

Add after the `fetchLyrics` function:

```javascript
function renderChordSheet() {
  const el = document.getElementById('chordSheetWrap');
  if (!el || !syncedLyrics) return;

  // Build a lookup: for each lyric line, find which chords fall within its time range
  // Each lyric line spans from its timestamp to the next line's timestamp
  const lines = syncedLyrics.filter(l => l.text.trim() !== '');
  if (!lines.length) { _showNoLyrics(); return; }

  let html = '';
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    const nextTime = li + 1 < lines.length ? lines[li + 1].time : Infinity;
    const lyricText = line.text;
    if (!lyricText) continue;

    // Find all chord changes that start within this lyric line's time range
    const lineChords = [];
    for (const bc of beatChords) {
      const chordTime = beatTimes[bc.beatStart] ?? 0;
      if (chordTime >= line.time && chordTime < nextTime) {
        const ch = (bc.chord === 'N') ? '' : transposeChord(bc.chord, transposeSteps);
        if (ch) lineChords.push({ chord: ch, time: chordTime });
      }
    }

    // Position chords above lyrics by mapping chord time to character position
    // Evenly distribute across the lyric text length based on time fraction
    const lineDuration = nextTime - line.time;
    let chordStr = '';
    if (lineChords.length > 0) {
      // Build chord line: place each chord at approximate character position
      const chars = new Array(Math.max(lyricText.length, 1)).fill(' ');
      for (const lc of lineChords) {
        const frac = lineDuration > 0 ? (lc.time - line.time) / lineDuration : 0;
        let pos = Math.round(frac * (lyricText.length - 1));
        pos = Math.max(0, Math.min(lyricText.length - 1, pos));
        // Write chord at position, don't overwrite previous chords
        const chordName = lc.chord;
        for (let ci = 0; ci < chordName.length && pos + ci < chars.length + chordName.length; ci++) {
          const idx = pos + ci;
          if (idx < chars.length) chars[idx] = chordName[ci];
          else chars.push(chordName[ci]);
        }
      }
      chordStr = chars.join('');
    }

    html += '<div class="cs-line-group" data-time="' + line.time + '">';
    if (chordStr.trim()) {
      html += '<div class="cs-line cs-chord-line">' + _escSheet(chordStr) + '</div>';
    }
    html += '<div class="cs-line cs-lyric-line">' + _escSheet(lyricText) + '</div>';
    html += '</div>';
  }

  el.innerHTML = html || '<div class="cs-no-lyrics">No lyrics available</div>';
}

function _escSheet(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
```

- [ ] **Step 2: Verify chord sheet renders**

Load a song in the web player. The chord sheet should appear below the audio controls with chord names on gold lines above white lyric lines. Chords should be roughly aligned to where they occur in the lyrics.

- [ ] **Step 3: Commit**

```bash
git add server/templates/play.html
git commit -m "feat(play): render chord sheet with chords above lyrics"
```

---

### Task 5: Auto-Scroll and Highlight Active Line During Playback

**Files:**
- Modify: `server/templates/play.html` — modify `goToBeat()` function and add scroll logic

- [ ] **Step 1: Add chord sheet line highlighting to the beat tracking loop**

Add a new function that highlights the current lyric line based on playback time:

```javascript
let _csActiveLine = null;

function updateChordSheetHighlight(currentTime) {
  const wrap = document.getElementById('chordSheetWrap');
  if (!wrap || !syncedLyrics) return;

  // Find active line group
  const groups = wrap.querySelectorAll('.cs-line-group');
  let activeGroup = null;
  for (let i = groups.length - 1; i >= 0; i--) {
    const t = parseFloat(groups[i].dataset.time);
    if (currentTime >= t) { activeGroup = groups[i]; break; }
  }

  if (activeGroup && activeGroup !== _csActiveLine) {
    if (_csActiveLine) _csActiveLine.classList.remove('active');
    activeGroup.classList.add('active');
    _csActiveLine = activeGroup;

    // Auto-scroll: keep active line in the upper third of the panel
    const wrapRect = wrap.getBoundingClientRect();
    const lineRect = activeGroup.getBoundingClientRect();
    const offset = lineRect.top - wrapRect.top - wrapRect.height * 0.3;
    wrap.scrollTop += offset;
  }
}
```

- [ ] **Step 2: Call it from the tracking loop**

In the `startTracking()` function (line ~1942), inside the `tick()` function, after the beat sync block (around line 1967), add:

```javascript
    // Sync chord sheet highlight
    updateChordSheetHighlight(cur);
```

So the block looks like:

```javascript
    // Sync beat display to player position
    if (beatTimes.length) {
      let bi = beatTimes.length - 1;
      for (let i = 0; i < beatTimes.length; i++) {
        if (beatTimes[i] > cur) { bi = Math.max(0, i - 1); break; }
      }
      if (bi !== currentBeatIdx) goToBeat(bi);
    }
    // Sync chord sheet highlight
    updateChordSheetHighlight(cur);
```

- [ ] **Step 3: Verify playback highlighting**

Play a song in the web player. The active lyric line should highlight brighter and auto-scroll into view as playback progresses.

- [ ] **Step 4: Commit**

```bash
git add server/templates/play.html
git commit -m "feat(play): auto-scroll and highlight active chord sheet line"
```

---

### Task 6: Re-render Chord Sheet on Transposition Change

**Files:**
- Modify: `server/templates/play.html` — `refreshDisplay()` function (line ~2313)

- [ ] **Step 1: Add `renderChordSheet()` call to `refreshDisplay()`**

In the `refreshDisplay()` function, at the end (after the `setCardContent` call around line 2338), add:

```javascript
  // Re-render chord sheet with new transposition
  renderChordSheet();
```

- [ ] **Step 2: Verify transposition updates the chord sheet**

Load a song, confirm chords show in the original key. Click the transpose up button — the chords in both the chord cards AND the chord sheet should update to the new key.

- [ ] **Step 3: Commit**

```bash
git add server/templates/play.html
git commit -m "feat(play): update chord sheet on transposition change"
```

---

### Task 7: Handle Edge Cases and Polish

**Files:**
- Modify: `server/templates/play.html`

- [ ] **Step 1: Handle songs with no beat data**

In `renderChordSheet()`, add an early return if there's no beat data:

```javascript
function renderChordSheet() {
  const el = document.getElementById('chordSheetWrap');
  if (!el || !syncedLyrics) return;
  if (!beatChords.length || !beatTimes.length) {
    el.innerHTML = '<div class="cs-no-lyrics">No chord timing data available</div>';
    return;
  }
  // ... rest of function
```

- [ ] **Step 2: Handle instrumental sections (empty lyric lines)**

In the `renderChordSheet` loop, handle empty/whitespace-only lines by showing chords-only rows. The current `filter(l => l.text.trim() !== '')` already skips blank lines. For instrumental breaks (where there are chord changes but no lyrics), the chords still appear above the next vocal line — this is acceptable behavior matching real chord sheets.

- [ ] **Step 3: Reset chord sheet state when loading a new song**

In `initPlayer()`, before the `fetchLyrics(title)` call, add:

```javascript
  syncedLyrics = null;
  _csActiveLine = null;
```

- [ ] **Step 4: Test with multiple songs**

Test with:
1. A song with synced lyrics on LRCLIB (e.g., popular English songs)
2. A song without lyrics on LRCLIB — should show "No lyrics available"
3. A Korean/non-English song — should gracefully show "No lyrics available" or foreign lyrics if available
4. Switch between songs — confirm the chord sheet resets and reloads

- [ ] **Step 5: Commit**

```bash
git add server/templates/play.html
git commit -m "feat(play): chord sheet edge cases and state reset"
```

---

### Task 8: Deploy and Verify

- [ ] **Step 1: Deploy to Fly**

```bash
fly deploy -c server/fly.toml
```

- [ ] **Step 2: End-to-end verification on production**

1. Open https://seechords.fly.dev/play on desktop browser
2. Load a song (e.g., from browse page)
3. Confirm chord sheet appears below audio controls
4. Confirm chords are positioned above lyrics
5. Play the song — confirm auto-scroll and line highlighting
6. Change transposition — confirm chords update in the sheet
7. Open on mobile — confirm chord sheet is NOT visible
8. Load a song with no LRCLIB lyrics — confirm graceful fallback message

- [ ] **Step 3: Commit any fixes**

```bash
git add server/templates/play.html
git commit -m "fix(play): chord sheet production fixes"
```
