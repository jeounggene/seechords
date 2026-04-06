# Enharmonic Chord Diagram Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 17 missing chord diagrams (Db/Gb-rooted) by extending the enharmonic fallback in `buildChordSVG` to also map flat→sharp.

**Architecture:** The existing `buildChordSVG` function has an enharmonic fallback that maps sharp→flat (D#→Eb, G#→Ab, A#→Bb). We add flat→sharp entries (Db→C#, Gb→F#) and widen the regex from `/^([A-G]#)/` to `/^([A-G][#b])/` so it also matches flat roots. The same change is applied to all 3 files containing this function.

**Tech Stack:** Vanilla JS (no build step)

**Spec:** `docs/superpowers/specs/2026-04-06-enharmonic-chord-diagrams-design.md`

---

### Task 1: Fix fallback in `extension/content.js`

**Files:**
- Modify: `extension/content.js:403-409`

- [ ] **Step 1: Edit the enharmonic fallback**

In `buildChordSVG` (~line 403-409), change:

```js
  // Enharmonic fallback: map sharp names to their flat equivalents
  if (!data) {
    const enharmonic = { 'D#':'Eb', 'G#':'Ab', 'A#':'Bb' };
    const re = /^([A-G]#)/;
    const m = key.match(re);
    if (m && enharmonic[m[1]]) {
      data = CHORD_DIAGRAMS[enharmonic[m[1]] + key.slice(m[1].length)];
    }
  }
```

To:

```js
  // Enharmonic fallback: sharp↔flat (covers Db→C#, Gb→F# and D#→Eb etc.)
  if (!data) {
    const enharmonic = { 'D#':'Eb', 'G#':'Ab', 'A#':'Bb', 'Db':'C#', 'Gb':'F#' };
    const re = /^([A-G][#b])/;
    const m = key.match(re);
    if (m && enharmonic[m[1]]) {
      data = CHORD_DIAGRAMS[enharmonic[m[1]] + key.slice(m[1].length)];
    }
  }
```

- [ ] **Step 2: Smoke-test in browser**

Load a YouTube video with the extension. Transpose until you see Db- or Gb-rooted chords. Verify the diagram renders (was previously blank).

Quick JS console check (paste in any page with the extension loaded):

```js
// Should return SVG markup, not empty string
console.log('Dbm6:', buildChordSVG('Dbm6') ? 'OK' : 'MISSING');
console.log('Gbdim:', buildChordSVG('Gbdim') ? 'OK' : 'MISSING');
console.log('C:', buildChordSVG('C') ? 'OK' : 'MISSING');  // existing still works
console.log('Db:', buildChordSVG('Db') ? 'OK' : 'MISSING'); // direct entry still works
```

Expected: all print "OK".

- [ ] **Step 3: Commit**

```bash
git add extension/content.js
git commit -m "fix: add flat→sharp enharmonic fallback for chord diagrams in extension"
```

---

### Task 2: Fix fallback in `server/templates/chords.html`

**Files:**
- Modify: `server/templates/chords.html:598-602`

- [ ] **Step 1: Edit the enharmonic fallback**

In `buildChordSVG` (~line 598-602), change:

```js
  if (!data) {
    const enh = { 'D#':'Eb', 'G#':'Ab', 'A#':'Bb' };
    const m = key.match(/^([A-G]#)/);
    if (m && enh[m[1]]) data = CHORD_DIAGRAMS[enh[m[1]] + key.slice(m[1].length)];
  }
```

To:

```js
  if (!data) {
    const enh = { 'D#':'Eb', 'G#':'Ab', 'A#':'Bb', 'Db':'C#', 'Gb':'F#' };
    const m = key.match(/^([A-G][#b])/);
    if (m && enh[m[1]]) data = CHORD_DIAGRAMS[enh[m[1]] + key.slice(m[1].length)];
  }
```

- [ ] **Step 2: Smoke-test in browser**

Open a song on the chords page (e.g. `http://localhost:5001/chords/<video_id>`). Transpose until Db or Gb chords appear. Verify diagrams render.

- [ ] **Step 3: Commit**

```bash
git add server/templates/chords.html
git commit -m "fix: add flat→sharp enharmonic fallback for chord diagrams in chords page"
```

---

### Task 3: Fix fallback in `server/templates/play.html`

**Files:**
- Modify: `server/templates/play.html:1423-1431`

- [ ] **Step 1: Edit the enharmonic fallback**

In `buildChordSVG` (~line 1423-1431), change:

```js
  // Enharmonic fallback: map sharp names to their flat equivalents
  if (!data) {
    const enharmonic = { 'D#':'Eb', 'G#':'Ab', 'A#':'Bb' };
    const re = /^([A-G]#)/;
    const m = key.match(re);
    if (m && enharmonic[m[1]]) {
      data = CHORD_DIAGRAMS[enharmonic[m[1]] + key.slice(m[1].length)];
    }
  }
```

To:

```js
  // Enharmonic fallback: sharp↔flat (covers Db→C#, Gb→F# and D#→Eb etc.)
  if (!data) {
    const enharmonic = { 'D#':'Eb', 'G#':'Ab', 'A#':'Bb', 'Db':'C#', 'Gb':'F#' };
    const re = /^([A-G][#b])/;
    const m = key.match(re);
    if (m && enharmonic[m[1]]) {
      data = CHORD_DIAGRAMS[enharmonic[m[1]] + key.slice(m[1].length)];
    }
  }
```

- [ ] **Step 2: Smoke-test in browser**

Open the web player (`http://localhost:5001/play`), upload or select a song. Transpose until Db/Gb chords appear. Verify diagrams render.

- [ ] **Step 3: Commit**

```bash
git add server/templates/play.html
git commit -m "fix: add flat→sharp enharmonic fallback for chord diagrams in web player"
```

---

### Task 4: Full verification across all 17 chords

- [ ] **Step 1: Verify all 17 chords resolve**

Open any page with `buildChordSVG` available (e.g. the chords page) and paste this in the browser console:

```js
const missing = [
  'Db6','Dbaug','Dbdim','Dbdim7','Dbm6','Dbm7b5','DbmM7','Dbsus2','Dbsus4',
  'Gb6','Gbaug','Gbdim','Gbdim7','Gbm6','Gbm7b5','GbmM7','Gbsus2'
];
const results = missing.map(c => `${c}: ${buildChordSVG(c) ? 'OK' : 'MISSING'}`);
console.log(results.join('\n'));
```

Expected: all 17 print "OK".

- [ ] **Step 2: Verify existing chords still work**

```js
const existing = ['C','Db','Dbm','Db7','Dbm7','Dbmaj7','Gb','Gbm','Gb7','Gbm7','Gbmaj7','Gbsus4','C#m6','F#dim','Am','E7'];
const results = existing.map(c => `${c}: ${buildChordSVG(c) ? 'OK' : 'MISSING'}`);
console.log(results.join('\n'));
```

Expected: all print "OK" — confirms direct entries are still used (no regression).
