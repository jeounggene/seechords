# Enharmonic Chord Diagram Fallback

**Date:** 2026-04-06
**Status:** Approved

## Problem

17 chord names that can appear after transposing have no entry in the `CHORD_DIAGRAMS` lookup object, causing blank diagram renders. All 17 are Db- or Gb-rooted variants whose enharmonic equivalents (C# / F#) already exist with correct fingering data.

### Missing chords

- **Db-rooted (9):** Db6, Dbaug, Dbdim, Dbdim7, Dbm6, Dbm7b5, DbmM7, Dbsus2, Dbsus4
- **Gb-rooted (8):** Gb6, Gbaug, Gbdim, Gbdim7, Gbm6, Gbm7b5, GbmM7, Gbsus2

## Solution: Extend enharmonic fallback in `buildChordSVG`

The existing fallback maps sharp→flat (`D#→Eb`, `G#→Ab`, `A#→Bb`). We add flat→sharp (`Db→C#`, `Gb→F#`) and widen the regex to match flat roots.

### Before

```js
const enharmonic = { 'D#':'Eb', 'G#':'Ab', 'A#':'Bb' };
const re = /^([A-G]#)/;
```

### After

```js
const enharmonic = { 'D#':'Eb', 'G#':'Ab', 'A#':'Bb', 'Db':'C#', 'Gb':'F#' };
const re = /^([A-G][#b])/;
```

The lookup-first-then-fallback pattern ensures existing chords with direct entries (Db, Dbm, Db7, Dbm7, Dbmaj7, Gb, Gbm, etc.) are never affected.

## Files changed

1. `extension/content.js` — `buildChordSVG` function (~line 404)
2. `server/templates/chords.html` — `buildChordSVG` function
3. `server/templates/play.html` — `buildChordSVG` function

## Approach evaluation

Three approaches were evaluated by independent agents:

| Approach | Score |
|---|---|
| A: Enhance fallback (chosen) | 43/50 |
| C: Programmatic alias generation | 42/50 |
| B: Add 17 explicit entries | 30/50 |

Approach A won on code size, simplicity, and leveraging the existing pattern. It also inherits the future-proofing benefit of C — any new C#/F# chord added to `CHORD_DIAGRAMS` will automatically resolve for Db/Gb via fallback.

## Risk

Zero regression risk. The fallback only fires when no direct match exists in `CHORD_DIAGRAMS`.
