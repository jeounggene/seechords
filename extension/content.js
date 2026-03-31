/* SeeChords – YouTube content script
 *
 * Detects the current YouTube video ID, checks for cached chords,
 * and injects the chord overlay UI or upload prompt.
 *
 * Syncs chord display to the YouTube video's current playback time
 * via the existing HTML5 <video> element on the page.
 */

'use strict';

// ─── State ────────────────────────────────────────────────
let currentVideoId  = null;
let chordData       = null;  // full API response
let chords          = [];    // [{chord, start, end}, …]
let beatTimes       = [];
let beatChords      = [];    // [{chord, beatStart, beatCount}, …]
let bpm             = 120;
let baseKey         = '';
let transposeSteps  = 0;
let currentBeatIdx  = -1;
let currentChordIdx = -1;
let overlayEl       = null;
let rafId           = null;
let observer        = null;
let videoEl         = null;
let offsetSeconds   = 0;     // user-adjustable timing offset
let lastAlignResult = null;  // cached alignment response for "Use Chord Sheet"
let currentVersionId = null; // active chord version ID
let currentRating   = 0;    // current star rating (0 = unrated)


const PX_PER_BEAT = 48;
/** Forces first applyChordDisplayFromVideoTime after tracking starts (avoids -1 === -1 skipping UI). */
const SYNC_UNSET = -999;

// ─── Isophonics → Display Converter ──────────────────────
const _ISO_TO_DISPLAY = {
  'maj': '', 'min': 'm', '7': '7', 'maj7': 'maj7', 'min7': 'm7',
  'dim': 'dim', 'dim7': 'dim7', 'hdim7': 'm7b5', 'aug': 'aug',
  'sus2': 'sus2', 'sus4': 'sus4', '6': '6', 'min6': 'm6', '9': '9',
  'min9': 'm9', '13': '13', '7b13': '7b13', '7add13': '7add13', '9sus4': '9sus4',
};

function _isoToDisplay(iso) {
  if (!iso || iso === 'N') return 'N';
  let bass = '';
  const slashIdx = iso.indexOf('/');
  if (slashIdx !== -1) { bass = iso.slice(slashIdx); iso = iso.slice(0, slashIdx); }
  const colonIdx = iso.indexOf(':');
  if (colonIdx === -1) return iso + bass;
  const root = iso.slice(0, colonIdx);
  const quality = iso.slice(colonIdx + 1);
  const suffix = _ISO_TO_DISPLAY[quality];
  if (suffix !== undefined) return root + suffix + bass;
  return root + quality + bass;
}

// ─── Chord Diagrams ───────────────────────────────────────
const CHORD_DIAGRAMS = {
  'C':   { f:[-1,3,2,0,1,0],     b:1 },
  'C#':  { f:[-1,1,3,3,3,1],     b:4 },
  'D':   { f:[-1,-1,0,2,3,2],    b:1 },
  'Eb':  { f:[-1,1,3,3,3,1],     b:6 },
  'E':   { f:[0,2,2,1,0,0],      b:1 },
  'F':   { f:[1,3,3,2,1,1],      b:1 },
  'F#':  { f:[1,3,3,2,1,1],      b:2 },
  'G':   { f:[3,2,0,0,0,3],      b:1 },
  'Ab':  { f:[1,3,3,2,1,1],      b:4 },
  'A':   { f:[-1,0,2,2,2,0],     b:1 },
  'Bb':  { f:[-1,1,3,3,3,1],     b:1 },
  'B':   { f:[-1,1,3,3,3,1],     b:2 },
  'Cm':  { f:[-1,1,3,3,2,1],     b:3 },
  'C#m': { f:[-1,1,3,3,2,1],     b:4 },
  'Dm':  { f:[-1,-1,0,2,3,1],    b:1 },
  'Ebm': { f:[-1,1,3,3,2,1],     b:6 },
  'Em':  { f:[0,2,2,0,0,0],      b:1 },
  'Fm':  { f:[1,3,3,1,1,1],      b:1 },
  'F#m': { f:[1,3,3,1,1,1],      b:2 },
  'Gm':  { f:[1,3,3,1,1,1],      b:3 },
  'Abm': { f:[1,3,3,1,1,1],      b:4 },
  'Am':  { f:[-1,0,2,2,1,0],     b:1 },
  'Bbm': { f:[-1,1,3,3,2,1],     b:1 },
  'Bm':  { f:[-1,1,3,3,2,1],     b:2 },
  'Db':  { f:[-1,1,3,3,3,1],     b:4 },
  'Gb':  { f:[1,3,3,2,1,1],      b:2 },
  'Dbm': { f:[-1,1,3,3,2,1],     b:4 },
  'Gbm': { f:[1,3,3,1,1,1],      b:2 },
  'C7':   { f:[-1,3,2,3,1,0],    b:1 },
  'C#7':  { f:[-1,1,3,1,3,1],    b:4 },
  'D7':   { f:[-1,-1,0,2,1,2],   b:1 },
  'Eb7':  { f:[-1,-1,1,3,2,3],   b:1 },
  'E7':   { f:[0,2,0,1,0,0],     b:1 },
  'F7':   { f:[1,3,1,2,1,1],     b:1 },
  'F#7':  { f:[1,3,1,2,1,1],     b:2 },
  'G7':   { f:[3,2,0,0,0,1],     b:1 },
  'Ab7':  { f:[-1,-1,1,1,1,2],   b:1 },
  'A7':   { f:[-1,0,2,0,2,0],    b:1 },
  'Bb7':  { f:[-1,1,3,1,3,1],    b:1 },
  'B7':   { f:[-1,2,1,2,0,2],    b:1 },
  'Db7':  { f:[-1,1,3,1,3,1],    b:4 },
  'Gb7':  { f:[1,3,1,2,1,1],     b:2 },
  'Cm7':  { f:[-1,1,3,1,2,1],    b:3 },
  'C#m7': { f:[-1,1,3,1,2,1],    b:4 },
  'Dm7':  { f:[-1,-1,0,2,1,1],   b:1 },
  'Ebm7': { f:[-1,-1,1,3,2,2],   b:1 },
  'Em7':  { f:[0,2,0,0,0,0],     b:1 },
  'Fm7':  { f:[1,3,1,1,1,1],     b:1 },
  'F#m7': { f:[1,3,1,1,1,1],     b:2 },
  'Gm7':  { f:[1,3,1,1,1,1],     b:3 },
  'Abm7': { f:[1,3,1,1,1,1],     b:4 },
  'Am7':  { f:[-1,0,2,0,1,0],    b:1 },
  'Bbm7': { f:[-1,1,3,1,2,1],    b:1 },
  'Bm7':  { f:[-1,2,0,2,0,2],    b:1 },
  'Dbm7': { f:[-1,1,3,1,2,1],    b:4 },
  'Gbm7': { f:[1,3,1,1,1,1],     b:2 },
  'Cmaj7':  { f:[-1,3,2,0,0,0],  b:1 },
  'C#maj7': { f:[-1,1,3,2,3,1],  b:4 },
  'Dmaj7':  { f:[-1,-1,0,2,2,2], b:1 },
  'Ebmaj7': { f:[-1,-1,1,3,3,3], b:1 },
  'Emaj7':  { f:[0,2,1,1,0,0],   b:1 },
  'Fmaj7':  { f:[1,3,2,2,1,0],   b:1 },
  'F#maj7': { f:[1,3,2,2,1,1],   b:2 },
  'Gmaj7':  { f:[3,2,0,0,0,2],   b:1 },
  'Abmaj7': { f:[-1,-1,1,1,1,3], b:1 },
  'Amaj7':  { f:[-1,0,2,1,2,0],  b:1 },
  'Bbmaj7': { f:[-1,1,3,2,3,1],  b:1 },
  'Bmaj7':  { f:[-1,2,1,3,0,2],  b:1 },
  'Dbmaj7': { f:[-1,1,3,2,3,1],  b:4 },
  'Gbmaj7': { f:[1,3,2,2,1,1],   b:2 },
  'Csus2':  { f:[-1,3,0,0,1,3],  b:1 },
  'C#sus2': { f:[-1,1,3,3,1,1],  b:4 },
  'Dsus2':  { f:[-1,-1,0,2,3,0], b:1 },
  'Ebsus2': { f:[-1,-1,1,3,4,1], b:1 },
  'Esus2':  { f:[0,2,4,4,0,0],   b:1 },
  'Fsus2':  { f:[-1,-1,3,0,1,1], b:1 },
  'F#sus2': { f:[-1,1,3,3,1,1],  b:2 },
  'Gsus2':  { f:[3,0,0,0,3,3],   b:1 },
  'Absus2': { f:[-1,1,3,3,1,1],  b:4 },
  'Asus2':  { f:[-1,0,2,2,0,0],  b:1 },
  'Bbsus2': { f:[-1,1,3,3,1,1],  b:1 },
  'Bsus2':  { f:[-1,2,4,4,2,2],  b:1 },
  'Csus4':  { f:[-1,3,3,0,1,1],  b:1 },
  'C#sus4': { f:[-1,1,3,3,4,1],  b:4 },
  'Dsus4':  { f:[-1,-1,0,2,3,3], b:1 },
  'Ebsus4': { f:[-1,-1,1,3,4,4], b:1 },
  'Esus4':  { f:[0,2,2,2,0,0],   b:1 },
  'Fsus4':  { f:[1,3,3,3,1,1],   b:1 },
  'F#sus4': { f:[1,3,3,3,1,1],   b:2 },
  'Gsus4':  { f:[3,3,0,0,1,3],   b:1 },
  'Absus4': { f:[1,3,3,3,1,1],   b:4 },
  'Asus4':  { f:[-1,0,2,2,3,0],  b:1 },
  'Bbsus4': { f:[-1,1,3,3,4,1],  b:1 },
  'Bsus4':  { f:[-1,2,4,4,0,0],  b:1 },
  'Cdim':  { f:[-1,3,4,2,4,-1],  b:1 },
  'C#dim': { f:[-1,-1,2,3,2,3],  b:1 },
  'Ddim':  { f:[-1,-1,0,1,3,1],  b:1 },
  'Ebdim': { f:[-1,-1,1,2,4,2],  b:1 },
  'Edim':  { f:[0,1,2,0,2,-1],   b:1 },
  'Fdim':  { f:[-1,-1,3,1,0,1],  b:1 },
  'F#dim': { f:[-1,-1,4,2,1,2],  b:1 },
  'Gdim':  { f:[3,4,2,3,-1,-1],  b:1 },
  'Abdim': { f:[4,2,0,1,-1,-1],  b:1 },
  'Adim':  { f:[-1,0,1,2,1,-1],  b:1 },
  'Bbdim': { f:[-1,1,2,3,2,-1],  b:1 },
  'Bdim':  { f:[-1,2,3,4,3,-1],  b:1 },
  'Caug':  { f:[-1,3,2,1,1,0],   b:1 },
  'C#aug': { f:[-1,0,3,2,2,1],   b:1 },
  'Daug':  { f:[-1,-1,0,3,3,2],  b:1 },
  'Ebaug': { f:[-1,-1,1,0,0,3],  b:1 },
  'Eaug':  { f:[0,3,2,1,1,0],    b:1 },
  'Faug':  { f:[1,0,3,2,2,1],    b:1 },
  'F#aug': { f:[-1,-1,4,3,3,2],  b:1 },
  'Gaug':  { f:[3,2,1,0,0,3],    b:1 },
  'Abaug': { f:[-1,-1,2,1,1,0],  b:1 },
  'Aaug':  { f:[-1,0,3,2,2,1],   b:1 },
  'Bbaug': { f:[-1,1,0,3,3,2],   b:1 },
  'Baug':  { f:[-1,2,1,0,0,3],   b:1 },

  // ── (b5) — major triad with flatted 5th ──
  'F(b5)': { f:[-1,-1,3,2,1,1],  b:1 },

  // ── Minor 9 ──
  'Cm9':  { f:[-1,3,1,3,3,3],    b:1 },
  'C#m9': { f:[-1,4,2,4,4,4],    b:1 },
  'Dm9':  { f:[-1,-1,0,2,1,0],   b:1 },
  'Em9':  { f:[0,2,0,0,0,2],     b:1 },
  'Fm9':  { f:[1,3,1,1,1,3],     b:1 },
  'F#m9': { f:[-1,-1,2,2,2,0],   b:1 },
  'Gm9':  { f:[3,1,0,3,3,1],     b:1 },
  'Am9':  { f:[-1,0,2,4,1,0],    b:1 },
  'Bm9':  { f:[-1,2,0,2,2,2],    b:1 },

  // ── 9sus4 ──
  'C9sus4':  { f:[-1,3,3,3,1,1],  b:1 },
  'D9sus4':  { f:[-1,-1,0,2,1,0],b:1 },
  'E9sus4':  { f:[0,2,0,2,0,2],   b:1 },
  'F9sus4':  { f:[1,1,1,3,1,1],   b:1 },
  'G9sus4':  { f:[3,3,0,0,1,1],   b:1 },
  'A9sus4':  { f:[-1,0,2,0,0,0],  b:1 },
  'B9sus4':  { f:[-1,2,2,2,2,0],  b:1 },

  // ── Dominant 13 ──
  'C13':  { f:[-1,3,2,3,3,0],     b:1 },
  'D13':  { f:[-1,-1,0,2,1,2],   b:1 },
  'E13':  { f:[0,2,0,1,2,2],     b:1 },
  'F13':  { f:[1,1,1,2,3,3],     b:1 },
  'F#7b13': { f:[2,-1,2,3,3,-1], b:1 },
  'G13':  { f:[3,2,0,0,0,2],     b:1 },
  'A13':  { f:[-1,0,2,0,2,2],    b:1 },
  'B13':  { f:[-1,2,1,2,0,4],    b:1 },
  'Gb7b13': { f:[2,-1,2,3,3,-1], b:1 },

  // ── Dominant 7b13 ──
  'C7b13':  { f:[8,-1,8,9,9,-1],  b:8 },
  'C#7b13': { f:[9,-1,9,10,10,-1], b:9 },
  'Db7b13': { f:[9,-1,9,10,10,-1], b:9 },
  'D7b13':  { f:[10,-1,10,11,11,-1], b:10 },
  'Eb7b13': { f:[11,-1,11,12,12,-1], b:11 },
  'E7b13':  { f:[12,-1,12,13,13,-1], b:12 },
  'F7b13':  { f:[1,-1,1,2,2,-1],  b:1 },
  'G7b13':  { f:[3,-1,3,4,4,-1],  b:3 },
  'Ab7b13': { f:[4,-1,4,5,5,-1],  b:4 },
  'A7b13':  { f:[5,-1,5,6,6,-1],  b:5 },
  'Bb7b13': { f:[6,-1,6,7,7,-1],  b:6 },
  'B7b13':  { f:[7,-1,7,8,8,-1],  b:7 },

  // ── Dominant 7add13 ──
  'C7add13':  { f:[-1,3,2,3,1,0],    b:1 },
  'C#7add13': { f:[-1,4,3,4,2,1],    b:1 },
  'Db7add13': { f:[-1,4,3,4,2,1],    b:1 },
  'D7add13':  { f:[-1,-1,0,2,1,2],   b:1 },
  'Eb7add13': { f:[-1,-1,1,3,2,3],   b:1 },
  'E7add13':  { f:[0,2,0,1,2,2],     b:1 },
  'F7add13':  { f:[1,3,1,2,1,3],     b:1 },
  'F#7add13': { f:[2,4,2,3,2,4],     b:1 },
  'Gb7add13': { f:[2,4,2,3,2,4],     b:1 },
  'G7add13':  { f:[3,2,0,0,0,2],     b:1 },
  'Ab7add13': { f:[4,3,1,1,1,3],     b:1 },
  'A7add13':  { f:[-1,0,2,0,2,2],    b:1 },
  'Bb7add13': { f:[-1,1,3,1,3,3],    b:1 },
  'B7add13':  { f:[-1,2,1,2,2,2],    b:1 },

  // ── Half-diminished (m7b5) ──
  'Cm7b5':  { f:[-1,3,4,3,4,-1],  b:1 },
  'C#m7b5': { f:[-1,4,5,4,5,-1],  b:1 },
  'Dm7b5':  { f:[-1,-1,0,1,1,1],  b:1 },
  'Ebm7b5': { f:[-1,-1,1,2,2,2],  b:1 },
  'Em7b5':  { f:[0,1,0,0,3,0],    b:1 },
  'Fm7b5':  { f:[1,2,3,1,4,-1],   b:1 },
  'F#m7b5': { f:[2,0,2,2,1,0],    b:1 },
  'Gm7b5':  { f:[3,4,3,3,3,-1],   b:1 },
  'Abm7b5': { f:[4,5,4,4,4,-1],   b:1 },
  'Am7b5':  { f:[-1,0,1,0,1,0],   b:1 },
  'Bbm7b5': { f:[-1,1,2,1,2,-1],  b:1 },
  'Bm7b5':  { f:[-1,2,3,2,3,-1],  b:1 },

  // ── Slash Chords ──
  'C/E':    { f:[0,3,2,0,1,0],     b:1 },
  'C/G':    { f:[3,3,2,0,1,0],     b:1 },
  'Cmaj7/G':{ f:[3,3,2,0,0,0],     b:1 },
  'D/F#':   { f:[2,0,0,2,3,2],     b:1 },
  'D7/F#':  { f:[2,0,0,2,1,2],     b:1 },
  'E/G#':   { f:[4,2,2,1,0,0],     b:1 },
  'F/A':    { f:[-1,0,3,2,1,1],    b:1 },
  'F/C':    { f:[-1,3,3,2,1,1],    b:1 },
  'G/B':    { f:[-1,2,0,0,0,3],    b:1 },
  'G/D':    { f:[-1,-1,0,0,0,3],   b:1 },
  'A/C#':   { f:[-1,4,2,2,2,0],    b:1 },
  'A/E':    { f:[0,0,2,2,2,0],     b:1 },
  'Bb/D':   { f:[-1,-1,0,3,3,1],   b:1 },
  'Am/E':   { f:[0,0,2,2,1,0],     b:1 },
  'Am/G':   { f:[3,0,2,2,1,0],     b:1 },
  'Am7/G':  { f:[3,0,2,0,1,0],     b:1 },
  'Am7/D':  { f:[-1,-1,0,2,1,0],   b:1 },
  'Dm/F':   { f:[1,0,0,2,3,1],     b:1 },
  'Em/B':   { f:[-1,2,2,0,0,0],    b:1 },
  'Bm/E':   { f:[0,2,4,4,3,0],     b:1 },
  'Bm7/E':  { f:[0,2,0,2,3,2],     b:1 },
};

// ─── Transposition ────────────────────────────────────────
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
  // Handle slash chords: transpose both root and bass note
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
  return raw
    .replace(/([A-G])#/, '$1♯')
    .replace(/([A-G])b/, '$1♭');
}

function formatChordHTML(raw) {
  const display = formatChord(raw);
  const m = display.match(/^([A-G][♯♭]?)(.*?)$/);
  if (!m) return display.replace(/♭/g, '<span class="sc-flat">♭</span>');
  const root = m[1].replace(/♭/g, '<span class="sc-flat">♭</span>');
  if (!m[2]) return root;
  return `${root}<span class="sc-chord-suffix">${m[2]}</span>`;
}

// ─── SVG Chord Diagram Builder ────────────────────────────
function buildChordSVG(chordName) {
  const key  = chordName.replace('♯','#').replace('♭','b');
  let data = CHORD_DIAGRAMS[key];
  // Enharmonic fallback: map sharp names to their flat equivalents
  if (!data) {
    const enharmonic = { 'D#':'Eb', 'G#':'Ab', 'A#':'Bb' };
    const re = /^([A-G]#)/;
    const m = key.match(re);
    if (m && enharmonic[m[1]]) {
      data = CHORD_DIAGRAMS[enharmonic[m[1]] + key.slice(m[1].length)];
    }
  }
  // Slash chord fallback: use base chord diagram
  if (!data && key.includes('/')) {
    data = CHORD_DIAGRAMS[key.split('/')[0]];
  }
  if (!data) return '';
  const { f: strings, b: baseFret } = data;

  const SX     = [6, 17, 28, 39, 50, 61];
  const NUT_Y  = 16;
  const FRET_H = 12;
  const N_FRETS = 4;
  const DOT_R  = 4.5;
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
      const x = SX[i];
      const y = NUT_Y + (fret - 0.5) * FRET_H;
      p.push(`<circle cx="${x}" cy="${y}" r="${DOT_R}" fill="currentColor" opacity=".88"/>`);
    }
  }

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-8 0 84 72" width="120" height="128" aria-hidden="true">${p.join('')}</svg>`;
}

// ─── Beat-Chord Building ──────────────────────────────────
function buildBeatChords() {
  if (!beatTimes.length || !chords.length) return [];
  const result = [];
  let bi = 0;
  let lastSegIdx = -1;
  while (bi < beatTimes.length) {
    const t = beatTimes[bi];
    const nextBt = beatTimes[bi + 1] ?? (t + (t - (beatTimes[bi - 1] ?? t - 0.5)));
    // Try beat start first, then midpoint — avoids skipping short segments
    const segIdxByBt = chords.findIndex(c => c.start <= t && c.end > t);
    const mid = (t + nextBt) / 2;
    const segIdxByMid = chords.findIndex(c => c.start <= mid && c.end > mid);
    const segIdx = segIdxByBt >= 0 ? segIdxByBt : segIdxByMid;
    const c = segIdx >= 0 ? chords[segIdx] : null;
    const actualIdx = segIdx >= 0 ? segIdx : -1;
    const chordName = c ? c.chord : 'N';
    // New block on chord change OR new segment (even if same chord)
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
  // Before the first beat time we are not "on" beat 0 yet (avoids showing the
  // first chord from t=0 while the first beat is later — intro / silence).
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
  for (let i = 0; i < beatChords.length; i++) {
    const bc = beatChords[i];
    if (bi >= bc.beatStart && bi < bc.beatStart + bc.beatCount) return i;
  }
  return beatChords.length - 1;
}

// ─── Video ID Extraction ──────────────────────────────────
function getVideoId() {
  const url = new URL(window.location.href);
  if (url.pathname === '/watch') {
    return url.searchParams.get('v') || null;
  }
  const m = url.pathname.match(/\/shorts\/([a-zA-Z0-9_-]{11})/);
  if (m) return m[1];
  return null;
}

// ─── YouTube <video> Element ──────────────────────────────
function findVideoElement() {
  return document.querySelector('video.html5-main-video') ||
         document.querySelector('video');
}

// ─── Overlay Injection ────────────────────────────────────
function injectOverlay() {
  if (overlayEl) return;

  overlayEl = document.createElement('div');
  overlayEl.id = 'seechords-overlay';
  overlayEl.innerHTML = `
    <div class="sc-header">
      <span class="sc-logo">🎸 SeeChords</span>
      <div class="sc-badges">
        <span class="sc-badge sc-badge-key" id="scKeyBadge">Key: —</span>
        <span class="sc-badge sc-badge-bpm" id="scBpmBadge">BPM: —</span>
      </div>
      <div class="sc-controls">
        <button class="sc-ctrl-btn" id="scTransposeDown" title="Transpose down">▼</button>
        <span class="sc-ctrl-label" id="scTransposeLabel">Original</span>
        <button class="sc-ctrl-btn" id="scTransposeUp" title="Transpose up">▲</button>
      </div>
      <div class="sc-sync-controls">
        <button class="sc-ctrl-btn" id="scSyncEarlier" title="Chords earlier (−50ms)">◁</button>
        <span class="sc-ctrl-label sc-sync-label" id="scSyncLabel">Sync</span>
        <button class="sc-ctrl-btn" id="scSyncLater" title="Chords later (+50ms)">▷</button>
      </div>
      <div class="sc-version-controls" id="scVersionControls" style="display:none;">
        <select class="sc-version-select" id="scVersionSelect" title="Switch chord version"></select>
        <button class="sc-reupload-btn" id="scReuploadBtn" title="Re-analyze chords">↻ Re-analyze</button>
      </div>
      <div class="sc-rating" id="scRating" style="display:none;">
        <span class="sc-star" data-star="1">★</span>
        <span class="sc-star" data-star="2">★</span>
        <span class="sc-star" data-star="3">★</span>
        <span class="sc-star" data-star="4">★</span>
        <span class="sc-star" data-star="5">★</span>
      </div>
      <button class="sc-toggle-btn" id="scToggle" title="Minimize/Expand">▾</button>
    </div>
    <div class="sc-body" id="scBody">
      <div class="sc-timeline" id="scTimeline">
        <div class="sc-tl-row" id="scTlRow"></div>
      </div>
      <div class="sc-chord-trio">
        <div class="sc-chord-card sc-card-prev" id="scCardPrev">
          <span class="sc-chord-name"></span>
          <span class="sc-chord-diagram"></span>
        </div>
        <div class="sc-chord-card sc-card-active" id="scCardActive">
          <span class="sc-chord-name"></span>
          <span class="sc-chord-diagram"></span>
        </div>
        <div class="sc-chord-card sc-card-next" id="scCardNext">
          <span class="sc-chord-name"></span>
          <span class="sc-chord-diagram"></span>
        </div>
      </div>
    </div>
    <div class="sc-compare-panel" id="scComparePanel" style="display:none;">
      <div class="sc-compare-header">
        <span class="sc-compare-title">Compare with chord sheet</span>
        <button class="sc-compare-close" id="scCompareClose">✕</button>
      </div>
      <textarea class="sc-compare-input" id="scCompareInput" rows="6"
        placeholder="Paste chords here. Accepts formats like:\n\nF | Em7 A7 | Dm | Bb C |\n\nor chord sheet with sections:\n\n[Verse]\nF       Em7  A7\nYesterday all my troubles\nDm      Bb     C\nseemed so far away"></textarea>
      <div class="sc-compare-btns">
        <button class="sc-compare-run" id="scCompareRun">Compare</button>
        <button class="sc-compare-run sc-align-run" id="scAlignRun">Align</button>
      </div>
      <div class="sc-compare-result" id="scCompareResult" style="display:none;"></div>
    </div>
    <div class="sc-status" id="scStatus"></div>
    <div class="sc-upload-prompt" id="scUploadPrompt" style="display:none;">
      <p class="sc-upload-msg">No chords found for this video.</p>
      <button class="sc-analyze-btn" id="scAutoExtractBtn">Analyze &amp; Generate Chords</button>
      <div class="sc-progress" id="scProgress" style="display:none;">
        <div class="sc-progress-bar" id="scProgressBar"></div>
        <span class="sc-progress-msg" id="scProgressMsg">Processing…</span>
      </div>
      <!-- Upload your own section (commented out)
      <p class="sc-upload-sub" style="margin: 12px 0 8px; position: relative; text-align: center;">
        <span style="background: rgba(30,30,30,0.9); padding: 0 8px; color: #666; font-size: 0.85em;">OR UPLOAD YOUR OWN</span>
        <hr style="position: absolute; top: 50%; left: 0; right: 0; border: none; border-top: 1px solid #444; z-index: -1; margin: 0;">
      </p>
      <label class="sc-upload-label">
        <input type="file" id="scFileInput" accept=".mp3,.wav,.m4a,.aac,.ogg,.flac,audio/*" />
        <span class="sc-upload-btn-text">Choose Audio File</span>
      </label>
      <p class="sc-file-name" id="scFileName" style="display:none;"></p>
      <label class="sc-rights-check">
        <input type="checkbox" id="scRightsConfirm" />
        <span>I confirm I have rights to upload this audio.</span>
      </label>
      <button class="sc-analyze-btn" id="scAnalyzeBtn" disabled>Upload &amp; Analyze</button>
      -->
    </div>
  `;

  // Insert below the video player — try multiple targets
  const targets = [
    { sel: '#below', method: 'prepend' },
    { sel: '#primary-inner', method: 'append' },
    { sel: 'ytd-watch-metadata', method: 'after' },
    { sel: '#player', method: 'after' },
    { sel: 'ytd-watch-flexy #primary', method: 'prepend' },
    { sel: 'ytd-watch-flexy', method: 'append' },
    { sel: '#content', method: 'prepend' },
  ];
  let inserted = false;
  for (const { sel, method } of targets) {
    const container = document.querySelector(sel);
    if (container) {
      try {
        if (method === 'prepend') container.prepend(overlayEl);
        else if (method === 'after') container.after(overlayEl);
        else container.appendChild(overlayEl);
        inserted = true;
        console.log('[SeeChords] Overlay injected via', method, 'into:', sel);
        break;
      } catch (e) {
        console.warn('[SeeChords] Failed to inject into', sel, e);
      }
    }
  }
  if (!inserted) {
    document.body.appendChild(overlayEl);
    console.log('[SeeChords] Overlay appended to body (fallback)');
  }
  // Force visibility after injection
  overlayEl.style.cssText = 'display:block!important;visibility:visible!important;opacity:1!important;';

  // Wire up events
  document.getElementById('scTransposeDown').addEventListener('click', () => {
    transposeSteps--;
    refreshDisplay();
  });
  document.getElementById('scTransposeUp').addEventListener('click', () => {
    transposeSteps++;
    refreshDisplay();
  });

  // Sync nudge
  const SYNC_STEP = 0.05; // 50 ms per click
  function updateSyncLabel() {
    const el = document.getElementById('scSyncLabel');
    if (!el) return;
    if (offsetSeconds === 0) { el.textContent = 'Sync'; return; }
    const ms = Math.round(offsetSeconds * 1000);
    el.textContent = (ms > 0 ? '+' : '') + ms + 'ms';
  }
  document.getElementById('scSyncEarlier').addEventListener('click', () => {
    offsetSeconds = Math.round((offsetSeconds - SYNC_STEP) * 1000) / 1000;
    updateSyncLabel();
    currentBeatIdx = SYNC_UNSET;
    currentChordIdx = SYNC_UNSET;
  });
  document.getElementById('scSyncLater').addEventListener('click', () => {
    offsetSeconds = Math.round((offsetSeconds + SYNC_STEP) * 1000) / 1000;
    updateSyncLabel();
    currentBeatIdx = SYNC_UNSET;
    currentChordIdx = SYNC_UNSET;
  });

  document.getElementById('scToggle').addEventListener('click', () => {
    const body = document.getElementById('scBody');
    const btn  = document.getElementById('scToggle');
    if (body.style.display === 'none') {
      body.style.display = '';
      btn.textContent = '▾';
    } else {
      body.style.display = 'none';
      btn.textContent = '▸';
    }
  });

  // Upload events
  const autoExtractBtn = document.getElementById('scAutoExtractBtn');

  function triggerAutoExtract() {
    autoExtractBtn.disabled = true;
    const progressDiv = document.getElementById('scProgress');
    const progressBar = document.getElementById('scProgressBar');
    const progressMsg = document.getElementById('scProgressMsg');
    
    progressDiv.style.display = 'block';
    progressMsg.textContent = 'Extracting audio…';
    progressBar.style.width = '10%';

    chrome.runtime.sendMessage({
      type: 'EXTRACT_AND_ANALYZE',
      videoId: currentVideoId,
      title: document.title
    }, (response) => {
      if (response && response.error) {
        progressMsg.textContent = response.error;
        autoExtractBtn.disabled = false;
        return;
      }
      if (response && response.job_id) {
        pollJob(response.job_id);
      }
    });
  }

  autoExtractBtn.addEventListener('click', triggerAutoExtract);

  /* Upload section commented out — listeners disabled
  const fileInput = document.getElementById('scFileInput');
  const analyzeBtn = document.getElementById('scAnalyzeBtn');
  const rightsCheck = document.getElementById('scRightsConfirm');
  fileInput.addEventListener('change', () => {
    const fileNameEl = document.getElementById('scFileName');
    if (fileInput.files.length) {
      fileNameEl.textContent = `${fileInput.files[0].name} selected`;
      fileNameEl.style.display = 'block';
    } else {
      fileNameEl.style.display = 'none';
    }
    analyzeBtn.disabled = !(fileInput.files.length && rightsCheck.checked);
  });
  rightsCheck.addEventListener('change', () => {
    analyzeBtn.disabled = !(fileInput.files.length && rightsCheck.checked);
  });
  analyzeBtn.addEventListener('click', () => startUploadAnalysis());
  */

  // Version select — switch to a different chord version
  document.getElementById('scVersionSelect').addEventListener('change', (e) => {
    const versionId = parseInt(e.target.value, 10);
    if (!versionId) return;
    chrome.runtime.sendMessage({ type: 'LOAD_VERSION', versionId }, (resp) => {
      if (resp && resp.found && resp.data) {
        loadChordData(resp.data);
      }
    });
  });

  // Re-upload button — shows the upload prompt over the chord display
  document.getElementById('scReuploadBtn').addEventListener('click', () => {
    showUploadPrompt();
  });

  // Compare panel events (panel opened via other triggers)
  document.getElementById('scCompareClose').addEventListener('click', () => {
    document.getElementById('scComparePanel').style.display = 'none';
  });
  document.getElementById('scCompareRun').addEventListener('click', runCompare);
  document.getElementById('scAlignRun').addEventListener('click', runAlign);

  // Star rating events
  document.querySelectorAll('#scRating .sc-star').forEach(star => {
    star.addEventListener('click', () => submitRating(parseInt(star.dataset.star, 10)));
    star.addEventListener('mouseenter', () => renderStars(parseInt(star.dataset.star, 10)));
    star.addEventListener('mouseleave', () => renderStars(currentRating));
  });
}

function removeOverlay() {
  if (overlayEl) {
    overlayEl.remove();
    overlayEl = null;
  }
  stopTracking();
}

// ─── Show States ──────────────────────────────────────────
function showStatus(msg) {
  const el = document.getElementById('scStatus');
  if (el) { el.textContent = msg; el.style.display = 'block'; }
  const prompt = document.getElementById('scUploadPrompt');
  if (prompt) prompt.style.display = 'none';
  const body = document.getElementById('scBody');
  if (body) body.style.display = 'none';
}

function showUploadPrompt() {
  console.log('[SeeChords] Showing upload prompt');
  const el = document.getElementById('scStatus');
  if (el) el.style.display = 'none';
  const prompt = document.getElementById('scUploadPrompt');
  if (prompt) {
    prompt.style.cssText = 'display:block!important;visibility:visible!important;';
  }
  const body = document.getElementById('scBody');
  if (body) body.style.display = 'none';

}

function showChords() {
  const el = document.getElementById('scStatus');
  if (el) el.style.display = 'none';
  const prompt = document.getElementById('scUploadPrompt');
  if (prompt) prompt.style.display = 'none';
  const body = document.getElementById('scBody');
  if (body) body.style.display = '';
}

// ─── Render Timeline & Cards ──────────────────────────────
function renderTimeline() {
  const row = document.getElementById('scTlRow');
  if (!row) return;
  row.innerHTML = '';
  if (!beatTimes.length) return;

  beatChords = buildBeatChords();

  const beatToGroup = new Array(beatTimes.length).fill(-1);
  beatChords.forEach((bc, gi) => {
    for (let b = bc.beatStart; b < bc.beatStart + bc.beatCount; b++) beatToGroup[b] = gi;
  });

  const tl = document.getElementById('scTimeline');
  const tlHalf = Math.ceil((tl ? tl.offsetWidth : 400) / 2 / PX_PER_BEAT);
  const makePad = () => { const p = document.createElement('div'); p.className = 'sc-beat-pad'; return p; };
  for (let p = 0; p < tlHalf; p++) row.appendChild(makePad());

  beatTimes.forEach((_, bi) => {
    const div = document.createElement('div');
    div.className = 'sc-beat-block';
    if (bi % 4 === 0) div.classList.add('sc-measure-start');
    div.dataset.bi = bi;

    const gi = beatToGroup[bi];
    const bc = beatChords[gi];
    if (bc && bc.beatStart === bi) {
      const name = document.createElement('span');
      if (bc.chord === 'N') {
        name.className = 'sc-beat-name sc-beat-n';
        name.textContent = 'N';
      } else {
        name.className = 'sc-beat-name';
        name.innerHTML = formatChordHTML(transposeChord(bc.chord, transposeSteps));
      }
      div.appendChild(name);
    }

    div.addEventListener('click', () => {
      const v = findVideoElement();
      if (v) { v.currentTime = beatTimes[bi]; v.play(); }
    });
    row.appendChild(div);
  });

  for (let p = 0; p < tlHalf; p++) row.appendChild(makePad());
}

function setCardContent(bci) {
  if (bci < 0 || !beatChords.length) {
    ['scCardPrev', 'scCardActive', 'scCardNext'].forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.querySelector('.sc-chord-name').textContent = '';
      el.querySelector('.sc-chord-diagram').innerHTML = '';
    });
    return;
  }
  [
    { id: 'scCardPrev',   i: bci - 1 },
    { id: 'scCardActive', i: bci },
    { id: 'scCardNext',   i: bci + 1 },
  ].forEach(({ id, i }) => {
    const el = document.getElementById(id);
    if (!el) return;
    const bc = beatChords[i];
    let label = '';
    let diagramChord = null;
    if (bc) {
      if (bc.chord === 'N') {
        label = 'N';
      } else {
        diagramChord = transposeChord(bc.chord, transposeSteps);
        label = formatChord(diagramChord);
      }
    }
    el.querySelector('.sc-chord-name').textContent = label;
    el.querySelector('.sc-chord-diagram').innerHTML = diagramChord ? buildChordSVG(diagramChord) : '';
  });
}

function refreshDisplay() {
  const tt = transposeSteps;
  const capo = tt < 0 ? Math.abs(tt) : 0;
  const lbl = tt === 0 ? 'Original' :
    (tt > 0 ? `▲${tt}` : `▼${Math.abs(tt)} (Capo ${capo})`);
  const el = document.getElementById('scTransposeLabel');
  if (el) el.textContent = lbl;

  if (baseKey) {
    const kb = document.getElementById('scKeyBadge');
    if (kb) kb.textContent = `Key: ${transposeChord(baseKey, tt)}`;
  }

  // Re-label beat-block names
  document.querySelectorAll('#scTlRow .sc-beat-name').forEach(nameEl => {
    const bi = +nameEl.closest('.sc-beat-block').dataset.bi;
    const gi = beatChords.findIndex(bc => bc.beatStart === bi);
    if (gi < 0) return;
    const ch = beatChords[gi].chord;
    if (ch === 'N') {
      nameEl.classList.add('sc-beat-n');
      nameEl.textContent = 'N';
    } else {
      nameEl.classList.remove('sc-beat-n');
      nameEl.innerHTML = formatChordHTML(transposeChord(ch, tt));
    }
  });

  if (currentChordIdx >= 0) setCardContent(currentChordIdx);
  else if (currentChordIdx === -1) setCardContent(-1);
}

/** Sync timeline highlight + chord cards to a video timeline position (seconds). */
function applyChordDisplayFromVideoTime(videoTime) {
  const t = videoTime + offsetSeconds;
  const chordIdx = findBeatChordAt(t);
  const bi = findBeatAt(t);

  if (bi !== currentBeatIdx) {
    currentBeatIdx = bi;
    document.querySelectorAll('#scTlRow .sc-beat-block').forEach((el, idx) => {
      el.classList.toggle('sc-active', bi >= 0 && idx === bi);
    });
    const active = document.querySelector('#scTlRow .sc-beat-block.sc-active');
    if (active) active.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
  }

  if (chordIdx !== currentChordIdx) {
    currentChordIdx = chordIdx;
    setCardContent(chordIdx);
    if (chordIdx >= 0) {
      const activeCard = document.getElementById('scCardActive');
      if (activeCard) {
        activeCard.classList.remove('sc-entering');
        void activeCard.offsetWidth;
        activeCard.classList.add('sc-entering');
      }
    }
  }
}

// ─── Tracking Loop ───────────────────────────────────────
function trackLoop() {
  const v = findVideoElement();
  if (!v) {
    rafId = requestAnimationFrame(trackLoop);
    return;
  }

  // Update while paused too (seek / scrub) — not only during playback.
  applyChordDisplayFromVideoTime(v.currentTime);

  rafId = requestAnimationFrame(trackLoop);
}

function startTracking() {
  stopTracking();
  currentBeatIdx = SYNC_UNSET;
  currentChordIdx = SYNC_UNSET;
  rafId = requestAnimationFrame(trackLoop);
}

function stopTracking() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
}

// ─── Upload & Analyze ─────────────────────────────────────
async function startUploadAnalysis() {
  const fileInput = document.getElementById('scFileInput');
  const f = fileInput.files && fileInput.files[0];
  if (!f || !currentVideoId) return;

  const progressDiv = document.getElementById('scProgress');
  const progressBar = document.getElementById('scProgressBar');
  const progressMsg = document.getElementById('scProgressMsg');
  const analyzeBtn  = document.getElementById('scAnalyzeBtn');

  progressDiv.style.display = 'block';
  analyzeBtn.disabled = true;
  progressMsg.textContent = 'Reading file…';

  // Read file as data URL for transfer to background
  const reader = new FileReader();
  reader.onload = async () => {
    progressMsg.textContent = 'Uploading…';
    progressBar.style.width = '10%';

    const response = await new Promise(resolve => {
      chrome.runtime.sendMessage({
        type: 'UPLOAD_AND_ANALYZE',
        videoId: currentVideoId,
        fileData: reader.result,
        fileName: f.name,
        title: document.title.replace(' - YouTube', '').trim(),
      }, resolve);
    });

    if (response.error) {
      progressMsg.textContent = `Error: ${response.error}`;
      analyzeBtn.disabled = false;
      return;
    }

    if (response.job_id) {
      pollJob(response.job_id);
    }
  };
  reader.readAsDataURL(f);
}

function pollJob(jobId) {
  const progressBar = document.getElementById('scProgressBar');
  const progressMsg = document.getElementById('scProgressMsg');

  const timer = setInterval(async () => {
    const data = await new Promise(resolve => {
      chrome.runtime.sendMessage({ type: 'POLL_STATUS', jobId }, resolve);
    });

    if (data.status === 'processing') {
      progressMsg.textContent = data.message || 'Processing…';
      progressBar.style.width = (data.progress || 0) + '%';
      return;
    }

    clearInterval(timer);

    if (data.status === 'done') {
      // Re-fetch from API to get full data with versions list
      const chordResponse = await new Promise(resolve => {
        chrome.runtime.sendMessage({ type: 'CHECK_CHORDS', videoId: currentVideoId }, resolve);
      });
      if (chordResponse && chordResponse.found && chordResponse.data) {
        loadChordData(chordResponse.data);
      } else {
        loadChordData(data);
      }
    } else {
      progressMsg.textContent = `Error: ${data.message || 'Unknown error'}`;
      document.getElementById('scAnalyzeBtn').disabled = false;
    }
  }, 2000);
}

// ─── Compare with chord sheet ─────────────────────────────
async function runCompare() {
  const input = document.getElementById('scCompareInput');
  const resultDiv = document.getElementById('scCompareResult');
  const runBtn = document.getElementById('scCompareRun');
  const text = input.value.trim();

  if (!text || !currentVideoId) return;

  runBtn.disabled = true;
  runBtn.textContent = 'Comparing…';
  resultDiv.style.display = 'none';

  try {
    const resp = await new Promise(resolve => {
      chrome.runtime.sendMessage({
        type: 'COMPARE_CHORDS',
        videoId: currentVideoId,
        referenceText: text,
      }, resolve);
    });

    if (resp.error) {
      resultDiv.innerHTML = `<div class="sc-compare-error">${resp.error}</div>`;
    } else {
      const pct = Math.round(resp.similarity * 100);
      const barColor = pct >= 70 ? '#4caf50' : pct >= 40 ? '#ff9800' : '#f44336';
      resultDiv.innerHTML = `
        <div class="sc-compare-score">
          <div class="sc-compare-bar-bg">
            <div class="sc-compare-bar-fill" style="width:${pct}%;background:${barColor}"></div>
          </div>
          <span class="sc-compare-pct">${pct}% match</span>
        </div>
        <div class="sc-compare-details">
          <div class="sc-compare-row"><span class="sc-compare-label">Found:</span> ${resp.chordsFound.join(', ') || '—'}</div>
          <div class="sc-compare-row sc-missing"><span class="sc-compare-label">Missing:</span> ${resp.chordsMissing.join(', ') || '—'}</div>
          <div class="sc-compare-row sc-extra"><span class="sc-compare-label">Extra:</span> ${resp.chordsExtra.join(', ') || '—'}</div>
          <div class="sc-compare-row"><span class="sc-compare-label">Predicted:</span> ${resp.predictedSequence.join(' → ')}</div>
          <div class="sc-compare-row"><span class="sc-compare-label">Reference:</span> ${resp.referenceSequence.join(' → ')}</div>
        </div>
      `;
    }
    resultDiv.style.display = 'block';
  } catch (e) {
    resultDiv.innerHTML = `<div class="sc-compare-error">Compare failed: ${e.message}</div>`;
    resultDiv.style.display = 'block';
  }

  runBtn.disabled = false;
  runBtn.textContent = 'Compare';
}

async function runAlign() {
  const input = document.getElementById('scCompareInput');
  const resultDiv = document.getElementById('scCompareResult');
  const runBtn = document.getElementById('scAlignRun');
  const text = input.value.trim();

  if (!text || !currentVideoId) return;

  runBtn.disabled = true;
  runBtn.textContent = 'Aligning…';
  resultDiv.style.display = 'none';

  try {
    const resp = await new Promise(resolve => {
      chrome.runtime.sendMessage({
        type: 'ALIGN_CHORDS',
        videoId: currentVideoId,
        referenceText: text,
      }, resolve);
    });

    if (resp.error) {
      resultDiv.innerHTML = `<div class="sc-compare-error">${resp.error}</div>`;
    } else {
      const beatPct = Math.round(resp.beatAccuracy * 100);
      const segPct = Math.round(resp.segmentAgreement * 100);
      const barColor = beatPct >= 70 ? '#4caf50' : beatPct >= 40 ? '#ff9800' : '#f44336';

      let rows = '';
      for (const seg of resp.segments) {
        const cls = seg.match ? 'sc-align-match' : 'sc-align-mismatch';
        const time = `${seg.start.toFixed(1)}–${seg.end.toFixed(1)}`;
        rows += `<tr class="${cls}">
          <td class="sc-align-time">${time}</td>
          <td class="sc-align-chord">${seg.sheetChord}</td>
          <td class="sc-align-chord">${seg.predChord}</td>
          <td class="sc-align-beats">${seg.matches}/${seg.nBeats}</td>
        </tr>`;
      }

      resultDiv.innerHTML = `
        <div class="sc-compare-score">
          <div class="sc-compare-bar-bg">
            <div class="sc-compare-bar-fill" style="width:${beatPct}%;background:${barColor}"></div>
          </div>
          <span class="sc-compare-pct">${beatPct}% beat match · ${segPct}% segments agree</span>
        </div>
        <div class="sc-align-table-wrap">
          <table class="sc-align-table">
            <thead><tr>
              <th>Time</th><th>Sheet</th><th>Model</th><th>Beats</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <button class="sc-use-sheet-btn" id="scUseSheetBtn">Use Chord Sheet</button>
      `;
      lastAlignResult = resp;
    }
    resultDiv.style.display = 'block';
  } catch (e) {
    resultDiv.innerHTML = `<div class="sc-compare-error">Align failed: ${e.message}</div>`;
    resultDiv.style.display = 'block';
  }

  runBtn.disabled = false;
  runBtn.textContent = 'Align';

  // Bind "Use Chord Sheet" button after render
  const useBtn = document.getElementById('scUseSheetBtn');
  if (useBtn) useBtn.addEventListener('click', useChordSheet);
}

function useChordSheet() {
  if (!lastAlignResult || !lastAlignResult.segments) return;
  // Convert aligned segments into the chords[] format the overlay expects
  chords = lastAlignResult.segments.map(seg => ({
    chord: seg.sheetChord,
    start: seg.start,
    end:   seg.end,
  }));
  transposeSteps = 0;
  currentBeatIdx  = SYNC_UNSET;
  currentChordIdx = SYNC_UNSET;
  document.getElementById('scTransposeLabel').textContent = 'Original';
  renderTimeline();
  applyChordDisplayFromVideoTime(findVideoElement() ? findVideoElement().currentTime : 0);
  // Close the compare panel
  document.getElementById('scComparePanel').style.display = 'none';
}

// ─── Load Chord Data ──────────────────────────────────────
function loadChordData(data) {
  chords    = (data.chords || []).map(c => {
    if (c.chord && c.chord.includes(':')) c.chord = _isoToDisplay(c.chord);
    return c;
  });
  beatTimes = data.beat_times || [];
  bpm       = data.bpm        || 120;
  baseKey   = data.key        || '';
  transposeSteps  = 0;
  offsetSeconds   = 0;
  currentBeatIdx  = SYNC_UNSET;
  currentChordIdx = SYNC_UNSET;
  document.getElementById('scKeyBadge').textContent = `Key: ${baseKey}`;
  document.getElementById('scBpmBadge').textContent = `BPM: ${bpm}`;
  document.getElementById('scTransposeLabel').textContent = 'Original';

  // Show the version controls
  const vc = document.getElementById('scVersionControls');
  if (vc) vc.style.display = 'flex';
  renderTimeline();
  showChords();

  videoEl = findVideoElement();
  applyChordDisplayFromVideoTime(videoEl ? videoEl.currentTime : 0);
  startTracking();

  // Populate version dropdown
  currentVersionId = data.versionId || null;
  if (currentVideoId) fetchVersionsList(currentVideoId, data.versionId);

  // Show rating widget and fetch existing rating
  if (currentVersionId) {
    const ratingEl = document.getElementById('scRating');
    if (ratingEl) ratingEl.style.display = 'flex';
    fetchRating(currentVersionId);
  }
}

function fetchVersionsList(videoId, activeVersionId) {
  chrome.runtime.sendMessage({ type: 'LIST_VERSIONS', videoId }, (resp) => {
    const sel = document.getElementById('scVersionSelect');
    if (!sel || !resp || !resp.versions) return;
    sel.innerHTML = '';
    if (resp.versions.length <= 1) {
      sel.style.display = 'none';
      return;
    }
    sel.style.display = '';
    resp.versions.forEach((v) => {
      const opt = document.createElement('option');
      opt.value = v.versionId;
      let date = '';
      if (v.analyzedAt) {
        const d = typeof v.analyzedAt === 'number'
          ? new Date(v.analyzedAt * 1000) : new Date(v.analyzedAt);
        if (!isNaN(d)) date = d.toISOString().slice(0, 16).replace('T', ' ');
      }
      const label = v.source === 'verified' ? '✓ Verified' : `v${v.versionId}`;
      opt.textContent = `${label} · ${v.key || '?'} · ${v.segmentCount} segs · ${date}`;
      if (v.versionId === activeVersionId || v.isActive) opt.selected = true;
      sel.appendChild(opt);
    });
  });
}

function fetchRating(versionId) {
  chrome.runtime.sendMessage({ type: 'GET_RATING', versionId }, (resp) => {
    currentRating = (resp && resp.stars) || 0;
    renderStars(currentRating);
  });
}

function renderStars(rating) {
  const stars = document.querySelectorAll('#scRating .sc-star');
  stars.forEach(s => {
    const val = parseInt(s.dataset.star, 10);
    s.classList.toggle('sc-star-active', val <= rating);
  });
}

function submitRating(stars) {
  if (!currentVersionId || !currentVideoId) return;
  currentRating = stars;
  renderStars(stars);
  chrome.runtime.sendMessage({
    type: 'RATE_CHORDS',
    versionId: currentVersionId,
    videoId: currentVideoId,
    stars,
  });
}


// ─── Main Check Flow ──────────────────────────────────────
async function checkForChords(videoId) {
  currentVideoId = videoId;
  transposeSteps  = 0;
  offsetSeconds   = 0;
  currentBeatIdx  = -1;
  currentChordIdx = -1;

  console.log('[SeeChords] Checking chords for:', videoId);
  injectOverlay();
  showStatus('Loading chords…');

  const response = await new Promise(resolve => {
    chrome.runtime.sendMessage({ type: 'CHECK_CHORDS', videoId }, resolve);
  });

  if (response && response.found && response.data) {
    loadChordData(response.data);
  } else {
    showUploadPrompt();
  }
}

// ─── URL Change Detection ─────────────────────────────────
function onNavigate() {
  const vid = getVideoId();
  if (!vid) {
    removeOverlay();
    currentVideoId = null;
    return;
  }
  if (vid !== currentVideoId) {
    removeOverlay();
    checkForChords(vid);
  }
}

// Watch for YouTube SPA navigation
let lastUrl = location.href;
observer = new MutationObserver(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    setTimeout(onNavigate, 800);
  }
});
observer.observe(document.body, { childList: true, subtree: true });

// Also listen for yt-navigate-finish (YouTube's SPA event)
window.addEventListener('yt-navigate-finish', () => {
  console.log('[SeeChords] yt-navigate-finish fired');
  setTimeout(onNavigate, 500);
});

// Initial check — retry until the page is ready
function tryInit(attempts = 0) {
  const vid = getVideoId();
  console.log(`[SeeChords] Init attempt ${attempts}, videoId=${vid}, url=${location.href}`);
  if (vid) {
    // Wait for YouTube's DOM to be ready
    const ready = document.querySelector('#below') ||
                  document.querySelector('ytd-watch-metadata') ||
                  document.querySelector('ytd-watch-flexy');
    if (ready) {
      onNavigate();
    } else if (attempts < 20) {
      setTimeout(() => tryInit(attempts + 1), 500);
    } else {
      // Force it even without ideal container
      onNavigate();
    }
  } else if (attempts < 10) {
    setTimeout(() => tryInit(attempts + 1), 1000);
  }
}
tryInit();
