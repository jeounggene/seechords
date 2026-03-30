#!/usr/bin/env python3
"""Lyric-aware chord alignment: parse chord sheets spatially, transcribe
audio with Whisper for word-level timestamps, then place chords at the
timestamps of the words they sit above.

Flow:
  1. parse_chord_sheet_spatial(text) → list of {chord, word, line_idx, ...}
  2. transcribe_words(audio_path) → list of {word, start, end}
  3. align_chords_to_words(spatial, transcript) → list of {chord, start, end}
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import re
import json

# ── Chord detection regex (same as compare_chords.py) ──────

_CHORD_RE = re.compile(
    r'^[A-G][b#]?'
    r'(m7b5|m9|m11|m13|mmaj7|min7|maj7|dim7|min9|9sus4|add\d+|M7|m7|m6|m|min|maj|dim|aug|sus[24]?|7|9|11|13|6)?'
    r'(/[A-G][b#]?)?$'
)

_ENHARMONIC = {
    'Db': 'C#', 'Eb': 'Eb', 'Gb': 'F#', 'Ab': 'Ab', 'Bb': 'Bb',
    'E#': 'F', 'B#': 'C', 'Cb': 'B', 'Fb': 'E',
}

_QUALITY_MAP = {
    '': '', 'maj': '', '7': '', 'maj7': '', '9': '', '11': '', '13': '',
    'sus2': '', 'sus4': '', 'sus': '', 'add9': '', '6': '',
    'm': 'm', 'min': 'm', 'm7': 'm', 'min7': 'm', 'mmaj7': 'm', 'm6': 'm',
    'dim': 'm', 'dim7': 'm', 'aug': '',
}


def _is_chord_token(token):
    return bool(_CHORD_RE.match(token))


def _normalize_chord(name):
    if not name or name in ('N', 'X', '-', 'NC', 'N.C.'):
        return 'N'
    if '/' in name:
        name = name.split('/')[0]
    m = re.match(r'^([A-G][b#]?)(.*)', name)
    if not m:
        return 'N'
    root, quality = m.group(1), m.group(2)
    root = _ENHARMONIC.get(root, root)
    quality = _QUALITY_MAP.get(quality, '')
    return root + quality


def _line_is_chords(line):
    """Heuristic: does this line consist mostly of chord tokens?"""
    tokens = line.split()
    if not tokens:
        return False
    clean = [t for t in tokens if t not in ('|', '/', '||', ':|', '|:', 'x2', 'x3', 'x4')]
    clean = [t for t in clean if not re.match(r'^\[.*\]$', t)]
    if not clean:
        return False
    chord_count = sum(1 for t in clean if _is_chord_token(t))
    return chord_count / len(clean) >= 0.5 and chord_count >= 1


def _clean_word(w):
    """Strip punctuation from a word for matching."""
    return re.sub(r'[^\w\']', '', w).lower().strip()


# ── Spatial Parser ──────────────────────────────────────────

def parse_chord_sheet_spatial(text):
    """Parse a chord sheet preserving the spatial relationship between
    chords and the lyrics below them.

    In UG-style chord sheets, chords are on a line by themselves and the
    next line contains lyrics.  Each chord's column position tells us
    which word it's above.

    Returns a list of dicts:
        { 'chord': str,          # normalised chord name
          'raw_chord': str,      # original chord text
          'word': str|None,      # the word the chord is above (if lyrics line present)
          'word_clean': str,     # lowercased, stripped of punctuation
          'line_idx': int,       # which chord-lyric pair (0-based)
          'col': int,            # column position of the chord
          'section': str|None }  # [Verse], [Chorus], etc.

    Also returns the full lyrics as a flat word list for transcript matching.
    """
    lines = text.split('\n')
    result = []
    lyrics_words = []
    current_section = None
    pair_idx = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Section header
        if re.match(r'^\[.*\]$', stripped):
            current_section = stripped
            i += 1
            continue

        # Empty line
        if not stripped:
            i += 1
            continue

        # Check if this is a chord line
        if _line_is_chords(stripped):
            # Find chord positions (column indices) in the ORIGINAL line
            chord_positions = _extract_chord_positions(line)

            # Check if next line is lyrics
            lyric_line = ''
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped and not _line_is_chords(next_stripped) and not re.match(r'^\[.*\]$', next_stripped):
                    lyric_line = lines[i + 1]
                    i += 1  # consume the lyric line too

            # Map each chord to the word at its column position
            if lyric_line:
                word_spans = _word_spans(lyric_line)
                for chord_name, col in chord_positions:
                    word, word_clean = _word_at_col(col, word_spans, lyric_line)
                    result.append({
                        'chord': _normalize_chord(chord_name),
                        'raw_chord': chord_name,
                        'word': word,
                        'word_clean': word_clean,
                        'line_idx': pair_idx,
                        'col': col,
                        'section': current_section,
                    })
                # Collect lyrics in order
                for _, _, w in word_spans:
                    lyrics_words.append(w)
            else:
                # Chord-only line (e.g. chorus chords without lyrics)
                for chord_name, col in chord_positions:
                    result.append({
                        'chord': _normalize_chord(chord_name),
                        'raw_chord': chord_name,
                        'word': None,
                        'word_clean': '',
                        'line_idx': pair_idx,
                        'col': col,
                        'section': current_section,
                    })
            pair_idx += 1
        else:
            # Standalone lyric line (no chords above) — still collect words
            word_spans = _word_spans(line)
            for _, _, w in word_spans:
                lyrics_words.append(w)
            i += 1
            continue

        i += 1

    return result, lyrics_words


def _extract_chord_positions(line):
    """Extract (chord_name, column_index) pairs from a chord line,
    preserving whitespace positions."""
    positions = []
    i = 0
    while i < len(line):
        if line[i].isspace():
            i += 1
            continue
        # Try to read a chord token starting at position i
        j = i
        while j < len(line) and not line[j].isspace():
            j += 1
        token = line[i:j]
        if _is_chord_token(token):
            positions.append((token, i))
        i = j
    return positions


def _word_spans(line):
    """Find (start_col, end_col, word) for each word in a line."""
    spans = []
    for m in re.finditer(r'\S+', line):
        spans.append((m.start(), m.end(), m.group()))
    return spans


def _word_at_col(col, word_spans, line):
    """Find the word that a chord at `col` is positioned above.

    Strategy: find the word whose span overlaps with or is closest to `col`.
    """
    if not word_spans:
        return None, ''

    # Direct overlap
    for start, end, word in word_spans:
        if start <= col < end:
            return word, _clean_word(word)

    # Closest word (prefer the word starting at or after the chord column)
    best = None
    best_dist = float('inf')
    for start, end, word in word_spans:
        dist = abs(start - col)
        if dist < best_dist:
            best_dist = dist
            best = word
    return best, _clean_word(best) if best else ''


# ── Whisper Transcription (subprocess-isolated) ────────────

def transcribe_words(audio_path, model_size='base'):
    """Transcribe audio and return word-level timestamps using faster-whisper.

    Runs in a **separate subprocess** to avoid library conflicts (essentia
    vs CTranslate2) in the main server process.  Pre-converts audio to
    16kHz mono WAV so the subprocess doesn't need ffmpeg at runtime.

    Returns list of dicts: { 'word': str, 'start': float, 'end': float }
    """
    import subprocess, sys, tempfile

    # 1. Pre-convert audio to 16kHz mono WAV
    wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(wav_fd)
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path,
        ]
        ff = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=120)
        if ff.returncode != 0:
            raise RuntimeError(f"ffmpeg pre-convert failed: {ff.stderr[-500:]}")

        # 2. Write a helper script that uses faster-whisper (CTranslate2)
        script = f"""
import os, json, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from faster_whisper import WhisperModel

model = WhisperModel({model_size!r}, device='cpu', compute_type='int8')
segments, info = model.transcribe({wav_path!r}, word_timestamps=True)

words = []
for seg in segments:
    for w in seg.words:
        words.append({{
            'word': w.word.strip(),
            'start': round(w.start, 3),
            'end': round(w.end, 3),
        }})
json.dump(words, sys.stdout)
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            env = os.environ.copy()
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            env['OMP_NUM_THREADS'] = '1'

            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=600, env=env,
            )
            if proc.returncode != 0:
                print(f"[lyric_align] Whisper subprocess stderr:\n{proc.stderr}",
                      flush=True)
                raise RuntimeError(
                    f"Whisper transcription failed (exit {proc.returncode})")
            raw_words = json.loads(proc.stdout)
            # Add word_clean field
            for w in raw_words:
                w['word_clean'] = _clean_word(w['word'])
            return raw_words
        finally:
            os.unlink(script_path)
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


# ── Lyric ↔ Transcript Alignment ───────────────────────────

def _fuzzy_match(a, b):
    """Simple fuzzy word match score. Returns 0.0-1.0."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Check if one contains the other
    if a in b or b in a:
        return 0.8
    # Character-level similarity (simple)
    common = sum(1 for c in a if c in b)
    return common / max(len(a), len(b))


def align_lyrics_to_transcript(sheet_words, transcript_words):
    """Align sheet lyrics to Whisper transcript using DP sequence matching.

    sheet_words:      list of str (lyrics words from chord sheet, in order)
    transcript_words: list of {word, word_clean, start, end}

    Returns: list of (sheet_idx, transcript_idx) mappings.
    Also returns a per-sheet-word timestamp dict: {sheet_idx: {start, end}}
    """
    N = len(sheet_words)
    M = len(transcript_words)

    if N == 0 or M == 0:
        return [], {}

    # DP: align sheet words to transcript, allowing gaps in transcript
    # Score matrix: higher = better match
    INF = float('-inf')
    dp = [[INF] * (M + 1) for _ in range(N + 1)]
    bp = [[(-1, -1)] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0

    SKIP_COST = -0.1  # cost of skipping a transcript word
    MATCH_BONUS = 1.0

    for i in range(N + 1):
        for j in range(M + 1):
            if dp[i][j] == INF:
                continue

            # Skip transcript word j (it's not in the sheet)
            if j < M:
                val = dp[i][j] + SKIP_COST
                if val > dp[i][j + 1]:
                    dp[i][j + 1] = val
                    bp[i][j + 1] = (i, j)

            # Match sheet word i to transcript word j
            if i < N and j < M:
                s_word = _clean_word(sheet_words[i])
                t_word = transcript_words[j]['word_clean']
                score = _fuzzy_match(s_word, t_word) * MATCH_BONUS
                val = dp[i][j] + score
                if val > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = val
                    bp[i + 1][j + 1] = (i, j)

    # Find best endpoint (all sheet words consumed)
    best_j = 0
    best_score = INF
    for j in range(M + 1):
        if dp[N][j] > best_score:
            best_score = dp[N][j]
            best_j = j

    # Backtrack
    pairs = []
    i, j = N, best_j
    while i > 0 or j > 0:
        pi, pj = bp[i][j]
        if pi == i - 1 and pj == j - 1:
            pairs.append((pi, pj))
        i, j = pi, pj
        if i < 0 or j < 0:
            break

    pairs.reverse()

    # Build per-sheet-word timestamps
    word_times = {}
    for sheet_idx, trans_idx in pairs:
        tw = transcript_words[trans_idx]
        word_times[sheet_idx] = {'start': tw['start'], 'end': tw['end']}

    return pairs, word_times


# ── Main Alignment Pipeline ────────────────────────────────

def align_chords_via_lyrics(chord_sheet_text, audio_path, model_size='base'):
    """Full pipeline: chord sheet + audio → time-stamped chord labels.

    Returns:
        segments: list of {chord, start, end, word, confidence}
        transcript_words: list of {word, start, end}
        spatial_chords: the parsed spatial chord data
        lyrics_words: the sheet lyrics word list
    """
    # 1. Parse chord sheet spatially
    spatial_chords, lyrics_words = parse_chord_sheet_spatial(chord_sheet_text)

    if not spatial_chords:
        return [], [], [], []

    # 2. Transcribe audio → word-level timestamps
    transcript_words = transcribe_words(audio_path, model_size=model_size)

    if not transcript_words:
        return [], transcript_words, spatial_chords, lyrics_words

    # 3. Align sheet lyrics to transcript
    _, word_times = align_lyrics_to_transcript(lyrics_words, transcript_words)

    # 4. Map chord positions to timestamps through their associated words
    #    Each spatial chord is above a word. Find that word's index in lyrics_words,
    #    then look up its timestamp from the transcript alignment.
    lyric_word_idx = 0  # running index into lyrics_words
    segments = []

    for chord_entry in spatial_chords:
        word = chord_entry['word']
        word_clean = chord_entry['word_clean']

        if word is None or not word_clean:
            # Chord-only line (no lyrics), we'll interpolate later
            segments.append({
                'chord': chord_entry['chord'],
                'raw_chord': chord_entry['raw_chord'],
                'start': None,
                'end': None,
                'word': word,
                'section': chord_entry['section'],
                'confidence': 0.0,
            })
            continue

        # Find this word's index in the lyrics_words list
        found_idx = None
        for k in range(lyric_word_idx, len(lyrics_words)):
            if _clean_word(lyrics_words[k]) == word_clean:
                found_idx = k
                lyric_word_idx = k + 1
                break

        if found_idx is not None and found_idx in word_times:
            ts = word_times[found_idx]
            segments.append({
                'chord': chord_entry['chord'],
                'raw_chord': chord_entry['raw_chord'],
                'start': ts['start'],
                'end': ts['end'],
                'word': word,
                'section': chord_entry['section'],
                'confidence': 1.0,
            })
        else:
            segments.append({
                'chord': chord_entry['chord'],
                'raw_chord': chord_entry['raw_chord'],
                'start': None,
                'end': None,
                'word': word,
                'section': chord_entry['section'],
                'confidence': 0.0,
            })

    # 5. Fill in gaps: interpolate timestamps for chords without matches
    _interpolate_timestamps(segments)

    # 6. Extend each chord's end to the next chord's start
    for i in range(len(segments) - 1):
        if segments[i + 1]['start'] is not None:
            segments[i]['end'] = segments[i + 1]['start']
    # Last segment: extend to end of audio (use last transcript word)
    if segments and transcript_words:
        segments[-1]['end'] = transcript_words[-1]['end']

    return segments, transcript_words, spatial_chords, lyrics_words


def _interpolate_timestamps(segments):
    """Fill in None timestamps by interpolating between known ones."""
    # Forward fill
    last_time = 0.0
    for seg in segments:
        if seg['start'] is not None:
            last_time = seg['start']
        else:
            seg['start'] = last_time

    # If there are still Nones at the start, backward fill
    next_time = segments[-1]['start'] if segments else 0
    for seg in reversed(segments):
        if seg['start'] is None or seg['start'] == 0.0:
            seg['start'] = next_time
        next_time = seg['start']
