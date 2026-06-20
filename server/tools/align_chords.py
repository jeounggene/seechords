#!/usr/bin/env python3
"""DP-based alignment of untimed chord sheets to beat-level predictions.

Core algorithm: given K untimed chords from a sheet and N beat-level
predicted chords, find the optimal mapping of sheet chords to contiguous
spans of beats that minimises disagreement with the model's predictions.

Usage as CLI:
    python align_chords.py <audio_file> <chord_sheet> [--save-lab out.lab]
    python align_chords.py <audio_file> --paste [--save-lab out.lab]

Usage as library:
    from align_chords import dp_align, beats_from_analysis
    beat_chords, beat_times = beats_from_analysis(analysis)
    segments = dp_align(sheet_chords, beat_chords, beat_times)
"""
import sys
import os
import re
import json
import argparse
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAINING_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, _TRAINING_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from tools.compare_chords import (
    parse_chord_sheet, _normalize_chord, run_analysis, _colour,
)

# ── DP alignment ─────────────────────────────────────────────

MIN_BEATS_PER_CHORD = 2
MAX_BEATS_FACTOR = 10  # at most 10× average beats-per-chord

# Chord similarity: same root but different quality costs less than total mismatch
_NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def _chord_root(chord):
    """Extract root note from a normalised chord (e.g. 'Am' → 'A')."""
    if not chord or chord == 'N':
        return chord
    if len(chord) >= 2 and chord[1] in '#b':
        return chord[:2]
    return chord[:1]


def _mismatch_cost(sheet_chord, beat_chord):
    """Cost of assigning sheet_chord to a beat predicted as beat_chord.

    0.0 = exact match, 0.3 = same root different quality, 1.0 = total mismatch.
    """
    if sheet_chord == beat_chord:
        return 0.0
    if _chord_root(sheet_chord) == _chord_root(beat_chord):
        return 0.3  # Am vs A, C vs Cm — same root
    return 1.0


def _collapse_consecutive(chords):
    """Collapse consecutive duplicate chords.  ['C','C','G','G'] → ['C','G']"""
    if not chords:
        return []
    collapsed = [chords[0]]
    for c in chords[1:]:
        if c != collapsed[-1]:
            collapsed.append(c)
    return collapsed


def beats_from_analysis(analysis):
    """Extract per-beat chord list from analysis output.

    The analysis merges consecutive identical chords into segments.
    This reconstructs one chord label per beat.

    Returns (beat_chords, beat_times) — both lists of length N.
    """
    merged = analysis['chords']          # [{chord, start, end}, ...]
    beat_times = analysis['beat_times']  # [float, ...]

    beat_chords = []
    seg_idx = 0
    for bt in beat_times:
        # Advance to the segment covering this beat
        while (seg_idx < len(merged) - 1
               and merged[seg_idx + 1]['start'] <= bt + 1e-4):
            seg_idx += 1
        beat_chords.append(_normalize_chord(merged[seg_idx]['chord']))

    return beat_chords, beat_times


def dp_align(sheet_chords, beat_chords, beat_times):
    """Align untimed sheet chords to predicted beat-level chords via DP.

    Args:
        sheet_chords: list of K normalised chord names (from chord sheet)
        beat_chords:  list of N predicted chord names (one per beat)
        beat_times:   list of N beat timestamps (seconds)

    Returns:
        list of dicts, one per (collapsed) sheet chord:
            { start, end, sheet_chord, pred_chord, n_beats, matches,
              beat_start_idx, beat_end_idx }
    """
    # Collapse consecutive duplicates — "C C G G Am Am" → "C G Am"
    sheet_chords = _collapse_consecutive(sheet_chords)

    K = len(sheet_chords)
    N = len(beat_chords)

    if K == 0 or N == 0:
        return []

    # If more chords than beats allow, truncate sheet
    if K * MIN_BEATS_PER_CHORD > N:
        K = N // MIN_BEATS_PER_CHORD
        sheet_chords = sheet_chords[:K]
    if K == 0:
        return []

    # --- Pre-compute cost prefix sums ---
    # cost_prefix[i][j] = sum of _mismatch_cost for beats 0..j-1 vs sheet_chords[i]
    cost_prefix = []
    for i in range(K):
        sc = sheet_chords[i]
        prefix = [0.0] * (N + 1)
        for j in range(N):
            prefix[j + 1] = prefix[j] + _mismatch_cost(sc, beat_chords[j])
        cost_prefix.append(prefix)

    # Also precompute exact-match prefix for stats output
    match_prefix = []
    for i in range(K):
        sc = sheet_chords[i]
        prefix = [0] * (N + 1)
        for j in range(N):
            prefix[j + 1] = prefix[j] + (1 if beat_chords[j] == sc else 0)
        match_prefix.append(prefix)

    avg_beats = N / K
    max_span = max(int(avg_beats * MAX_BEATS_FACTOR), MIN_BEATS_PER_CHORD + 1)

    # Regularisation: small penalty for deviating from average span length.
    # Keeps chords from getting squeezed into 1-2 beats while others hog 20.
    REG_WEIGHT = 0.05

    INF = float('inf')
    # dp[i][j] = minimum cost to align first i sheet chords to first j beats
    dp = [[INF] * (N + 1) for _ in range(K + 1)]
    parent = [[-1] * (N + 1) for _ in range(K + 1)]
    dp[0][0] = 0

    for i in range(1, K + 1):
        remaining = K - i
        j_lo = i * MIN_BEATS_PER_CHORD
        j_hi = N - remaining * MIN_BEATS_PER_CHORD
        for j in range(j_lo, j_hi + 1):
            best_cost = INF
            best_k = -1
            # chord i covers beats [k, j)
            k_lo = max((i - 1) * MIN_BEATS_PER_CHORD, j - max_span)
            k_hi = j - MIN_BEATS_PER_CHORD
            for k in range(k_lo, k_hi + 1):
                if dp[i - 1][k] >= INF:
                    continue
                span = j - k
                # Weighted mismatch cost for this span
                span_cost = cost_prefix[i - 1][j] - cost_prefix[i - 1][k]
                # Regularisation: penalise spans far from average
                reg = REG_WEIGHT * abs(span - avg_beats)
                total = dp[i - 1][k] + span_cost + reg
                if total < best_cost:
                    best_cost = total
                    best_k = k
            if best_cost < INF:
                dp[i][j] = best_cost
                parent[i][j] = best_k

    if dp[K][N] >= INF:
        return []

    # --- Backtrack to recover boundaries ---
    boundaries = []
    j = N
    for i in range(K, 0, -1):
        k = parent[i][j]
        boundaries.append((k, j))
        j = k
    boundaries.reverse()

    # --- Build output segments ---
    segments = []
    for idx, (bstart, bend) in enumerate(boundaries):
        start_time = beat_times[bstart]
        if bend < N:
            end_time = beat_times[bend]
        else:
            end_time = beat_times[-1] + (beat_times[-1] - beat_times[-2]
                                         if N >= 2 else 0.5)

        span_preds = beat_chords[bstart:bend]
        pred_majority = Counter(span_preds).most_common(1)[0][0]
        n_matches = match_prefix[idx][bend] - match_prefix[idx][bstart]

        segments.append({
            'start': round(start_time, 3),
            'end': round(end_time, 3),
            'sheet_chord': sheet_chords[idx],
            'pred_chord': pred_majority,
            'n_beats': bend - bstart,
            'matches': n_matches,
            'beat_start_idx': bstart,
            'beat_end_idx': bend,
        })

    return segments


# ── Output helpers ───────────────────────────────────────────

def write_lab(segments, path):
    """Write alignment as an Isophonics-format .lab file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for seg in segments:
            chord = seg['sheet_chord']
            # Convert to Isophonics notation
            if chord == 'N':
                iso = 'N'
            elif chord.endswith('m'):
                iso = chord[:-1] + ':min'
            else:
                iso = chord + ':maj'
            f.write(f"{seg['start']:.6f} {seg['end']:.6f} {iso}\n")


def print_alignment(segments, *, verbose=False):
    """Pretty-print the aligned segments with match info."""
    total_beats = sum(s['n_beats'] for s in segments)
    total_matches = sum(s['matches'] for s in segments)

    print(f"\n{'Time':>12s}  {'Sheet':>7s}  {'Model':>7s}  "
          f"{'Beats':>5s}  {'Match':>5s}")
    print('-' * 50)

    for seg in segments:
        t = f"{seg['start']:5.1f}-{seg['end']:5.1f}"
        sc = seg['sheet_chord']
        pc = seg['pred_chord']
        match = sc == pc
        match_str = _colour('✓', True) if match else _colour('✗', False)
        pc_col = _colour(pc, match)
        pct = (seg['matches'] / seg['n_beats'] * 100
               if seg['n_beats'] else 0)
        beats_str = f"{seg['n_beats']:>3d}"
        match_detail = f"{seg['matches']}/{seg['n_beats']}"

        print(f"{t:>12s}  {sc:>7s}  {pc_col:>7s}  "
              f"{beats_str:>5s}  {match_detail:>5s} {match_str}")

    accuracy = total_matches / total_beats if total_beats else 0
    agree = sum(1 for s in segments if s['sheet_chord'] == s['pred_chord'])
    print(f"\n  Beat accuracy: {accuracy:.1%} ({total_matches}/{total_beats})")
    print(f"  Segment agreement: {agree}/{len(segments)} "
          f"({agree/len(segments):.1%})")


def alignment_to_json(segments):
    """Convert alignment segments to JSON-serialisable list."""
    return [
        {
            'start': s['start'],
            'end': s['end'],
            'sheetChord': s['sheet_chord'],
            'predChord': s['pred_chord'],
            'nBeats': s['n_beats'],
            'matches': s['matches'],
            'match': s['sheet_chord'] == s['pred_chord'],
        }
        for s in segments
    ]


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Align a chord sheet to model predictions using DP')
    parser.add_argument('audio', help='Path to audio file')
    parser.add_argument('chords', nargs='?', help='Chord sheet text file')
    parser.add_argument('--paste', action='store_true',
                        help='Paste chord sheet from stdin')
    parser.add_argument('--save-lab', metavar='PATH',
                        help='Save aligned result as .lab file')
    parser.add_argument('--json', action='store_true',
                        help='Output alignment as JSON')
    args = parser.parse_args()

    if not args.chords and not args.paste:
        parser.error('Provide a chord sheet file or --paste')

    # 1. Run model analysis
    print(f"Analyzing: {args.audio}", file=sys.stderr)
    analysis = run_analysis(args.audio)
    print(f"  Key: {analysis['key']}, BPM: {analysis['bpm']}, "
          f"Beats: {len(analysis['beat_times'])}", file=sys.stderr)

    # 2. Parse chord sheet
    if args.paste:
        print("Paste chord sheet (Ctrl+D when done):", file=sys.stderr)
        text = sys.stdin.read()
    else:
        with open(args.chords) as f:
            text = f.read()

    sheet_chords = parse_chord_sheet(text)
    if not sheet_chords:
        print("No chords found in input.", file=sys.stderr)
        sys.exit(1)

    unique = list(dict.fromkeys(sheet_chords))
    print(f"  Sheet chords: {len(sheet_chords)} "
          f"({', '.join(unique)})", file=sys.stderr)

    # 3. DP align
    beat_chords, beat_times = beats_from_analysis(analysis)
    segments = dp_align(sheet_chords, beat_chords, beat_times)

    if not segments:
        print("Alignment failed — too few beats for number of chords.",
              file=sys.stderr)
        sys.exit(1)

    # 4. Output
    if args.json:
        print(json.dumps(alignment_to_json(segments), indent=2))
    else:
        print_alignment(segments)

    if args.save_lab:
        write_lab(segments, args.save_lab)
        print(f"\nSaved .lab: {args.save_lab}", file=sys.stderr)


if __name__ == '__main__':
    main()
