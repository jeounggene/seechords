#!/usr/bin/env python3
"""Compare model chord predictions against a reference chord sheet.

Accepts several input formats:

1. **Simple chords** — one chord per line with time:
       0.0  F
       5.2  Em7
       8.0  A7

2. **Bar-based** — chords separated by | for bars:
       F | Em7 A7 | Dm | Bb C |

3. **UG / section-based** — paste from Ultimate Guitar:
       [Verse]
       F   Em7  A7
       Yesterday all my troubles
       Dm        Bb    C    F
       seemed so far away

   (Lines with chords are auto-detected; lyrics lines are ignored.)

Usage:
    python compare_chords.py <audio_file> <chord_sheet>
    python compare_chords.py <audio_file> --paste   # paste chords interactively
    python compare_chords.py <audio_file> --lab <lab_file>  # Isophonics .lab format

The tool runs analysis, aligns predictions to the reference, and shows a
colour-coded comparison with accuracy stats.
"""
import sys
import os
import re
import json
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(SCRIPT_DIR, '..', 'server')
VENV_PYTHON = os.path.join(SCRIPT_DIR, '..', '..', 'ezchords', '.venv312', 'bin', 'python')

# Chord name regex: root + optional quality
_CHORD_RE = re.compile(
    r'^[A-G][b#]?'
    r'(m|min|maj|dim|aug|sus[24]?|add\d+|7|9|11|13|maj7|m7|min7|mmaj7|dim7|6|m6)?'
    r'(/[A-G][b#]?)?$'
)

# Map enharmonics and normalize chord names
_ENHARMONIC = {
    'Db': 'C#', 'Eb': 'Eb', 'Gb': 'F#', 'Ab': 'Ab', 'Bb': 'Bb',
    'E#': 'F', 'B#': 'C', 'Cb': 'B', 'Fb': 'E',
}

# Normalize quality suffixes to canonical short forms (preserve extended types)
_QUALITY_MAP = {
    '': '', 'maj': '', '7': '7', 'maj7': 'maj7', '9': '9', '11': '11', '13': '13',
    'sus2': 'sus2', 'sus4': 'sus4', 'sus': 'sus4', 'add9': 'add9', '6': '6',
    'm': 'm', 'min': 'm', 'm7': 'm7', 'min7': 'm7', 'mmaj7': 'mmaj7', 'm6': 'm6',
    'dim': 'dim', 'dim7': 'dim7', 'aug': 'aug',
}


def _is_chord_token(token):
    """Check if a token looks like a chord name."""
    return bool(_CHORD_RE.match(token))


def _normalize_chord(name):
    """Normalize a chord name — preserve extended quality (A7, Cmaj7, etc.).

    Handles both standard names (Am7, Fmaj7) and Isophonics notation
    (A:min7, F:maj, D:min/b7, etc.).
    Normalizes root enharmonics and strips bass notes, but keeps the quality.
    """
    if not name or name in ('N', 'X', '-', 'NC', 'N.C.'):
        return 'N'

    # Handle Isophonics notation: "Root:quality" or "Root:quality/bass"
    if ':' in name:
        parts = name.split(':')
        root = parts[0]
        rest = parts[1] if len(parts) > 1 else ''
        # Strip bass note and modifiers like (*3)
        rest = re.sub(r'\(.*?\)', '', rest)
        if '/' in rest:
            rest = rest.split('/')[0]
        rest = rest.strip()

        root = _ENHARMONIC.get(root, root)
        _ISO_MAP = {
            '': '', 'maj': '', 'min': 'm', 'min7': 'm7', 'min6': 'm6',
            'maj7': 'maj7', '7': '7', 'dim': 'dim', 'dim7': 'dim7',
            'hdim': 'dim', 'hdim7': 'dim7', 'aug': 'aug',
            'sus2': 'sus2', 'sus4': 'sus4',
        }
        suffix = _ISO_MAP.get(rest, '')
        return root + suffix

    # Strip bass note
    if '/' in name:
        name = name.split('/')[0]

    # Parse root
    m = re.match(r'^([A-G][b#]?)(.*)', name)
    if not m:
        return 'N'
    root, quality = m.group(1), m.group(2)

    # Normalize root enharmonics
    root = _ENHARMONIC.get(root, root)

    # Normalize quality
    quality = _QUALITY_MAP.get(quality, quality)

    return root + quality


def _line_is_chords(line):
    """Heuristic: does this line consist mostly of chord tokens?"""
    tokens = line.split()
    if not tokens:
        return False
    # Filter out section markers like [Verse], bar lines, etc.
    clean = [t for t in tokens if t not in ('|', '/', '||', ':|', '|:', 'x2', 'x3', 'x4')]
    clean = [t for t in clean if not re.match(r'^\[.*\]$', t)]
    if not clean:
        return False
    chord_count = sum(1 for t in clean if _is_chord_token(t))
    return chord_count / len(clean) >= 0.5 and chord_count >= 1


def parse_chord_sheet(text):
    """Parse a chord sheet into a flat list of chord names.

    Returns list of normalized chord names in order of appearance.
    """
    chords = []
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip section headers like [Verse], [Chorus]
        if re.match(r'^\[.*\]$', line):
            continue

        # Format 1: "time chord" (lab-style)
        m = re.match(r'^[\d:.]+\s+[\d:.]+\s+(.+)', line)
        if m:
            chord = m.group(1).strip()
            chords.append(_normalize_chord(chord))
            continue

        m = re.match(r'^[\d:.]+\s+(.+)', line)
        if m:
            chord = m.group(1).strip()
            if _is_chord_token(chord):
                chords.append(_normalize_chord(chord))
                continue

        # Format 2: bar-based (has | separators)
        if '|' in line:
            # Split on bars, then on spaces within bars
            bars = re.split(r'\|+', line)
            for bar in bars:
                tokens = bar.split()
                for t in tokens:
                    if _is_chord_token(t):
                        chords.append(_normalize_chord(t))
            continue

        # Format 3: chord line from UG (heuristic)
        if _line_is_chords(line):
            tokens = line.split()
            for t in tokens:
                if _is_chord_token(t):
                    chords.append(_normalize_chord(t))
            continue

        # Otherwise skip (lyrics, etc.)

    return chords


def parse_lab_file(path):
    """Parse an Isophonics .lab file into (start, end, chord) tuples."""
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                chord = parts[2]
                segments.append((start, end, _normalize_chord(chord)))
    return segments


def run_analysis(audio_path):
    """Run the chord analysis via the server's analyze_chords.py."""
    analyze_script = os.path.join(SERVER_DIR, 'analyze_chords.py')
    result = subprocess.run(
        [VENV_PYTHON, analyze_script, audio_path],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"Analysis failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def compare_with_lab(analysis, lab_segments):
    """Compare analysis output against .lab ground truth (time-aligned)."""
    chords = analysis['chords']  # [{chord, start, end}, ...]

    # Build a chord lookup from analysis
    def get_predicted_chord(t):
        for seg in chords:
            if seg['start'] <= t < seg['end']:
                return _normalize_chord(seg['chord'])
        return 'N'

    total_time = 0.0
    correct_time = 0.0
    results = []

    for start, end, expected in lab_segments:
        duration = end - start
        mid = (start + end) / 2
        predicted = get_predicted_chord(mid)
        match = predicted == expected
        total_time += duration
        if match:
            correct_time += duration

        results.append({
            'start': start, 'end': end, 'duration': duration,
            'expected': expected, 'predicted': predicted, 'match': match,
        })

    accuracy = correct_time / total_time if total_time > 0 else 0
    return results, accuracy


def compare_with_sequence(analysis, ref_chords):
    """Compare analysis chord sequence against a reference sequence.

    Since the reference has no timing, we compare the unique chord sequence
    (collapsing consecutive duplicates on both sides).
    """
    # Collapse analysis into sequence of unique chords
    pred_chords = []
    for seg in analysis['chords']:
        c = _normalize_chord(seg['chord'])
        if not pred_chords or pred_chords[-1] != c:
            pred_chords.append(c)

    # Collapse reference
    ref_collapsed = []
    for c in ref_chords:
        if not ref_collapsed or ref_collapsed[-1] != c:
            ref_collapsed.append(c)

    return pred_chords, ref_collapsed


def _colour(text, match):
    """ANSI colour: green if match, red if not."""
    if match:
        return f'\033[92m{text}\033[0m'
    return f'\033[91m{text}\033[0m'


def print_lab_comparison(results, accuracy):
    """Print time-aligned comparison results."""
    print(f"\n{'Time':>10s}  {'Expected':>8s}  {'Predicted':>8s}  {'Match':>5s}")
    print('-' * 40)

    for r in results:
        t = f"{r['start']:6.1f}-{r['end']:5.1f}"
        match_str = _colour('✓', True) if r['match'] else _colour('✗', False)
        exp = r['expected']
        pred = _colour(r['predicted'], r['match'])
        print(f"{t:>10s}  {exp:>8s}  {pred:>8s}  {match_str}")

    print(f"\n  Weighted accuracy (WCSR): {accuracy:.1%}")
    correct = sum(1 for r in results if r['match'])
    print(f"  Segment accuracy: {correct}/{len(results)} ({correct/len(results):.1%})")


def print_sequence_comparison(pred_seq, ref_seq):
    """Print sequence comparison using LCS alignment."""
    # Simple sequence comparison: show side by side
    print(f"\nReference sequence ({len(ref_seq)} changes):")
    print('  ' + ' → '.join(ref_seq[:30]))
    if len(ref_seq) > 30:
        print(f'  ... ({len(ref_seq) - 30} more)')

    print(f"\nPredicted sequence ({len(pred_seq)} changes):")
    print('  ' + ' → '.join(pred_seq[:30]))
    if len(pred_seq) > 30:
        print(f'  ... ({len(pred_seq) - 30} more)')

    # Compute LCS for sequence similarity
    m, n = len(ref_seq), len(pred_seq)
    if m > 500 or n > 500:
        print("\n  (Sequences too long for detailed alignment)")
        return

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_seq[i-1] == pred_seq[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    similarity = lcs_len / max(m, n) if max(m, n) > 0 else 0
    print(f"\n  Sequence similarity (LCS): {similarity:.1%} ({lcs_len}/{max(m, n)})")

    # Show which reference chords were found
    ref_set = set(ref_seq)
    pred_set = set(pred_seq)
    found = ref_set & pred_set
    missing = ref_set - pred_set
    extra = pred_set - ref_set
    if found:
        print(f"  Chords found:   {', '.join(sorted(found))}")
    if missing:
        print(f"  Chords missing: {', '.join(sorted(missing))}")
    if extra:
        print(f"  Extra chords:   {', '.join(sorted(extra))}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare chord predictions against a reference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('audio', help='Path to audio file (mp3, wav, etc.)')
    parser.add_argument('chords', nargs='?', help='Path to chord sheet text file')
    parser.add_argument('--lab', help='Path to .lab file (Isophonics format)')
    parser.add_argument('--paste', action='store_true',
                        help='Paste chord sheet from clipboard / stdin')
    args = parser.parse_args()

    if not args.chords and not args.lab and not args.paste:
        parser.error('Provide a chord sheet file, --lab file, or --paste')

    # Run analysis
    print(f"Analyzing: {args.audio}")
    analysis = run_analysis(args.audio)
    print(f"  Key: {analysis['key']}, BPM: {analysis['bpm']}")
    print(f"  Segments: {len(analysis['chords'])}")
    print(f"  Beats: {len(analysis['beat_times'])}")

    # Show predicted chord summary
    from collections import Counter
    pred_counts = Counter()
    for seg in analysis['chords']:
        dur = seg['end'] - seg['start']
        pred_counts[seg['chord']] += dur
    total_dur = sum(pred_counts.values())
    print(f"\n  Predicted chord distribution:")
    for chord, dur in pred_counts.most_common(8):
        print(f"    {chord:>6s}: {dur/total_dur:5.1%}")

    # Compare
    if args.lab:
        lab_segments = parse_lab_file(args.lab)
        results, accuracy = compare_with_lab(analysis, lab_segments)
        print_lab_comparison(results, accuracy)
    else:
        if args.paste:
            print("\nPaste chord sheet (Ctrl+D when done):")
            text = sys.stdin.read()
        else:
            with open(args.chords) as f:
                text = f.read()

        ref_chords = parse_chord_sheet(text)
        if not ref_chords:
            print("No chords found in input. Check the format.", file=sys.stderr)
            sys.exit(1)
        print(f"\n  Reference chords parsed: {len(ref_chords)}")
        print(f"  Unique: {', '.join(dict.fromkeys(ref_chords))}")

        pred_seq, ref_seq = compare_with_sequence(analysis, ref_chords)
        print_sequence_comparison(pred_seq, ref_seq)


if __name__ == '__main__':
    main()
