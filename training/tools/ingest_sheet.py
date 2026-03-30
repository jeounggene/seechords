#!/usr/bin/env python3
"""Convert a chord sheet + audio into a DP-aligned .lab file for training.

Runs the chord model on the audio to get beat-level predictions, then
uses dynamic programming to optimally align the untimed chord sheet
tokens to those beats.  The result is a silver-tier .lab file that
can be reviewed and corrected before training.

Supported chord sheet formats:
  1. Bar-based:   | F | Em7 A7 | Dm | Bb C |
  2. Plain list:   F  Em7  A7  Dm  Bb  C   (space-separated)
  3. UG paste:     Lines with chord tokens auto-detected

Usage:
    python ingest_sheet.py <audio_file> <chord_sheet_file> [--out-dir data/silver]
    python ingest_sheet.py <audio_file> --paste [--out-dir data/silver]
    python ingest_sheet.py <audio_file> <chord_sheet_file> --review  # interactive

This creates:
    data/silver/audio/<name>.mp3   (symlink to original)
    data/silver/labels/<name>.lab  (DP-aligned chord annotations)
"""
import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAINING_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, _TRAINING_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from tools.compare_chords import parse_chord_sheet, run_analysis
from tools.align_chords import dp_align, beats_from_analysis, write_lab, print_alignment


def review_alignment(segments):
    """Interactive review: let user correct individual aligned chords.

    Returns the (possibly modified) segments list.
    """
    print("\n── Interactive Review ──")
    print("For each segment, press Enter to accept, or type a new chord.")
    print("Type 's' to split a segment, 'd' to delete, 'q' to finish.\n")

    reviewed = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        match = seg['sheet_chord'] == seg['pred_chord']
        marker = '✓' if match else '✗'
        print(f"[{i+1}/{len(segments)}] "
              f"{seg['start']:5.1f}-{seg['end']:5.1f}  "
              f"sheet={seg['sheet_chord']:>6s}  "
              f"model={seg['pred_chord']:>6s}  "
              f"beats={seg['n_beats']}  {marker}")

        ans = input("  > ").strip()
        if ans == '' or ans == 'y':
            reviewed.append(seg)
        elif ans == 'q':
            reviewed.extend(segments[i:])
            break
        elif ans == 'd':
            # Delete: merge beats into previous or next segment
            if reviewed:
                prev = reviewed[-1]
                prev['end'] = seg['end']
                prev['n_beats'] += seg['n_beats']
                prev['beat_end_idx'] = seg['beat_end_idx']
            # else: just skip
        elif ans == 's':
            # Split: ask for mid-beat
            mid = seg['n_beats'] // 2
            mid_idx = seg['beat_start_idx'] + mid
            if mid < 1 or seg['n_beats'] < 2:
                print("    Too few beats to split.")
                continue  # re-prompt same segment
            chord2 = input(f"    Second half chord [{seg['sheet_chord']}]: ").strip()
            if not chord2:
                chord2 = seg['sheet_chord']
            # First half
            from tools.align_chords import _normalize_chord  # noqa: F811
            seg1 = dict(seg)
            seg1['beat_end_idx'] = mid_idx
            seg1['n_beats'] = mid
            seg1['end'] = segments[i]['start']  # placeholder
            # We'd need beat_times to compute properly, but approximate:
            if mid_idx < seg['beat_end_idx']:
                # rough midpoint
                seg1['end'] = round(
                    seg['start'] + (seg['end'] - seg['start']) * mid / seg['n_beats'], 3)
            reviewed.append(seg1)
            # Second half
            seg2 = dict(seg)
            seg2['sheet_chord'] = chord2
            seg2['start'] = seg1['end']
            seg2['beat_start_idx'] = mid_idx
            seg2['n_beats'] = seg['n_beats'] - mid
            reviewed.append(seg2)
        else:
            # Treat as chord correction
            from tools.compare_chords import _normalize_chord
            new_chord = _normalize_chord(ans)
            corrected = dict(seg)
            corrected['sheet_chord'] = new_chord
            reviewed.append(corrected)
        i += 1

    return reviewed


def main():
    parser = argparse.ArgumentParser(
        description='DP-align chord sheet to model predictions for training')
    parser.add_argument('audio', help='Path to audio file')
    parser.add_argument('chords', nargs='?', help='Path to chord sheet text file')
    parser.add_argument('--paste', action='store_true',
                        help='Paste chord sheet from stdin')
    parser.add_argument('--out-dir', default='data/silver',
                        help='Output directory (default: data/silver)')
    parser.add_argument('--name', type=str, default=None,
                        help='Override song name (default: audio filename stem)')
    parser.add_argument('--review', action='store_true',
                        help='Interactive review/correction after alignment')
    parser.add_argument('--json', action='store_true',
                        help='Print alignment as JSON and exit')
    args = parser.parse_args()

    if not args.chords and not args.paste:
        parser.error('Provide a chord sheet file or --paste')

    # 1. Read chord sheet
    if args.paste:
        print("Paste chord sheet (Ctrl+D when done):")
        text = sys.stdin.read()
    else:
        with open(args.chords) as f:
            text = f.read()

    sheet_chords = parse_chord_sheet(text)
    if not sheet_chords:
        print("No chords found in input.", file=sys.stderr)
        sys.exit(1)

    unique = list(dict.fromkeys(sheet_chords))
    print(f"Parsed {len(sheet_chords)} chord tokens "
          f"({', '.join(unique)})")

    # 2. Run model analysis
    print(f"Analyzing: {args.audio}")
    analysis = run_analysis(args.audio)
    beat_chords, beat_times = beats_from_analysis(analysis)
    print(f"  Key: {analysis['key']}, BPM: {analysis['bpm']}, "
          f"Beats: {len(beat_times)}")

    # 3. DP alignment
    segments = dp_align(sheet_chords, beat_chords, beat_times)
    if not segments:
        print("Alignment failed — check chord count vs song length.",
              file=sys.stderr)
        sys.exit(1)

    # 4. Show alignment
    if args.json:
        import json
        from tools.align_chords import alignment_to_json
        print(json.dumps(alignment_to_json(segments), indent=2))
        return

    print_alignment(segments)

    # 5. Optional interactive review
    if args.review:
        segments = review_alignment(segments)
        print("\n── After review ──")
        print_alignment(segments)

    # 6. Save output
    name = args.name or os.path.splitext(os.path.basename(args.audio))[0]
    audio_dir = os.path.join(args.out_dir, 'audio')
    label_dir = os.path.join(args.out_dir, 'labels')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Symlink audio
    ext = os.path.splitext(args.audio)[1]
    audio_link = os.path.join(audio_dir, name + ext)
    audio_abs = os.path.abspath(args.audio)
    if os.path.exists(audio_link):
        os.remove(audio_link)
    os.symlink(audio_abs, audio_link)

    # Write .lab
    lab_path = os.path.join(label_dir, name + '.lab')
    write_lab(segments, lab_path)

    print(f"\nCreated:")
    print(f"  Audio: {audio_link} -> {audio_abs}")
    print(f"  Labels: {lab_path}")
    print(f"\nTo include in training, run:")
    print(f"  ./run_pipeline.sh --extra {args.out_dir}")

    # Preview
    print(f"\nFirst 10 segments:")
    for seg in segments[:10]:
        print(f"  {seg['start']:6.2f} - {seg['end']:6.2f}  {seg['sheet_chord']}")


if __name__ == '__main__':
    main()
