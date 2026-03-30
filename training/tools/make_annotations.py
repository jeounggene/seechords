#!/usr/bin/env python3
"""Create .lab annotation files from simple chord sheets.

Converts human-readable chord sheets into timed .lab annotation files
by combining them with beat detection from the audio.

Usage:
    # From a simple chord list (one chord per line, applied to beats):
    python make_annotations.py --audio song.mp3 --chords chords.txt --out song.lab

    # From a section-based chord sheet:
    python make_annotations.py --audio song.mp3 --chords chords.txt --out song.lab --format sections

Input formats:

1. "beats" format (default) — one chord per beat:
    C
    C
    Am
    Am
    F
    F
    G
    G

2. "sections" format — chord with optional bar count (default 4 beats per bar):
    # Intro
    N x2
    # Verse
    C x4
    Am x4
    F x4
    G x4
    # Chorus
    F x4
    Am x4
    C x4
    G x4

3. "timed" format — start_time end_time chord:
    0.0 2.5 C
    2.5 5.0 Am
    5.0 7.5 F
    7.5 10.0 G

Output: standard .lab file (tab-separated: start end chord_label)
"""
import sys
import os
import argparse
import numpy as np

# Essentia for beat detection
from essentia.standard import MonoLoader, RhythmExtractor2013

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)
from shared.chord_vocab import parse_chord_label, NOTES

SR = 44100


def detect_beats(audio_path):
    """Detect beat times in audio."""
    audio = MonoLoader(filename=audio_path, sampleRate=SR)()
    rhythm = RhythmExtractor2013(method='multifeature')
    bpm, beats, _, _, _ = rhythm(audio)

    beat_times = beats.tolist()
    if len(beat_times) > 1:
        filtered = [beat_times[0]]
        for bt in beat_times[1:]:
            if bt - filtered[-1] >= 0.15:
                filtered.append(bt)
        beat_times = filtered

    duration = len(audio) / SR
    return beat_times, float(bpm), duration


def parse_beats_format(chord_text):
    """Parse one-chord-per-beat format."""
    chords = []
    for line in chord_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        chords.append(line)
    return chords


def parse_sections_format(chord_text, beats_per_bar=4):
    """Parse section-based format with optional repeat counts.

    Each line: CHORD [xN] where N is number of bars.
    A bar = beats_per_bar beats.
    """
    chords = []
    for line in chord_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        chord = parts[0]
        n_bars = 1
        if len(parts) > 1 and parts[1].startswith('x'):
            try:
                n_bars = int(parts[1][1:])
            except ValueError:
                n_bars = 1

        for _ in range(n_bars * beats_per_bar):
            chords.append(chord)

    return chords


def parse_timed_format(chord_text):
    """Parse timed format (start end chord)."""
    annotations = []
    for line in chord_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 3:
            annotations.append((float(parts[0]), float(parts[1]), parts[2]))
    return annotations


def chords_to_lab(beat_times, chords, duration):
    """Convert chord list + beat times to .lab annotations.

    Each chord is mapped to the corresponding beat interval.
    If there are more chords than beats, excess chords are ignored.
    If there are fewer chords than beats, the last chord repeats.
    """
    annotations = []
    n = min(len(chords), len(beat_times))

    for i in range(n):
        start = beat_times[i]
        if i + 1 < len(beat_times):
            end = beat_times[i + 1]
        else:
            end = duration

        # Normalize chord to Harte format
        chord = chords[i]
        if chord == 'N' or chord == '-' or chord == 'NC':
            harte = 'N'
        else:
            # Simple conversion: C → C:maj, Am → A:min, etc.
            parsed = parse_chord_label(chord)
            if parsed == 'N':
                harte = 'N'
            elif parsed.endswith('m'):
                harte = parsed[:-1] + ':min'
            else:
                harte = parsed + ':maj'

        annotations.append((start, end, harte))

    return annotations


def write_lab(annotations, out_path):
    """Write .lab annotation file."""
    with open(out_path, 'w') as f:
        for start, end, chord in annotations:
            f.write(f"{start:.6f}\t{end:.6f}\t{chord}\n")


def main():
    parser = argparse.ArgumentParser(description='Create .lab chord annotations from chord sheets + audio')
    parser.add_argument('--audio', required=True, help='Audio file path (for beat detection)')
    parser.add_argument('--chords', required=True, help='Chord sheet text file')
    parser.add_argument('--out', required=True, help='Output .lab file')
    parser.add_argument('--format', choices=['beats', 'sections', 'timed'], default='beats',
                        help='Input chord sheet format')
    parser.add_argument('--bpb', type=int, default=4, help='Beats per bar (for sections format)')
    args = parser.parse_args()

    # Read chord sheet
    with open(args.chords, 'r') as f:
        chord_text = f.read()

    if args.format == 'timed':
        # Already has timing — no need for beat detection
        annotations = parse_timed_format(chord_text)
        write_lab(annotations, args.out)
        print(f"Wrote {len(annotations)} timed annotations to {args.out}")
        return

    # Need beat detection for beats/sections formats
    print(f"Detecting beats in {args.audio}...")
    beat_times, bpm, duration = detect_beats(args.audio)
    print(f"  BPM: {bpm:.1f}, Beats: {len(beat_times)}, Duration: {duration:.1f}s")

    if args.format == 'beats':
        chords = parse_beats_format(chord_text)
    elif args.format == 'sections':
        chords = parse_sections_format(chord_text, args.bpb)

    print(f"  Chord entries: {len(chords)}")
    if len(chords) > len(beat_times):
        print(f"  WARNING: {len(chords)} chords but only {len(beat_times)} beats — truncating")
    elif len(chords) < len(beat_times):
        print(f"  NOTE: {len(chords)} chords for {len(beat_times)} beats — last chord will extend")

    annotations = chords_to_lab(beat_times, chords, duration)
    write_lab(annotations, args.out)
    print(f"Wrote {len(annotations)} annotations to {args.out}")


if __name__ == '__main__':
    main()
