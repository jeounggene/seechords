#!/usr/bin/env python3
"""Analyze Billboard annotations and select the best ~150 songs for training."""
import csv
import json
import os
import collections
import sys

LABELS_DIR = "data/billboard/labels"
METADATA_PATH = "data/billboard/metadata.csv"

# Our model's chord vocabulary (from shared/chord_vocab.py)
TARGET_QUALITIES = {'maj', 'min', '7', 'maj7', 'min7'}
# Map Billboard annotation qualities to our model's quality set
QUALITY_MAP = {
    'maj': 'maj', 'min': 'min', '7': '7', 'maj7': 'maj7', 'min7': 'min7',
    'hdim7': 'other', 'dim': 'other', 'dim7': 'other', 'aug': 'other',
    'sus4': 'other', 'sus2': 'other', '1': 'other', '5': 'other',
    'min6': 'other', 'maj6': 'other', '9': '7', 'min9': 'min7',
    'maj9': 'maj7', '11': '7', 'min11': 'min7', '13': '7',
}

def simplify_quality(full_qual):
    """Map a Billboard chord quality to our simplified set."""
    base = full_qual.split('(')[0].split('/')[0]
    return QUALITY_MAP.get(base, 'other')

def analyze_song(lab_path):
    """Analyze a .lab file and return stats."""
    chords = collections.Counter()
    qual_counter = collections.Counter()
    root_set = set()
    total_dur = 0
    chord_dur = 0  # non-N duration
    n_segments = 0
    n_changes = 0
    prev_chord = None

    with open(lab_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end, chord = float(parts[0]), float(parts[1]), parts[2]
            dur = end - start
            total_dur += dur
            n_segments += 1

            if chord != 'N':
                chords[chord] += 1
                chord_dur += dur
                if ':' in chord:
                    root, qual = chord.split(':', 1)
                    root_set.add(root)
                    sq = simplify_quality(qual)
                    qual_counter[sq] += 1

            if chord != prev_chord:
                n_changes += 1
            prev_chord = chord

    unique_chords = len(chords)
    pct_chord = chord_dur / total_dur if total_dur > 0 else 0
    n_target_quals = sum(1 for q in qual_counter if q in TARGET_QUALITIES)

    return {
        'unique_chords': unique_chords,
        'n_roots': len(root_set),
        'n_target_quals': n_target_quals,
        'qual_counter': qual_counter,
        'n_changes': n_changes,
        'duration': total_dur,
        'pct_chord': pct_chord,
        'n_segments': n_segments,
        'chords': chords,
    }

def main():
    # Load metadata
    meta = {}
    with open(METADATA_PATH) as f:
        for row in csv.DictReader(f):
            tid = row['id'].strip().zfill(4)
            if row['title'].strip():
                meta[tid] = row

    # Analyze all songs
    songs = []
    for lab_file in sorted(os.listdir(LABELS_DIR)):
        if not lab_file.endswith('.lab'):
            continue
        tid = lab_file[:-4]
        info = meta.get(tid, {})
        if not info:
            continue  # skip songs without metadata

        analysis = analyze_song(os.path.join(LABELS_DIR, lab_file))

        songs.append({
            'tid': tid,
            'title': info.get('title', '?'),
            'artist': info.get('artist', '?'),
            'chart_date': info.get('chart_date', ''),
            'peak_rank': info.get('peak_rank', ''),
            **analysis,
        })

    print(f"Analyzed {len(songs)} songs\n")

    # ── Scoring ──────────────────────────────────────────────
    # Score each song for training value:
    # 1. Chord diversity (unique chords, roots, qualities)
    # 2. Duration (longer = more training beats, but diminishing returns)
    # 3. Harmonic richness (chord changes per minute)
    # 4. Coverage of target qualities (maj, min, 7, maj7, min7)
    # 5. Low silence ratio (high pct_chord)

    for s in songs:
        dur_min = s['duration'] / 60
        changes_per_min = s['n_changes'] / dur_min if dur_min > 0 else 0

        score = 0
        score += min(s['unique_chords'], 15) * 3     # up to 45 pts for diversity
        score += s['n_roots'] * 2                      # up to 24 pts for root coverage
        score += s['n_target_quals'] * 8               # up to 40 pts for quality coverage
        score += min(changes_per_min, 20) * 1.5        # up to 30 pts for harmonic richness
        score += s['pct_chord'] * 20                   # up to 20 pts for low silence
        score += min(dur_min, 5) * 4                   # up to 20 pts for duration

        # Bonus for having min7/maj7 (less common, more valuable)
        if s['qual_counter'].get('min7', 0) > 2:
            score += 10
        if s['qual_counter'].get('maj7', 0) > 2:
            score += 10
        if s['qual_counter'].get('7', 0) > 2:
            score += 5

        s['score'] = score

    # Sort by score descending
    songs.sort(key=lambda s: s['score'], reverse=True)

    # ── Deduplicate by title+artist ──────────────────────────
    seen_songs = set()
    deduped = []
    for s in songs:
        key = (s['title'].lower().strip(), s['artist'].lower().strip())
        if key not in seen_songs:
            seen_songs.add(key)
            deduped.append(s)
    print(f"After dedup: {len(deduped)} unique songs (removed {len(songs)-len(deduped)} duplicates)\n")
    songs = deduped

    # ── Decade diversity ─────────────────────────────────────
    # Ensure we get songs from different decades
    decade_counts = collections.Counter()
    selected = []
    TARGET = 150

    # First pass: take top scorers but cap per-decade
    MAX_PER_DECADE = 35
    for s in songs:
        if len(selected) >= TARGET:
            break
        year = s['chart_date'][:4] if s['chart_date'] else '????'
        decade = year[:3] + '0s' if year != '????' else 'unknown'
        if decade_counts[decade] < MAX_PER_DECADE:
            selected.append(s)
            decade_counts[decade] += 1

    # If we need more, relax the cap
    if len(selected) < TARGET:
        already = {s['tid'] for s in selected}
        for s in songs:
            if len(selected) >= TARGET:
                break
            if s['tid'] not in already:
                selected.append(s)

    # ── Summary ──────────────────────────────────────────────
    print(f"Selected {len(selected)} songs for training\n")

    # Decade distribution
    decade_dist = collections.Counter()
    for s in selected:
        year = s['chart_date'][:4] if s['chart_date'] else '????'
        decade = year[:3] + '0s' if year != '????' else 'unknown'
        decade_dist[decade] += 1
    print("Decade distribution:")
    for dec in sorted(decade_dist):
        print(f"  {dec}: {decade_dist[dec]}")

    # Quality coverage
    qual_total = collections.Counter()
    for s in selected:
        for q, c in s['qual_counter'].items():
            qual_total[q] += c
    print(f"\nQuality distribution across selected songs:")
    for q, c in qual_total.most_common():
        print(f"  {q:8s}: {c:5d} segments")

    # Print the selected list
    print(f"\n{'─' * 90}")
    print(f"  {'ID':>4s}  {'Score':>5s}  {'Chords':>6s}  {'Dur':>5s}  {'Artist':<25s}  Title")
    print(f"{'─' * 90}")
    for s in selected:
        dur_str = f"{s['duration']/60:.1f}m"
        print(f"  {s['tid']:>4s}  {s['score']:5.0f}  {s['unique_chords']:6d}  {dur_str:>5s}  {s['artist'][:25]:<25s}  {s['title'][:40]}")

    # Output the track IDs as a JSON list for easy use
    ids = [s['tid'] for s in selected]
    with open('data/billboard/training_subset.json', 'w') as f:
        json.dump({'track_ids': ids, 'count': len(ids)}, f, indent=2)
    print(f"\nSaved track IDs to data/billboard/training_subset.json")

if __name__ == '__main__':
    main()
