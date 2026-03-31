#!/usr/bin/env python3
"""V1 feature extraction: beat-synchronous 12-dim HPCP features from audio + annotations.

Supports two modes:

1. Isophonics mode (--isophonics):
   Uses Isophonics annotation directory structure with gold beat/key annotations.
   Auto-detects the artist subdirectory under chordlab/.

2. Flat mode (--audio-dir + --lab-dir):
   Simple flat directories of audio and .lab files, auto-detects beats.

Feature extraction pipeline:
   audio -> EqualLoudness -> HighPass(100Hz) -> HPCP(12-bin, harmonics=4)
   -> flatness weighting -> temporal smoothing -> beat synchronization

Output .npz:
    X:           (N_beats, 12) float32  -- beat-synchronous HPCP
    y:           (N_beats,)    int32    -- chord class indices (Tier 1: 25 classes)
    song_ids:    (N_beats,)    int32    -- song index per beat
    beat_times:  object array           -- beat times per song
    filenames:   object array           -- song identifiers
    key_indices: (N_songs,)    int32    -- key index per song (0-11, relative major)

Usage:
    python -m v1.prepare_data \\
        --isophonics "data/beatles/annotations" \\
        --isophonics "data/queen/annotations" \\
        --audio-dir data/audio --out data/features.npz
"""""
import sys
import os
import argparse
import numpy as np

from essentia.standard import (
    MonoLoader, RhythmExtractor2013,
    HPCP, Key,
    FrameGenerator, Windowing, Spectrum, SpectralPeaks,
    Flatness, HighPass, EqualLoudness,
)

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)
from shared.chord_vocab import parse_chord_label, label_to_idx, NOTES

SR = 44100
FRAME_SIZE = 8192
HOP_SIZE = 2048

_ENHARMONIC_TO_SHARP = {
    'Db': 'C#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
}
_SHARP_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ── Annotation parsers ───────────────────────────────────────

def parse_lab_file(lab_path):
    """Parse a chord .lab file (space or tab separated)."""
    annotations = []
    with open(lab_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            annotations.append((float(parts[0]), float(parts[1]), parts[2]))
    return annotations


def parse_beat_file(beat_path):
    """Parse Isophonics beat file (.txt): 'time beat_number'."""
    beat_times = []
    with open(beat_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                beat_times.append(float(parts[0]))
    return beat_times


def parse_key_file(key_path):
    """Parse Isophonics key file: 'start end Key|Silence [key_name]'.
    Returns (key_name, mode) for the dominant key segment."""
    keys = []
    with open(key_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[2] == 'Key':
                duration = float(parts[1]) - float(parts[0])
                key_name = parts[3]
                keys.append((key_name, duration))
    if not keys:
        return None, None
    key_name, _ = max(keys, key=lambda x: x[1])
    mode = 'major'
    if 'minor' in key_name:
        key_name = key_name.replace('minor', '')
        mode = 'minor'
    elif 'major' in key_name:
        key_name = key_name.replace('major', '')
    return key_name, mode


def key_to_index(key_name, mode):
    """Convert key name + mode to relative major index (0-11)."""
    if key_name is None:
        return 0
    kn = _ENHARMONIC_TO_SHARP.get(key_name, key_name)
    if kn not in _SHARP_NOTES:
        return 0
    idx = _SHARP_NOTES.index(kn)
    if mode == 'minor':
        idx = (idx + 3) % 12
    return idx


def get_chord_at_time(annotations, t):
    for start, end, label in annotations:
        if start <= t < end:
            return label
    return 'N'


# ── Feature extraction ───────────────────────────────────────

def extract_hpcp(audio_path):
    """Extract frame-level HPCP + flatness weights."""
    audio = MonoLoader(filename=audio_path, sampleRate=SR)()
    audio = EqualLoudness(sampleRate=SR)(audio)
    audio = HighPass(cutoffFrequency=100, sampleRate=SR)(audio)

    windowing = Windowing(type='blackmanharris62', size=FRAME_SIZE)
    spectrum_algo = Spectrum(size=FRAME_SIZE)
    peaks = SpectralPeaks(
        orderBy='magnitude', magnitudeThreshold=0.0001, maxPeaks=40,
        minFrequency=80, maxFrequency=4000, sampleRate=SR,
    )
    hpcp_algo = HPCP(
        size=12, referenceFrequency=440, harmonics=4,
        bandPreset=False, minFrequency=80, maxFrequency=4000,
        weightType='cosine', nonLinear=True, windowSize=1.0, sampleRate=SR,
    )
    flatness_algo = Flatness()

    hpcps, hpcps_native, flatness_weights = [], [], []
    for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE,
                                startFromZero=True):
        spec = spectrum_algo(windowing(frame))
        freqs, mags = peaks(spec)
        hpcp = hpcp_algo(freqs, mags)
        hpcps_native.append(hpcp)
        # Essentia HPCP bin 0 = A (ref 440Hz). Rotate so bin 0 = C.
        hpcps.append(np.roll(hpcp, -3))
        flatness_weights.append(max(0.2, 1.0 - flatness_algo(spec) * 2.0))

    hpcps = np.array(hpcps, dtype=np.float32)
    hpcps_native = np.array(hpcps_native, dtype=np.float32)
    flatness_weights = np.array(flatness_weights, dtype=np.float32)

    if len(hpcps) >= 5:
        kernel = np.ones(5) / 5.0
        smoothed = np.zeros_like(hpcps)
        for b in range(12):
            smoothed[:, b] = np.convolve(hpcps[:, b], kernel, mode='same')
        hpcps = smoothed

    return hpcps, hpcps_native, flatness_weights


_BEAT_THIS_MODEL = None


def detect_beats(audio_path, use_beat_this=False):
    """Auto-detect beats. Uses Beat This! transformer if requested, else Essentia."""
    if use_beat_this:
        try:
            return _detect_beats_beat_this(audio_path)
        except Exception:
            pass
    audio = MonoLoader(filename=audio_path, sampleRate=SR)()
    rhythm = RhythmExtractor2013(method='multifeature')
    _, beats, _, _, _ = rhythm(audio)
    bt = beats.tolist()
    if len(bt) > 1:
        filtered = [bt[0]]
        for b in bt[1:]:
            if b - filtered[-1] >= 0.15:
                filtered.append(b)
        bt = filtered
    return bt


def _detect_beats_beat_this(audio_path):
    """Detect beats using Beat This! transformer model (ISMIR 2024)."""
    global _BEAT_THIS_MODEL
    if _BEAT_THIS_MODEL is None:
        from beat_this.inference import File2Beats
        _BEAT_THIS_MODEL = File2Beats(
            checkpoint_path="small0", device="cpu", dbn=False)
    beats, _downbeats = _BEAT_THIS_MODEL(audio_path)
    return beats.tolist()


def detect_key(hpcps_native):
    """Auto-detect key (fallback when no gold key). Expects A-referenced HPCP."""
    key_algo = Key(profileType='bgate')
    key_name, scale, _, _ = key_algo(np.mean(hpcps_native, axis=0))
    return key_to_index(key_name, 'minor' if scale == 'minor' else 'major')


def recurrence_smooth(beat_chroma, threshold=0.92, self_weight=2.0):
    """Smooth beat-level HPCP using a recurrence (self-similarity) matrix.

    Beats in repeated song sections (verse 1 ≈ verse 2) get averaged
    together, reinforcing the true harmonic content and suppressing noise.

    Args:
        beat_chroma: (n_beats, 12) HPCP array
        threshold: cosine similarity threshold for recurrence links
        self_weight: extra weight for the beat itself vs its recurrent peers

    Returns:
        (n_beats, 12) smoothed HPCP array
    """
    n = len(beat_chroma)
    if n < 4:
        return beat_chroma

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(beat_chroma, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = beat_chroma / norms

    # Cosine similarity matrix
    sim = normed @ normed.T  # (n, n)

    # Zero out the main diagonal band (±2 beats) to avoid local smoothing
    # which is already handled by temporal smoothing
    for d in range(-2, 3):
        diag_idx = np.arange(max(0, d), min(n, n + d))
        off_idx = diag_idx - d
        sim[diag_idx, off_idx] = 0.0

    # Build weights: self_weight for identity, sim value for recurrent beats
    smoothed = np.zeros_like(beat_chroma)
    for i in range(n):
        recurrent = np.where(sim[i] >= threshold)[0]
        if len(recurrent) == 0:
            smoothed[i] = beat_chroma[i]
        else:
            weights = sim[i, recurrent]
            total_w = self_weight + weights.sum()
            smoothed[i] = (self_weight * beat_chroma[i] +
                           (beat_chroma[recurrent] * weights[:, np.newaxis]).sum(axis=0)
                           ) / total_w

    return smoothed


def sync_to_beats(hpcps, beat_times, flatness_weights):
    """Aggregate frame-level HPCP to beat-level."""
    n_frames = len(hpcps)
    beat_frames = [int(round(bt * SR / HOP_SIZE)) for bt in beat_times]
    n_beats = len(beat_frames)
    beat_chroma = np.zeros((n_beats, 12), dtype=np.float32)

    for bi in range(n_beats):
        start = min(beat_frames[bi], n_frames - 1)
        end = beat_frames[bi + 1] if bi + 1 < n_beats else n_frames
        end = min(end, n_frames)
        if end <= start:
            end = start + 1
        seg = hpcps[start:end]
        w = flatness_weights[start:end]
        w_sum = w.sum()
        if w_sum > 0:
            beat_chroma[bi] = (seg * w[:, np.newaxis]).sum(axis=0) / w_sum
        else:
            beat_chroma[bi] = np.median(seg, axis=0)
    return beat_chroma


def process_song(audio_path, chord_annotations, beat_times, key_idx, tier=1):
    """Extract beat-sync HPCP + assign chord labels per beat."""
    if len(beat_times) < 2:
        return None, None, beat_times, key_idx

    hpcps, _, flatness_weights = extract_hpcp(audio_path)
    beat_chroma = sync_to_beats(hpcps, beat_times, flatness_weights)

    # Recurrence smoothing disabled — it tends to homogenise chroma toward
    # the dominant chord (e.g. tonic C absorbs Am/F beats in pop songs).
    # Raw beat-level chroma gives the classifier more discriminative features.
    # beat_chroma = recurrence_smooth(beat_chroma)

    labels = []
    for bi in range(len(beat_times)):
        if bi + 1 < len(beat_times):
            mid = (beat_times[bi] + beat_times[bi + 1]) / 2.0
        else:
            mid = beat_times[bi] + 0.25
        raw_label = get_chord_at_time(chord_annotations, mid)
        parsed = parse_chord_label(raw_label, tier=tier)
        labels.append(label_to_idx(parsed, tier=tier))

    return beat_chroma, np.array(labels, dtype=np.int32), beat_times, key_idx


# ── Dataset discovery ────────────────────────────────────────

def find_isophonics_songs(annotations_root, audio_root):
    """Find songs from Isophonics structure with gold beats/keys.

    annotations_root: Isophonics annotations directory (e.g. "The Beatles Annotations")
    audio_root: directory with album subdirs containing audio

    Auto-detects the artist subdirectory under chordlab/.
    """
    chordlab_root = os.path.join(annotations_root, 'chordlab')
    if not os.path.isdir(chordlab_root):
        return []
    # Auto-detect artist subdirectory (e.g. "The Beatles" or "Queen")
    artist_dirs = [d for d in os.listdir(chordlab_root)
                   if os.path.isdir(os.path.join(chordlab_root, d))]
    if not artist_dirs:
        return []
    artist = artist_dirs[0]
    chord_base = os.path.join(annotations_root, 'chordlab', artist)
    beat_base = os.path.join(annotations_root, 'beat', artist)
    key_base = os.path.join(annotations_root, 'keylab', artist)

    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    results = []

    for album in sorted(os.listdir(chord_base)):
        chord_dir = os.path.join(chord_base, album)
        if not os.path.isdir(chord_dir):
            continue
        for fname in sorted(os.listdir(chord_dir)):
            if not fname.endswith('.lab'):
                continue
            stem = fname[:-4]
            chord_path = os.path.join(chord_dir, fname)
            beat_path = os.path.join(beat_base, album, stem + '.txt')
            key_path = os.path.join(key_base, album, stem + '.lab')

            # Find audio
            audio_path = None
            audio_album_dir = os.path.join(audio_root, album)
            if os.path.isdir(audio_album_dir):
                for ext in audio_exts:
                    candidate = os.path.join(audio_album_dir, stem + ext)
                    if os.path.exists(candidate):
                        audio_path = candidate
                        break
            if audio_path is None:
                continue

            results.append({
                'audio': audio_path,
                'chords': chord_path,
                'beats': beat_path if os.path.exists(beat_path) else None,
                'key': key_path if os.path.exists(key_path) else None,
                'stem': f"{album}/{stem}",
            })
    return results


def find_flat_pairs(audio_dir, lab_dir):
    """Find matching audio/.lab pairs from flat directories."""
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = {}
    for f in sorted(os.listdir(audio_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in audio_exts:
            audio_files[stem] = os.path.join(audio_dir, f)

    lab_files = {}
    for f in sorted(os.listdir(lab_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() == '.lab':
            lab_files[stem] = os.path.join(lab_dir, f)

    return [{'audio': audio_files[s], 'chords': lab_files[s],
             'beats': None, 'key': None, 'stem': s}
            for s in sorted(set(audio_files) & set(lab_files))]


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract training features from audio + chord annotations')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to weights.json config (auto-discovers all datasets)')
    parser.add_argument('--isophonics', type=str, action='append', default=None,
                        help='Path to Isophonics annotations root (can be specified multiple times)')
    parser.add_argument('--audio-dir', type=str, action='append', default=None,
                        help='Directory with audio files (repeatable, pairs with --isophonics)')
    parser.add_argument('--lab-dir', type=str, default=None,
                        help='Directory with .lab files (flat mode only)')
    parser.add_argument('--out', default='data/features.npz', help='Output .npz file')
    parser.add_argument('--tier', type=int, default=1, choices=[1, 2])
    args = parser.parse_args()

    import json as _json

    songs = []
    mode = 'config'

    if args.config:
        config_dir = os.path.dirname(os.path.abspath(args.config))
        base_dir = os.path.dirname(config_dir)  # training/ root
        with open(args.config) as f:
            cfg = _json.load(f)
        for name, ds in cfg['datasets'].items():
            if ds['type'] == 'isophonics':
                ann_root = os.path.join(base_dir, ds['annotations'])
                aud_dir = os.path.join(base_dir, ds['audio_dir'])
                found = find_isophonics_songs(ann_root, aud_dir)
                print(f"  {name}: {len(found)} songs (isophonics)")
                songs.extend(found)
            elif ds['type'] == 'flat':
                aud_dir = os.path.join(base_dir, ds['audio_dir'])
                lab_dir = os.path.join(base_dir, ds['labels_dir'])
                if os.path.isdir(aud_dir) and os.path.isdir(lab_dir):
                    found = find_flat_pairs(aud_dir, lab_dir)
                    print(f"  {name}: {len(found)} songs (flat)")
                    songs.extend(found)
                else:
                    print(f"  {name}: skipped (dirs not found)")
    elif args.isophonics:
        audio_dirs = args.audio_dir or []
        if not audio_dirs:
            print("ERROR: --audio-dir required with --isophonics")
            sys.exit(1)
        # If fewer audio-dirs than isophonics, repeat the last one
        while len(audio_dirs) < len(args.isophonics):
            audio_dirs.append(audio_dirs[-1])
        for iso_root, aud_dir in zip(args.isophonics, audio_dirs):
            found = find_isophonics_songs(iso_root, aud_dir)
            songs.extend(found)
        mode = 'isophonics'
    elif args.audio_dir and args.lab_dir:
        songs = find_flat_pairs(args.audio_dir[0], args.lab_dir)
        mode = 'flat'
    else:
        print("ERROR: provide --config, --isophonics + --audio-dir, or --audio-dir + --lab-dir")
        sys.exit(1)

    if not songs:
        print("No matching audio/annotation pairs found!")
        if mode == 'isophonics':
            print(f"  Annotations: {args.isophonics}")
            print(f"  Audio dir:   {args.audio_dir}")
            print("  Audio should be in ALBUM/SONG.{mp3,wav,...} subdirectories")
        sys.exit(1)

    n_gold_beats = sum(1 for s in songs if s['beats'])
    n_gold_keys = sum(1 for s in songs if s['key'])
    print(f"Found {len(songs)} songs ({mode} mode)")
    print(f"  Gold beats: {n_gold_beats}, Gold keys: {n_gold_keys}")

    all_X, all_y, all_song_ids = [], [], []
    all_beat_times, all_filenames, all_key_indices = [], [], []

    for song_idx, song in enumerate(songs):
        print(f"  [{song_idx+1}/{len(songs)}] {song['stem']}...", end=' ', flush=True)
        try:
            chord_annotations = parse_lab_file(song['chords'])

            if song['beats']:
                beat_times = parse_beat_file(song['beats'])
                beat_src = 'gold'
            else:
                beat_times = detect_beats(song['audio'])
                beat_src = 'auto'

            if song['key']:
                kn, km = parse_key_file(song['key'])
                key_idx = key_to_index(kn, km)
                key_src = 'gold'
            else:
                _, hpcps_native, _ = extract_hpcp(song['audio'])
                key_idx = detect_key(hpcps_native)
                key_src = 'auto'

            X, y, bt, ki = process_song(
                song['audio'], chord_annotations, beat_times, key_idx, args.tier)

            if X is None or len(X) == 0:
                print("SKIP (no beats)")
                continue

            all_X.append(X)
            all_y.append(y)
            all_song_ids.append(np.full(len(y), song_idx, dtype=np.int32))
            all_beat_times.append(np.array(bt, dtype=np.float32))
            all_filenames.append(song['stem'])
            all_key_indices.append(key_idx)

            key_note = NOTES[key_idx] if key_idx < len(NOTES) else '?'
            print(f"OK ({len(X)} beats, key={key_note}[{key_src}], beats={beat_src})")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_X:
        print("\nNo features extracted!")
        sys.exit(1)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    song_ids = np.concatenate(all_song_ids, axis=0)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(
        args.out, X=X, y=y, song_ids=song_ids,
        beat_times=np.array(all_beat_times, dtype=object),
        filenames=np.array(all_filenames, dtype=object),
        key_indices=np.array(all_key_indices, dtype=np.int32),
    )

    from chord_vocab import TIER1_VOCAB, TIER2_VOCAB
    vocab = TIER1_VOCAB if args.tier == 1 else TIER2_VOCAB
    print(f"\nSaved {len(X)} beat-level features to {args.out}")
    print(f"Songs: {len(all_filenames)}, Vocabulary: {len(vocab)} classes (tier {args.tier})")
    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for idx, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        pct = 100.0 * cnt / len(y)
        if pct >= 0.5:
            print(f"  {vocab[idx]:8s}: {cnt:6d} ({pct:5.1f}%)")


if __name__ == '__main__':
    main()
