#!/usr/bin/env python3
"""V2 feature extraction: 36-dim contextual HPCP + factorized root/quality labels.

Extends v1 features with:
  - 36-dim context: [prev_beat_HPCP, curr_beat_HPCP, next_beat_HPCP]
  - Factorized labels: separate root (13) and quality (7) class indices
  - Gold/silver sample weighting via --weight per --isophonics source
  - Provenance tracking (gold = has beat annotations, silver = auto beats)

Outputs both v1-compatible (12-dim) and v2 (36-dim) features for ablation.

Output .npz:
    X_12:           (N, 12)  beat HPCP (v1 baseline)
    X_36:           (N, 36)  contextual HPCP [prev, curr, next]
    root_labels:    (N,)     root class indices (13 classes)
    quality_labels: (N,)     quality class indices (7 classes)
    tier1_labels:   (N,)     tier-1 flat class indices (25 classes, for comparison)
    sample_weights: (N,)     gold/silver weights
    song_ids:       (N,)     song index per beat
    beat_times:     object   per-song beat times
    filenames:      object   song identifiers
    key_indices:    (S,)     key index per song
    provenance:     object   'gold' or 'silver' per song

Usage:
    python -m v2.prepare_data \\
        --isophonics "data/beatles/annotations" --weight 1.0 \\
        --isophonics "data/queen/annotations"       --weight 0.5 \\
        --audio-dir data/audio --out data/features_v2.npz
"""""
import sys
import os
import argparse
import numpy as np

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

# Reuse audio processing from v1
from v1.prepare_data import (
    extract_hpcp, detect_beats, detect_key,
    sync_to_beats, parse_lab_file, parse_beat_file, parse_key_file,
    key_to_index, get_chord_at_time, find_isophonics_songs, find_flat_pairs,
    SR, HOP_SIZE,
)
from shared.chord_vocab import parse_chord_label, label_to_idx
from v2.chord_schema import parse_chord_v2, ROOT_VOCAB, QUALITY_VOCAB

FRAME_SIZE = 8192  # same as v1


def extract_bass_hpcp(audio_path):
    """Extract frame-level bass-register HPCP (50-350 Hz).

    Separate pipeline optimised for bass frequencies:
      - No high-pass filter (preserves bass fundamentals)
      - Narrower frequency range (50-350 Hz)
      - Fewer harmonics (2 vs 4) to focus on fundamentals
    Returns (n_frames, 12) float32 array (C-referenced).
    """
    from essentia.standard import (
        MonoLoader, HPCP, FrameGenerator, Windowing, Spectrum,
        SpectralPeaks, EqualLoudness,
    )

    audio = MonoLoader(filename=audio_path, sampleRate=SR)()
    audio = EqualLoudness(sampleRate=SR)(audio)
    # NO high-pass — we want bass frequencies

    windowing = Windowing(type='blackmanharris62', size=FRAME_SIZE)
    spectrum_algo = Spectrum(size=FRAME_SIZE)
    peaks = SpectralPeaks(
        orderBy='magnitude', magnitudeThreshold=0.0001, maxPeaks=20,
        minFrequency=50, maxFrequency=350, sampleRate=SR,
    )
    hpcp_algo = HPCP(
        size=12, referenceFrequency=440, harmonics=2,
        bandPreset=False, minFrequency=50, maxFrequency=350,
        weightType='cosine', nonLinear=True, windowSize=1.0, sampleRate=SR,
    )

    hpcps = []
    for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE,
                                startFromZero=True):
        spec = spectrum_algo(windowing(frame))
        freqs, mags = peaks(spec)
        hpcp = hpcp_algo(freqs, mags)
        hpcps.append(np.roll(hpcp, -3))  # rotate A-ref → C-ref

    hpcps = np.array(hpcps, dtype=np.float32)

    if len(hpcps) >= 5:
        kernel = np.ones(5) / 5.0
        smoothed = np.zeros_like(hpcps)
        for b in range(12):
            smoothed[:, b] = np.convolve(hpcps[:, b], kernel, mode='same')
        hpcps = smoothed

    return hpcps


def compute_delta_chroma(beat_chroma, song_ids=None):
    """First-order chroma difference per beat.

    delta[i] = beat_chroma[i] − beat_chroma[i−1].
    Song boundaries (where song_ids changes) and index 0 get zero deltas.
    Returns (n_beats, 12) float32.
    """
    delta = np.zeros_like(beat_chroma, dtype=np.float32)
    delta[1:] = beat_chroma[1:] - beat_chroma[:-1]
    if song_ids is not None:
        boundaries = np.where(song_ids[1:] != song_ids[:-1])[0] + 1
        delta[boundaries] = 0.0
    return delta


def compute_third_ratio(beat_chroma):
    """Compute major/minor 3rd energy ratio for each possible root.

    For each beat and each of the 12 possible roots, computes:
        (chroma[root+4] - chroma[root+3]) / (chroma[root+4] + chroma[root+3] + eps)
    where root+4 = major 3rd interval, root+3 = minor 3rd interval.

    Returns (n_beats, 12) float32 in [-1, 1] range.
    Positive = major 3rd dominant, negative = minor 3rd dominant.
    """
    n_beats = len(beat_chroma)
    ratios = np.zeros((n_beats, 12), dtype=np.float32)
    for root in range(12):
        maj3 = beat_chroma[:, (root + 4) % 12]
        min3 = beat_chroma[:, (root + 3) % 12]
        ratios[:, root] = (maj3 - min3) / (maj3 + min3 + 1e-8)
    return ratios


def build_context_features(beat_chroma, radius=1):
    """Build contextual features: concatenate (2*radius+1) beats of 12-dim HPCP.

    radius=1 → 36-dim [prev, curr, next]   (default, backward-compatible)
    radius=2 → 60-dim [t-2, t-1, curr, t+1, t+2]

    Boundary beats are clamped (duplicated from nearest valid beat).
    """
    n_beats = len(beat_chroma)
    width = 2 * radius + 1
    feat_dim = width * 12
    X = np.zeros((n_beats, feat_dim), dtype=np.float32)

    for i in range(n_beats):
        for k, offset in enumerate(range(-radius, radius + 1)):
            j = max(0, min(n_beats - 1, i + offset))
            X[i, k * 12:(k + 1) * 12] = beat_chroma[j]

    return X


def process_song_v2(audio_path, chord_annotations, beat_times, key_idx,
                    context_radius=1, use_delta=False, use_bass=False):
    """Extract beat-sync HPCP + assign v2 labels per beat.

    Returns:
        X_12:    (n_beats, 12)  standard HPCP
        X_full:  (n_beats, D)   full feature vector (context + optional delta/bass)
        roots:   (n_beats,)     root indices
        quals:   (n_beats,)     quality indices
        tier1:   (n_beats,)     tier-1 flat indices
        bt:      list           beat times
        key_idx: int
    """
    if len(beat_times) < 2:
        return None, None, None, None, None, beat_times, key_idx

    hpcps, _, flatness_weights = extract_hpcp(audio_path)
    beat_chroma = sync_to_beats(hpcps, beat_times, flatness_weights)

    X_ctx = build_context_features(beat_chroma, radius=context_radius)

    # Optional extra features
    extra = []
    if use_delta:
        delta = np.zeros_like(beat_chroma, dtype=np.float32)
        delta[1:] = beat_chroma[1:] - beat_chroma[:-1]
        extra.append(delta)
    if use_bass:
        bass_hpcps = extract_bass_hpcp(audio_path)
        bass_chroma = sync_to_beats(bass_hpcps, beat_times,
                                    np.ones(len(bass_hpcps), dtype=np.float32))
        extra.append(bass_chroma)

    X_full = np.concatenate([X_ctx] + extra, axis=1) if extra else X_ctx

    roots, quals, tier1 = [], [], []
    for bi in range(len(beat_times)):
        if bi + 1 < len(beat_times):
            mid = (beat_times[bi] + beat_times[bi + 1]) / 2.0
        else:
            mid = beat_times[bi] + 0.25
        raw_label = get_chord_at_time(chord_annotations, mid)

        # v2 labels
        ri, qi = parse_chord_v2(raw_label)
        roots.append(ri)
        quals.append(qi)

        # tier-1 labels (for comparison)
        parsed = parse_chord_label(raw_label, tier=1)
        tier1.append(label_to_idx(parsed, tier=1))

    return (
        beat_chroma,
        X_full,
        np.array(roots, dtype=np.int32),
        np.array(quals, dtype=np.int32),
        np.array(tier1, dtype=np.int32),
        beat_times,
        key_idx,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Extract v2 training features (36-dim context + root/quality labels)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to weights.json config (auto-discovers all datasets)')
    parser.add_argument('--isophonics', type=str, action='append', default=None,
                        help='Isophonics annotations root (repeatable)')
    parser.add_argument('--weight', type=float, action='append', default=None,
                        help='Sample weight for corresponding --isophonics (repeatable)')
    parser.add_argument('--audio-dir', type=str, action='append', default=None,
                        help='Directory with audio files (repeatable, pairs with --isophonics)')
    parser.add_argument('--context-radius', type=int, default=1,
                        help='Context window radius: 1=3 beats (36-dim), 2=5 beats (60-dim)')
    parser.add_argument('--delta', action='store_true',
                        help='Add 12-dim chroma delta features (temporal change)')
    parser.add_argument('--bass', action='store_true',
                        help='Add 12-dim bass-register HPCP features (50-350 Hz)')
    parser.add_argument('--out', default='data/features_v2.npz',
                        help='Output .npz file')
    args = parser.parse_args()

    import json as _json

    all_songs = []

    if args.config:
        # Load datasets from weights.json config
        config_dir = os.path.dirname(os.path.abspath(args.config))
        base_dir = os.path.dirname(config_dir)  # training/ root
        with open(args.config) as f:
            cfg = _json.load(f)
        for name, ds in cfg['datasets'].items():
            w = ds.get('weight', 1.0)
            if ds['type'] == 'isophonics':
                ann_root = os.path.join(base_dir, ds['annotations'])
                aud_dir = os.path.join(base_dir, ds['audio_dir'])
                found = find_isophonics_songs(ann_root, aud_dir)
                for s in found:
                    s['weight'] = w
                    s['provenance'] = 'gold' if s['beats'] else 'silver'
                    s['_dataset'] = name
                print(f"  {name}: {len(found)} songs (isophonics, weight={w})")
                all_songs.extend(found)
            elif ds['type'] == 'flat':
                aud_dir = os.path.join(base_dir, ds['audio_dir'])
                lab_dir = os.path.join(base_dir, ds['labels_dir'])
                if os.path.isdir(aud_dir) and os.path.isdir(lab_dir):
                    found = find_flat_pairs(aud_dir, lab_dir)
                    # Filter by subset file if specified
                    subset_file = ds.get('subset_file')
                    if subset_file:
                        sf_path = os.path.join(base_dir, subset_file)
                        with open(sf_path) as _sf:
                            subset_data = _json.load(_sf)
                        allowed = set(subset_data.get('track_ids', subset_data))
                        before = len(found)
                        found = [s for s in found if s['stem'] in allowed]
                        print(f"  {name}: {len(found)}/{before} songs after subset filter (flat, weight={w})")
                    else:
                        print(f"  {name}: {len(found)} songs (flat, weight={w})")

                    # Load per-song tier scores if tier_file specified
                    tier_data = None
                    tier_file = ds.get('tier_file')
                    if tier_file:
                        tf_path = os.path.join(base_dir, tier_file)
                        if os.path.exists(tf_path):
                            with open(tf_path) as _tf:
                                tier_data = _json.load(_tf)
                            # Exclude rejected/untrusted songs
                            before_tier = len(found)
                            found = [s for s in found
                                     if tier_data.get(s['stem'], {}).get('weight', 0) > 0]
                            n_dropped = before_tier - len(found)
                            if n_dropped:
                                print(f"    tier gating: dropped {n_dropped} songs (rejected/untrusted)")

                    for s in found:
                        # Per-song weight = dataset_weight × tier_score
                        if tier_data and s['stem'] in tier_data:
                            s['weight'] = tier_data[s['stem']]['weight']
                        else:
                            s['weight'] = w
                        s['provenance'] = name
                        s['_dataset'] = name
                    all_songs.extend(found)
                else:
                    print(f"  {name}: skipped (dirs not found)")
    else:
        # CLI mode
        if not args.isophonics:
            print("ERROR: provide --config or at least one --isophonics")
            sys.exit(1)
        audio_dirs = args.audio_dir or []
        if not audio_dirs:
            print("ERROR: --audio-dir required with --isophonics")
            sys.exit(1)
        # If fewer audio-dirs than isophonics, repeat the last one
        while len(audio_dirs) < len(args.isophonics):
            audio_dirs.append(audio_dirs[-1])

        weights = args.weight or []
        while len(weights) < len(args.isophonics):
            weights.append(1.0)

        for iso_root, aud_dir, w in zip(args.isophonics, audio_dirs, weights):
            found = find_isophonics_songs(iso_root, aud_dir)
            for s in found:
                s['weight'] = w
                s['provenance'] = 'gold' if s['beats'] else 'silver'
            all_songs.extend(found)

    if not all_songs:
        print("No matching audio/annotation pairs found!")
        sys.exit(1)

    n_gold = sum(1 for s in all_songs if s['provenance'] == 'gold')
    n_silver = len(all_songs) - n_gold
    print(f"Found {len(all_songs)} songs (gold={n_gold}, silver={n_silver})")

    # ── Hard reject: permanently exclude blacklisted songs ──
    if args.config:
        hard_reject_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), 'hard_reject.json')
        if os.path.exists(hard_reject_path):
            with open(hard_reject_path) as _hrf:
                hr_data = _json.load(_hrf)
            reject_stems = set(hr_data.get('reject', {}).keys())
            exclude_stems = set(hr_data.get('exclude_from_beat_training', {}).keys())
            blacklist = reject_stems | exclude_stems
            if blacklist:
                before_hr = len(all_songs)
                all_songs = [s for s in all_songs if s['stem'] not in blacklist]
                n_hr = before_hr - len(all_songs)
                if n_hr:
                    print(f"  hard reject: dropped {n_hr} songs ({', '.join(sorted(blacklist))})")

    # ── Beat cap: limit capped datasets so they don't exceed gold beat count ──
    if args.config:
        # Estimate gold beats (assume ~350 beats/song average for gold)
        # We'll enforce the cap after extraction by truncating if needed
        gold_songs = [s for s in all_songs if s['provenance'] == 'gold']
        for name, ds in cfg['datasets'].items():
            max_beats = ds.get('max_beats')
            if max_beats == 'gold':
                capped_songs = [s for s in all_songs if s.get('_dataset') == name]
                if capped_songs and gold_songs:
                    # We can't know exact beat counts before extraction, so cap by song count
                    # proportional to gold. Gold averages ~370 beats/song across Beatles+Queen.
                    # Billboard averages ~440 beats/song. So cap billboard songs at:
                    #   n_billboard <= n_gold * (370/440) ≈ n_gold * 0.84
                    max_songs = max(1, int(len(gold_songs) * 0.84))
                    if len(capped_songs) > max_songs:
                        rng = np.random.RandomState(42)
                        # Prefer higher-weight (higher-quality) songs
                        capped_songs.sort(key=lambda s: -s['weight'])
                        keep = set(id(s) for s in capped_songs[:max_songs])
                        before_cap = len(all_songs)
                        all_songs = [s for s in all_songs
                                     if s.get('_dataset') != name or id(s) in keep]
                        n_dropped = before_cap - len(all_songs)
                        print(f"  beat cap ({name}): kept {max_songs}/{max_songs + n_dropped} songs (matching gold count)")

    # Process each song
    all_X12, all_X36 = [], []
    all_roots, all_quals, all_tier1 = [], [], []
    all_weights, all_song_ids = [], []
    all_beat_times, all_filenames, all_key_indices = [], [], []
    all_provenance = []

    from v2.chord_schema import NOTES as V2_NOTES

    for song_idx, song in enumerate(all_songs):
        print(f"  [{song_idx+1}/{len(all_songs)}] {song['stem']}...", end=' ', flush=True)
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

            result = process_song_v2(
                song['audio'], chord_annotations, beat_times, key_idx,
                context_radius=args.context_radius,
                use_delta=args.delta, use_bass=args.bass)
            X12, X_ctx, roots, quals, tier1, bt, ki = result

            if X12 is None or len(X12) == 0:
                print("SKIP (no beats)")
                continue

            n = len(X12)
            all_X12.append(X12)
            all_X36.append(X_ctx)
            all_roots.append(roots)
            all_quals.append(quals)
            all_tier1.append(tier1)
            all_weights.append(np.full(n, song['weight'], dtype=np.float32))
            all_song_ids.append(np.full(n, song_idx, dtype=np.int32))
            all_beat_times.append(np.array(bt, dtype=np.float32))
            all_filenames.append(song['stem'])
            all_key_indices.append(ki)
            all_provenance.append(song['provenance'])

            key_note = V2_NOTES[ki] if ki < len(V2_NOTES) else '?'
            print(f"OK ({n} beats, key={key_note}[{key_src}], beats={beat_src}, w={song['weight']})")

        except Exception as e:
            print(f"ERROR: {e}")

    if not all_X12:
        print("\nNo features extracted!")
        sys.exit(1)

    # Concatenate
    X_12 = np.concatenate(all_X12, axis=0)
    X_ctx = np.concatenate(all_X36, axis=0)
    ctx_dim = X_ctx.shape[1]
    ctx_key = f'X_{ctx_dim}'
    root_labels = np.concatenate(all_roots, axis=0)
    quality_labels = np.concatenate(all_quals, axis=0)
    tier1_labels = np.concatenate(all_tier1, axis=0)
    sample_weights = np.concatenate(all_weights, axis=0)
    song_ids = np.concatenate(all_song_ids, axis=0)

    # Compute third-interval ratio features from raw HPCP
    X_third_ratio = compute_third_ratio(X_12)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    save_dict = dict(
        X_12=X_12,
        X_third_ratio=X_third_ratio,
        root_labels=root_labels,
        quality_labels=quality_labels,
        tier1_labels=tier1_labels,
        sample_weights=sample_weights,
        song_ids=song_ids,
        beat_times=np.array(all_beat_times, dtype=object),
        filenames=np.array(all_filenames, dtype=object),
        key_indices=np.array(all_key_indices, dtype=np.int32),
        provenance=np.array(all_provenance, dtype=object),
    )
    save_dict[ctx_key] = X_ctx
    # Store feature composition for train.py → model → server
    import json as _json2
    feature_config = {
        'context_radius': args.context_radius,
        'has_delta': args.delta,
        'has_bass': args.bass,
    }
    save_dict['feature_config'] = np.array(_json2.dumps(feature_config))
    np.savez_compressed(args.out, **save_dict)

    n_beats = len(X_12)
    print(f"\nSaved {n_beats} beat-level features to {args.out}")
    print(f"Songs: {len(all_filenames)}")
    print(f"  Feature dims: X_12={X_12.shape}, {ctx_key}={X_ctx.shape}")
    feat_parts = [f"ctx_{args.context_radius}={2*args.context_radius+1}×12"]
    if args.delta:
        feat_parts.append("delta=12")
    if args.bass:
        feat_parts.append("bass=12")
    print(f"  Feature composition: {' + '.join(feat_parts)} = {ctx_dim}")
    print(f"  Gold: {n_gold} songs, Silver: {n_silver} songs")

    # Root distribution
    print("\nRoot distribution:")
    for ri in range(len(ROOT_VOCAB)):
        cnt = np.sum(root_labels == ri)
        if cnt > 0:
            pct = 100.0 * cnt / n_beats
            print(f"  {ROOT_VOCAB[ri]:3s}: {cnt:6d} ({pct:5.1f}%)")

    # Quality distribution
    print("\nQuality distribution:")
    for qi in range(len(QUALITY_VOCAB)):
        cnt = np.sum(quality_labels == qi)
        if cnt > 0:
            pct = 100.0 * cnt / n_beats
            print(f"  {QUALITY_VOCAB[qi]:5s}: {cnt:6d} ({pct:5.1f}%)")


if __name__ == '__main__':
    main()
