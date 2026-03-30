#!/usr/bin/env python3
"""Score silver (billboard) songs on data quality for training weight assignment.

Two-stage pipeline:
  Stage A — Hard gates:  binary reject on obviously broken songs
  Stage B — Weighted scoring:  7-dimension quality score for survivors

Hard gates (reject if ANY triggers):
  - Duration mismatch > 20%
  - Beat density < 0.5 or > 5.0 beats/sec
  - Beat interval CV > 0.45
  - Silence ratio > 60%
  - Label coverage < 40%

Scoring weights:
  Duration agreement   30%    Beat regularity   25%
  Annotation quality   20%    Model-agreement   10%
  Spectral quality      5%    Key plausibility   5%
  Chord diversity       5%

Output: data/billboard/silver_scores.json

Usage:
    python -m tools.score_silver --audio-dir data/billboard/audio --lab-dir data/billboard/labels
    python -m tools.score_silver --audio-dir data/billboard/audio --lab-dir data/billboard/labels \
        --model models/chord_model_v2.pkl --data data/features_v2.npz
"""
import sys
import os
import argparse
import json
import numpy as np

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from v1.prepare_data import (
    extract_hpcp, detect_beats, detect_key,
    sync_to_beats, parse_lab_file, find_flat_pairs,
    SR, HOP_SIZE,
)

# ── Constants ────────────────────────────────────────────────

WEIGHTS = {
    'duration':   0.30,
    'beats':      0.25,
    'annotation': 0.20,
    'model':      0.10,
    'spectral':   0.05,
    'key':        0.05,
    'diversity':  0.05,
}

# Hard gate thresholds
GATE_DURATION_MISMATCH = 0.20
GATE_BEAT_DENSITY_LO = 0.5
GATE_BEAT_DENSITY_HI = 5.0
GATE_BEAT_CV = 0.45
GATE_SILENCE_RATIO = 0.60
GATE_LABEL_COVERAGE = 0.40

# Circle of fifths for key plausibility
_COF_ORDER = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F']
_NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _cof_distance(note_a, note_b):
    """Circle-of-fifths distance between two note names (0–6)."""
    if note_a not in _COF_ORDER or note_b not in _COF_ORDER:
        return 6  # worst case
    ia = _COF_ORDER.index(note_a)
    ib = _COF_ORDER.index(note_b)
    d = abs(ia - ib)
    return min(d, 12 - d)


# ── Metric computation ───────────────────────────────────────

def compute_raw_metrics(audio_path, lab_path, beat_times, audio_dur,
                        hpcps, hpcps_native, flatness_weights, key_idx):
    """Compute all raw metrics needed for gates and scoring."""
    annotations = parse_lab_file(lab_path)
    if not annotations:
        return None

    # Duration metrics
    label_end = annotations[-1][1] if annotations else 0.0
    mismatch_pct = abs(audio_dur - label_end) / max(audio_dur, label_end, 1e-6)

    # Beat metrics
    n_beats = len(beat_times)
    beat_density = n_beats / max(audio_dur, 1e-6)
    if n_beats > 1:
        intervals = np.diff(beat_times)
        beat_cv = float(np.std(intervals) / max(np.mean(intervals), 1e-6))
    else:
        beat_cv = 1.0

    # Annotation metrics
    total_silence = sum(end - start for start, end, lbl in annotations
                        if lbl.strip() in ('N', 'X', 'silence'))
    label_span = label_end - annotations[0][0] if annotations else 0.0
    silence_ratio = total_silence / max(label_span, 1e-6)
    label_coverage = label_end / max(audio_dur, 1e-6)

    # Segment metrics
    n_segments = len(annotations)
    avg_segment_dur = label_span / max(n_segments, 1) if label_span > 0 else 0.0
    seg_density = n_segments / max(label_span, 1e-6)

    # Consecutive identical labels span
    max_consec_span = 0.0
    if len(annotations) > 1:
        run_start = 0
        for i in range(1, len(annotations)):
            if annotations[i][2] != annotations[run_start][2]:
                span = annotations[i-1][1] - annotations[run_start][0]
                max_consec_span = max(max_consec_span, span)
                run_start = i
        # final run
        span = annotations[-1][1] - annotations[run_start][0]
        max_consec_span = max(max_consec_span, span)
    consec_ratio = max_consec_span / max(label_span, 1e-6)

    # Chord diversity
    non_silence = [(s, e, l) for s, e, l in annotations if l.strip() not in ('N', 'X', 'silence')]
    unique_labels = set(l for _, _, l in non_silence)
    unique_roots = set()
    for lbl in unique_labels:
        root = lbl.split(':')[0].split('/')[0] if ':' in lbl else lbl.rstrip('m7').rstrip('m').rstrip('7')
        if root and root[0].isupper():
            unique_roots.add(root)
    n_unique_chords = len(unique_labels)
    n_unique_roots = len(unique_roots)

    # Spectral quality (flatness)
    if flatness_weights is not None and len(flatness_weights) > 0:
        # flatness_weights are already 1.0 - flatness*2, clamped at 0.2
        # So "noisy" frames have low weight. Reverse: noisy = weight < 0.5
        noisy_frac = float(np.mean(flatness_weights < 0.5))
    else:
        noisy_frac = 0.0

    # Key plausibility
    detected_key_note = _NOTES[key_idx % 12] if key_idx is not None else None
    # Find dominant root in labels
    root_counts = {}
    for _, _, lbl in non_silence:
        root = lbl.split(':')[0] if ':' in lbl else lbl
        # Clean root
        for n in _NOTES:
            if root.startswith(n):
                root_counts[n] = root_counts.get(n, 0) + 1
                break
    dominant_root = max(root_counts, key=root_counts.get) if root_counts else None

    return {
        'audio_dur': round(audio_dur, 2),
        'label_end': round(label_end, 2),
        'mismatch_pct': round(mismatch_pct, 4),
        'n_beats': n_beats,
        'beat_density': round(beat_density, 3),
        'beat_cv': round(beat_cv, 4),
        'silence_ratio': round(silence_ratio, 4),
        'label_coverage': round(label_coverage, 4),
        'n_segments': n_segments,
        'avg_segment_dur': round(avg_segment_dur, 3),
        'seg_density': round(seg_density, 3),
        'consec_ratio': round(consec_ratio, 4),
        'n_unique_chords': n_unique_chords,
        'n_unique_roots': n_unique_roots,
        'noisy_frac': round(noisy_frac, 4),
        'detected_key': detected_key_note,
        'dominant_root': dominant_root,
    }


# ── Hard gates ───────────────────────────────────────────────

def check_hard_gates(metrics):
    """Return (pass, fail_reason) tuple."""
    if metrics['mismatch_pct'] > GATE_DURATION_MISMATCH:
        return False, 'duration_mismatch'
    if metrics['beat_density'] < GATE_BEAT_DENSITY_LO:
        return False, 'beat_density_low'
    if metrics['beat_density'] > GATE_BEAT_DENSITY_HI:
        return False, 'beat_density_high'
    if metrics['beat_cv'] > GATE_BEAT_CV:
        return False, 'beat_cv'
    if metrics['silence_ratio'] > GATE_SILENCE_RATIO:
        return False, 'silence_ratio'
    if metrics['label_coverage'] < GATE_LABEL_COVERAGE:
        return False, 'label_coverage'
    return True, None


# ── Sub-scores ───────────────────────────────────────────────

def score_duration(metrics):
    """Duration agreement: linear decay from 0% to 20% mismatch."""
    return 1.0 - _clamp(metrics['mismatch_pct'] / GATE_DURATION_MISMATCH)


def score_beats(metrics):
    """Beat regularity: CV penalty + density edge penalty."""
    cv_score = 1.0 - _clamp(metrics['beat_cv'] / GATE_BEAT_CV)

    # Mild penalty for density at edges of [0.5, 5.0]
    d = metrics['beat_density']
    if d < 1.0:
        edge_penalty = 0.15 * (1.0 - (d - 0.5) / 0.5)  # 0.15 at 0.5, 0 at 1.0
    elif d > 3.5:
        edge_penalty = 0.15 * ((d - 3.5) / 1.5)  # 0 at 3.5, 0.15 at 5.0
    else:
        edge_penalty = 0.0

    return _clamp(cv_score - edge_penalty)


def score_annotation(metrics):
    """Annotation quality: silence + micro-segment + coverage + consec check."""
    # Silence ratio penalty
    silence_score = 1.0 - _clamp(metrics['silence_ratio'] / GATE_SILENCE_RATIO)

    # Micro-segment penalty (avg < 0.5s or density > 1.5/sec)
    micro_penalty = 0.0
    if metrics['avg_segment_dur'] < 0.5 and metrics['n_segments'] > 5:
        micro_penalty = 0.3
    elif metrics['seg_density'] > 1.5:
        micro_penalty = 0.2

    # Label coverage
    coverage_score = _clamp(metrics['label_coverage'])

    # Consecutive identical label penalty
    consec_penalty = 0.0
    if metrics['consec_ratio'] > 0.30:
        consec_penalty = 0.2 * _clamp((metrics['consec_ratio'] - 0.30) / 0.30)

    return _clamp((silence_score * 0.4 + coverage_score * 0.4 + 0.2) - micro_penalty - consec_penalty)


def score_spectral(metrics):
    """Spectral quality: fraction of noisy frames."""
    return 1.0 - _clamp(metrics['noisy_frac'] / 0.5)


def score_key(metrics):
    """Key plausibility: circle-of-fifths distance between detected key and dominant root."""
    if metrics['detected_key'] is None or metrics['dominant_root'] is None:
        return 0.5  # unknown = neutral
    dist = _cof_distance(metrics['detected_key'], metrics['dominant_root'])
    if dist <= 1:
        return 1.0  # tonic or dominant — perfectly normal
    return _clamp(1.0 - (dist - 1) / 5.0)  # linear decay from dist=1 to dist=6


def score_diversity(metrics):
    """Chord diversity: unique chords and roots."""
    chord_score = _clamp(metrics['n_unique_chords'] / 15.0)
    root_score = _clamp(metrics['n_unique_roots'] / 8.0)
    return (chord_score + root_score) / 2.0


def score_model_agreement(stem, model_data):
    """Model-agreement: compare hybrid predictions vs labels at beat level.

    model_data is a dict mapping stem -> {agreement_ratio, flip_ratio_excess}.
    Returns sub-score or None if unavailable.
    """
    if model_data is None or stem not in model_data:
        return None
    d = model_data[stem]
    agreement = d['agreement_ratio']
    flip_excess = d.get('flip_ratio_excess', 0.0)
    return _clamp(agreement * 0.7 + (1.0 - _clamp(flip_excess)) * 0.3)


# ── Model agreement pre-computation ─────────────────────────

def precompute_model_agreement(model_path, data_path):
    """Run hybrid decode on all songs and compare predictions vs labels.

    Returns dict mapping song stem -> {agreement_ratio, flip_ratio_excess}.
    """
    import pickle
    from v2.decode import decode_hybrid, smooth_isolated

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data = np.load(data_path, allow_pickle=True)
    feat_dim = model.get('feature_dim', 12)
    X = data[f'X_{feat_dim}'].astype(np.float32) if f'X_{feat_dim}' in data else data['X_12'].astype(np.float32)
    tier1_labels = data['tier1_labels'].astype(np.int32)
    song_ids = data['song_ids'].astype(np.int32)
    key_indices = data['key_indices']
    filenames = list(data['filenames'])
    provenance = list(data['provenance'])

    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    results = {}
    unique_songs = np.unique(song_ids)

    for song_id in unique_songs:
        prov = provenance[song_id] if song_id < len(provenance) else 'unknown'
        if prov == 'gold':
            continue  # skip gold songs

        mask = song_ids == song_id
        X_song = X_norm[mask]
        y_song = tier1_labels[mask]
        n_beats = len(X_song)
        if n_beats == 0:
            continue

        ki = key_indices[song_id] if song_id < len(key_indices) else 0
        stem = filenames[song_id] if song_id < len(filenames) else f"song_{song_id}"

        path_h, _, _ = decode_hybrid(model, X_song, ki)
        path_h = smooth_isolated(path_h)

        # Root agreement (ignore quality — just check root match)
        def _to_root(p):
            if p == 0:
                return 0
            return ((p - 1) % 12 + 1) if 1 <= p <= 24 else 0

        pred_roots = np.array([_to_root(p) for p in path_h])
        true_roots = np.array([_to_root(y) for y in y_song])
        agreement_ratio = float(np.mean(pred_roots == true_roots))

        # Flip rate comparison
        pred_flips = sum(1 for i in range(1, n_beats) if path_h[i] != path_h[i-1])
        true_flips = sum(1 for i in range(1, n_beats) if y_song[i] != y_song[i-1])
        pred_flip_rate = pred_flips / max(n_beats - 1, 1)
        true_flip_rate = true_flips / max(n_beats - 1, 1)
        flip_ratio = pred_flip_rate / max(true_flip_rate, 1e-6)
        flip_ratio_excess = max(0.0, (flip_ratio - 3.0) / 3.0)  # excess beyond 3x

        results[stem] = {
            'agreement_ratio': round(agreement_ratio, 4),
            'flip_ratio': round(flip_ratio, 3),
            'flip_ratio_excess': round(flip_ratio_excess, 4),
        }

    return results


# ── Main scoring ─────────────────────────────────────────────

def score_song(audio_path, lab_path, stem, model_data=None, fast=False):
    """Score a single song. Returns dict with score, tier, sub_scores, metrics.

    If fast=True, skip HPCP extraction (spectral/key = neutral 0.7). ~5x faster.
    """
    from essentia.standard import MonoLoader, RhythmExtractor2013

    # Load audio ONCE
    audio = MonoLoader(filename=audio_path, sampleRate=SR)()
    audio_dur = len(audio) / SR

    # Detect beats from the already-loaded audio
    rhythm = RhythmExtractor2013(method='multifeature')
    _, beats_arr, _, _, _ = rhythm(audio)
    beat_times = beats_arr.tolist()
    if len(beat_times) > 1:
        filtered = [beat_times[0]]
        for b in beat_times[1:]:
            if b - filtered[-1] >= 0.15:
                filtered.append(b)
        beat_times = filtered

    if fast:
        hpcps, hpcps_native, flatness_weights = None, None, None
        key_idx = None
    else:
        hpcps, hpcps_native, flatness_weights = extract_hpcp(audio_path)
        key_idx = detect_key(hpcps_native)

    # Compute all raw metrics
    metrics = compute_raw_metrics(
        audio_path, lab_path, beat_times, audio_dur,
        hpcps, hpcps_native, flatness_weights, key_idx)

    if metrics is None:
        return {
            'score': 0.0,
            'tier': 'rejected',
            'hard_gate_fail': 'no_annotations',
            'sub_scores': {},
            'metrics': {},
        }

    # Stage A: Hard gates
    passed, fail_reason = check_hard_gates(metrics)
    if not passed:
        return {
            'score': 0.0,
            'tier': 'rejected',
            'hard_gate_fail': fail_reason,
            'sub_scores': {},
            'metrics': metrics,
        }

    # Stage B: Weighted scoring
    sub_scores = {
        'duration':   round(score_duration(metrics), 4),
        'beats':      round(score_beats(metrics), 4),
        'annotation': round(score_annotation(metrics), 4),
        'diversity':  round(score_diversity(metrics), 4),
    }

    if not fast:
        sub_scores['spectral'] = round(score_spectral(metrics), 4)
        sub_scores['key'] = round(score_key(metrics), 4)

    # Model agreement (optional)
    model_score = score_model_agreement(stem, model_data)
    if model_score is not None:
        sub_scores['model'] = round(model_score, 4)

    # Compute composite score with weight normalization
    active_weights = {k: WEIGHTS[k] for k in sub_scores}
    w_total = sum(active_weights.values())
    composite = sum(active_weights[k] * sub_scores[k] for k in sub_scores) / max(w_total, 1e-6)
    composite = round(composite, 4)

    # Assign tier
    if composite >= 0.7:
        tier = 'trusted'
    elif composite >= 0.4:
        tier = 'marginal'
    else:
        tier = 'untrusted'

    return {
        'score': composite,
        'tier': tier,
        'hard_gate_fail': None,
        'sub_scores': sub_scores,
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Score silver data quality')
    parser.add_argument('--audio-dir', required=True, help='Billboard audio directory')
    parser.add_argument('--lab-dir', required=True, help='Billboard labels directory')
    parser.add_argument('--subset', type=str, default=None,
                        help='Path to training_subset.json (score only subset)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model .pkl for model-agreement scoring')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to features .npz for model-agreement scoring')
    parser.add_argument('--out', default='data/billboard/silver_scores.json',
                        help='Output JSON path')
    parser.add_argument('--fast', action='store_true',
                        help='Skip HPCP extraction (spectral/key get neutral scores). ~5x faster.')
    args = parser.parse_args()

    # Discover songs
    pairs = find_flat_pairs(args.audio_dir, args.lab_dir)
    print(f"Found {len(pairs)} audio/label pairs")

    # Optionally filter to subset
    if args.subset:
        with open(args.subset) as f:
            subset_data = json.load(f)
        allowed = set(subset_data.get('track_ids', subset_data))
        pairs = [p for p in pairs if p['stem'] in allowed]
        print(f"Filtered to {len(pairs)} songs via subset")

    # Pre-compute model agreement if model provided
    model_data = None
    if args.model and args.data:
        if os.path.exists(args.model) and os.path.exists(args.data):
            print(f"\nPre-computing model agreement...")
            model_data = precompute_model_agreement(args.model, args.data)
            print(f"  Got agreement data for {len(model_data)} silver songs")
        else:
            print(f"  WARNING: model or data file not found, skipping model-agreement")

    # Score each song
    results = {}
    tier_counts = {'rejected': 0, 'untrusted': 0, 'marginal': 0, 'trusted': 0}

    for i, song in enumerate(pairs):
        stem = song['stem']
        print(f"  [{i+1}/{len(pairs)}] {stem}...", end=' ', flush=True)
        try:
            result = score_song(song['audio'], song['chords'], stem, model_data, fast=args.fast)
            results[stem] = result
            tier_counts[result['tier']] += 1
            if result['hard_gate_fail']:
                print(f"REJECTED ({result['hard_gate_fail']})")
            else:
                print(f"{result['tier']} (score={result['score']:.3f})")
        except Exception as e:
            print(f"ERROR: {e}")
            results[stem] = {
                'score': 0.0,
                'tier': 'rejected',
                'hard_gate_fail': f'error: {str(e)[:100]}',
                'sub_scores': {},
                'metrics': {},
            }
            tier_counts['rejected'] += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"Silver scoring summary ({len(results)} songs):")
    print(f"  Trusted:   {tier_counts['trusted']}")
    print(f"  Marginal:  {tier_counts['marginal']}")
    print(f"  Untrusted: {tier_counts['untrusted']}")
    print(f"  Rejected:  {tier_counts['rejected']}")

    scored = [r for r in results.values() if r['tier'] not in ('rejected',)]
    if scored:
        scores = [r['score'] for r in scored]
        print(f"\nScore distribution (non-rejected):")
        print(f"  Min:    {min(scores):.3f}")
        print(f"  Median: {np.median(scores):.3f}")
        print(f"  Mean:   {np.mean(scores):.3f}")
        print(f"  Max:    {max(scores):.3f}")

    # Build silver_tiers.json alongside
    tiers = {}
    for stem, r in results.items():
        if r['tier'] in ('trusted', 'marginal'):
            weight = 0.8 * r['score']
        else:
            weight = 0.0
        tiers[stem] = {
            'tier': r['tier'],
            'score': r['score'],
            'weight': round(weight, 4),
        }

    # Save
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved to {args.out}")

    tiers_path = os.path.join(os.path.dirname(args.out), 'silver_tiers.json')
    with open(tiers_path, 'w') as f:
        json.dump(tiers, f, indent=2)
    print(f"Tiers saved to {tiers_path}")


if __name__ == '__main__':
    main()
