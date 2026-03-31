#!/usr/bin/env python3
"""Evaluate beat detection: Beat This! vs Essentia vs gold beat annotations.

Compares beat trackers against gold beat annotations from Isophonics (Beatles/Queen).
Reports F1 at 70ms tolerance (standard MIR beat evaluation) and optionally measures
downstream chord accuracy with each beat tracker.

Usage:
    python -m v2.eval_beats --config data/weights.json --datasets beatles
    python -m v2.eval_beats --config data/weights.json --datasets beatles --chord-eval \
        --checkpoint-btc ../server/btc_model/btc_model_best.pth
"""
import sys
import os
import argparse
import time
import json
import numpy as np

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SERVER_ROOT = os.path.join(os.path.dirname(_TRAINING_ROOT), 'server')
sys.path.insert(0, _TRAINING_ROOT)
sys.path.insert(0, _SERVER_ROOT)

from v1.prepare_data import (
    parse_beat_file, find_isophonics_songs, detect_beats, _detect_beats_beat_this,
)


def beat_f1(ref_beats, est_beats, tolerance=0.070):
    """Compute F1 score for beat tracking at given tolerance (seconds).

    Standard MIR evaluation: a predicted beat is a true positive if there is a
    reference beat within `tolerance` seconds. Each reference beat can be matched
    at most once.
    """
    ref = np.array(ref_beats, dtype=np.float64)
    est = np.array(est_beats, dtype=np.float64)

    if len(ref) == 0 and len(est) == 0:
        return 1.0, 1.0, 1.0
    if len(ref) == 0:
        return 0.0, 0.0, 0.0
    if len(est) == 0:
        return 0.0, 0.0, 0.0

    matched_ref = set()
    tp = 0

    for e in est:
        dists = np.abs(ref - e)
        closest_idx = np.argmin(dists)
        if dists[closest_idx] <= tolerance and closest_idx not in matched_ref:
            tp += 1
            matched_ref.add(closest_idx)

    precision = tp / len(est) if len(est) > 0 else 0.0
    recall = tp / len(ref) if len(ref) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall


def median_offset(ref_beats, est_beats, tolerance=0.070):
    """Compute median signed offset of matched beats (est - ref)."""
    ref = np.array(ref_beats, dtype=np.float64)
    est = np.array(est_beats, dtype=np.float64)
    offsets = []

    matched_ref = set()
    for e in est:
        dists = np.abs(ref - e)
        closest_idx = np.argmin(dists)
        if dists[closest_idx] <= tolerance and closest_idx not in matched_ref:
            offsets.append(e - ref[closest_idx])
            matched_ref.add(closest_idx)

    return float(np.median(offsets)) if offsets else 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate beat detection: Beat This! vs Essentia vs gold')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to weights.json dataset config')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated dataset names (default: all with gold beats)')
    parser.add_argument('--max-songs', type=int, default=None,
                        help='Limit number of songs')
    parser.add_argument('--tolerance', type=float, default=0.070,
                        help='Beat matching tolerance in seconds (default: 0.070)')
    parser.add_argument('--chord-eval', action='store_true',
                        help='Also evaluate downstream chord accuracy with each beat tracker')
    parser.add_argument('--checkpoint-btc', type=str, default=None,
                        help='BTC checkpoint for chord-eval (requires --chord-eval)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    args = parser.parse_args()

    # Discover songs with gold beats
    base_dir = _TRAINING_ROOT
    with open(args.config) as f:
        cfg = json.load(f)

    filter_datasets = args.datasets.split(',') if args.datasets else None
    all_songs = []
    for name, ds in cfg['datasets'].items():
        if filter_datasets and name not in filter_datasets:
            continue
        if ds['type'] == 'isophonics':
            ann_root = os.path.join(base_dir, ds['annotations'])
            aud_dir = os.path.join(base_dir, ds['audio_dir'])
            found = find_isophonics_songs(ann_root, aud_dir)
            gold_beat_songs = [s for s in found if s.get('beats')]
            for s in gold_beat_songs:
                s['dataset'] = name
            print(f"  {name}: {len(gold_beat_songs)} songs with gold beats")
            all_songs.extend(gold_beat_songs)

    if not all_songs:
        print("No songs with gold beat annotations found.")
        sys.exit(1)

    if args.max_songs and len(all_songs) > args.max_songs:
        all_songs = all_songs[:args.max_songs]

    # Optional: load BTC for chord evaluation
    btc_model, btc_mean, btc_std = None, 0, 1
    if args.chord_eval and args.checkpoint_btc:
        import torch
        from v2.eval_btc_vs_freeze5 import (
            load_btc_model, extract_cqt, run_btc_inference, btc_beat_sync,
            get_ground_truth_at_beats, compute_metrics,
        )
        from v1.prepare_data import parse_lab_file
        device = torch.device('cpu') if args.cpu else (
            torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
        btc_model, btc_mean, btc_std = load_btc_model(args.checkpoint_btc, device)
        print(f"  BTC loaded for chord eval (device={device})")

    print(f"\nEvaluating {len(all_songs)} songs (tolerance={args.tolerance*1000:.0f}ms)")
    print(f"{'='*80}")

    essentia_f1s = []
    beat_this_f1s = []
    essentia_offsets = []
    beat_this_offsets = []

    # For chord eval
    chord_gold_beats_results = []
    chord_essentia_beats_results = []
    chord_beat_this_beats_results = []

    for i, song in enumerate(all_songs):
        ds = song.get('dataset', '?')
        print(f"  [{i+1}/{len(all_songs)}] [{ds}] {song['stem']}...", end=' ', flush=True)

        gold_beats = parse_beat_file(song['beats'])
        if len(gold_beats) < 2:
            print("SKIP (too few gold beats)")
            continue

        t0 = time.time()

        # Essentia
        essentia_beats = detect_beats(song['audio'], use_beat_this=False)
        e_f1, e_p, e_r = beat_f1(gold_beats, essentia_beats, args.tolerance)
        e_off = median_offset(gold_beats, essentia_beats, args.tolerance)
        essentia_f1s.append(e_f1)
        essentia_offsets.append(e_off)

        # Beat This!
        try:
            bt_beats = _detect_beats_beat_this(song['audio'])
            bt_f1, bt_p, bt_r = beat_f1(gold_beats, bt_beats, args.tolerance)
            bt_off = median_offset(gold_beats, bt_beats, args.tolerance)
            beat_this_f1s.append(bt_f1)
            beat_this_offsets.append(bt_off)
        except Exception as e:
            bt_f1, bt_beats = None, None
            print(f"beat_this error: {e}", end=' ')

        dt = time.time() - t0
        parts = [f"Ess={e_f1:.3f}"]
        if bt_f1 is not None:
            parts.append(f"BT={bt_f1:.3f}")
        print(f"{' | '.join(parts)} ({dt:.1f}s)")

        # Downstream chord eval
        if args.chord_eval and btc_model is not None:
            try:
                chord_annotations = parse_lab_file(song['chords'])
                cqt = extract_cqt(song['audio'])
                frame_preds = run_btc_inference(btc_model, cqt, btc_mean, btc_std, device)

                gt_gold = get_ground_truth_at_beats(chord_annotations, gold_beats)
                btc_gold = btc_beat_sync(frame_preds, gold_beats)
                n = len(gt_gold)
                btc_gold = (btc_gold[:n] + ['N'] * max(0, n - len(btc_gold)))[:n]
                chord_gold_beats_results.append(compute_metrics(gt_gold, btc_gold))

                gt_ess = get_ground_truth_at_beats(chord_annotations, essentia_beats)
                btc_ess = btc_beat_sync(frame_preds, essentia_beats)
                n = len(gt_ess)
                btc_ess = (btc_ess[:n] + ['N'] * max(0, n - len(btc_ess)))[:n]
                chord_essentia_beats_results.append(compute_metrics(gt_ess, btc_ess))

                if bt_beats is not None:
                    gt_bt = get_ground_truth_at_beats(chord_annotations, bt_beats)
                    btc_bt = btc_beat_sync(frame_preds, bt_beats)
                    n = len(gt_bt)
                    btc_bt = (btc_bt[:n] + ['N'] * max(0, n - len(btc_bt)))[:n]
                    chord_beat_this_beats_results.append(compute_metrics(gt_bt, btc_bt))
            except Exception:
                pass

    # ── Aggregate beat metrics ──
    print(f"\n{'='*80}")
    print(f"BEAT DETECTION RESULTS ({len(all_songs)} songs, tolerance={args.tolerance*1000:.0f}ms)")
    print(f"{'='*80}")

    if essentia_f1s:
        print(f"\n  Essentia RhythmExtractor2013:")
        print(f"    F1:  mean={np.mean(essentia_f1s):.3f}  median={np.median(essentia_f1s):.3f}  "
              f"min={np.min(essentia_f1s):.3f}  max={np.max(essentia_f1s):.3f}")
        print(f"    Median offset: {np.median(essentia_offsets)*1000:+.1f}ms")
        print(f"    Songs with F1 >= 0.9: {sum(1 for f in essentia_f1s if f >= 0.9)}/{len(essentia_f1s)}")

    if beat_this_f1s:
        print(f"\n  Beat This! (small0):")
        print(f"    F1:  mean={np.mean(beat_this_f1s):.3f}  median={np.median(beat_this_f1s):.3f}  "
              f"min={np.min(beat_this_f1s):.3f}  max={np.max(beat_this_f1s):.3f}")
        print(f"    Median offset: {np.median(beat_this_offsets)*1000:+.1f}ms")
        print(f"    Songs with F1 >= 0.9: {sum(1 for f in beat_this_f1s if f >= 0.9)}/{len(beat_this_f1s)}")

    if essentia_f1s and beat_this_f1s and len(essentia_f1s) == len(beat_this_f1s):
        diffs = np.array(beat_this_f1s) - np.array(essentia_f1s)
        bt_wins = np.sum(diffs > 0.01)
        ess_wins = np.sum(diffs < -0.01)
        ties = len(diffs) - bt_wins - ess_wins
        print(f"\n  Head-to-head:")
        print(f"    Beat This! wins: {bt_wins}, Essentia wins: {ess_wins}, Ties: {ties}")
        print(f"    Avg F1 diff (BT - Ess): {np.mean(diffs):+.3f}")

    # ── Downstream chord results ──
    if args.chord_eval and chord_gold_beats_results:
        print(f"\n{'='*80}")
        print(f"DOWNSTREAM CHORD ACCURACY (BTC model, different beat sources)")
        print(f"{'='*80}")

        for label, results in [
            ('Gold beats', chord_gold_beats_results),
            ('Essentia beats', chord_essentia_beats_results),
            ('Beat This! beats', chord_beat_this_beats_results),
        ]:
            if not results:
                continue
            total = sum(r['n_beats'] for r in results)
            wcsr = sum(r['wcsr'] * r['n_beats'] for r in results) / max(total, 1)
            print(f"  {label:20s}: WCSR={wcsr:.3f} ({len(results)} songs, {total} beats)")


if __name__ == '__main__':
    main()
