#!/usr/bin/env python3
"""Evaluate BTC chord accuracy using the production inference pipeline.

No Essentia dependency -- uses librosa for CQT, Beat This! for beats.
Tests against Isophonics ground-truth chord annotations (Beatles/Queen).

Metrics:
  WCSR  - Weighted Chord Symbol Recall (exact match at Tier 1 level)
  Root  - Root note accuracy (ignoring quality)
  Maj   - Major chord accuracy
  Min   - Minor chord accuracy

Usage:
    python -m v2.eval_btc_production --max-songs 20
    python -m v2.eval_btc_production --per-song
    python -m v2.eval_btc_production --datasets beatles --per-song
"""
import sys
import os
import argparse
import time
import json
import glob
import numpy as np

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SERVER_ROOT = os.path.join(os.path.dirname(_TRAINING_ROOT), 'server')
sys.path.insert(0, _TRAINING_ROOT)
sys.path.insert(0, _SERVER_ROOT)

from shared.chord_vocab import TIER1_VOCAB, TIER1_TO_IDX, parse_chord_label


def parse_lab_file(lab_path):
    """Parse a .lab chord annotation file into [(start, end, label), ...]."""
    annotations = []
    with open(lab_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                annotations.append((float(parts[0]), float(parts[1]), parts[2]))
    return annotations


def get_chord_at_time(annotations, t):
    """Look up the chord label at time t from annotations."""
    for start, end, label in annotations:
        if start <= t < end:
            return label
    return 'N'


def find_isophonics_songs(ann_root, audio_root):
    """Find songs with matching audio and chord annotations.

    Handles the Isophonics layout where annotations are under
    chordlab/The Beatles/Album/Song.lab but audio is under audio/Album/Song.mp3
    (i.e. the artist subdirectory may need to be stripped).
    """
    songs = []
    chordlab_root = os.path.join(ann_root, 'chordlab')
    if not os.path.isdir(chordlab_root):
        return songs

    for lab_path in sorted(glob.glob(os.path.join(chordlab_root, '**', '*.lab'), recursive=True)):
        rel = os.path.relpath(lab_path, chordlab_root)
        stem = os.path.splitext(os.path.basename(lab_path))[0]

        # rel might be "The Beatles/Album/Song.lab" -- try stripping artist prefix
        rel_parts = rel.replace('\\', '/').split('/')
        search_rels = [os.path.join(*rel_parts)]
        if len(rel_parts) >= 3:
            search_rels.append(os.path.join(*rel_parts[1:]))

        audio_path = None
        for search_rel in search_rels:
            album_dir = os.path.dirname(search_rel)
            for ext in ('.mp3', '.wav', '.flac', '.m4a'):
                candidate = os.path.join(audio_root, album_dir, stem + ext)
                if os.path.isfile(candidate):
                    audio_path = candidate
                    break
            if audio_path:
                break

        if audio_path is None:
            continue

        beat_path = None
        beat_candidate = os.path.join(ann_root, 'beat', rel)
        if os.path.isfile(beat_candidate):
            beat_path = beat_candidate

        album_name = rel_parts[-2] if len(rel_parts) >= 2 else '?'
        songs.append({
            'stem': f"{album_name}/{stem}",
            'audio': audio_path,
            'chords': lab_path,
            'beats': beat_path,
        })

    return songs


def gt_at_beats(annotations, beat_times):
    """Get Tier 1 ground-truth chord label at each beat midpoint."""
    labels = []
    for bi in range(len(beat_times)):
        mid = ((beat_times[bi] + beat_times[bi + 1]) / 2.0
               if bi + 1 < len(beat_times) else beat_times[bi] + 0.25)
        raw = get_chord_at_time(annotations, mid)
        labels.append(parse_chord_label(raw, tier=1))
    return labels


def compute_metrics(gt_names, pred_names):
    """Compute WCSR, root, major, minor accuracy."""
    gt_idx = np.array([TIER1_TO_IDX.get(n, 0) for n in gt_names], dtype=np.int64)
    pred_idx = np.array([TIER1_TO_IDX.get(n, 0) for n in pred_names], dtype=np.int64)
    total = len(gt_idx)
    if total == 0:
        return {'wcsr': 0, 'major_acc': 0, 'minor_acc': 0, 'root_acc': 0, 'n_beats': 0}

    wcsr = float(np.sum(gt_idx == pred_idx) / total)

    maj_mask = (gt_idx >= 1) & (gt_idx <= 12)
    min_mask = (gt_idx >= 13) & (gt_idx <= 24)
    major_acc = (float(np.sum(gt_idx[maj_mask] == pred_idx[maj_mask]) / maj_mask.sum())
                 if maj_mask.sum() > 0 else 0.0)
    minor_acc = (float(np.sum(gt_idx[min_mask] == pred_idx[min_mask]) / min_mask.sum())
                 if min_mask.sum() > 0 else 0.0)

    pred_roots = np.where(pred_idx == 0, 0, np.where(pred_idx <= 12, pred_idx, pred_idx - 12))
    true_roots = np.where(gt_idx == 0, 0, np.where(gt_idx <= 12, gt_idx, gt_idx - 12))
    root_acc = float(np.sum(true_roots == pred_roots) / total)

    return {
        'wcsr': wcsr, 'major_acc': major_acc, 'minor_acc': minor_acc,
        'root_acc': root_acc, 'n_beats': total,
    }


def btc_predict_to_tier1(btc_idx):
    """Map a BTC 170-class index to Tier 1 label."""
    from btc_model.vocab import btc_idx_to_tier1
    return btc_idx_to_tier1(btc_idx)


def main():
    parser = argparse.ArgumentParser(description='Evaluate BTC production pipeline accuracy')
    parser.add_argument('--config', type=str, default=os.path.join(_TRAINING_ROOT, 'data', 'weights.json'))
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(_SERVER_ROOT, 'btc_model', 'btc_model_best.pth'))
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--max-songs', type=int, default=None)
    parser.add_argument('--per-song', action='store_true')
    parser.add_argument('--use-gold-beats', action='store_true',
                        help='Use gold beat annotations where available (isolates chord accuracy from beat errors)')
    parser.add_argument('--beat-aggregation', type=str, default=None,
                        choices=['logit', 'majority'],
                        help='Per-beat mapping: logit=sum smoothed logits (default); '
                             'majority=legacy frame vote. Overrides BTC_BEAT_AGGREGATION.')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if args.beat_aggregation:
        os.environ['BTC_BEAT_AGGREGATION'] = args.beat_aggregation

    import torch

    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device('mps')
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load BTC via server code (matches production exactly)
    from analyze_chords import (
        _load_btc_model,
        _extract_cqt,
        _gaussian_smooth_logits,
        _majority_filter,
        _btc_beat_aggregation_mode,
    )
    from btc_model.vocab import btc_idx_to_tier1

    beat_mode = _btc_beat_aggregation_mode()
    print(f"Device: {device}")
    print(f"BTC beat aggregation: {beat_mode} (set BTC_BEAT_AGGREGATION or --beat-aggregation)")
    model, _, btc_mean, btc_std = _load_btc_model()
    print(f"BTC loaded (mean={btc_mean:.4f}, std={btc_std:.4f})")

    # Beat This!
    from analyze_chords import _detect_beats_beat_this

    # Discover songs
    with open(args.config) as f:
        cfg = json.load(f)
    filter_ds = args.datasets.split(',') if args.datasets else None

    all_songs = []
    for name, ds in cfg['datasets'].items():
        if filter_ds and name not in filter_ds:
            continue
        if ds['type'] != 'isophonics':
            continue
        ann_root = os.path.join(_TRAINING_ROOT, ds['annotations'])
        aud_dir = os.path.join(_TRAINING_ROOT, ds['audio_dir'])
        found = find_isophonics_songs(ann_root, aud_dir)
        for s in found:
            s['dataset'] = name
        print(f"  {name}: {len(found)} songs with audio")
        all_songs.extend(found)

    if args.max_songs and len(all_songs) > args.max_songs:
        all_songs = all_songs[:args.max_songs]

    print(f"\nEvaluating {len(all_songs)} songs...")
    print('=' * 80)

    results = []
    for i, song in enumerate(all_songs):
        ds = song.get('dataset', '?')
        print(f"  [{i+1}/{len(all_songs)}] [{ds}] {song['stem']}...", end=' ', flush=True)

        t0 = time.time()
        try:
            annotations = parse_lab_file(song['chords'])

            if args.use_gold_beats and song.get('beats'):
                beat_times = []
                with open(song['beats']) as bf:
                    for line in bf:
                        parts = line.strip().split()
                        if parts:
                            try:
                                beat_times.append(float(parts[0]))
                            except ValueError:
                                pass
            else:
                beat_times, _ = _detect_beats_beat_this(song['audio'])
            if len(beat_times) < 4:
                print("SKIP (too few beats)")
                continue

            gt = gt_at_beats(annotations, beat_times)

            # Run BTC production pipeline
            cqt = _extract_cqt(song['audio'])
            cqt_norm = (cqt - btc_mean) / max(btc_std, 1e-6)
            n_frames = cqt_norm.shape[0]

            seq_len = 108
            stride = max(1, int(seq_len * 0.25))
            logit_sum = np.zeros((n_frames, 170), dtype=np.float32)
            logit_count = np.zeros(n_frames, dtype=np.float32)

            with torch.no_grad():
                pos = 0
                while pos < n_frames:
                    end = min(pos + seq_len, n_frames)
                    chunk = cqt_norm[pos:end]
                    actual_len = chunk.shape[0]
                    if actual_len < seq_len:
                        pad = np.zeros((seq_len - actual_len, 144), dtype=np.float32)
                        chunk = np.concatenate([chunk, pad], axis=0)
                    x = torch.from_numpy(chunk).unsqueeze(0).to(device)
                    out = model(x)
                    logits = out[0, :actual_len].cpu().numpy()
                    logit_sum[pos:pos + actual_len] += logits
                    logit_count[pos:pos + actual_len] += 1.0
                    pos += stride
                    if pos >= n_frames:
                        break

            logit_count[logit_count == 0] = 1.0
            avg_logits = logit_sum / logit_count[:, np.newaxis]
            avg_logits = _gaussian_smooth_logits(avg_logits, kernel_size=9)

            hop_dur = 2048 / 22050.0
            pred_names = []
            if beat_mode == 'majority':
                frame_preds = avg_logits.argmax(axis=1).astype(np.int64)
                frame_preds = _majority_filter(frame_preds, kernel_size=9)
                for bi in range(len(beat_times)):
                    t_start = beat_times[bi]
                    t_end = beat_times[bi + 1] if bi + 1 < len(beat_times) else t_start + 0.5
                    f_start = max(0, int(round(t_start / hop_dur)))
                    f_end = min(n_frames, int(round(t_end / hop_dur)))
                    if f_end <= f_start:
                        f_end = f_start + 1
                    if f_start >= n_frames:
                        pred_names.append('N')
                        continue
                    seg = frame_preds[f_start:min(f_end, n_frames)]
                    if len(seg) == 0:
                        pred_names.append('N')
                        continue
                    counts = np.bincount(seg, minlength=170)
                    winner = int(counts.argmax())
                    pred_names.append(btc_idx_to_tier1(winner))
            else:
                for bi in range(len(beat_times)):
                    t_start = beat_times[bi]
                    t_end = beat_times[bi + 1] if bi + 1 < len(beat_times) else t_start + 0.5
                    f_start = max(0, int(round(t_start / hop_dur)))
                    f_end = min(n_frames, int(round(t_end / hop_dur)))
                    if f_end <= f_start:
                        f_end = f_start + 1
                    if f_start >= n_frames:
                        pred_names.append('N')
                        continue
                    window = avg_logits[f_start:min(f_end, n_frames)]
                    if len(window) == 0:
                        pred_names.append('N')
                        continue
                    winner = int(window.sum(axis=0).argmax())
                    pred_names.append(btc_idx_to_tier1(winner))

            n = len(gt)
            pred_names = (pred_names[:n] + ['N'] * max(0, n - len(pred_names)))[:n]
            m = compute_metrics(gt, pred_names)
            results.append({**m, 'name': song['stem'], 'dataset': ds})

            dt = time.time() - t0
            print(f"WCSR={m['wcsr']:.3f}  root={m['root_acc']:.3f} ({dt:.1f}s)")

        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    # Aggregate
    print(f"\n{'=' * 80}")
    print(f"AGGREGATE ({len(results)} songs)")
    print(f"{'=' * 80}")

    if results:
        total_beats = sum(r['n_beats'] for r in results)
        wcsr = sum(r['wcsr'] * r['n_beats'] for r in results) / max(total_beats, 1)
        maj = sum(r['major_acc'] * r['n_beats'] for r in results) / max(total_beats, 1)
        minor = sum(r['minor_acc'] * r['n_beats'] for r in results) / max(total_beats, 1)
        root = sum(r['root_acc'] * r['n_beats'] for r in results) / max(total_beats, 1)
        print(f"\n  BTC (production pipeline)")
        print(f"    WCSR={wcsr:.3f}  maj={maj:.3f}  min={minor:.3f}  root={root:.3f}")
        print(f"    ({len(results)} songs, {total_beats} beats)")

        datasets = sorted(set(r['dataset'] for r in results))
        if len(datasets) > 1:
            for ds in datasets:
                dr = [r for r in results if r['dataset'] == ds]
                db = sum(r['n_beats'] for r in dr)
                dw = sum(r['wcsr'] * r['n_beats'] for r in dr) / max(db, 1)
                print(f"    {ds:12s}: WCSR={dw:.3f} ({len(dr)} songs, {db} beats)")

        if args.per_song:
            print(f"\n  {'Song':<55s} {'WCSR':>6s} {'Root':>6s} {'Maj':>6s} {'Min':>6s}")
            print(f"  {'-'*55} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
            for r in sorted(results, key=lambda x: x['wcsr']):
                print(f"  {r['name']:<55s} {r['wcsr']:.3f}  {r['root_acc']:.3f}  "
                      f"{r['major_acc']:.3f}  {r['minor_acc']:.3f}")

        # Worst performers
        worst = sorted(results, key=lambda x: x['wcsr'])[:10]
        print(f"\n  Bottom 10 (worst WCSR):")
        for r in worst:
            print(f"    {r['name']:<55s} WCSR={r['wcsr']:.3f}  root={r['root_acc']:.3f}")

        best = sorted(results, key=lambda x: -x['wcsr'])[:10]
        print(f"\n  Top 10 (best WCSR):")
        for r in best:
            print(f"    {r['name']:<55s} WCSR={r['wcsr']:.3f}  root={r['root_acc']:.3f}")


if __name__ == '__main__':
    main()
