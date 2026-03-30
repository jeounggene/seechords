#!/usr/bin/env python3
"""Evaluate a v2 chord model with Viterbi decoding on held-out songs.

Runs direct, factorized, and hybrid decoding, comparing accuracy across
multiple dimensions:
  - Overall WCSR (weighted chord symbol recall)
  - Root accuracy, quality accuracy
  - Major vs minor chord accuracy
  - Gold vs silver data slices
  - Per-song breakdown
  - UX metrics (segment length, flip rate)
  - Confusion matrices

Usage:
    python -m v2.evaluate --model models/chord_model_v2.pkl --data data/features_v2.npz
"""
import sys
import os
import argparse
import pickle
import numpy as np
from collections import defaultdict

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from shared.chord_vocab import TIER1_VOCAB
from v2.chord_schema import (
    ROOT_VOCAB, QUALITY_VOCAB, v2_to_tier1_idx, compose_chord,
)
from v2.decode import (
    decode_tier1_direct, decode_factorized, decode_hybrid,
    decode_hybrid_crf, smooth_isolated, compute_ux_metrics,
)


def weighted_chord_symbol_recall(y_true, y_pred, beat_durations=None):
    """MIREX-style WCSR: weight each beat by duration (or uniform)."""
    if beat_durations is None:
        beat_durations = np.ones(len(y_true))
    total = beat_durations.sum()
    if total == 0:
        return 0.0
    correct = sum(beat_durations[i] for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / total


def print_confusion(y_true, y_pred, vocab, top_n=10):
    """Print confusion matrix for the top N most frequent true classes."""
    counts = defaultdict(int)
    for t in y_true:
        counts[t] += 1
    top_classes = sorted(counts.keys(), key=lambda c: -counts[c])[:top_n]

    print(f"\nConfusion matrix (top {len(top_classes)} classes):")
    header = "        " + "".join(f"{vocab[c]:>7s}" for c in top_classes)
    print(header)
    print("        " + "-" * (7 * len(top_classes)))
    for true_cls in top_classes:
        row_mask = y_true == true_cls
        if row_mask.sum() == 0:
            continue
        preds = y_pred[row_mask]
        row = f"{vocab[true_cls]:>7s}|"
        for pred_cls in top_classes:
            pct = 100.0 * np.sum(preds == pred_cls) / row_mask.sum()
            row += f"{pct:6.1f}%" if pct >= 1.0 else "      ."
        print(row)


def evaluate_model(model, X, tier1_labels, root_labels, quality_labels,
                   song_ids, key_indices, filenames, provenance,
                   sample_weights, key_prior_mode='none', quality_agg=False,
                   quality_src='tier1'):
    """Run full evaluation: direct + factorized + hybrid + CRF, per-song and per-slice."""
    n_tier1 = len(TIER1_VOCAB)
    n_roots = len(ROOT_VOCAB)
    n_quals = len(QUALITY_VOCAB)
    has_crf = 'tuned_params' in model or 'crf_log_trans' in model

    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    unique_songs = np.unique(song_ids)
    direct_pred = np.zeros(len(tier1_labels), dtype=int)
    factor_pred = np.zeros(len(tier1_labels), dtype=int)
    hybrid_pred = np.zeros(len(tier1_labels), dtype=int)
    crf_pred = np.zeros(len(tier1_labels), dtype=int) if has_crf else None

    song_results = []

    for song_id in unique_songs:
        mask = song_ids == song_id
        X_song = X_norm[mask]
        y_song = tier1_labels[mask]
        n_beats = len(X_song)
        if n_beats == 0:
            continue

        ki = key_indices[song_id] if song_id < len(key_indices) else 0
        prov = provenance[song_id] if song_id < len(provenance) else 'unknown'
        name = filenames[song_id] if song_id < len(filenames) else f"song_{song_id}"

        # Direct Tier-1 decode
        path_d, _, _ = decode_tier1_direct(model, X_song, ki)
        path_d = smooth_isolated(path_d)
        direct_pred[mask] = path_d

        # Factorized decode
        path_f, root_probs, qual_probs, _, _ = decode_factorized(model, X_song, ki)
        path_f = smooth_isolated(path_f)
        factor_pred[mask] = path_f

        # Hybrid decode (factorized root + direct quality)
        path_h, _, _ = decode_hybrid(model, X_song, ki,
                                      key_prior_mode=key_prior_mode,
                                      quality_agg=quality_agg,
                                      quality_src=quality_src)
        path_h = smooth_isolated(path_h)
        hybrid_pred[mask] = path_h

        # CRF decode
        if has_crf:
            path_c, _, _ = decode_hybrid_crf(model, X_song, ki)
            path_c = smooth_isolated(path_c)
            crf_pred[mask] = path_c

        # Per-song metrics
        d_acc = np.mean(path_d == y_song)
        f_acc = np.mean(path_f == y_song)
        h_acc = np.mean(path_h == y_song)
        ux_h = compute_ux_metrics(path_h, np.arange(n_beats))

        result = {
            'song_id': int(song_id),
            'name': name,
            'provenance': prov,
            'n_beats': n_beats,
            'direct_acc': float(d_acc),
            'factor_acc': float(f_acc),
            'hybrid_acc': float(h_acc),
            'ux': ux_h,
        }
        if has_crf:
            result['crf_acc'] = float(np.mean(path_c == y_song))
            result['ux_crf'] = compute_ux_metrics(path_c, np.arange(n_beats))
        song_results.append(result)

    return direct_pred, factor_pred, hybrid_pred, crf_pred, song_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate v2 chord model')
    parser.add_argument('--model', required=True, help='Path to v2 model .pkl')
    parser.add_argument('--data', required=True, help='Path to features_v2.npz')
    parser.add_argument('--songs', type=str, default=None,
                        help='Comma-separated song indices to evaluate')
    parser.add_argument('--split', type=str, default='test',
                        choices=['all', 'train', 'val', 'test'],
                        help='Which split to evaluate (default: test)')
    parser.add_argument('--worst', type=int, default=0,
                        help='Show N worst songs in detail')
    parser.add_argument('--key-prior', type=str, default='none',
                        choices=['tier1', 'root', 'none'],
                        help='Key prior mode for hybrid decoder (default: none)')
    parser.add_argument('--quality-agg', action='store_true',
                        help='(deprecated) Aggregate 7-class quality into 3-class')
    parser.add_argument('--quality-src', type=str, default='tier1',
                        choices=['tier1', 'agg7', 'clf3'],
                        help='Quality source for hybrid decoder (default: tier1)')
    args = parser.parse_args()

    # Load model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    print(f"Model: {args.model}")
    print(f"  Version: {model.get('version', 1)}")
    print(f"  Feature dim: {model.get('feature_dim', 12)}")
    print(f"  Metadata: {model.get('metadata', {})}")

    # Load data
    data = np.load(args.data, allow_pickle=True)
    feat_dim = model.get('feature_dim', 12)
    has_third_ratio = model.get('third_ratio', False)

    # Determine base feature dim (before third-ratio was appended)
    base_dim = feat_dim - 12 if has_third_ratio else feat_dim

    if f'X_{base_dim}' in data:
        X = data[f'X_{base_dim}'].astype(np.float32)
    else:
        X = data['X_12'].astype(np.float32)

    if has_third_ratio:
        X_12 = data['X_12'].astype(np.float32)
        if 'X_third_ratio' in data:
            X_tr = data['X_third_ratio'].astype(np.float32)
        else:
            n_beats = len(X_12)
            X_tr = np.zeros((n_beats, 12), dtype=np.float32)
            for root in range(12):
                maj3 = X_12[:, (root + 4) % 12]
                min3 = X_12[:, (root + 3) % 12]
                X_tr[:, root] = (maj3 - min3) / (maj3 + min3 + 1e-8)
        X = np.hstack([X, X_tr])
        print(f"  Third-ratio features: appended ({X.shape[1]} dims total)")
    tier1_labels = data['tier1_labels'].astype(np.int32)
    root_labels = data['root_labels'].astype(np.int32)
    quality_labels = data['quality_labels'].astype(np.int32)
    song_ids = data['song_ids'].astype(np.int32)
    key_indices = data['key_indices']
    filenames = list(data['filenames'])
    provenance = list(data['provenance'])
    sample_weights = data['sample_weights'].astype(np.float32)

    if args.songs:
        selected = set(int(s) for s in args.songs.split(','))
        mask = np.array([s in selected for s in song_ids])
        X = X[mask]
        tier1_labels, root_labels, quality_labels = tier1_labels[mask], root_labels[mask], quality_labels[mask]
        song_ids, sample_weights = song_ids[mask], sample_weights[mask]
    elif args.split != 'all':
        # Filter by split using song IDs stored in model
        split_key = f'{args.split}_song_ids'
        if split_key in model:
            split_songs = set(model[split_key])
            mask = np.array([s in split_songs for s in song_ids])
            X = X[mask]
            tier1_labels, root_labels, quality_labels = tier1_labels[mask], root_labels[mask], quality_labels[mask]
            song_ids, sample_weights = song_ids[mask], sample_weights[mask]
            print(f"  Split: {args.split} ({len(split_songs)} songs)")
        else:
            print(f"  WARNING: model has no '{split_key}' — evaluating all songs")
            print(f"  (retrain with updated train.py to get split info)")

    n_songs = len(np.unique(song_ids))
    print(f"\nEvaluating {len(X)} beats from {n_songs} songs")
    print(f"  Key prior mode: {args.key_prior}")
    print(f"  Quality source: {args.quality_src}")
    if args.quality_agg:
        print(f"  Quality aggregation: 7-class → 3-class (N/maj/min) (deprecated, use --quality-src agg7)")

    # Run evaluation
    direct_pred, factor_pred, hybrid_pred, crf_pred, song_results = evaluate_model(
        model, X, tier1_labels, root_labels, quality_labels,
        song_ids, key_indices, filenames, provenance, sample_weights,
        key_prior_mode=args.key_prior, quality_agg=args.quality_agg,
        quality_src=args.quality_src)

    from sklearn.metrics import accuracy_score
    has_crf = crf_pred is not None

    # ── Overall metrics ──
    d_acc = accuracy_score(tier1_labels, direct_pred)
    f_acc = accuracy_score(tier1_labels, factor_pred)
    h_acc = accuracy_score(tier1_labels, hybrid_pred)
    d_wcsr = weighted_chord_symbol_recall(tier1_labels, direct_pred)
    f_wcsr = weighted_chord_symbol_recall(tier1_labels, factor_pred)
    h_wcsr = weighted_chord_symbol_recall(tier1_labels, hybrid_pred)

    if has_crf:
        c_acc = accuracy_score(tier1_labels, crf_pred)
        c_wcsr = weighted_chord_symbol_recall(tier1_labels, crf_pred)
        w = 86
        print(f"\n{'='*w}")
        print(f"{'Metric':<32s} {'Direct':>10s} {'Factorized':>12s} {'Hybrid':>10s} {'CRF':>10s}")
        print(f"{'-'*w}")
        print(f"{'Beat-level accuracy':<32s} {d_acc:>10.3f} {f_acc:>12.3f} {h_acc:>10.3f} {c_acc:>10.3f}")
        print(f"{'WCSR':<32s} {d_wcsr:>10.3f} {f_wcsr:>12.3f} {h_wcsr:>10.3f} {c_wcsr:>10.3f}")
    else:
        print(f"\n{'='*74}")
        print(f"{'Metric':<32s} {'Direct':>10s} {'Factorized':>12s} {'Hybrid':>10s}")
        print(f"{'-'*74}")
        print(f"{'Beat-level accuracy':<32s} {d_acc:>10.3f} {f_acc:>12.3f} {h_acc:>10.3f}")
        print(f"{'WCSR':<32s} {d_wcsr:>10.3f} {f_wcsr:>12.3f} {h_wcsr:>10.3f}")

    # Major / minor / N breakdown
    maj_mask = np.array([1 <= y <= 12 for y in tier1_labels])
    min_mask = np.array([13 <= y <= 24 for y in tier1_labels])
    n_mask = tier1_labels == 0

    for label, m in [('Major chords', maj_mask), ('Minor chords', min_mask), ('No-chord (N)', n_mask)]:
        if m.sum() > 0:
            da = accuracy_score(tier1_labels[m], direct_pred[m])
            fa = accuracy_score(tier1_labels[m], factor_pred[m])
            ha = accuracy_score(tier1_labels[m], hybrid_pred[m])
            if has_crf:
                ca = accuracy_score(tier1_labels[m], crf_pred[m])
                print(f"{label + f' ({m.sum()} beats)':<32s} {da:>10.3f} {fa:>12.3f} {ha:>10.3f} {ca:>10.3f}")
            else:
                print(f"{label + f' ({m.sum()} beats)':<32s} {da:>10.3f} {fa:>12.3f} {ha:>10.3f}")

    # Root accuracy (from factorized and hybrid predictions)
    def _pred_roots(pred):
        return np.array([0 if p == 0 else ((p - 1) % 12 + 1) if 1 <= p <= 24 else 0 for p in pred])

    f_root_acc = accuracy_score(root_labels, _pred_roots(factor_pred))
    h_root_acc = accuracy_score(root_labels, _pred_roots(hybrid_pred))
    d_root_acc = accuracy_score(root_labels, _pred_roots(direct_pred))
    if has_crf:
        c_root_acc = accuracy_score(root_labels, _pred_roots(crf_pred))
        print(f"{'Root accuracy':<32s} {d_root_acc:>10.3f} {f_root_acc:>12.3f} {h_root_acc:>10.3f} {c_root_acc:>10.3f}")
        print(f"{'='*86}")
    else:
        print(f"{'Root accuracy':<32s} {d_root_acc:>10.3f} {f_root_acc:>12.3f} {h_root_acc:>10.3f}")
        print(f"{'='*74}")

    # ── Gold vs Silver slice ──
    gold_mask = np.zeros(len(tier1_labels), dtype=bool)
    silver_mask = np.zeros(len(tier1_labels), dtype=bool)
    for song_id in np.unique(song_ids):
        m = song_ids == song_id
        prov = provenance[song_id] if song_id < len(provenance) else 'unknown'
        if prov == 'gold':
            gold_mask |= m
        else:
            silver_mask |= m

    print(f"\nProvenance slices:")
    for label, m in [('Gold', gold_mask), ('Silver', silver_mask)]:
        if m.sum() > 0:
            da = accuracy_score(tier1_labels[m], direct_pred[m])
            fa = accuracy_score(tier1_labels[m], factor_pred[m])
            ha = accuracy_score(tier1_labels[m], hybrid_pred[m])
            n = m.sum()
            ns = len(set(song_ids[m]))
            if has_crf:
                ca = accuracy_score(tier1_labels[m], crf_pred[m])
                print(f"  {label} ({ns} songs, {n} beats): direct={da:.3f}  factorized={fa:.3f}  hybrid={ha:.3f}  crf={ca:.3f}")
            else:
                print(f"  {label} ({ns} songs, {n} beats): direct={da:.3f}  factorized={fa:.3f}  hybrid={ha:.3f}")

    # ── UX metrics (aggregate, for hybrid and CRF) ──
    all_ux = [r['ux'] for r in song_results if r['ux']]
    if all_ux:
        avg_seg = np.mean([u['avg_segment_beats'] for u in all_ux])
        avg_flip = np.mean([u['flip_rate'] for u in all_ux])
        total_one = sum(u['n_one_beat_flips'] for u in all_ux)
        print(f"\nUX metrics (hybrid, averaged over songs):")
        print(f"  Avg segment length: {avg_seg:.1f} beats")
        print(f"  Avg flip rate:      {avg_flip:.3f}")
        print(f"  Total 1-beat flips: {total_one}")

    if has_crf:
        all_ux_crf = [r['ux_crf'] for r in song_results if r.get('ux_crf')]
        if all_ux_crf:
            avg_seg_c = np.mean([u['avg_segment_beats'] for u in all_ux_crf])
            avg_flip_c = np.mean([u['flip_rate'] for u in all_ux_crf])
            total_one_c = sum(u['n_one_beat_flips'] for u in all_ux_crf)
            print(f"\nUX metrics (CRF, averaged over songs):")
            print(f"  Avg segment length: {avg_seg_c:.1f} beats")
            print(f"  Avg flip rate:      {avg_flip_c:.3f}")
            print(f"  Total 1-beat flips: {total_one_c}")

    # ── Per-song results ──
    sort_key = 'crf_acc' if has_crf else 'hybrid_acc'
    print(f"\nPer-song breakdown (sorted by {sort_key}):")
    sorted_results = sorted(song_results, key=lambda x: x.get(sort_key, x['hybrid_acc']))
    for r in sorted_results:
        prov_tag = 'G' if r['provenance'] == 'gold' else 'S'
        line = (f"  [{prov_tag}] {r['name']:40s}  {r['n_beats']:4d} beats  "
                f"direct={r['direct_acc']:.3f}  factor={r['factor_acc']:.3f}  hybrid={r['hybrid_acc']:.3f}")
        if has_crf:
            line += f"  crf={r['crf_acc']:.3f}"
        print(line)

    # ── Worst-N detail ──
    if args.worst > 0:
        print(f"\n{'='*74}")
        print(f"Worst {args.worst} songs by hybrid accuracy:")
        print(f"{'='*74}")
        for r in sorted_results[:args.worst]:
            prov_tag = 'Gold' if r['provenance'] == 'gold' else 'Silver'
            print(f"\n  {r['name']} ({prov_tag}, {r['n_beats']} beats)")
            print(f"    Direct={r['direct_acc']:.3f}  Factorized={r['factor_acc']:.3f}  Hybrid={r['hybrid_acc']:.3f}")
            if r['ux']:
                print(f"    Flip rate={r['ux']['flip_rate']:.3f}  Avg segment={r['ux']['avg_segment_beats']:.1f} beats")

    # ── Confusion matrices ──
    print("\n── Direct Tier-1 confusion ──")
    print_confusion(tier1_labels, direct_pred, TIER1_VOCAB)

    print("\n── Hybrid Tier-1 confusion ──")
    print_confusion(tier1_labels, hybrid_pred, TIER1_VOCAB)

    if has_crf:
        print("\n── CRF Tier-1 confusion ──")
        print_confusion(tier1_labels, crf_pred, TIER1_VOCAB)

    # ── Same-root quality confusion (best decoder: CRF or hybrid) ──
    best_pred = crf_pred if has_crf else hybrid_pred
    best_name = "CRF" if has_crf else "hybrid"
    print(f"\n── Same-root quality confusion ({best_name}) ──")
    print(f"  When true label is minor, how often does {best_name} predict same-root major?")
    total_min = 0
    same_root_maj = 0
    for note_idx in range(12):
        min_idx = 13 + note_idx
        maj_idx = 1 + note_idx
        m = tier1_labels == min_idx
        n = m.sum()
        if n == 0:
            continue
        total_min += n
        to_maj = (best_pred[m] == maj_idx).sum()
        same_root_maj += to_maj
        note = TIER1_VOCAB[min_idx]
        maj_note = TIER1_VOCAB[maj_idx]
        print(f"  {note:>3s} → {maj_note:<3s}: {to_maj:4d}/{n:4d} ({100*to_maj/n:.1f}%)")
    if total_min > 0:
        print(f"  {'Total':>7s}: {same_root_maj:4d}/{total_min:4d} ({100*same_root_maj/total_min:.1f}%)")

    # ── Major vs Minor 2x2 confusion (best decoder) ──
    print(f"\n── Major vs Minor 2×2 confusion ({best_name}) ──")
    maj_mask_t = np.array([1 <= y <= 12 for y in tier1_labels])
    min_mask_t = np.array([13 <= y <= 24 for y in tier1_labels])
    maj_mask_p = np.array([1 <= y <= 12 for y in best_pred])
    min_mask_p = np.array([13 <= y <= 24 for y in best_pred])
    mm = maj_mask_t.sum()
    mn = min_mask_t.sum()
    if mm > 0 and mn > 0:
        mm_mm = (maj_mask_t & maj_mask_p).sum()
        mm_mn = (maj_mask_t & min_mask_p).sum()
        mn_mm = (min_mask_t & maj_mask_p).sum()
        mn_mn = (min_mask_t & min_mask_p).sum()
        print(f"             Pred Maj  Pred Min")
        print(f"  True Maj   {mm_mm:6d}    {mm_mn:6d}   ({100*mm_mm/mm:.1f}% / {100*mm_mn/mm:.1f}%)")
        print(f"  True Min   {mn_mm:6d}    {mn_mn:6d}   ({100*mn_mm/mn:.1f}% / {100*mn_mn/mn:.1f}%)")


if __name__ == '__main__':
    main()
