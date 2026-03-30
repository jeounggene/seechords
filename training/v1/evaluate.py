#!/usr/bin/env python3
"""V1 evaluation: Viterbi decoding with WCSR and per-song metrics.

Runs the trained v1 model on feature data, applying Viterbi decoding per song
with softened transitions (SELF_PROB=0.40, FLOOR=0.015) and key-biased emissions.

Metrics reported:
    - Beat-level accuracy
    - Weighted chord symbol recall (WCSR) -- MIREX standard
    - Per-song accuracy breakdown
    - Major/minor/N accuracy breakdown
    - Confusion matrix for top N classes

Usage:
    python -m v1.evaluate --model models/chord_model.pkl --data data/features.npz
"""
import sys
import os
import argparse
import pickle
import numpy as np
from collections import defaultdict

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)
from shared.chord_vocab import (
    TIER1_VOCAB, TIER2_VOCAB, NOTES, parse_chord_label, label_to_idx,
)


def viterbi_decode(log_emission, log_transition, log_prior=None):
    """Run Viterbi decoding on a sequence.

    Args:
        log_emission: (n_classes, T) log-probabilities
        log_transition: (n_classes, n_classes) log P(j|i)
        log_prior: (n_classes,) log prior for first frame

    Returns:
        path: (T,) best chord indices
    """
    n_classes, T = log_emission.shape
    if log_prior is None:
        log_prior = np.full(n_classes, -np.log(n_classes))

    viterbi = np.full((n_classes, T), -np.inf)
    backptr = np.zeros((n_classes, T), dtype=int)

    viterbi[:, 0] = log_prior + log_emission[:, 0]

    for t in range(1, T):
        for s in range(n_classes):
            scores = viterbi[:, t - 1] + log_transition[:, s]
            bp = int(np.argmax(scores))
            viterbi[s, t] = scores[bp] + log_emission[s, t]
            backptr[s, t] = bp

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(viterbi[:, -1]))
    for t in range(T - 2, -1, -1):
        path[t] = backptr[path[t + 1], t + 1]

    return path


def evaluate_with_viterbi(model, X, y, song_ids, key_indices):
    """Evaluate using classifier + Viterbi decoding per song."""
    clf = model['classifier']
    trans = model['transition_probs']
    key_priors = model['key_priors']
    vocab = model['vocab']
    n_classes = len(vocab)

    # Soften transitions to match server decode
    trans_soft = trans.copy()
    SELF_PROB = 0.40
    FLOOR = 0.015
    for i in range(n_classes):
        trans_soft[i, i] = SELF_PROB
        off_diag = trans[i].copy()
        off_diag[i] = 0
        off_sum = off_diag.sum()
        if off_sum > 0:
            off_diag = np.maximum(off_diag / off_sum * (1 - SELF_PROB), FLOOR)
            off_diag = off_diag / off_diag.sum() * (1 - SELF_PROB)
        trans_soft[i] = off_diag
        trans_soft[i, i] = SELF_PROB

    log_trans = np.log(np.clip(trans_soft, 1e-10, None))

    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    unique_songs = np.unique(song_ids)
    all_pred = np.zeros(len(y), dtype=int)

    song_results = []

    for song_id in unique_songs:
        mask = song_ids == song_id
        X_song = X_norm[mask]
        y_song = y[mask]
        n_beats = len(X_song)

        if n_beats == 0:
            continue

        # Classifier takes HPCP only (12 dims) — no key priors as input
        ki = key_indices[song_id] if song_id < len(key_indices) else 0
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X_song)
            full_probs = np.full((n_beats, n_classes), 1e-10)
            for ci, cls in enumerate(clf.classes_):
                full_probs[:, cls] = probs[:, ci]
        else:
            pred = clf.predict(X_song)
            full_probs = np.full((n_beats, n_classes), 1e-10)
            for i, p in enumerate(pred):
                full_probs[i, p] = 1.0

        log_emit = np.log(np.clip(full_probs.T, 1e-10, None))  # (n_classes, n_beats)

        # Add key prior as emission bias (not classifier input)
        log_key_prior = np.log(np.clip(key_priors[ki], 1e-10, None))
        log_emit += log_key_prior[:, np.newaxis] * 0.5

        # Run Viterbi
        path = viterbi_decode(log_emit, log_trans)
        all_pred[mask] = path

        # Per-song accuracy
        correct = np.sum(path == y_song)
        acc = correct / len(y_song) if len(y_song) > 0 else 0
        song_results.append({
            'song_id': int(song_id),
            'n_beats': n_beats,
            'accuracy': acc,
        })

    return all_pred, song_results


def weighted_chord_symbol_recall(y_true, y_pred, beat_durations=None):
    """MIREX-style weighted chord symbol recall.

    Weight each beat by its duration (or uniform if not provided).
    """
    if beat_durations is None:
        beat_durations = np.ones(len(y_true))

    correct_weight = 0.0
    total_weight = beat_durations.sum()

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_weight += beat_durations[i]

    return correct_weight / total_weight if total_weight > 0 else 0.0


def print_confusion(y_true, y_pred, vocab, top_n=10):
    """Print confusion matrix for the top N most frequent classes."""
    counts = defaultdict(int)
    for t in y_true:
        counts[t] += 1

    top_classes = sorted(counts.keys(), key=lambda c: -counts[c])[:top_n]
    print(f"\nConfusion matrix (top {len(top_classes)} classes):")

    # Header
    header = "        " + "".join(f"{vocab[c]:>7s}" for c in top_classes)
    print(header)
    print("        " + "-" * (7 * len(top_classes)))

    for true_cls in top_classes:
        row_mask = y_true == true_cls
        if row_mask.sum() == 0:
            continue
        preds_for_class = y_pred[row_mask]
        row = f"{vocab[true_cls]:>7s}|"
        for pred_cls in top_classes:
            cnt = np.sum(preds_for_class == pred_cls)
            pct = 100.0 * cnt / row_mask.sum()
            if pct >= 1.0:
                row += f"{pct:6.1f}%"
            else:
                row += "      ."
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Evaluate chord recognition model')
    parser.add_argument('--model', required=True, help='Path to trained model .pkl')
    parser.add_argument('--data', required=True, help='Path to features.npz (can be same as training data for sanity check)')
    parser.add_argument('--songs', type=str, default=None, help='Comma-separated song indices to evaluate (default: all)')
    args = parser.parse_args()

    # Load model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    vocab = model['vocab']
    tier = model['tier']
    print(f"Model: {args.model}")
    print(f"  Tier: {tier}, Classes: {len(vocab)}")
    print(f"  Training stats: {model['metadata']}")

    # Load data
    data = np.load(args.data, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int32)
    song_ids = data['song_ids'].astype(np.int32)
    key_indices = data['key_indices']
    filenames = list(data['filenames'])

    if args.songs:
        selected = set(int(s) for s in args.songs.split(','))
        mask = np.array([s in selected for s in song_ids])
        X, y, song_ids = X[mask], y[mask], song_ids[mask]

    print(f"\nEvaluating on {len(X)} beats from {len(np.unique(song_ids))} songs")

    # Run evaluation
    predictions, song_results = evaluate_with_viterbi(model, X, y, song_ids, key_indices)

    # Overall metrics
    from sklearn.metrics import accuracy_score
    overall_acc = accuracy_score(y, predictions)
    wcsr = weighted_chord_symbol_recall(y, predictions)

    print(f"\n{'='*50}")
    print(f"Overall beat-level accuracy:  {overall_acc:.3f}")
    print(f"Weighted chord symbol recall: {wcsr:.3f}")
    print(f"{'='*50}")

    # Per-song results
    print(f"\nPer-song breakdown:")
    for res in sorted(song_results, key=lambda r: r['accuracy']):
        sid = res['song_id']
        name = filenames[sid] if sid < len(filenames) else f"song_{sid}"
        print(f"  {name:40s}  {res['n_beats']:4d} beats  acc={res['accuracy']:.3f}")

    # Confusion matrix
    print_confusion(y, predictions, vocab, top_n=10)

    # Major/minor breakdown
    maj_mask = np.array([1 <= yi <= 12 for yi in y])  # major chords
    min_mask = np.array([13 <= yi <= 24 for yi in y])  # minor chords
    n_mask = y == 0

    if maj_mask.sum() > 0:
        maj_acc = accuracy_score(y[maj_mask], predictions[maj_mask])
        print(f"\nMajor chord accuracy: {maj_acc:.3f} ({maj_mask.sum()} beats)")
    if min_mask.sum() > 0:
        min_acc = accuracy_score(y[min_mask], predictions[min_mask])
        print(f"Minor chord accuracy: {min_acc:.3f} ({min_mask.sum()} beats)")
    if n_mask.sum() > 0:
        n_acc = accuracy_score(y[n_mask], predictions[n_mask])
        print(f"No-chord accuracy:    {n_acc:.3f} ({n_mask.sum()} beats)")


if __name__ == '__main__':
    main()
