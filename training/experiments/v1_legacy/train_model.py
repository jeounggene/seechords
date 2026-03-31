#!/usr/bin/env python3
"""V1 model training: flat 25-class Random Forest + HMM transition learning.

Trains a Random Forest classifier on 12-dim HPCP features to predict Tier 1
chord classes (N + 12 major + 12 minor = 25). Also learns:
  - Class-centroid templates (for fallback template matching)
  - HMM transition matrix from annotated chord sequences
  - Per-key chord prior probabilities from music theory

Key priors are NOT used as classifier inputs — they caused the model to ignore
chroma. Instead they are applied as emission bias during Viterbi decoding.

Output .pkl contains: classifier, templates, transition_probs, key_profiles,
vocab, tier, feature_dim, metadata.

Usage:
    python -m v1.train_model --data data/features.npz --out models/chord_model.pkl
"""""
import sys
import os
import argparse
import pickle
import numpy as np
from collections import Counter

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)
from shared.chord_vocab import (
    TIER1_VOCAB, TIER2_VOCAB, NOTES,
    TIER1_TO_IDX, label_to_idx, idx_to_label,
)

# Key-chord relationship tables
_DIATONIC_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_DIATONIC_QUALITIES_T1 = ['', 'm', 'm', '', '', 'm', 'm']  # maj/min only


def _build_key_priors(tier=1):
    """Build per-key chord prior probabilities from music theory.

    Returns: (12, n_classes) array where [key_idx, chord_idx] = prior probability.
    """
    vocab = TIER1_VOCAB if tier == 1 else TIER2_VOCAB
    n_classes = len(vocab)
    priors = np.ones((12, n_classes)) * 0.01  # small base prob for all chords

    for key_idx in range(12):
        for offset, quality in zip(_DIATONIC_INTERVALS, _DIATONIC_QUALITIES_T1):
            note = NOTES[(key_idx + offset) % 12]
            chord_name = note + quality
            if chord_name in vocab:
                ci = label_to_idx(chord_name, tier)
                priors[key_idx, ci] = 1.0  # strong prior for diatonic chords

        # Tonic and dominant get extra weight (major quality only)
        tonic_note = NOTES[key_idx]
        dom_note = NOTES[(key_idx + 7) % 12]
        if tonic_note in vocab:
            priors[key_idx, label_to_idx(tonic_note, tier)] = 2.0
        if dom_note in vocab:
            priors[key_idx, label_to_idx(dom_note, tier)] = 1.5

        # N (no-chord) gets moderate prior
        priors[key_idx, 0] = 0.3

    # Normalize rows
    priors /= priors.sum(axis=1, keepdims=True)
    return priors


def _learn_transition_matrix(y, song_ids, n_classes, smoothing=1.0):
    """Learn chord transition probabilities from annotated data.

    Uses Laplace smoothing. Transitions across song boundaries are excluded.
    """
    trans = np.full((n_classes, n_classes), smoothing)

    for i in range(1, len(y)):
        if song_ids[i] == song_ids[i - 1]:  # same song
            trans[y[i - 1], y[i]] += 1.0

    # Normalize rows
    trans /= trans.sum(axis=1, keepdims=True)
    return trans


def _learn_templates(X, y, n_classes):
    """Compute class-centroid templates (mean HPCP per chord class)."""
    templates = np.zeros((n_classes, 12))
    counts = np.zeros(n_classes)
    for i in range(len(X)):
        templates[y[i]] += X[i]
        counts[y[i]] += 1

    for c in range(n_classes):
        if counts[c] > 0:
            templates[c] /= counts[c]
            norm = np.linalg.norm(templates[c])
            if norm > 0:
                templates[c] /= norm

    return templates


def train(args):
    # Load data
    data = np.load(args.data, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int32)
    song_ids = data['song_ids'].astype(np.int32)
    key_indices = data['key_indices']
    filenames = list(data['filenames'])

    tier = args.tier
    vocab = TIER1_VOCAB if tier == 1 else TIER2_VOCAB
    n_classes = len(vocab)

    print(f"Training data: {len(X)} beats from {len(filenames)} songs")
    print(f"Vocabulary: {n_classes} classes (tier {tier})")

    # ── Normalize features ──
    # L2 normalize each beat's HPCP vector
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    # ── Learn components ──
    print("\n1. Learning templates (class centroids)...")
    templates = _learn_templates(X_norm, y, n_classes)
    non_empty = np.sum(np.any(templates > 0, axis=1))
    print(f"   {non_empty}/{n_classes} classes have training examples")

    print("2. Learning transition matrix...")
    trans_matrix = _learn_transition_matrix(y, song_ids, n_classes, smoothing=1.0)
    # Show top transitions
    for ci in range(min(5, n_classes)):
        top3 = np.argsort(trans_matrix[ci])[-3:][::-1]
        probs = [f"{vocab[t]}:{trans_matrix[ci,t]:.2f}" for t in top3]
        if ci < len(vocab):
            print(f"   {vocab[ci]:6s} → {', '.join(probs)}")

    print("3. Building key priors...")
    key_priors = _build_key_priors(tier)

    print("4. Training classifier...")

    # Split: leave out ~15% of songs for validation
    unique_songs = np.unique(song_ids)
    n_val = max(1, int(len(unique_songs) * 0.15))
    rng = np.random.RandomState(42)
    rng.shuffle(unique_songs)
    val_songs = set(unique_songs[:n_val])
    train_mask = np.array([s not in val_songs for s in song_ids])
    val_mask = ~train_mask

    X_train, y_train = X_norm[train_mask], y[train_mask]
    X_val, y_val = X_norm[val_mask], y[val_mask]
    print(f"   Train: {len(X_train)} beats, Val: {len(X_val)} beats")

    # Train classifier on HPCP features only (12 dims).
    # Key priors are NOT used as classifier inputs — they caused the model
    # to ignore chroma and predict solely based on key. Instead, key priors
    # are applied as emission bias during Viterbi decoding.
    X_train_clf = X_train
    X_val_clf = X_val

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_clf, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train_clf))
    val_pred = clf.predict(X_val_clf)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"\n   Train accuracy: {train_acc:.3f}")
    print(f"   Val accuracy:   {val_acc:.3f}")

    # Per-class metrics (only classes present in validation)
    present = sorted(set(y_val))
    target_names = [vocab[i] for i in present]
    print(f"\n   Per-class (validation):")
    report = classification_report(y_val, val_pred, labels=present,
                                   target_names=target_names, zero_division=0)
    print(report)

    # ── Template accuracy (baseline) ──
    template_sim = templates @ X_val.T  # (n_classes, n_val)
    template_pred = np.argmax(template_sim, axis=0)
    template_acc = accuracy_score(y_val, template_pred)
    print(f"   Template-only baseline accuracy: {template_acc:.3f}")

    # ── Save model ──
    model = {
        'classifier': clf,
        'templates': templates,
        'transition_probs': trans_matrix,
        'key_priors': key_priors,
        'vocab': vocab,
        'tier': tier,
        'feature_dim': 12,  # HPCP only (key priors used as emission bias, not classifier input)
        'metadata': {
            'n_songs': len(filenames),
            'n_beats': len(X),
            'n_classes': n_classes,
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'template_acc': float(template_acc),
        },
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(model, f, protocol=4)

    print(f"\nModel saved to {args.out}")
    print(f"File size: {os.path.getsize(args.out) / 1024:.0f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train chord recognition model')
    parser.add_argument('--data', required=True, help='Path to features.npz from prepare_data.py')
    parser.add_argument('--out', default='models/chord_model.pkl', help='Output model path')
    parser.add_argument('--tier', type=int, default=1, choices=[1, 2], help='Vocabulary tier')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
