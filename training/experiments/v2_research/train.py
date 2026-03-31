#!/usr/bin/env python3
"""V2 model training: factorized root + quality Random Forest classifiers.

Trains three classifiers:
  1. clf_tier1:   flat 25-class Tier-1 (for Models A/B and comparison)
  2. clf_root:    13-class root classifier
  3. clf_quality: 7-class quality classifier

Also learns:
  - Root transition matrix (13x13)
  - Quality transition matrix (7x7)
  - Tier-1 transition matrix (25x25)
  - Key priors for root (12x13) and tier-1 (12x25)
  - Class-centroid templates (12-dim, for fallback)

RF hyperparameters: n_estimators=300, max_depth=20, min_samples_leaf=2,
class_weight='balanced_subsample'.

Supports ablation via:
  --feature-dim 12|36  (v1 baseline vs v2 context features)
  --no-weights         (disable gold/silver sample weighting)

Usage:
    python -m v2.train --data data/features_v2.npz --out models/chord_model_v2.pkl
"""""
import sys
import os
import argparse
import pickle
import numpy as np
from collections import Counter

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)
from v2.chord_schema import (
    ROOT_VOCAB, QUALITY_VOCAB, IDX_TO_ROOT, IDX_TO_QUALITY,
    QUALITY_PARENT, NOTES, v2_to_tier1_idx, compose_chord,
    QUALITY3_VOCAB, QUALITY7_TO_3,
)
from shared.chord_vocab import TIER1_VOCAB, label_to_idx as tier1_label_to_idx

# ── Diatonic key-chord tables ────────────────────────────────
_DIATONIC_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_DIATONIC_ROOT_INDICES = []  # populated per-key below


def _build_root_key_priors():
    """Build per-key root prior probabilities (12, 13).

    For each key, boost diatonic roots.
    """
    n_roots = len(ROOT_VOCAB)
    priors = np.ones((12, n_roots)) * 0.01

    for key_idx in range(12):
        for offset in _DIATONIC_INTERVALS:
            note = NOTES[(key_idx + offset) % 12]
            ri = ROOT_VOCAB.index(note) if note in ROOT_VOCAB else -1
            if ri >= 0:
                priors[key_idx, ri] = 1.0

        # Tonic and dominant extra boost
        tonic = NOTES[key_idx]
        dominant = NOTES[(key_idx + 7) % 12]
        if tonic in ROOT_VOCAB:
            priors[key_idx, ROOT_VOCAB.index(tonic)] = 2.0
        if dominant in ROOT_VOCAB:
            priors[key_idx, ROOT_VOCAB.index(dominant)] = 1.5

        # N gets moderate prior
        priors[key_idx, 0] = 0.3

    priors /= priors.sum(axis=1, keepdims=True)
    return priors


def _build_tier1_key_priors():
    """Build per-key Tier-1 chord priors (12, 25) — same as v1."""
    from shared.chord_vocab import TIER1_VOCAB as vocab
    _DIATONIC_QUALITIES_T1 = ['', 'm', 'm', '', '', 'm', 'm']
    n_classes = len(vocab)
    priors = np.ones((12, n_classes)) * 0.01

    for key_idx in range(12):
        for offset, quality in zip(_DIATONIC_INTERVALS, _DIATONIC_QUALITIES_T1):
            note = NOTES[(key_idx + offset) % 12]
            chord_name = note + quality
            if chord_name in vocab:
                ci = vocab.index(chord_name)
                priors[key_idx, ci] = 1.0
        tonic_note = NOTES[key_idx]
        dom_note = NOTES[(key_idx + 7) % 12]
        if tonic_note in vocab:
            priors[key_idx, vocab.index(tonic_note)] = 2.0
        if dom_note in vocab:
            priors[key_idx, vocab.index(dom_note)] = 1.5
        priors[key_idx, 0] = 0.3

    priors /= priors.sum(axis=1, keepdims=True)
    return priors


def _learn_transition_matrix(labels, song_ids, n_classes, smoothing=1.0):
    """Learn transition matrix from label sequence."""
    trans = np.full((n_classes, n_classes), smoothing)
    for i in range(1, len(labels)):
        if song_ids[i] == song_ids[i - 1]:
            trans[labels[i - 1], labels[i]] += 1.0
    trans /= trans.sum(axis=1, keepdims=True)
    return trans


def _learn_templates(X, y, n_classes):
    """Compute L2-normalized class centroids."""
    dim = X.shape[1]
    templates = np.zeros((n_classes, dim))
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
    # Load v2 data
    data = np.load(args.data, allow_pickle=True)

    # Load feature config (new: tracks delta/bass composition)
    import json as _json
    _fc_raw = data['feature_config'].item() if 'feature_config' in data else '{}'
    feature_config = _json.loads(_fc_raw) if _fc_raw else {}

    if args.feature_dim == 12:
        X = data['X_12'].astype(np.float32)
    else:
        feat_key = f'X_{args.feature_dim}'
        if feat_key in data:
            X = data[feat_key].astype(np.float32)
        elif args.feature_dim == 36 and 'X_36' in data:
            X = data['X_36'].astype(np.float32)
        else:
            avail = [k for k in data.files if k.startswith('X_')]
            raise ValueError(f"No {feat_key} in data. Available: {avail}")

    # Append third-ratio features if requested
    if args.third_ratio:
        X_12 = data['X_12'].astype(np.float32)
        if 'X_third_ratio' in data:
            X_tr = data['X_third_ratio'].astype(np.float32)
        else:
            # Compute on-the-fly from raw HPCP
            n_beats = len(X_12)
            X_tr = np.zeros((n_beats, 12), dtype=np.float32)
            for root in range(12):
                maj3 = X_12[:, (root + 4) % 12]
                min3 = X_12[:, (root + 3) % 12]
                X_tr[:, root] = (maj3 - min3) / (maj3 + min3 + 1e-8)
        X = np.hstack([X, X_tr])
        print(f"Third-ratio features appended: {X.shape[1]} dims")

    root_labels = data['root_labels'].astype(np.int32)
    quality_labels = data['quality_labels'].astype(np.int32)
    tier1_labels = data['tier1_labels'].astype(np.int32)
    sample_weights = data['sample_weights'].astype(np.float32)
    song_ids = data['song_ids'].astype(np.int32)
    key_indices = data['key_indices']
    filenames = list(data['filenames'])
    provenance = list(data['provenance'])

    if args.no_weights:
        sample_weights = np.ones_like(sample_weights)

    # ── Gold-only filter ──
    if args.gold_only:
        gold_mask = np.array([provenance[s] == 'gold' for s in song_ids])
        X = X[gold_mask]
        root_labels = root_labels[gold_mask]
        quality_labels = quality_labels[gold_mask]
        tier1_labels = tier1_labels[gold_mask]
        sample_weights = sample_weights[gold_mask]
        # Remap song_ids to contiguous range
        old_song_ids = song_ids[gold_mask]
        kept_songs = sorted(set(old_song_ids))
        remap = {old: new for new, old in enumerate(kept_songs)}
        song_ids = np.array([remap[s] for s in old_song_ids], dtype=np.int32)
        key_indices = np.array([key_indices[s] for s in kept_songs])
        filenames = [filenames[s] for s in kept_songs]
        provenance = [provenance[s] for s in kept_songs]
        print(f"Gold-only: kept {len(kept_songs)} songs, {len(X)} beats")

    feat_dim = X.shape[1]
    n_songs = len(filenames)
    n_beats = len(X)
    n_roots = len(ROOT_VOCAB)
    n_quals = len(QUALITY_VOCAB)
    n_tier1 = len(TIER1_VOCAB)

    print(f"Training data: {n_beats} beats from {n_songs} songs")
    print(f"Feature dim: {feat_dim}")
    print(f"Weights: {'enabled' if not args.no_weights else 'disabled'}")
    gold_songs = sum(1 for p in provenance if p == 'gold')
    print(f"Provenance: {gold_songs} gold, {n_songs - gold_songs} silver")

    # ── L2 normalize ──
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    # ── Song-level train/val/test split (70/15/15), stratified by provenance ──
    unique_songs = np.unique(song_ids)
    # Group songs by provenance for stratified split
    gold_songs_list = [s for s in unique_songs if provenance[int(s)] == 'gold']
    silver_songs_list = [s for s in unique_songs if provenance[int(s)] != 'gold']
    rng = np.random.RandomState(42)
    rng.shuffle(gold_songs_list)
    rng.shuffle(silver_songs_list)

    def _split_ids(ids, val_frac=0.15, test_frac=0.15):
        n = len(ids)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        return ids[n_test + n_val:], ids[n_test:n_test + n_val], ids[:n_test]

    g_train, g_val, g_test = _split_ids(np.array(gold_songs_list))
    s_train, s_val, s_test = _split_ids(np.array(silver_songs_list))

    train_songs_arr = np.concatenate([g_train, s_train])
    val_songs_arr = np.concatenate([g_val, s_val])
    test_songs_arr = np.concatenate([g_test, s_test])

    train_song_set = set(train_songs_arr.tolist())
    val_song_set = set(val_songs_arr.tolist())
    test_song_set = set(test_songs_arr.tolist())

    train_mask = np.array([s in train_song_set for s in song_ids])
    val_mask = np.array([s in val_song_set for s in song_ids])
    test_mask = np.array([s in test_song_set for s in song_ids])

    X_train, X_val = X_norm[train_mask], X_norm[val_mask]
    w_train = sample_weights[train_mask]

    roots_train, roots_val = root_labels[train_mask], root_labels[val_mask]
    quals_train, quals_val = quality_labels[train_mask], quality_labels[val_mask]
    t1_train, t1_val = tier1_labels[train_mask], tier1_labels[val_mask]

    n_train_g = sum(1 for s in train_songs_arr if provenance[int(s)] == 'gold')
    n_val_g = sum(1 for s in val_songs_arr if provenance[int(s)] == 'gold')
    n_test_g = sum(1 for s in test_songs_arr if provenance[int(s)] == 'gold')
    print(f"\nSplit (stratified by provenance):")
    print(f"  Train: {len(train_songs_arr)} songs ({n_train_g}G/{len(train_songs_arr)-n_train_g}S), {train_mask.sum()} beats")
    print(f"  Val:   {len(val_songs_arr)} songs ({n_val_g}G/{len(val_songs_arr)-n_val_g}S), {val_mask.sum()} beats")
    print(f"  Test:  {len(test_songs_arr)} songs ({n_test_g}G/{len(test_songs_arr)-n_test_g}S), {test_mask.sum()} beats")

    # ── Minor weight boost ──
    if args.minor_boost > 1.0:
        minor_mask_train = np.isin(quals_train, [2, 5])  # min=2, min7=5
        n_minor = minor_mask_train.sum()
        w_train[minor_mask_train] *= args.minor_boost
        print(f"\n  Minor boost: {n_minor} minor beats × {args.minor_boost:.1f}x weight")

    # ── Pitch-shift augmentation for minor beats ──
    if args.augment_minor:
        minor_mask_train = np.isin(quals_train, [2, 5])  # min=2, min7=5
        X_minor = X_train[minor_mask_train]
        r_minor = roots_train[minor_mask_train]
        q_minor = quals_train[minor_mask_train]
        t1_minor = t1_train[minor_mask_train]
        w_minor = w_train[minor_mask_train]
        n_orig_minor = len(X_minor)

        aug_X, aug_r, aug_q, aug_t1, aug_w = [], [], [], [], []
        for shift in range(1, 12):
            # Circular-shift each 12-dim chroma segment
            X_shifted = np.empty_like(X_minor)
            n_segments = feat_dim // 12
            for seg in range(n_segments):
                s = seg * 12
                X_shifted[:, s:s+12] = np.roll(X_minor[:, s:s+12], shift, axis=1)

            # Shift root labels (1-12 range, 0=N stays 0)
            r_shifted = r_minor.copy()
            nonzero = r_shifted > 0
            r_shifted[nonzero] = (r_shifted[nonzero] - 1 + shift) % 12 + 1

            # Recompute tier1 labels from shifted root + same quality
            t1_shifted = np.array([v2_to_tier1_idx(r, q)
                                   for r, q in zip(r_shifted, q_minor)], dtype=np.int32)

            aug_X.append(X_shifted)
            aug_r.append(r_shifted)
            aug_q.append(q_minor.copy())
            aug_t1.append(t1_shifted)
            aug_w.append(w_minor.copy())

        # Concatenate augmented data with original training set
        X_train = np.vstack([X_train] + aug_X)
        roots_train = np.concatenate([roots_train] + aug_r)
        quals_train = np.concatenate([quals_train] + aug_q)
        t1_train = np.concatenate([t1_train] + aug_t1)
        w_train = np.concatenate([w_train] + aug_w)

        n_aug = sum(len(a) for a in aug_X)
        n_minor_after = np.isin(quals_train, [2, 5]).sum()
        n_major_after = np.isin(quals_train, [1, 3, 4, 6]).sum()
        print(f"\n  Pitch-shift augmentation: +{n_aug} minor beats ({n_orig_minor} × 11 shifts)")
        print(f"  Training set: {len(X_train)} beats (minor={n_minor_after}, major={n_major_after}, ratio={n_minor_after/max(1,n_major_after):.2f})")

    # ── Learn transition matrices ──
    print("\n1. Learning transition matrices...")
    root_trans = _learn_transition_matrix(root_labels, song_ids, n_roots)
    qual_trans = _learn_transition_matrix(quality_labels, song_ids, n_quals)
    tier1_trans = _learn_transition_matrix(tier1_labels, song_ids, n_tier1)

    # Show top root transitions
    for ri in range(min(5, n_roots)):
        top3 = np.argsort(root_trans[ri])[-3:][::-1]
        probs = [f"{ROOT_VOCAB[t]}:{root_trans[ri,t]:.2f}" for t in top3]
        print(f"   {ROOT_VOCAB[ri]:3s} → {', '.join(probs)}")

    # ── Key priors ──
    print("2. Building key priors...")
    root_key_priors = _build_root_key_priors()
    tier1_key_priors = _build_tier1_key_priors()

    # ── Templates for 12-dim features ──
    print("3. Learning templates...")
    if feat_dim == 12:
        templates_12 = _learn_templates(X_norm, tier1_labels, n_tier1)
    else:
        # Use center 12 dims from 36-dim
        X_center = X_norm[:, 12:24]
        c_norms = np.linalg.norm(X_center, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1.0
        X_center_norm = X_center / c_norms
        templates_12 = _learn_templates(X_center_norm, tier1_labels, n_tier1)

    # ── Train classifiers ──
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    rf_params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
    )
    print(f"\nRF params: n_estimators={args.n_estimators}, max_depth={args.max_depth}")

    # Model A/B: Tier-1 flat classifier
    print(f"\n4. Training Tier-1 flat classifier ({feat_dim}-dim)...")
    clf_tier1 = RandomForestClassifier(**rf_params)
    clf_tier1.fit(X_train, t1_train, sample_weight=w_train)
    t1_val_pred = clf_tier1.predict(X_val)
    t1_val_acc = accuracy_score(t1_val, t1_val_pred)
    print(f"   Tier-1 val accuracy: {t1_val_acc:.3f}")

    # Model C: Root classifier
    print(f"5. Training root classifier ({feat_dim}-dim)...")
    clf_root = RandomForestClassifier(**rf_params)
    clf_root.fit(X_train, roots_train, sample_weight=w_train)
    root_val_pred = clf_root.predict(X_val)
    root_val_acc = accuracy_score(roots_val, root_val_pred)
    print(f"   Root val accuracy: {root_val_acc:.3f}")

    present_roots = sorted(set(roots_val))
    root_names = [ROOT_VOCAB[i] for i in present_roots]
    print(classification_report(roots_val, root_val_pred, labels=present_roots,
                                target_names=root_names, zero_division=0))

    # Model C: Quality classifier
    print(f"6. Training quality classifier ({feat_dim}-dim)...")
    clf_quality = RandomForestClassifier(**rf_params)
    clf_quality.fit(X_train, quals_train, sample_weight=w_train)
    qual_val_pred = clf_quality.predict(X_val)
    qual_val_acc = accuracy_score(quals_val, qual_val_pred)
    print(f"   Quality val accuracy: {qual_val_acc:.3f}")

    present_quals = sorted(set(quals_val))
    qual_names = [QUALITY_VOCAB[i] for i in present_quals]
    print(classification_report(quals_val, qual_val_pred, labels=present_quals,
                                target_names=qual_names, zero_division=0))

    # Model D: 3-class quality classifier (N/maj/min)
    print(f"7. Training 3-class quality classifier ({feat_dim}-dim)...")
    q3_map = np.array(QUALITY7_TO_3, dtype=np.int32)
    quals3_train = q3_map[quals_train]
    quals3_val = q3_map[quals_val]
    n_quals3 = len(QUALITY3_VOCAB)

    clf_quality_3 = RandomForestClassifier(**rf_params)
    clf_quality_3.fit(X_train, quals3_train, sample_weight=w_train)
    q3_val_pred = clf_quality_3.predict(X_val)
    q3_val_acc = accuracy_score(quals3_val, q3_val_pred)
    print(f"   Quality-3 val accuracy: {q3_val_acc:.3f}")

    present_q3 = sorted(set(quals3_val))
    q3_names = [QUALITY3_VOCAB[i] for i in present_q3]
    print(classification_report(quals3_val, q3_val_pred, labels=present_q3,
                                target_names=q3_names, zero_division=0))

    # ── Compose v2 predictions and compare ──
    # Convert root+quality back to tier1 for comparison
    v2_tier1_pred = np.array([v2_to_tier1_idx(r, q)
                              for r, q in zip(root_val_pred, qual_val_pred)])
    v2_tier1_acc = accuracy_score(t1_val, v2_tier1_pred)
    print(f"\n   Factorized → Tier-1 val accuracy: {v2_tier1_acc:.3f}")
    print(f"   Direct Tier-1 val accuracy:       {t1_val_acc:.3f}")

    # Major/minor breakdown
    maj_mask = np.array([1 <= yi <= 12 for yi in t1_val])
    min_mask = np.array([13 <= yi <= 24 for yi in t1_val])
    if maj_mask.sum() > 0:
        print(f"\n   Major accuracy (direct):     {accuracy_score(t1_val[maj_mask], t1_val_pred[maj_mask]):.3f}")
        print(f"   Major accuracy (factorized): {accuracy_score(t1_val[maj_mask], v2_tier1_pred[maj_mask]):.3f}")
    if min_mask.sum() > 0:
        print(f"   Minor accuracy (direct):     {accuracy_score(t1_val[min_mask], t1_val_pred[min_mask]):.3f}")
        print(f"   Minor accuracy (factorized): {accuracy_score(t1_val[min_mask], v2_tier1_pred[min_mask]):.3f}")

    # ── Save model ──
    model = {
        # Classifiers
        'clf_tier1': clf_tier1,
        'clf_root': clf_root,
        'clf_quality': clf_quality,
        'clf_quality_3': clf_quality_3,
        # Transitions
        'root_transition': root_trans,
        'quality_transition': qual_trans,
        'tier1_transition': tier1_trans,
        # Key priors
        'root_key_priors': root_key_priors,
        'tier1_key_priors': tier1_key_priors,
        # Templates (12-dim, for fallback)
        'templates_12': templates_12,
        # Vocabularies
        'root_vocab': ROOT_VOCAB,
        'quality_vocab': QUALITY_VOCAB,
        'quality3_vocab': QUALITY3_VOCAB,
        'tier1_vocab': TIER1_VOCAB,
        # Config
        'feature_dim': feat_dim,
        'third_ratio': args.third_ratio,
        'feature_config': feature_config,
        'version': 2,
        # Split info (for evaluation filtering)
        'val_song_ids': sorted(val_song_set),
        'test_song_ids': sorted(test_song_set),
        'train_song_ids': sorted(train_song_set),
        # Metadata
        'metadata': {
            'n_songs': n_songs,
            'n_beats': n_beats,
            'n_gold': gold_songs,
            'n_silver': n_songs - gold_songs,
            'feature_dim': feat_dim,
            'weighted': not args.no_weights,
            'split': '70/15/15',
            'n_train': len(train_songs_arr),
            'n_val': len(val_songs_arr),
            'n_test': len(test_songs_arr),
            'tier1_val_acc': float(t1_val_acc),
            'root_val_acc': float(root_val_acc),
            'quality_val_acc': float(qual_val_acc),
            'quality3_val_acc': float(q3_val_acc),
            'v2_tier1_val_acc': float(v2_tier1_acc),
            'minor_boost': args.minor_boost,
            'augment_minor': args.augment_minor,
            'gold_only': args.gold_only,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
        },
    }

    # ── Optional CRF / decoder tuning ──
    if args.crf:
        from v2.decode import train_crf
        print(f"\n8. Tuning decoder parameters (on validation set)...")
        crf_result = train_crf(
            model, X_val, t1_val, song_ids[val_mask],
            key_indices,
        )
        model.update(crf_result)
        model['metadata']['crf'] = True
        model['metadata']['tuned_params'] = crf_result.get('tuned_params', {})

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(model, f, protocol=4)

    print(f"\nModel saved to {args.out}")
    print(f"File size: {os.path.getsize(args.out) / 1024:.0f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train v2 chord model (root + quality)')
    parser.add_argument('--data', required=True, help='Path to features_v2.npz')
    parser.add_argument('--out', default='models/chord_model_v2.pkl', help='Output model path')
    parser.add_argument('--feature-dim', type=int, default=36,
                        help='Feature dimension (12=v1, 36=3-beat ctx, 60=5-beat ctx)')
    parser.add_argument('--third-ratio', action='store_true',
                        help='Append 12-dim major/minor 3rd interval ratio features')
    parser.add_argument('--no-weights', action='store_true',
                        help='Disable sample weighting (all weights=1.0)')
    parser.add_argument('--minor-boost', type=float, default=1.0,
                        help='Multiply sample weights of minor-quality beats by this factor (default: 1.0)')
    parser.add_argument('--augment-minor', action='store_true',
                        help='Pitch-shift augment minor beats (11 transpositions, 12x minor data)')
    parser.add_argument('--gold-only', action='store_true',
                        help='Train on gold-provenance songs only (drop all silver)')
    parser.add_argument('--n-estimators', type=int, default=300,
                        help='Number of RF trees (default: 300)')
    parser.add_argument('--max-depth', type=int, default=20,
                        help='Max tree depth (default: 20)')
    parser.add_argument('--crf', action='store_true',
                        help='Learn CRF transition weights after RF training')
    parser.add_argument('--crf-reg', type=float, default=0.01,
                        help='CRF L2 regularization (default: 0.01)')
    parser.add_argument('--crf-iter', type=int, default=50,
                        help='CRF max L-BFGS iterations (default: 50)')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
