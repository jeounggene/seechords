"""Dataset and batching for Transformer chord training.

Loads features_v2.npz and provides:
    - 24-dim features per beat (12 HPCP + 12 bass HPCP)
    - Song-level sequences chunked into fixed-length windows
    - Same train/val/test split as RF pipeline (seed=42, stratified)
    - Gold-only filtering option
    - Class weights for quality loss
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _compute_third_ratio(beat_chroma):
    """Major/minor 3rd energy ratio for each possible root. Pure numpy.

    Returns (n_beats, 12) float32 in [-1, 1].
    Positive = major 3rd dominant, negative = minor 3rd dominant.
    """
    n_beats = len(beat_chroma)
    ratios = np.zeros((n_beats, 12), dtype=np.float32)
    for root in range(12):
        maj3 = beat_chroma[:, (root + 4) % 12]
        min3 = beat_chroma[:, (root + 3) % 12]
        ratios[:, root] = (maj3 - min3) / (maj3 + min3 + 1e-8)
    return ratios


def load_data(npz_path, gold_only=False, feature_dim=24):
    """Load features and labels from NPZ.

    Args:
        npz_path: path to features_v2.npz
        gold_only: if True, keep only gold-provenance beats
        feature_dim: 24 (HPCP + bass) or 48 (full X_48)

    Returns dict with:
        X:              (n_beats, feature_dim) float32
        root_labels:    (n_beats,) int32
        quality3_labels:(n_beats,) int32 — 3-class (N/maj/min)
        tier1_labels:   (n_beats,) int32
        song_ids:       (n_beats,) int32
        key_indices:    (n_songs,) int32
        provenance:     (n_songs,) str
        filenames:      (n_songs,) str
    """
    data = np.load(npz_path, allow_pickle=True)

    X_12 = data['X_12'].astype(np.float32)      # (n_beats, 12)
    # v2.prepare_data stores context (+ optional delta/bass) as X_36, X_48, or X_60.
    # Transformer paths expect a 48-wide block (36-dim temporal ctx + 12-dim bass HPCP).
    if 'X_48' in data:
        X_48 = data['X_48'].astype(np.float32)
    elif 'X_36' in data:
        x36 = data['X_36'].astype(np.float32)
        n = x36.shape[0]
        pad = np.zeros((n, 12), dtype=np.float32)
        X_48 = np.hstack([x36, pad])
    elif 'X_60' in data:
        x60 = data['X_60'].astype(np.float32)
        if x60.shape[1] < 48:
            raise ValueError(f'X_60 has width {x60.shape[1]}; need >= 48 for Transformer')
        X_48 = x60[:, :48].astype(np.float32)
    else:
        raise KeyError(
            'features npz must contain X_48, X_36, or X_60. '
            'Rebuild with: python -m v2.prepare_data ... --bass '
            '(recommended for 48-dim bass columns).'
        )

    if feature_dim == 24:
        bass = X_48[:, 36:48]                     # (n_beats, 12) — last 12 cols
        X = np.hstack([X_12, bass])               # (n_beats, 24)
    elif feature_dim == 48:
        X = X_48                                   # (n_beats, 48) — context + bass
    elif feature_dim == 60:
        X = np.hstack([X_12, X_48])               # (n_beats, 60) — HPCP + context + bass
    elif feature_dim == 72:
        if 'X_third_ratio' in data:
            X_tr = data['X_third_ratio'].astype(np.float32)
        else:
            X_tr = _compute_third_ratio(X_12)       # compute on-the-fly from raw HPCP
        X_hpcp = np.hstack([X_12, X_48])           # (n_beats, 60) — HPCP + context + bass
        # Normalize HPCP portion before concatenating third-ratio features
        norms = np.linalg.norm(X_hpcp, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_hpcp = X_hpcp / norms
        X = np.hstack([X_hpcp, X_tr])              # (n_beats, 72)
    elif feature_dim == 144:
        if 'X_cqt' in data:
            X = data['X_cqt'].astype(np.float32)      # (n_beats, 144) — beat-synced CQT
        else:
            raise KeyError(
                'features npz must contain X_cqt for feature_dim=144. '
                'Rebuild with: python -m v2.prepare_data ... --cqt'
            )
    else:
        raise ValueError(f"feature_dim must be 24, 48, 60, 72, or 144, got {feature_dim}")

    if feature_dim not in (72, 144):
        # L2 normalize per beat (72-dim handles normalization above)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

    root_labels = data['root_labels'].astype(np.int64)
    quality_labels = data['quality_labels'].astype(np.int64)  # 7-class v2
    tier1_labels = data['tier1_labels'].astype(np.int64)
    sample_weights = data['sample_weights'].astype(np.float32)
    song_ids = data['song_ids'].astype(np.int64)
    key_indices = data['key_indices'].astype(np.int64)
    provenance = list(data['provenance'])
    filenames = list(data['filenames'])

    # Quality 7 → 3 mapping: N=0, maj=1, min=2
    QUALITY7_TO_3 = np.array([0, 1, 2, 1, 1, 2, 1], dtype=np.int64)
    quality3_labels = QUALITY7_TO_3[quality_labels]

    # Gold-only filter
    if gold_only:
        gold_beat_mask = np.array([provenance[s] == 'gold' for s in song_ids])
        X = X[gold_beat_mask]
        root_labels = root_labels[gold_beat_mask]
        quality_labels = quality_labels[gold_beat_mask]
        quality3_labels = quality3_labels[gold_beat_mask]
        tier1_labels = tier1_labels[gold_beat_mask]
        sample_weights = sample_weights[gold_beat_mask]
        old_song_ids = song_ids[gold_beat_mask]

        # Remap song IDs to contiguous range
        kept_songs = sorted(set(old_song_ids.tolist()))
        remap = {old: new for new, old in enumerate(kept_songs)}
        song_ids = np.array([remap[s] for s in old_song_ids], dtype=np.int64)
        key_indices = np.array([key_indices[s] for s in kept_songs], dtype=np.int64)
        provenance = [provenance[s] for s in kept_songs]
        filenames = [filenames[s] for s in kept_songs]

    return {
        'X': X,
        'root_labels': root_labels,
        'quality_labels': quality_labels,
        'quality3_labels': quality3_labels,
        'tier1_labels': tier1_labels,
        'sample_weights': sample_weights,
        'song_ids': song_ids,
        'key_indices': key_indices,
        'provenance': provenance,
        'filenames': filenames,
    }


def split_songs(data, val_frac=0.15, test_frac=0.15, seed=42):
    """Song-level train/val/test split, stratified by provenance.

    Exactly matches the split logic in v2/train.py.
    Returns: (train_songs, val_songs, test_songs) as sets of song IDs.
    """
    song_ids = data['song_ids']
    provenance = data['provenance']
    unique_songs = sorted(set(song_ids.tolist()))

    gold_songs = [s for s in unique_songs if provenance[int(s)] == 'gold']
    silver_songs = [s for s in unique_songs if provenance[int(s)] != 'gold']

    rng = np.random.RandomState(seed)
    rng.shuffle(gold_songs)
    rng.shuffle(silver_songs)

    def _split_ids(ids):
        n = len(ids)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        return ids[n_test + n_val:], ids[n_test:n_test + n_val], ids[:n_test]

    g_train, g_val, g_test = _split_ids(np.array(gold_songs))
    s_train, s_val, s_test = _split_ids(np.array(silver_songs))

    train_set = set(g_train.tolist()) | set(s_train.tolist())
    val_set = set(g_val.tolist()) | set(s_val.tolist())
    test_set = set(g_test.tolist()) | set(s_test.tolist())

    return train_set, val_set, test_set


def compute_quality7_weights(data, song_set):
    """Inverse-frequency class weights for 7-class quality CE (train songs only)."""
    mask = np.array([s in song_set for s in data['song_ids']])
    q = data['quality_labels'][mask]
    counts = np.bincount(q, minlength=7).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    inv = inv * (7.0 / inv.sum())
    return torch.FloatTensor(inv)


def compute_quality3_weights(data, song_set, minor_boost=4.0):
    """Compute class weights for quality3 loss.

    Uses a targeted approach: set minor weight to compensate for class imbalance
    between major and minor, boosted by minor_boost.

    Args:
        data: dict from load_data
        song_set: set of song IDs to compute weights from (training set)
        minor_boost: additional multiplier on minor weight beyond parity

    Returns: torch.FloatTensor of shape (3,) — weights for [N, maj, min]
    """
    mask = np.array([s in song_set for s in data['song_ids']])
    q3 = data['quality3_labels'][mask]
    counts = np.bincount(q3, minlength=3).astype(np.float64)
    counts[counts == 0] = 1.0
    # Minor gets boosted relative to major to compensate for class imbalance
    # N gets moderate weight (it's very rare but we don't want it to dominate)
    n_maj = counts[1]
    n_min = counts[2]
    ratio = n_maj / max(n_min, 1)  # e.g., ~4x
    weights = np.array([1.0, 1.0, ratio * minor_boost / ratio], dtype=np.float64)
    # simplifies to: weights = [1.0, 1.0, minor_boost]
    weights = np.array([1.0, 1.0, minor_boost], dtype=np.float64)
    return torch.FloatTensor(weights)


def _pitch_shift(X, roots, q3, t1, shift, feature_dim):
    """Pitch-shift features and labels by `shift` semitones.

    HPCP features are groups of 12 dims that can be circularly rolled.
    Root labels are shifted accordingly. Quality stays the same.
    """
    if shift == 0:
        return X, roots, q3, t1

    X_shifted = X.copy()
    if feature_dim == 144:
        # CQT: roll by bins_per_semitone (2 bins per semitone for 24 bins/octave)
        X_shifted = np.roll(X, shift * 2, axis=1)
    else:
        # HPCP: roll each group of 12 chroma dims
        for start in range(0, feature_dim, 12):
            end = min(start + 12, feature_dim)
            if end - start == 12:
                X_shifted[:, start:end] = np.roll(X[:, start:end], shift, axis=1)

    # Shift root: 0=N stays, 1-12 shift circularly
    roots_shifted = roots.copy()
    nonzero = roots > 0
    roots_shifted[nonzero] = ((roots[nonzero] - 1 + shift) % 12) + 1

    # Recompute tier1 from shifted root + same quality
    t1_shifted = t1.copy()
    for i in range(len(t1)):
        if t1[i] == 0:
            continue  # N
        elif t1[i] <= 12:
            # major: root index = t1[i]
            t1_shifted[i] = roots_shifted[i]
        else:
            # minor: root index = t1[i] - 12
            t1_shifted[i] = roots_shifted[i] + 12

    return X_shifted, roots_shifted, q3, t1_shifted


class ChordWindowDataset(Dataset):
    """Dataset of fixed-length beat windows for training.

    Each item is a window of consecutive beats from one song.
    Supports pitch-shift augmentation (shift=0..11) to multiply data 12x.

    Args:
        data:       dict from load_data
        song_set:   set of song IDs to include
        window:     window length in beats
        stride:     stride between windows
        augment_shifts: list of pitch shifts to apply (e.g. range(12) for all)
    """

    def __init__(self, data, song_set, window=128, stride=64, augment_shifts=None):
        self.window = window
        self.feature_dim = data['X'].shape[1]
        self.windows = []

        if augment_shifts is None:
            augment_shifts = [0]

        for sid in sorted(song_set):
            mask = data['song_ids'] == sid
            X = data['X'][mask]
            roots = data['root_labels'][mask]
            q3 = data['quality3_labels'][mask]
            q7 = data['quality_labels'][mask]
            t1 = data['tier1_labels'][mask]
            sw = data['sample_weights'][mask]
            ki = data['key_indices'][sid]
            n_beats = len(X)

            if n_beats == 0:
                continue

            # Create overlapping windows
            starts = list(range(0, max(1, n_beats - window + 1), stride))
            if starts[-1] + window < n_beats:
                starts.append(max(0, n_beats - window))

            for shift in augment_shifts:
                X_s, roots_s, q3_s, t1_s = _pitch_shift(
                    X, roots, q3, t1, shift, self.feature_dim
                )
                # quality7 invariant under pitch shift (same as q3)
                q7_s = q7.copy()

                for start in starts:
                    end = min(start + window, n_beats)
                    w_X = X_s[start:end]
                    w_roots = roots_s[start:end]
                    w_q3 = q3_s[start:end]
                    w_q7 = q7_s[start:end]
                    w_t1 = t1_s[start:end]
                    w_sw = sw[start:end].astype(np.float32)
                    w_sw_mean = float(w_sw.mean())
                    n_real = len(w_X)

                    if n_real < window:
                        pad_len = window - n_real
                        w_X = np.pad(w_X, ((0, pad_len), (0, 0)), mode='constant')
                        w_roots = np.pad(w_roots, (0, pad_len), constant_values=0)
                        w_q3 = np.pad(w_q3, (0, pad_len), constant_values=0)
                        w_q7 = np.pad(w_q7, (0, pad_len), constant_values=0)
                        w_t1 = np.pad(w_t1, (0, pad_len), constant_values=0)
                        w_sw = np.pad(w_sw, (0, pad_len), constant_values=0.0)

                    self.windows.append((
                        w_X.astype(np.float32),
                        w_roots.astype(np.int64),
                        w_q3.astype(np.int64),
                        w_q7.astype(np.int64),
                        w_t1.astype(np.int64),
                        int(ki),
                        n_real,
                        w_sw_mean,
                        w_sw.astype(np.float32),
                    ))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        X, roots, q3, q7, t1, ki, n_real, sw_mean, sw_beats = self.windows[idx]
        # Build padding mask: True where padded
        mask = np.zeros(self.window, dtype=bool)
        mask[n_real:] = True
        return {
            'features': torch.from_numpy(X),
            'root_labels': torch.from_numpy(roots),
            'quality3_labels': torch.from_numpy(q3),
            'quality_labels': torch.from_numpy(q7),
            'tier1_labels': torch.from_numpy(t1),
            'key_idx': ki,
            'padding_mask': torch.from_numpy(mask),
            'n_real': n_real,
            'sample_weight': sw_mean,
            'beat_weights': torch.from_numpy(sw_beats),
        }


def get_song_data(data, song_set):
    """Get per-song feature arrays for full-song inference.

    Returns: list of dicts with keys: features, root_labels, quality3_labels,
             tier1_labels, key_idx, song_id, name, provenance
    """
    songs = []
    for sid in sorted(song_set):
        mask = data['song_ids'] == sid
        X = data['X'][mask].astype(np.float32)
        if len(X) == 0:
            continue
        songs.append({
            'features': X,
            'root_labels': data['root_labels'][mask],
            'quality_labels': data['quality_labels'][mask],
            'quality3_labels': data['quality3_labels'][mask],
            'tier1_labels': data['tier1_labels'][mask],
            'key_idx': int(data['key_indices'][sid]),
            'song_id': int(sid),
            'name': data['filenames'][sid],
            'provenance': data['provenance'][sid],
        })
    return songs
