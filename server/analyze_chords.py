#!/usr/bin/env python3
"""Chord & beat analysis pipeline.

Called as a subprocess by app.py (needs Python 3.12 + essentia-tensorflow).
Reads an audio file path from argv[1], writes JSON to stdout:
  { "chords": [...], "bpm": float, "key": str, "beat_times": [...] }

Inference cascade (first success wins):
  1. BTC transformer (170-class, CQT features) -- primary
  2. Freeze5 Transformer+CRF (25-class, HPCP features) -- fallback
  3. Template HMM (hand-crafted chroma templates) -- last resort

Beat detection: Beat This! transformer (ISMIR 2024) with Essentia fallback.
"""
import sys
import os
import json
import numpy as np


def _essentia():
    """Lazy-load Essentia to avoid importing TensorFlow on the BTC path."""
    import essentia.standard as es
    return es

FRAME_SIZE = 8192   # Larger frame for better low-frequency resolution
HOP_SIZE   = 2048
SR         = 44100

# ── Chord template definitions ──────────────────────────────
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

SIMPLE_INTERVALS = [
    ('',  [0, 4, 7]),
    ('m', [0, 3, 7]),
]

EXTENDED_INTERVALS = [
    ('7',    [0, 4, 7, 10]),
    ('m7',   [0, 3, 7, 10]),
    ('maj7', [0, 4, 7, 11]),
    ('sus2', [0, 2, 7]),
    ('sus4', [0, 5, 7]),
    ('dim',  [0, 3, 6]),
    ('aug',  [0, 4, 8]),
]

ALL_INTERVALS = SIMPLE_INTERVALS + EXTENDED_INTERVALS

ROOT_WEIGHT  = 1.8   # Stronger root emphasis
THIRD_WEIGHT = 1.0
FIFTH_WEIGHT = 1.3


def _build_templates(intervals_list):
    templates = {}
    for i, note in enumerate(NOTES):
        for chord_type, intervals in intervals_list:
            t = np.zeros(12)
            for k, iv in enumerate(intervals):
                if k == 0:
                    w = ROOT_WEIGHT
                elif iv in (7, 6, 8):  # fifth (or dim/aug fifth)
                    w = FIFTH_WEIGHT
                else:
                    w = THIRD_WEIGHT
                t[(i + iv) % 12] = w
            t /= np.linalg.norm(t)
            templates[note + chord_type] = t
    return templates


SIMPLE_TEMPLATES = _build_templates(SIMPLE_INTERVALS)
SIMPLE_CHORDS    = list(SIMPLE_TEMPLATES.keys())
SIMPLE_MATRIX    = np.array([SIMPLE_TEMPLATES[c] for c in SIMPLE_CHORDS])

CHORD_TEMPLATES  = _build_templates(ALL_INTERVALS)
ALL_CHORDS       = list(CHORD_TEMPLATES.keys())
TEMPLATE_MATRIX  = np.array([CHORD_TEMPLATES[c] for c in ALL_CHORDS])

_EXTENDED_TO_SIMPLE = {}
for i, note in enumerate(NOTES):
    for suffix, _ in EXTENDED_INTERVALS:
        parent = note + ('m' if 'm' in suffix and suffix != 'maj7' else '')
        _EXTENDED_TO_SIMPLE[note + suffix] = parent

_DIATONIC_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_DIATONIC_QUALITIES = ['', 'm', 'm', '', '', 'm', 'dim']


def _diatonic_set(key_idx):
    s = set()
    for offset, quality in zip(_DIATONIC_INTERVALS, _DIATONIC_QUALITIES):
        note = NOTES[(key_idx + offset) % 12]
        s.add(note + quality)
    return s


def _secondary_dominants(key_idx):
    """Return dominant 7th chords that resolve to diatonic chords (V7/x)."""
    sd = set()
    for offset in _DIATONIC_INTERVALS:
        # The V7 of each diatonic chord is a major chord a fifth above it
        target = (key_idx + offset) % 12
        dom_root = (target + 7) % 12  # a fifth above = V of target
        sd.add(NOTES[dom_root] + '7')
        sd.add(NOTES[dom_root])       # also the triad
    return sd


# ── Viterbi HMM ─────────────────────────────────────────────
def _viterbi_decode(template_matrix, chord_list, beat_chroma,
                    self_prob=0.95, emission_bias=None):
    n_chords = len(chord_list)
    n_beats  = beat_chroma.shape[1]

    norms = np.linalg.norm(beat_chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    sim = template_matrix @ (beat_chroma / norms)

    log_emit = np.log(np.clip(sim, 1e-10, None))
    if emission_bias is not None:
        log_emit += emission_bias

    switch_prob = (1.0 - self_prob) / max(n_chords - 1, 1)
    log_self   = np.log(self_prob)
    log_switch = np.log(switch_prob)

    viterbi = np.full((n_chords, n_beats), -np.inf)
    backptr = np.zeros((n_chords, n_beats), dtype=int)
    viterbi[:, 0] = np.log(1.0 / n_chords) + log_emit[:, 0]

    for t in range(1, n_beats):
        prev = viterbi[:, t - 1]
        for s in range(n_chords):
            candidates = prev + log_switch
            candidates[s] = prev[s] + log_self
            bp = int(np.argmax(candidates))
            viterbi[s, t] = candidates[bp] + log_emit[s, t]
            backptr[s, t] = bp

    path = np.zeros(n_beats, dtype=int)
    path[-1] = int(np.argmax(viterbi[:, -1]))
    for t in range(n_beats - 2, -1, -1):
        path[t] = backptr[path[t + 1], t + 1]

    return [chord_list[ci] for ci in path]


# ── Temporal smoothing ───────────────────────────────────────
def _smooth_hpcp(hpcps, window=5):
    """Apply causal moving-average smoothing to HPCP matrix (n_frames x 12)."""
    if len(hpcps) < window:
        return hpcps
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(hpcps)
    for b in range(12):
        smoothed[:, b] = np.convolve(hpcps[:, b], kernel, mode='same')
    return smoothed


def _recurrence_smooth(beat_chroma, threshold=0.92, self_weight=2.0):
    """Smooth beat-level HPCP using a recurrence (self-similarity) matrix.

    Beats in repeated sections (verse 1 ≈ verse 2) get averaged together,
    reinforcing the true harmonic content and suppressing noise.

    Args:
        beat_chroma: (12, n_beats) column-oriented HPCP array
        threshold: cosine similarity threshold for recurrence links
        self_weight: extra weight for the beat itself vs recurrent peers

    Returns:
        (12, n_beats) smoothed HPCP array
    """
    n = beat_chroma.shape[1]
    if n < 4:
        return beat_chroma

    # Transpose to row-per-beat for easier similarity computation
    bc = beat_chroma.T  # (n, 12)
    norms = np.linalg.norm(bc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = bc / norms

    sim = normed @ normed.T  # (n, n)

    # Zero out diagonal band to avoid local smoothing (already done by temporal filter)
    for d in range(-2, 3):
        diag_idx = np.arange(max(0, d), min(n, n + d))
        off_idx = diag_idx - d
        sim[diag_idx, off_idx] = 0.0

    smoothed = np.zeros_like(bc)
    for i in range(n):
        recurrent = np.where(sim[i] >= threshold)[0]
        if len(recurrent) == 0:
            smoothed[i] = bc[i]
        else:
            weights = sim[i, recurrent]
            total_w = self_weight + weights.sum()
            smoothed[i] = (self_weight * bc[i] +
                           (bc[recurrent] * weights[:, np.newaxis]).sum(axis=0)
                           ) / total_w

    return smoothed.T  # back to (12, n)


# ── Beat-synchronised HPCP aggregation ───────────────────────
def _sync_hpcp_to_beats(hpcps_12, beat_frames, n_frames, weights=None):
    """Aggregate frame-level 12-bin HPCP to beat-level using weighted median."""
    n_beats = len(beat_frames)
    beat_chroma = np.zeros((12, n_beats))
    for bi in range(n_beats):
        start = beat_frames[bi]
        end = beat_frames[bi + 1] if bi + 1 < n_beats else n_frames
        start = min(start, n_frames - 1)
        end = min(end, n_frames)
        if end <= start:
            end = start + 1
        segment = hpcps_12[start:end]
        if weights is not None:
            w = weights[start:end]
            # Weighted average (more robust than trying weighted median)
            w_sum = w.sum()
            if w_sum > 0:
                beat_chroma[:, bi] = (segment * w[:, np.newaxis]).sum(axis=0) / w_sum
            else:
                beat_chroma[:, bi] = np.median(segment, axis=0)
        else:
            beat_chroma[:, bi] = np.median(segment, axis=0)
    return beat_chroma


# ── Trained model support ────────────────────────────────────
def _training_root():
    """Directory that contains ``v2/`` on ``sys.path`` (repo or Docker layout)."""
    here = os.path.dirname(os.path.abspath(__file__))
    parent_training = os.path.normpath(os.path.join(here, '..', 'training'))
    if os.path.isdir(os.path.join(parent_training, 'v2')):
        return parent_training
    flat = os.path.join(here, 'training')
    if os.path.isdir(os.path.join(flat, 'v2')):
        return flat
    return parent_training


def _model_dir():
    override = os.environ.get('SEECHORDS_MODEL_DIR')
    if override:
        return override
    return os.path.normpath(os.path.join(_training_root(), 'models'))


_MODEL_DIR = _model_dir()
_TRAINING_ROOT = _training_root()

_FREEZE5_MODEL = None
_FREEZE5_DEVICE = None


def _freeze5_checkpoint_path():
    return os.environ.get(
        'FREEZE5_CHECKPOINT',
        os.path.join(_MODEL_DIR, 'chord_transformer_crf_freeze5.pt'),
    )


def _load_freeze5_crf():
    """Lazy-load Transformer+CRF freeze5 checkpoint (CPU)."""
    global _FREEZE5_MODEL, _FREEZE5_DEVICE
    if _FREEZE5_MODEL is not None:
        return _FREEZE5_MODEL, _FREEZE5_DEVICE
    import torch
    sys.path.insert(0, _TRAINING_ROOT)
    from v2.transformer_model import ChordTransformerCRF

    path = _freeze5_checkpoint_path()
    device = torch.device('cpu')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp = ckpt['hyperparams']
    model = ChordTransformerCRF(
        input_dim=hp['input_dim'],
        d_model=hp['d_model'],
        nhead=hp['nhead'],
        num_layers=hp['num_layers'],
        d_ff=hp['d_ff'],
        dropout=hp['dropout'],
        crf_self_bias=hp.get('crf_self_bias', 2.0),
        emission_temp=hp.get('emission_temp', 1.0),
        emission_dropout=hp.get('emission_dropout', 0.0),
        emission_noise_std=hp.get('emission_noise_std', 0.0),
        em_emission_bias=hp.get('em_emission_bias', 0.0),
        emission_mode=hp.get('emission_mode', 'direct'),
        n_qualities=hp.get('n_qualities', 3),
        use_key_aux=hp.get('key_aux_weight', 0.0) > 0.0,
        key_condition_quality=hp.get('key_condition_quality', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    _FREEZE5_MODEL = model
    _FREEZE5_DEVICE = device
    return model, device


def _freeze5_build_features(beat_chroma_cols):
    """Build 60-dim features from (12, n_beats) beat chroma (matches training feature_dim=60)."""
    bc = beat_chroma_cols.T.astype(np.float32)  # (n_beats, 12)
    ctx60 = _build_context_features(beat_chroma_cols, radius=2)  # (n_beats, 60)
    x48 = ctx60[:, :48]
    x = np.hstack([bc, x48])  # (n_beats, 60)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


def _freeze5_get_emissions(model, x_np, device):
    """Compute CRF emissions from features. Returns (emissions_tensor, softmax_probs_np)."""
    import torch
    with torch.no_grad():
        xt = torch.from_numpy(x_np).unsqueeze(0).to(device)
        h = model._encode(xt)
        tier1_logits = model.tier1_head(h)
        root_logits = model.root_head(h)

        if model.emission_mode == 'hybrid':
            emissions = model._tier1_emissions_hybrid(root_logits, tier1_logits)
        else:
            emissions = model._tier1_emissions_direct(tier1_logits)

        emission_probs = torch.softmax(emissions[0], dim=-1).cpu().numpy()
    return emissions, emission_probs


def _freeze5_viterbi_with_bias(model, emissions, key_bias_logits):
    """Run CRF Viterbi on emissions + per-class key bias. Returns (path, log_score)."""
    import torch
    with torch.no_grad():
        biased = emissions.clone()
        if key_bias_logits is not None:
            biased = biased + key_bias_logits.unsqueeze(0).unsqueeze(0)

        paths = model.crf.decode(biased)
        path = np.array(paths[0], dtype=np.int64)

        # Viterbi log-score of the best path
        e = biased[0]
        crf = model.crf
        v = crf.start_transitions + e[0]
        for t in range(1, e.shape[0]):
            scores = v.unsqueeze(1) + crf.transitions
            best_scores, _ = scores.max(dim=0)
            v = best_scores + e[t]
        v = v + crf.end_transitions
        log_score = float(v.max())

    return path, log_score


def _build_key_bias_logits(key_idx, vocab, bias_strength=0.5):
    """Build per-class emission bias for a given key hypothesis.

    Diatonic chords get +bias_strength, non-diatonic get 0, N gets 0.
    """
    import torch
    diatonic = _diatonic_set(key_idx)
    sec_doms = _secondary_dominants(key_idx)
    bias = torch.zeros(len(vocab))
    for ci, name in enumerate(vocab):
        if name == 'N':
            continue
        elif name in diatonic:
            bias[ci] = bias_strength
        elif name in sec_doms:
            bias[ci] = bias_strength * 0.3
    return bias


def _freeze5_decode_chords(beat_chroma_cols, key_idx, beat_times):
    """Full freeze5 inference: 12-key search with diatonic emission bias, long-run breaking.

    Computes CRF emissions once, then tests all 12 key hypotheses by adding
    a diatonic bias to the emissions before Viterbi decoding. Picks the key
    with the highest Viterbi score.

    Args:
        beat_chroma_cols: (12, n_beats) column-oriented beat HPCP.
        key_idx: estimated key index from Essentia (0-11, relative major).
        beat_times: list of beat times in seconds.

    Returns:
        (list of chord name strings, best_key_idx)
    """
    import torch
    sys.path.insert(0, _TRAINING_ROOT)
    from v2.decode import smooth_isolated
    from shared.chord_vocab import TIER1_VOCAB as vocab

    model, device = _load_freeze5_crf()

    # Build features and compute emissions once (key-independent)
    x = _freeze5_build_features(beat_chroma_cols)
    emissions, emission_probs = _freeze5_get_emissions(model, x, device)

    best_score = -np.inf
    best_path = None
    best_key = key_idx

    for ki in range(12):
        key_bias = _build_key_bias_logits(ki, vocab, bias_strength=0.5).to(device)
        path, score = _freeze5_viterbi_with_bias(model, emissions, key_bias)

        if score > best_score:
            best_score = score
            best_path = path.copy()
            best_key = ki

    best_path = smooth_isolated(best_path)
    chord_names = [vocab[int(i)] for i in best_path]

    # Long-run breaking using original emission probabilities
    if emission_probs is not None and beat_times:
        chord_names = _break_long_runs(chord_names, emission_probs, vocab,
                                       beat_times, best_key)

    return chord_names, best_key


# ── Beat This! transformer beat tracker (ISMIR 2024) ────────
_BEAT_THIS_MODEL = None


def _detect_beats_beat_this(audio_path):
    """Detect beats using Beat This! transformer model (ISMIR 2024).

    Returns (beat_times, downbeat_times) as Python lists of float seconds.
    """
    global _BEAT_THIS_MODEL
    if _BEAT_THIS_MODEL is None:
        from beat_this.inference import File2Beats
        _BEAT_THIS_MODEL = File2Beats(
            checkpoint_path="small0", device="cpu", dbn=False)
    beats, downbeats = _BEAT_THIS_MODEL(audio_path)
    return beats.tolist(), downbeats.tolist()


# ── BTC (pre-trained 170-class model) inference ─────────────
_BTC_MODEL = None
_BTC_DEVICE = None
_BTC_MEAN = None
_BTC_STD = None


def _btc_checkpoint_path():
    return os.environ.get(
        'BTC_CHECKPOINT',
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'btc_model', 'btc_model_best.pth'),
    )


def _load_btc_model():
    """Lazy-load BTC model checkpoint (CPU)."""
    global _BTC_MODEL, _BTC_DEVICE, _BTC_MEAN, _BTC_STD
    if _BTC_MODEL is not None:
        return _BTC_MODEL, _BTC_DEVICE, _BTC_MEAN, _BTC_STD
    import torch
    from btc_model.btc_model import BTC_model

    path = _btc_checkpoint_path()
    device = torch.device('cpu')
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Handle both checkpoint formats (large_voca vs best)
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    norm = ckpt.get('normalization', {})
    if isinstance(norm, dict) and 'mean' in norm:
        _BTC_MEAN = float(norm['mean'])
        _BTC_STD = float(norm['std'])
    else:
        mean_val = ckpt.get('mean')
        std_val = ckpt.get('std')
        _BTC_MEAN = float(mean_val) if mean_val is not None else -2.37
        _BTC_STD = float(std_val) if std_val is not None else 1.96

    model = BTC_model()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _BTC_MODEL = model
    _BTC_DEVICE = device
    return model, device, _BTC_MEAN, _BTC_STD


def _extract_cqt(audio_path, sr=22050, hop_length=2048, n_bins=144,
                 bins_per_octave=24):
    """Extract CQT spectrogram using librosa (log-magnitude, matching ChordMini).

    Returns (n_frames, 144) float32 array.
    """
    import librosa
    y, _ = librosa.load(audio_path, sr=sr)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length,
                       n_bins=n_bins, bins_per_octave=bins_per_octave,
                       fmin=librosa.note_to_hz('C1'))
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)


def _gaussian_smooth_logits(logits_np, kernel_size=9):
    """Apply 1D Gaussian smoothing to frame-level logits (ChordMini-style).

    Each of the 170 class channels is convolved independently with a normalized
    Gaussian kernel (sigma = kernel_size / 6, per the three-sigma rule).
    Boundary frames use replicate padding.
    """
    if logits_np.shape[0] < kernel_size:
        return logits_np
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 6.0
    half = kernel_size // 2
    x = np.arange(kernel_size, dtype=np.float32) - half
    gauss = np.exp(-0.5 * (x / sigma) ** 2)
    gauss /= gauss.sum()

    n_frames, n_classes = logits_np.shape
    padded = np.pad(logits_np, ((half, half), (0, 0)), mode='edge')
    smoothed = np.zeros_like(logits_np)
    for c in range(n_classes):
        smoothed[:, c] = np.convolve(padded[:, c], gauss, mode='valid')
    return smoothed


def _majority_filter(preds, kernel_size=9):
    """Replace each frame's prediction with the majority class in a local window."""
    n = len(preds)
    if n < kernel_size:
        return preds
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    padded = np.pad(preds, (half, half), mode='edge')
    filtered = np.empty_like(preds)
    for i in range(n):
        window = padded[i:i + kernel_size]
        labels, counts = np.unique(window, return_counts=True)
        max_count = counts.max()
        candidates = labels[counts == max_count]
        filtered[i] = preds[i] if preds[i] in candidates else candidates[0]
    return filtered


def _btc_decode_chords(audio_path, beat_times):
    """Run BTC model on CQT features with ChordMini-style inference pipeline.

    Improvements over basic argmax:
    - 75% overlap sliding windows (stride = seq_len * 0.25)
    - Logit accumulation across overlapping windows
    - Gaussian temporal smoothing on averaged logits (kernel=9, sigma=1.5)
    - Majority filter on final frame predictions
    - Per-beat majority vote to sync frames to beats
    """
    import torch
    from btc_model.vocab import btc_idx_to_display

    model, device, mean, std = _load_btc_model()

    cqt = _extract_cqt(audio_path)
    cqt = (cqt - mean) / max(std, 1e-6)

    n_frames = cqt.shape[0]
    if n_frames == 0:
        return None

    seq_len = 108
    stride = max(1, int(seq_len * 0.25))  # 75% overlap (ChordMini-style)

    logit_sum = np.zeros((n_frames, 170), dtype=np.float32)
    logit_count = np.zeros(n_frames, dtype=np.float32)

    with torch.no_grad():
        pos = 0
        while pos < n_frames:
            end = min(pos + seq_len, n_frames)
            chunk = cqt[pos:end]
            actual_len = chunk.shape[0]

            if actual_len < seq_len:
                pad = np.zeros((seq_len - actual_len, 144), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            x = torch.from_numpy(chunk).unsqueeze(0).to(device)
            out = model(x)  # (1, seq_len, 170)
            logits = out[0, :actual_len].cpu().numpy()

            logit_sum[pos:pos + actual_len] += logits
            logit_count[pos:pos + actual_len] += 1.0

            pos += stride
            if pos >= n_frames:
                break

    logit_count[logit_count == 0] = 1.0
    avg_logits = logit_sum / logit_count[:, np.newaxis]

    # Gaussian temporal smoothing on logits before argmax
    avg_logits = _gaussian_smooth_logits(avg_logits, kernel_size=9)

    frame_preds = avg_logits.argmax(axis=1).astype(np.int64)

    # Majority filter to clean up isolated spurious predictions
    frame_preds = _majority_filter(frame_preds, kernel_size=9)

    # Sync frame-level predictions to beat times via majority vote
    hop_dur = 2048 / 22050.0  # ~0.093s per CQT frame
    beat_chords = []
    for bi in range(len(beat_times)):
        t_start = beat_times[bi]
        t_end = beat_times[bi + 1] if bi + 1 < len(beat_times) else t_start + 0.5
        f_start = max(0, int(round(t_start / hop_dur)))
        f_end = min(n_frames, int(round(t_end / hop_dur)))
        if f_end <= f_start:
            f_end = f_start + 1
        if f_start >= n_frames:
            beat_chords.append('N')
            continue
        f_end = min(f_end, n_frames)
        segment = frame_preds[f_start:f_end]
        if len(segment) == 0:
            beat_chords.append('N')
            continue
        counts = np.bincount(segment, minlength=170)
        winner = int(counts.argmax())
        beat_chords.append(btc_idx_to_display(winner))

    return beat_chords



def _build_context_features(beat_chroma_cols, radius=1):
    """Build contextual features: concatenate (2*radius+1) beats of 12-dim HPCP.

    beat_chroma_cols: (12, n_beats) column-oriented HPCP.
    radius: 1 → 36-dim, 2 → 60-dim.
    Returns: (n_beats, (2*radius+1)*12) feature matrix.
    """
    bc = beat_chroma_cols.T  # (n_beats, 12)
    n_beats = len(bc)
    width = 2 * radius + 1
    feat_dim = width * 12
    X = np.zeros((n_beats, feat_dim), dtype=np.float32)
    for i in range(n_beats):
        for k, offset in enumerate(range(-radius, radius + 1)):
            j = max(0, min(n_beats - 1, i + offset))
            X[i, k * 12:(k + 1) * 12] = bc[j]
    return X




def _break_long_runs(path, full_probs, vocab, beat_times, key_idx, max_dur=4.0):
    """Break up chord runs that persist longer than max_dur seconds.

    For each run exceeding max_dur, find the beat(s) where the current
    chord's classifier probability is weakest and insert the best
    diatonic alternative chord there.  Repeats until no run exceeds max_dur.
    """
    result = list(path)
    n = len(result)
    if n < 4:
        return result

    chord_to_idx = {c: i for i, c in enumerate(vocab)}

    # Only allow insertions of diatonic chords (in-key)
    diatonic = _diatonic_set(key_idx)
    diatonic_indices = {chord_to_idx[c] for c in diatonic if c in chord_to_idx}

    for _iteration in range(10):  # safety cap
        changed = False
        i = 0
        while i < n:
            j = i + 1
            while j < n and result[j] == result[i]:
                j += 1
            run_len = j - i

            start_t = beat_times[i] if i < len(beat_times) else 0
            end_t = beat_times[j] if j < len(beat_times) else beat_times[-1] + 0.5
            run_dur = end_t - start_t

            if run_dur > max_dur and run_len >= 4:
                current_chord = result[i]
                ci = chord_to_idx.get(current_chord, -1)

                if ci >= 0:
                    inner_start = i + 1
                    inner_end = j - 1

                    if inner_end > inner_start:
                        best_split = -1
                        best_alt_score = -1.0
                        best_alt_idx = -1

                        for bi in range(inner_start, inner_end):
                            bp = full_probs[bi].copy()
                            bp[ci] = 0
                            # Only consider diatonic alternatives
                            for idx in range(len(bp)):
                                if idx not in diatonic_indices:
                                    bp[idx] = 0
                            alt_idx = int(np.argmax(bp))
                            alt_prob = bp[alt_idx]
                            cur_prob = full_probs[bi, ci]

                            if alt_prob < 0.03:
                                continue

                            score = alt_prob / max(cur_prob, 0.01)
                            if score > best_alt_score:
                                best_alt_score = score
                                best_split = bi
                                best_alt_idx = alt_idx

                        if best_split >= 0 and best_alt_idx >= 0:
                            best_alt_chord = vocab[best_alt_idx]
                            # Expand from split point while alternative
                            # has decent probability
                            lo = best_split
                            hi = best_split
                            while lo > inner_start and full_probs[lo - 1, best_alt_idx] > 0.08:
                                lo -= 1
                            while hi < inner_end - 1 and full_probs[hi + 1, best_alt_idx] > 0.08:
                                hi += 1
                            for bi in range(lo, hi + 1):
                                result[bi] = best_alt_chord
                            changed = True
            i = j

        if not changed:
            break

    return result


def _gate_leading_silence_beats(audio_eq, beat_times, chord_labels, sr, rel_frac=0.08):
    """Set chord to N for beats from the start until RMS exceeds rel_frac × track peak.

    The chord model can label beats while the mix is still inaudible (silent intro,
    video lead-in). Only **leading** beats are changed so quiet verses mid-song are
    not wiped. Uses equal-loudness audio so bass in the first bar still counts.
    """
    if not beat_times or not chord_labels or len(chord_labels) != len(beat_times):
        return chord_labels
    n = len(beat_times)
    n_audio = len(audio_eq)
    rms_list = []
    for bi in range(n):
        t0 = float(beat_times[bi])
        t1 = float(beat_times[bi + 1]) if bi + 1 < n else t0 + 0.5
        i0 = int(max(0, min(n_audio - 1, round(t0 * sr))))
        i1 = int(max(i0 + 1, min(n_audio, round(t1 * sr))))
        seg = audio_eq[i0:i1]
        rms_list.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    peak = max(rms_list) if rms_list else 0.0
    if peak < 1e-10:
        return chord_labels
    rel_frac = max(0.02, min(0.5, float(rel_frac)))
    thresh = rel_frac * peak
    out = list(chord_labels)
    for bi in range(n):
        if rms_list[bi] >= thresh:
            break
        out[bi] = 'N'
    return out


def _uniformize_beat_times(beat_times, bpm):
    """Replace beat times with a fixed grid t_i = t0 + i * (60/bpm).

    Chord labels stay index-aligned; only timestamps change. Used so the client
    timeline scrolls at constant speed (equal px per beat ↔ equal seconds per beat).

    Set UNIFORM_BEAT_GRID=0 to keep detector-native irregular beat times.
    """
    if not beat_times or len(beat_times) < 2:
        return list(beat_times) if beat_times else []
    if os.environ.get('UNIFORM_BEAT_GRID', '1').lower() in ('0', 'false', 'no'):
        return list(beat_times)
    bpm = float(max(40.0, min(300.0, float(bpm))))
    interval = 60.0 / bpm
    t0 = float(beat_times[0])
    n = len(beat_times)
    return [round(t0 + i * interval, 3) for i in range(n)]


def _postprocess_and_format(final_path, beat_times, bpm, key_str, audio_path,
                            skip_silence_gate=False):
    """Shared post-processing: silence gate, smoothing, merge segments."""
    if not skip_silence_gate and \
       os.environ.get('SILENCE_GATE', '1').lower() not in ('0', 'false', 'no'):
        try:
            import librosa
            y_gate, _sr = librosa.load(audio_path, sr=SR, mono=True)
            try:
                rel = float(os.environ.get('SILENCE_GATE_FRAC', '0.08'))
            except ValueError:
                rel = 0.08
            final_path = _gate_leading_silence_beats(
                y_gate, beat_times, final_path, SR, rel_frac=rel)
        except Exception:
            pass

    # Remove isolated single-beat chords surrounded by the same chord
    if len(final_path) >= 3:
        smoothed = list(final_path)
        for i in range(1, len(smoothed) - 1):
            if smoothed[i - 1] == smoothed[i + 1] and smoothed[i] != smoothed[i - 1]:
                smoothed[i] = smoothed[i - 1]
        final_path = smoothed

    beat_times = _uniformize_beat_times(beat_times, bpm)
    iv_tail = (beat_times[1] - beat_times[0]) if len(beat_times) >= 2 else 0.5

    # Merge consecutive identical chords into timed segments
    merged = []
    for i, chord_name in enumerate(final_path):
        start = beat_times[i]
        end = beat_times[i + 1] if i + 1 < len(beat_times) else start + iv_tail
        if merged and merged[-1]['chord'] == chord_name:
            merged[-1]['end'] = round(end, 3)
        else:
            merged.append({'chord': chord_name, 'start': round(start, 3), 'end': round(end, 3)})

    return {
        'chords':     merged,
        'bpm':        round(float(bpm), 1),
        'key':        key_str,
        'beat_times': [round(b, 3) for b in beat_times],
    }


def _estimate_key_from_chroma(beat_chroma_cols):
    """Estimate key from beat-level chroma using profile correlation (no Essentia)."""
    profile_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    profile_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    avg_chroma = np.mean(beat_chroma_cols, axis=1)
    if np.max(avg_chroma) < 1e-8:
        return 'C', 0
    best_score, best_key, best_scale = -1, 0, 'major'
    for shift in range(12):
        rolled = np.roll(avg_chroma, -shift)
        corr_maj = float(np.corrcoef(rolled, profile_major)[0, 1])
        corr_min = float(np.corrcoef(rolled, profile_minor)[0, 1])
        if corr_maj > best_score:
            best_score, best_key, best_scale = corr_maj, shift, 'major'
        if corr_min > best_score:
            best_score, best_key, best_scale = corr_min, shift, 'minor'
    key_str = NOTES[best_key] if best_scale == 'major' else NOTES[best_key] + 'm'
    return key_str, best_key


def analyze(audio_path):
    # ── Fast path: Beat This! + BTC (no Essentia needed) ──
    use_btc = os.environ.get('USE_BTC', '1').lower() not in ('0', 'false', 'no')
    use_beat_this = os.environ.get('USE_BEAT_THIS', '1').lower() not in ('0', 'false', 'no')
    btc_result = None

    if use_btc and use_beat_this and os.path.isfile(_btc_checkpoint_path()):
        try:
            print('[SeeChords] Fast path: Beat This! + BTC', flush=True)
            beat_times, _downbeats = _detect_beats_beat_this(audio_path)
            print(f'[SeeChords] Beat This! found {len(beat_times)} beats', flush=True)
            if len(beat_times) >= 2:
                bpm = 60.0 / np.median(np.diff(beat_times))
            else:
                bpm = 120.0
            btc_chords = _btc_decode_chords(audio_path, beat_times)
            if btc_chords is not None and len(btc_chords) == len(beat_times):
                import librosa
                y_key, _ = librosa.load(audio_path, sr=22050, mono=True)
                chroma = librosa.feature.chroma_cqt(y=y_key, sr=22050)
                key_str, _ = _estimate_key_from_chroma(chroma)
                btc_result = _postprocess_and_format(
                    btc_chords, beat_times, bpm, key_str, audio_path)
                print(f'[SeeChords] Fast path success: {len(btc_result["chords"])} segments',
                      flush=True)
        except Exception as e:
            import traceback
            print(f'[SeeChords] Fast path failed: {e}', flush=True)
            traceback.print_exc()
            btc_result = None

    if btc_result is not None:
        return btc_result

    # ── Full path: Essentia (lazy-loaded) for HPCP/key/beats/fallback models ──
    es = _essentia()

    audio = es.MonoLoader(filename=audio_path, sampleRate=SR)()
    audio_eq = es.EqualLoudness(sampleRate=SR)(audio)
    audio = es.HighPass(cutoffFrequency=100, sampleRate=SR)(audio_eq)

    # ── Beat tracking ──
    beat_times = None

    if use_beat_this:
        try:
            beat_times, _downbeats = _detect_beats_beat_this(audio_path)
            if len(beat_times) >= 2:
                bpm = 60.0 / np.median(np.diff(beat_times))
            else:
                bpm = 120.0
        except Exception:
            beat_times = None

    if beat_times is None:
        audio_raw = es.MonoLoader(filename=audio_path, sampleRate=SR)()
        rhythm = es.RhythmExtractor2013(method='multifeature')
        bpm, beats, beats_confidence, _, beats_intervals = rhythm(audio_raw)
        beat_times = beats.tolist()

        try:
            beat_offset = float(os.environ.get('BEAT_OFFSET_MS', '50')) / 1000.0
        except ValueError:
            beat_offset = 0.05
        if beat_offset != 0:
            beat_times = [max(0.0, t + beat_offset) for t in beat_times]

        if len(beat_times) > 1:
            filtered = [beat_times[0]]
            for bt in beat_times[1:]:
                if bt - filtered[-1] >= 0.15:
                    filtered.append(bt)
            beat_times = filtered

    # ── Compute 12-bin HPCP per frame + spectral flatness ──
    windowing = es.Windowing(type='blackmanharris62', size=FRAME_SIZE)
    spectrum_algo = es.Spectrum(size=FRAME_SIZE)
    peaks = es.SpectralPeaks(
        orderBy='magnitude',
        magnitudeThreshold=0.0001,
        maxPeaks=40,
        minFrequency=80,
        maxFrequency=4000,
        sampleRate=SR,
    )
    hpcp_algo = es.HPCP(
        size=12,
        referenceFrequency=440,
        harmonics=4,
        bandPreset=False,
        minFrequency=80,
        maxFrequency=4000,
        weightType='cosine',
        nonLinear=True,
        windowSize=1.0,
        sampleRate=SR,
    )
    flatness_algo = es.Flatness()

    hpcps_12 = []
    hpcps_native = []
    flatness_weights = []
    for frame in es.FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE,
                                   startFromZero=True):
        spec = spectrum_algo(windowing(frame))
        freqs, mags = peaks(spec)
        h = hpcp_algo(freqs, mags)
        hpcps_native.append(h)
        hpcps_12.append(np.roll(h, -3))
        sf = flatness_algo(spec)
        w = max(0.2, 1.0 - sf * 2.0)
        flatness_weights.append(w)

    hpcps_12 = np.array(hpcps_12)
    flatness_weights = np.array(flatness_weights)
    n_frames = len(hpcps_12)

    hpcps_12 = _smooth_hpcp(hpcps_12, window=5)

    # ── Key detection (uses native A-referenced HPCP) ──
    avg_hpcp_native = np.mean(np.array(hpcps_native), axis=0)
    key_algo = es.Key(profileType='bgate')
    key_name, scale, key_strength, _ = key_algo(avg_hpcp_native)
    key_str = key_name if scale == 'major' else key_name + 'm'

    _ENHARMONIC_TO_SHARP = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    }
    kn = _ENHARMONIC_TO_SHARP.get(key_name, key_name)
    sharp_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = sharp_notes.index(kn) if kn in sharp_notes else 0
    if scale == 'minor':
        key_idx = (key_idx + 3) % 12

    diatonic = _diatonic_set(key_idx)
    sec_doms = _secondary_dominants(key_idx)

    # ── Beat-synchronise HPCP ──
    beat_frames = [int(round(bt * SR / HOP_SIZE)) for bt in beat_times]
    beat_chroma = _sync_hpcp_to_beats(hpcps_12, beat_frames, n_frames, flatness_weights)

    if beat_chroma.shape[1] == 0:
        return {
            'chords': [], 'bpm': round(float(bpm), 1),
            'key': key_str, 'beat_times': [],
        }

    # ── Try BTC, then freeze5, then template fallback ──
    use_freeze5 = os.environ.get('USE_FREEZE5', '1').lower() not in ('0', 'false', 'no')
    final_path = None

    if use_btc and os.path.isfile(_btc_checkpoint_path()):
        try:
            btc_chords = _btc_decode_chords(audio_path, beat_times)
            if btc_chords is not None and len(btc_chords) == len(beat_times):
                final_path = btc_chords
        except Exception:
            final_path = None

    if final_path is None and use_freeze5 and os.path.isfile(_freeze5_checkpoint_path()):
        try:
            final_path, best_key = _freeze5_decode_chords(beat_chroma, key_idx, beat_times)
            _IDX_TO_NOTE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
            key_str = _IDX_TO_NOTE[best_key]
        except Exception:
            final_path = None

    if final_path is None:
        # ── Template fallback: hand-crafted templates + diatonic bias ──
            KEY_BOOST = 1.0
            SEC_DOM_BOOST = 0.3
            simple_bias = np.zeros((len(SIMPLE_CHORDS), 1))
            for ci, name in enumerate(SIMPLE_CHORDS):
                if name in diatonic:
                    simple_bias[ci, 0] = KEY_BOOST
                elif name in sec_doms:
                    simple_bias[ci, 0] = SEC_DOM_BOOST

            simple_path = _viterbi_decode(
                SIMPLE_MATRIX, SIMPLE_CHORDS, beat_chroma,
                self_prob=0.95, emission_bias=simple_bias,
            )

            PROMOTE_THRESH = 0.15
            norms = np.linalg.norm(beat_chroma, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            bc_normed = beat_chroma / norms
            full_sim = TEMPLATE_MATRIX @ bc_normed

            final_path = []
            for bi, simple_name in enumerate(simple_path):
                simple_score = full_sim[ALL_CHORDS.index(simple_name), bi]
                best_ext_name  = simple_name
                best_ext_score = simple_score
                for ext_name, parent in _EXTENDED_TO_SIMPLE.items():
                    if parent == simple_name:
                        ext_score = full_sim[ALL_CHORDS.index(ext_name), bi]
                        if ext_score > best_ext_score + PROMOTE_THRESH:
                            best_ext_name  = ext_name
                            best_ext_score = ext_score
                final_path.append(best_ext_name)

    # Leading silence gate (uses equal-loudness audio from Essentia path)
    if os.environ.get('SILENCE_GATE', '1').lower() not in ('0', 'false', 'no'):
        try:
            rel = float(os.environ.get('SILENCE_GATE_FRAC', '0.08'))
        except ValueError:
            rel = 0.08
        final_path = _gate_leading_silence_beats(
            audio_eq, beat_times, final_path, SR, rel_frac=rel)

    return _postprocess_and_format(final_path, beat_times, bpm, key_str, audio_path,
                                   skip_silence_gate=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: analyze_chords.py <audio_path>'}))
        sys.exit(1)

    result = analyze(sys.argv[1])
    print(json.dumps(result))
