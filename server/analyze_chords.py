#!/usr/bin/env python3
"""Chord & beat analysis using Essentia HPCP + beat-synchronised Viterbi HMM.

Called as a subprocess by app.py (needs Python 3.12 + essentia-tensorflow).
Reads an audio file path from argv[1], writes JSON to stdout:
  { "chords": [...], "bpm": float, "key": str, "beat_times": [...] }

If a trained model exists at ../training/models/chord_model.pkl, uses it
for chord classification with learned transitions. Otherwise falls back
to hand-crafted templates.

Accuracy pipeline:
  1. High-pass filter removes bass drum / sub-bass from corrupting chroma
  2. HPCP with 4 harmonics (not 8) avoids pitch-class bleeding
  3. Spectral-flatness weighting down-weights percussive frames
  4. Temporal smoothing of HPCP before beat aggregation
  5. Beat-synced median aggregation
  6. Viterbi HMM with learned or template-based emission + transition probs
  7. Post-processing: minimum chord duration filter
"""
import sys
import os
import json
import pickle
import numpy as np
from essentia.standard import (
    MonoLoader, RhythmExtractor2013,
    HPCP, Key,
    FrameGenerator, Windowing, Spectrum, SpectralPeaks,
    Flatness, HighPass, EqualLoudness,
)

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
_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'training', 'models')
_MODEL_PATH_V1 = os.path.join(_MODEL_DIR, 'chord_model.pkl')
_MODEL_PATH_V2 = os.path.join(_MODEL_DIR, 'chord_model_v2.pkl')
_TRAINING_ROOT = os.path.join(os.path.dirname(__file__), '..', 'training')
_TRAINED_MODEL = None


def _load_trained_model():
    """Load trained model if available. Prefers v2, falls back to v1."""
    global _TRAINED_MODEL
    if _TRAINED_MODEL is not None:
        return _TRAINED_MODEL
    # Try v2 first, then v1
    for path in [_MODEL_PATH_V2, _MODEL_PATH_V1]:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    _TRAINED_MODEL = pickle.load(f)
                return _TRAINED_MODEL
            except Exception:
                continue
    return None


def _extract_bass_beat_chroma(audio, beat_frames, n_frames):
    """Extract bass-register HPCP (50-350 Hz) and sync to beats.

    Args:
        audio: pre-loaded audio array (EqualLoudness applied, NO high-pass)
        beat_frames: list of beat frame indices
        n_frames: total number of frames

    Returns:
        (12, n_beats) bass beat chroma array
    """
    windowing = Windowing(type='blackmanharris62', size=FRAME_SIZE)
    spectrum_algo = Spectrum(size=FRAME_SIZE)
    bass_peaks = SpectralPeaks(
        orderBy='magnitude', magnitudeThreshold=0.0001, maxPeaks=20,
        minFrequency=50, maxFrequency=350, sampleRate=SR,
    )
    bass_hpcp_algo = HPCP(
        size=12, referenceFrequency=440, harmonics=2,
        bandPreset=False, minFrequency=50, maxFrequency=350,
        weightType='cosine', nonLinear=True, windowSize=1.0, sampleRate=SR,
    )

    bass_hpcps = []
    for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE,
                                startFromZero=True):
        spec = spectrum_algo(windowing(frame))
        freqs, mags = bass_peaks(spec)
        h = bass_hpcp_algo(freqs, mags)
        bass_hpcps.append(np.roll(h, -3))  # A-ref → C-ref

    bass_hpcps = np.array(bass_hpcps)
    bass_hpcps = _smooth_hpcp(bass_hpcps, window=5)
    return _sync_hpcp_to_beats(bass_hpcps, beat_frames, len(bass_hpcps))


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


def _model_viterbi_decode_v2(model, beat_chroma_cols, key_idx, decode_mode='direct',
                             bass_beat_chroma=None):
    """Decode chords using v2 model with configurable decode strategy.

    Args:
        model: v2 model dict (version=2)
        beat_chroma_cols: (12, n_beats) array
        key_idx: estimated key index (0-11)
        decode_mode: 'direct' (default, safe), 'hybrid', or 'factorized'
        bass_beat_chroma: (12, n_beats) bass HPCP array, or None

    Returns:
        (list of chord name strings, best_key_idx, full_probs)
    """
    sys.path.insert(0, _TRAINING_ROOT)
    from v2.decode import (
        decode_tier1_direct, decode_factorized, decode_hybrid,
        smooth_isolated, get_classifier_probs, soften_transitions,
        viterbi_decode,
    )
    from shared.chord_vocab import TIER1_VOCAB as vocab

    feat_dim = model.get('feature_dim', 12)
    feature_config = model.get('feature_config', {})
    n_beats = beat_chroma_cols.shape[1]

    # Build context HPCP features
    ctx_radius = feature_config.get('context_radius', 1) if feature_config else 1
    if feat_dim >= 36 and not feature_config:
        # Legacy model: infer radius from feature_dim (assumes HPCP-only)
        ctx_radius = (feat_dim // 12 - 1) // 2

    if feat_dim >= 36:
        X = _build_context_features(beat_chroma_cols, radius=ctx_radius)
    else:
        X = beat_chroma_cols.T  # (n_beats, 12)

    # Append optional extra features (must match training order)
    bc = beat_chroma_cols.T  # (n_beats, 12)
    extras = []
    if feature_config.get('has_delta'):
        delta = np.zeros_like(bc, dtype=np.float32)
        delta[1:] = bc[1:] - bc[:-1]
        extras.append(delta)
    if feature_config.get('has_bass') and bass_beat_chroma is not None:
        extras.append(bass_beat_chroma.T)  # (n_beats, 12)
    if extras:
        X = np.concatenate([X] + extras, axis=1)

    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    # Try all 12 key hypotheses (same as v1) — pick best Viterbi score
    best_score = -np.inf
    best_path = None
    best_key = key_idx
    best_probs = None

    n_classes = len(vocab)
    log_prior = np.full(n_classes, -np.log(n_classes))

    for ki in range(12):
        if decode_mode == 'hybrid':
            path, full_probs, log_lik = decode_hybrid(model, X_norm, ki)
        elif decode_mode == 'factorized':
            path, _, _, full_probs, log_lik = decode_factorized(model, X_norm, ki)
        else:  # direct (default)
            path, full_probs, log_lik = decode_tier1_direct(model, X_norm, ki)

        path = smooth_isolated(path)

        # Score: Viterbi log-likelihood (proper key comparison metric)
        if log_lik > best_score:
            best_score = log_lik
            best_path = path
            best_key = ki
            best_probs = full_probs

    return [vocab[ci] for ci in best_path], best_key, best_probs


def _model_viterbi_decode(model, beat_chroma_cols, key_idx):
    """Decode chords using trained model + learned transitions.

    Tries all 12 major key hypotheses and picks the one with the highest
    overall Viterbi log-likelihood.  This makes the system robust to
    wrong key detection (e.g. Bb detected when the true key is F).

    Args:
        model: dict with 'classifier', 'transition_probs', 'key_priors', 'vocab'
        beat_chroma_cols: (12, n_beats) array
        key_idx: int, estimated key from Essentia (used only as hint, all 12 tried)

    Returns:
        (list of chord name strings, best_key_idx, full_probs)
    """
    clf = model['classifier']
    trans = model['transition_probs']
    key_priors = model['key_priors']
    vocab = model['vocab']
    n_classes = len(vocab)

    n_beats = beat_chroma_cols.shape[1]
    X = beat_chroma_cols.T  # (n_beats, 12)

    # L2 normalize (shared across all key hypotheses)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    # Soften transitions (shared across all key hypotheses)
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
    log_prior = np.full(n_classes, -np.log(n_classes))

    best_score = -np.inf
    best_path = None
    best_key = key_idx

    # Classifier takes only HPCP features (12 dims) — no key priors
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X_norm)
        full_probs = np.full((n_beats, n_classes), 1e-10)
        for ci, cls in enumerate(clf.classes_):
            full_probs[:, cls] = probs[:, ci]
    else:
        pred = clf.predict(X_norm)
        full_probs = np.full((n_beats, n_classes), 1e-10)
        for i, p in enumerate(pred):
            full_probs[i, p] = 1.0

    log_emit_base = np.log(np.clip(full_probs.T, 1e-10, None))  # (n_classes, n_beats)

    for ki in range(12):
        # Apply key prior as emission bias only
        log_key_prior = np.log(np.clip(key_priors[ki], 1e-10, None))
        log_emit = log_emit_base + log_key_prior[:, np.newaxis] * 0.3

        # Viterbi
        viterbi = np.full((n_classes, n_beats), -np.inf)
        backptr = np.zeros((n_classes, n_beats), dtype=int)
        viterbi[:, 0] = log_prior + log_emit[:, 0]

        for t in range(1, n_beats):
            for s in range(n_classes):
                scores = viterbi[:, t - 1] + log_trans[:, s]
                bp = int(np.argmax(scores))
                viterbi[s, t] = scores[bp] + log_emit[s, t]
                backptr[s, t] = bp

        # Total score = best final state score
        final_score = float(np.max(viterbi[:, -1]))

        if final_score > best_score:
            best_score = final_score
            best_key = ki
            # Backtrace
            path = np.zeros(n_beats, dtype=int)
            path[-1] = int(np.argmax(viterbi[:, -1]))
            for t in range(n_beats - 2, -1, -1):
                path[t] = backptr[path[t + 1], t + 1]
            best_path = path

    return [vocab[ci] for ci in best_path], best_key, full_probs


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


def analyze(audio_path):
    audio = MonoLoader(filename=audio_path, sampleRate=SR)()

    # ── Pre-processing: equal loudness + high-pass to remove bass drum ──
    audio_eq = EqualLoudness(sampleRate=SR)(audio)
    audio = HighPass(cutoffFrequency=100, sampleRate=SR)(audio_eq)

    # ── Beat tracking (use original audio for rhythm) ──
    audio_raw = MonoLoader(filename=audio_path, sampleRate=SR)()
    rhythm = RhythmExtractor2013(method='multifeature')
    bpm, beats, beats_confidence, _, beats_intervals = rhythm(audio_raw)
    beat_times = beats.tolist()

    # Filter out beats that are too close together (< 0.15s)
    if len(beat_times) > 1:
        filtered = [beat_times[0]]
        for bt in beat_times[1:]:
            if bt - filtered[-1] >= 0.15:
                filtered.append(bt)
        beat_times = filtered

    # ── Compute 12-bin HPCP per frame + spectral flatness ──
    windowing = Windowing(type='blackmanharris62', size=FRAME_SIZE)
    spectrum_algo = Spectrum(size=FRAME_SIZE)
    peaks = SpectralPeaks(
        orderBy='magnitude',
        magnitudeThreshold=0.0001,   # Higher threshold to reject noise
        maxPeaks=40,                 # Fewer peaks = less noise
        minFrequency=80,             # Above bass drum fundamentals
        maxFrequency=4000,           # Below cymbal noise
        sampleRate=SR,
    )
    hpcp_algo = HPCP(
        size=12,
        referenceFrequency=440,
        harmonics=4,                 # Reduced from 8 — less pitch bleeding
        bandPreset=False,
        minFrequency=80,
        maxFrequency=4000,
        weightType='cosine',
        nonLinear=True,
        windowSize=1.0,              # Must be >= 12/size (=1.0 for 12 bins)
        sampleRate=SR,
    )
    flatness_algo = Flatness()

    hpcps_12 = []
    hpcps_native = []  # A-referenced (native Essentia) for Key detection
    flatness_weights = []  # 1.0 = tonal, 0.0 = percussive
    for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE,
                                startFromZero=True):
        spec = spectrum_algo(windowing(frame))
        freqs, mags = peaks(spec)
        h = hpcp_algo(freqs, mags)
        hpcps_native.append(h)
        # Essentia HPCP bin 0 = A (ref 440Hz). Rotate so bin 0 = C.
        hpcps_12.append(np.roll(h, -3))
        # Spectral flatness: low = tonal (good), high = noise/percussive (bad)
        sf = flatness_algo(spec)
        # Convert: tonal frames get weight ~1.0, percussive frames get ~0.2
        w = max(0.2, 1.0 - sf * 2.0)
        flatness_weights.append(w)

    hpcps_12 = np.array(hpcps_12)
    flatness_weights = np.array(flatness_weights)
    n_frames = len(hpcps_12)

    # ── Temporal smoothing of HPCP ──
    hpcps_12 = _smooth_hpcp(hpcps_12, window=5)

    # ── Key detection (uses native A-referenced HPCP) ──
    avg_hpcp_native = np.mean(np.array(hpcps_native), axis=0)
    key_algo = Key(profileType='bgate')
    key_name, scale, key_strength, _ = key_algo(avg_hpcp_native)
    key_str = key_name if scale == 'major' else key_name + 'm'

    # Map key name to NOTES index
    _ENHARMONIC_TO_SHARP = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    }
    kn = _ENHARMONIC_TO_SHARP.get(key_name, key_name)
    sharp_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = sharp_notes.index(kn) if kn in sharp_notes else 0
    if scale == 'minor':
        key_idx = (key_idx + 3) % 12  # relative major

    diatonic = _diatonic_set(key_idx)
    sec_doms = _secondary_dominants(key_idx)

    # ── Beat-synchronise HPCP (weighted by tonal-ness) ──
    beat_frames = [int(round(bt * SR / HOP_SIZE)) for bt in beat_times]
    beat_chroma = _sync_hpcp_to_beats(hpcps_12, beat_frames, n_frames, flatness_weights)

    # Recurrence smoothing disabled — it tends to homogenise chroma toward
    # the dominant chord (e.g. tonic C absorbs Am/F beats in pop songs).
    # Raw beat-level chroma gives the classifier more discriminative features.
    # beat_chroma = _recurrence_smooth(beat_chroma)

    if beat_chroma.shape[1] == 0:
        return {
            'chords': [], 'bpm': round(float(bpm), 1),
            'key': key_str, 'beat_times': [],
        }

    # ── Try trained model first, fall back to templates ──
    trained_model = _load_trained_model()

    if trained_model is not None:
        model_version = trained_model.get('version', 1)
        _IDX_TO_NOTE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

        if model_version >= 2:
            # ── V2 model: hybrid decode by default ──
            # Hybrid = factorized root (best root acc) + direct quality (preserves minors)
            # Override via CHORD_DECODE_MODE env var: 'direct', 'hybrid', 'factorized'
            decode_mode = os.environ.get('CHORD_DECODE_MODE', 'hybrid')

            # Extract bass HPCP if model requires it
            bass_bc = None
            feature_config = trained_model.get('feature_config', {})
            if feature_config and feature_config.get('has_bass'):
                bass_bc = _extract_bass_beat_chroma(audio_eq, beat_frames, n_frames)

            final_path, best_key_idx, full_probs = _model_viterbi_decode_v2(
                trained_model, beat_chroma, key_idx, decode_mode=decode_mode,
                bass_beat_chroma=bass_bc)
            vocab_list = trained_model.get('tier1_vocab', NOTES)
            # Break up long runs using direct probs
            final_path = _break_long_runs(final_path, full_probs,
                                          vocab_list, beat_times, best_key_idx)
        else:
            # ── V1 model: original decode path ──
            final_path, best_key_idx, full_probs = _model_viterbi_decode(
                trained_model, beat_chroma, key_idx)
            final_path = _break_long_runs(final_path, full_probs,
                                          trained_model['vocab'], beat_times,
                                          best_key_idx)
        key_str = _IDX_TO_NOTE[best_key_idx]
    else:
        # ── Template fallback: hand-crafted templates + diatonic bias ──
        KEY_BOOST = 1.0              # Strong diatonic preference
        SEC_DOM_BOOST = 0.3          # Mild boost for secondary dominants
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

        # Promote to extended chords where evidence is strong
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

    # ── Post-processing: remove very short chord segments (< 1 beat) ──
    # Replace isolated single-beat chords surrounded by the same chord
    if len(final_path) >= 3:
        smoothed = list(final_path)
        for i in range(1, len(smoothed) - 1):
            if smoothed[i - 1] == smoothed[i + 1] and smoothed[i] != smoothed[i - 1]:
                smoothed[i] = smoothed[i - 1]
        final_path = smoothed

    # ── Merge consecutive identical chords into timed segments ──
    merged = []
    for i, chord_name in enumerate(final_path):
        start = beat_times[i]
        end = beat_times[i + 1] if i + 1 < len(beat_times) else start + 0.5
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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: analyze_chords.py <audio_path>'}))
        sys.exit(1)

    result = analyze(sys.argv[1])
    print(json.dumps(result))
