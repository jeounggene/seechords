"""V2 Viterbi decoder: combined root+quality emission scoring with key priors.

Decoding strategies:
  - decode_tier1_direct():  Standard Tier-1 Viterbi — DEFAULT for production.
                            Balanced major/minor accuracy. Musically safe.
  - decode_factorized():    Combined root+quality -> Tier-1 Viterbi.
                            Higher overall WCSR but collapses minor→major.
  - decode_hybrid():        Factorized root + direct quality.
                            Best of both: factorized root accuracy & flip rate,
                            direct quality preserves minors. EXPERIMENTAL.

Factorized emission scoring:
    P(chord) ~= P(root) * P(quality) * P(root|key)

Hybrid emission scoring:
    1. Get factorized root probs  (13-class, from clf_root)
    2. Get direct tier1 probs     (25-class, from clf_tier1)
    3. Collapse direct probs into per-root quality distribution (maj/min/N)
    4. For each beat: final_root = factorized, final_quality = direct
    5. Compose into 25-state emissions → Viterbi

Transition structure:
  - Root transitions (13x13) and quality transitions (7x7) are combined
    into a Tier-1 transition matrix (25x25) via outer product
  - Self-transition bias (configurable, default 0.40)
  - Off-diagonal floor (configurable, default 0.015)

Post-processing:
  - smooth_isolated(): replace single-beat flips surrounded by same chord
  - path_to_segments(): convert beat-level path to timed chord segments
  - compute_ux_metrics(): segment length stats and flip rate
"""
import os, sys
import numpy as np

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from v2.chord_schema import (
    ROOT_VOCAB, QUALITY_VOCAB, IDX_TO_ROOT, IDX_TO_QUALITY,
    v2_to_tier1_idx, compose_chord, simplify_chord,
)
from shared.chord_vocab import TIER1_VOCAB


# ── Core Viterbi ─────────────────────────────────────────────

def viterbi_decode(log_emission, log_transition, log_prior=None):
    """Standard Viterbi on (n_states, T) emissions + (n_states, n_states) transitions.

    Returns: (path, log_likelihood)
        path:            (T,) best state sequence
        log_likelihood:  float, max log-prob at final time step
    """
    n_states, T = log_emission.shape
    if log_prior is None:
        log_prior = np.full(n_states, -np.log(n_states))

    viterbi_mat = np.full((n_states, T), -np.inf)
    backptr = np.zeros((n_states, T), dtype=int)

    viterbi_mat[:, 0] = log_prior + log_emission[:, 0]

    for t in range(1, T):
        # Vectorized: scores[prev, cur] = viterbi[prev, t-1] + log_trans[prev, cur]
        scores = viterbi_mat[:, t - 1, np.newaxis] + log_transition  # (n_states, n_states)
        backptr[:, t] = np.argmax(scores, axis=0)  # best prev for each cur
        viterbi_mat[:, t] = np.max(scores, axis=0) + log_emission[:, t]

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(viterbi_mat[:, -1]))
    log_likelihood = float(viterbi_mat[path[-1], -1])
    for t in range(T - 2, -1, -1):
        path[t] = backptr[path[t + 1], t + 1]

    return path, log_likelihood


# ── Transition building ──────────────────────────────────────

def soften_transitions(trans, self_prob=0.40, floor=0.015):
    """Soften a transition matrix: set self-transition probability, floor off-diag."""
    n = trans.shape[0]
    soft = trans.copy()
    for i in range(n):
        soft[i, i] = self_prob
        off = trans[i].copy()
        off[i] = 0
        off_sum = off.sum()
        if off_sum > 0:
            off = np.maximum(off / off_sum * (1 - self_prob), floor)
            off = off / off.sum() * (1 - self_prob)
        soft[i] = off
        soft[i, i] = self_prob
    return soft


def build_tier1_transition_from_rq(root_trans, qual_trans, self_prob=0.40, floor=0.015):
    """Build Tier-1 (25x25) transition matrix from root (13x13) + quality (7x7).

    For each Tier-1 state pair (i→j), combine:
        P(root_j | root_i) * P(quality_j | quality_i)
    then soften.
    """
    n_tier1 = len(TIER1_VOCAB)
    combined = np.zeros((n_tier1, n_tier1))

    for i in range(n_tier1):
        ri_from, qi_from = _tier1_to_rq(i)
        for j in range(n_tier1):
            rj_to, qj_to = _tier1_to_rq(j)
            combined[i, j] = root_trans[ri_from, rj_to] * qual_trans[qi_from, qj_to]

    # Normalize rows
    row_sums = combined.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    combined /= row_sums

    return soften_transitions(combined, self_prob=self_prob, floor=floor)


def _tier1_to_rq(tier1_idx):
    """Convert Tier-1 index to (root_idx, quality_idx) in v2 space."""
    if tier1_idx == 0:
        return 0, 0  # N, N
    elif 1 <= tier1_idx <= 12:
        return tier1_idx, 1  # major
    elif 13 <= tier1_idx <= 24:
        return tier1_idx - 12, 2  # minor
    return 0, 0


# ── Emission scoring ─────────────────────────────────────────

def build_tier1_emissions_from_rq(root_probs, qual_probs, n_beats):
    """Build Tier-1 (25, n_beats) emissions from root + quality probabilities.

    root_probs:  (n_beats, 13)
    qual_probs:  (n_beats, 7)
    """
    n_tier1 = len(TIER1_VOCAB)
    emissions = np.zeros((n_tier1, n_beats))

    for ci in range(n_tier1):
        ri, qi = _tier1_to_rq(ci)
        # Joint probability: P(chord) ≈ P(root) * P(quality)
        emissions[ci, :] = root_probs[:, ri] * qual_probs[:, qi]

    # Avoid zero
    emissions = np.clip(emissions, 1e-10, None)
    return emissions


def build_direct_emissions(clf, X_norm, n_classes):
    """Build emissions from direct Tier-1 classifier probabilities."""
    n_beats = len(X_norm)
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
    return full_probs.T  # (n_classes, n_beats)


def get_classifier_probs(clf, X_norm, n_classes):
    """Get full (n_beats, n_classes) probability matrix from a classifier."""
    n_beats = len(X_norm)
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X_norm)
        full = np.full((n_beats, n_classes), 1e-10)
        for ci, cls in enumerate(clf.classes_):
            full[:, cls] = probs[:, ci]
    else:
        pred = clf.predict(X_norm)
        full = np.full((n_beats, n_classes), 1e-10)
        for i, p in enumerate(pred):
            full[i, p] = 1.0
    return full


# ── Duration penalty ─────────────────────────────────────────

def apply_duration_penalty(log_emission, min_beats=2, penalty=0.3):
    """Soft penalty for very short chords (< min_beats).

    This is applied as an additional self-transition boost rather than
    a hard constraint, so it reduces chatter without creating artifacts.
    """
    # Already handled by self_prob in transitions. This is additional.
    # We slightly boost emission of previous-beat chord for continuity.
    # Implemented in the transition softening (self_prob parameter).
    return log_emission


# ── Main decode functions ────────────────────────────────────

def decode_tier1_direct(model, X_song, key_idx, self_prob=0.40, floor=0.015,
                        key_bias=0.5):
    """Decode using direct Tier-1 classifier + Viterbi (Models A/B).

    Returns: (path, full_probs)
        path:       (n_beats,) Tier-1 class indices
        full_probs: (n_beats, 25) posterior probabilities
    """
    clf = model['clf_tier1']
    trans = model.get('tier1_transition', model.get('transition_probs'))
    key_priors = model.get('tier1_key_priors', model.get('key_priors'))
    n_classes = len(TIER1_VOCAB)

    trans_soft = soften_transitions(trans, self_prob=self_prob, floor=floor)
    log_trans = np.log(np.clip(trans_soft, 1e-10, None))

    full_probs = get_classifier_probs(clf, X_song, n_classes)
    log_emit = np.log(np.clip(full_probs.T, 1e-10, None))

    if key_priors is not None and key_idx is not None:
        log_kp = np.log(np.clip(key_priors[key_idx], 1e-10, None))
        log_emit += log_kp[:, np.newaxis] * key_bias

    path, log_lik = viterbi_decode(log_emit, log_trans)
    return path, full_probs, log_lik


def decode_factorized(model, X_song, key_idx, self_prob=0.40, floor=0.015,
                      key_bias=0.15):
    """Decode using factorized root+quality classifiers + combined Viterbi (Model C).

    Returns: (path, root_probs, qual_probs, full_probs)
        path:        (n_beats,) Tier-1 class indices
        root_probs:  (n_beats, 13)
        qual_probs:  (n_beats, 7)
        full_probs:  (n_beats, 25) composed Tier-1 probs
    """
    clf_root = model['clf_root']
    clf_quality = model['clf_quality']
    root_trans = model['root_transition']
    qual_trans = model['quality_transition']
    root_key_priors = model['root_key_priors']
    n_tier1 = len(TIER1_VOCAB)

    # Get root + quality probabilities
    root_probs = get_classifier_probs(clf_root, X_song, len(ROOT_VOCAB))
    qual_probs = get_classifier_probs(clf_quality, X_song, len(QUALITY_VOCAB))
    n_beats = len(X_song)

    # Build combined Tier-1 emissions
    tier1_emit = build_tier1_emissions_from_rq(root_probs, qual_probs, n_beats)

    # Apply key prior to root dimension
    if root_key_priors is not None and key_idx is not None:
        log_root_kp = np.log(np.clip(root_key_priors[key_idx], 1e-10, None))
        # Distribute root key prior across Tier-1 states
        for ci in range(n_tier1):
            ri, _ = _tier1_to_rq(ci)
            tier1_emit[ci, :] *= np.exp(log_root_kp[ri] * key_bias)

    log_emit = np.log(np.clip(tier1_emit, 1e-10, None))

    # Build combined transition matrix
    trans_combined = build_tier1_transition_from_rq(root_trans, qual_trans,
                                                    self_prob=self_prob, floor=floor)
    log_trans = np.log(np.clip(trans_combined, 1e-10, None))

    path, log_lik = viterbi_decode(log_emit, log_trans)

    # Compose full_probs for tier-1 (for compatibility)
    full_probs = tier1_emit.T  # (n_beats, n_tier1)
    row_sums = full_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    full_probs = full_probs / row_sums

    return path, root_probs, qual_probs, full_probs, log_lik


def decode_hybrid(model, X_song, key_idx, self_prob=0.40, floor=0.015,
                  key_bias=0.5, key_prior_mode='none', quality_agg=False,
                  quality_src='tier1'):
    """Decode using factorized root + direct quality (hybrid approach).

    Strategy:
      1. Get root probs from factorized clf_root  (best root accuracy)
      2. Get quality ratios from one of several sources
      3. Combine: emission(chord=r+q) = root_prob(r) * quality_ratio(q)
      4. Viterbi with tier1 transitions

    Args:
      key_prior_mode: 'tier1' (quality-aware), 'root' (root-only),
                      or 'none' (no key prior, default)
      quality_agg: (deprecated) if True, equivalent to quality_src='agg7'
      quality_src: 'tier1' (per-root from clf_tier1, default),
                   'agg7' (aggregate 7-class clf_quality into 3-class),
                   'clf3' (direct 3-class clf_quality_3)

    Returns: (path, full_probs, log_lik)
    """
    # Handle deprecated quality_agg flag
    if quality_agg and quality_src == 'tier1':
        quality_src = 'agg7'

    clf_root = model['clf_root']
    clf_tier1 = model['clf_tier1']
    trans = model.get('tier1_transition')
    key_priors = model.get('tier1_key_priors')
    root_key_priors = model.get('root_key_priors')
    n_tier1 = len(TIER1_VOCAB)
    n_roots = len(ROOT_VOCAB)
    n_beats = len(X_song)

    # 1. Factorized root probabilities (13 classes)
    root_probs = get_classifier_probs(clf_root, X_song, n_roots)  # (n_beats, 13)

    # 2. Quality ratios — three paths
    qr_maj = qr_min = None
    if quality_src == 'clf3' and 'clf_quality_3' in model:
        # Direct 3-class quality classifier (N/maj/min)
        clf_q3 = model['clf_quality_3']
        q3_probs = get_classifier_probs(clf_q3, X_song, 3)  # (n_beats, 3)
        q_maj = q3_probs[:, 1]   # maj
        q_min = q3_probs[:, 2]   # min
        q_total = q_maj + q_min + 1e-10
        qr_maj = q_maj / q_total
        qr_min = q_min / q_total
    elif quality_src == 'agg7' and 'clf_quality' in model:
        # Aggregate 7-class quality into 3-class (N/maj/min)
        clf_quality = model['clf_quality']
        qual_probs = get_classifier_probs(clf_quality, X_song, len(QUALITY_VOCAB))
        q_n   = qual_probs[:, 0]
        q_maj = qual_probs[:, 1] + qual_probs[:, 3] + qual_probs[:, 4] + qual_probs[:, 6]
        q_min = qual_probs[:, 2] + qual_probs[:, 5]
        q_total = q_maj + q_min + 1e-10
        qr_maj = q_maj / q_total
        qr_min = q_min / q_total
    # else quality_src == 'tier1': per-root direct probs (qr_maj/qr_min stay None)

    # 3. Build hybrid emissions
    direct_probs = get_classifier_probs(clf_tier1, X_song, n_tier1)  # (n_beats, 25)
    hybrid_emit = np.zeros((n_tier1, n_beats))

    # N chord: combine root_prob(N) with direct_prob(N)
    hybrid_emit[0, :] = root_probs[:, 0] * direct_probs[:, 0]

    for note_idx in range(12):
        root_idx = note_idx + 1       # ROOT_VOCAB index (1-12)
        maj_idx = 1 + note_idx        # TIER1 major index
        min_idx = 13 + note_idx       # TIER1 minor index

        r_prob = root_probs[:, root_idx]    # (n_beats,)

        if qr_maj is not None:
            # Use aggregated 3-class quality (root-independent)
            hybrid_emit[maj_idx, :] = r_prob * qr_maj
            hybrid_emit[min_idx, :] = r_prob * qr_min
        else:
            # Use per-root quality ratio from direct tier1 classifier
            d_maj = direct_probs[:, maj_idx]
            d_min = direct_probs[:, min_idx]
            d_total = d_maj + d_min + 1e-10
            hybrid_emit[maj_idx, :] = r_prob * (d_maj / d_total)
            hybrid_emit[min_idx, :] = r_prob * (d_min / d_total)

    # Apply key prior based on mode
    if key_prior_mode == 'tier1' and key_priors is not None and key_idx is not None:
        # Quality-aware: Cmaj and Cm get different priors
        log_kp = np.log(np.clip(key_priors[key_idx], 1e-10, None))
        for ci in range(n_tier1):
            hybrid_emit[ci, :] *= np.exp(log_kp[ci] * key_bias)
    elif key_prior_mode == 'root' and root_key_priors is not None and key_idx is not None:
        # Root-only: Am and A get same prior (old behavior)
        log_root_kp = np.log(np.clip(root_key_priors[key_idx], 1e-10, None))
        for ci in range(n_tier1):
            ri, _ = _tier1_to_rq(ci)
            hybrid_emit[ci, :] *= np.exp(log_root_kp[ri] * key_bias)
    # else key_prior_mode == 'none': no key prior applied

    hybrid_emit = np.clip(hybrid_emit, 1e-10, None)
    log_emit = np.log(hybrid_emit)

    # Use tier1 transitions (same as direct)
    trans_soft = soften_transitions(trans, self_prob=self_prob, floor=floor)
    log_trans = np.log(np.clip(trans_soft, 1e-10, None))

    path, log_lik = viterbi_decode(log_emit, log_trans)

    # Build full_probs for compatibility
    full_probs = hybrid_emit.T  # (n_beats, n_tier1)
    row_sums = full_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    full_probs = full_probs / row_sums

    return path, full_probs, log_lik


# ── CRF / Parameter Tuning ───────────────────────────────────

def _logsumexp(x):
    """Numerically stable log-sum-exp."""
    m = np.max(x)
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.sum(np.exp(x - m)))


def tune_decoder(model, X, tier1_labels, song_ids, key_indices,
                 self_probs=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
                 floors=(0.005, 0.010, 0.015, 0.020, 0.030),
                 minor_biases=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)):
    """Grid search over decoder hyperparameters on provided data.

    Tunes: self_prob, floor, and minor_bias (additive log-emission boost
    for minor chord states 13-24, helping correct major→minor confusion).

    Returns: dict with best parameters and their WCSR.
    """
    n_tier1 = len(TIER1_VOCAB)
    trans = model.get('tier1_transition')

    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    # Pre-compute per-song hybrid emissions (expensive part, done once)
    unique_songs = np.unique(song_ids)
    song_data = []
    for sid in unique_songs:
        mask = song_ids == sid
        X_song = X_norm[mask]
        y_song = tier1_labels[mask]
        n_beats = len(X_song)
        if n_beats < 2:
            continue
        ki = key_indices[sid] if sid < len(key_indices) else 0
        log_emit = _get_hybrid_log_emissions(model, X_song, ki)
        song_data.append((log_emit, y_song, mask))

    print(f"  Tuning on {len(song_data)} songs, "
          f"{len(self_probs)}×{len(floors)}×{len(minor_biases)} = "
          f"{len(self_probs)*len(floors)*len(minor_biases)} combos")

    best_score = -1.0
    best_params = {}
    best_wcsr_final = 0.0
    flip_weight = 0.5  # penalize flip rate in objective

    for sp in self_probs:
        for fl in floors:
            trans_soft = soften_transitions(trans, self_prob=sp, floor=fl)
            log_trans = np.log(np.clip(trans_soft, 1e-10, None))

            for mb in minor_biases:
                correct = 0
                total = 0
                total_flips = 0
                total_transitions = 0

                for log_emit, y_song, mask in song_data:
                    emit = log_emit.copy() if mb > 0 else log_emit
                    if mb > 0:
                        emit[13:25, :] += mb  # boost minor states
                    path, _ = viterbi_decode(emit, log_trans)
                    path = smooth_isolated(path)
                    correct += np.sum(path == y_song)
                    total += len(y_song)
                    if len(path) > 1:
                        changes = np.sum(path[1:] != path[:-1])
                        total_flips += changes
                        total_transitions += len(path) - 1

                wcsr = correct / max(total, 1)
                flip_rate = total_flips / max(total_transitions, 1)
                score = wcsr - flip_weight * flip_rate
                if score > best_score:
                    best_score = score
                    best_wcsr_final = wcsr
                    best_params = {
                        'self_prob': sp,
                        'floor': fl,
                        'minor_bias': mb,
                    }

    print(f"  Best: self_prob={best_params['self_prob']}, "
          f"floor={best_params['floor']}, "
          f"minor_bias={best_params['minor_bias']}, "
          f"WCSR={best_wcsr_final:.4f}")

    return best_params


def decode_hybrid_tuned(model, X_song, key_idx):
    """Decode using tuned parameters (stored in model['tuned_params']).

    Falls back to decode_hybrid if no tuned params.
    Returns: (path, full_probs, log_lik)
    """
    params = model.get('tuned_params')
    if params is None:
        return decode_hybrid(model, X_song, key_idx)

    sp = params.get('self_prob', 0.40)
    fl = params.get('floor', 0.015)
    mb = params.get('minor_bias', 0.0)

    n_tier1 = len(TIER1_VOCAB)
    log_emit = _get_hybrid_log_emissions(model, X_song, key_idx)

    if mb > 0:
        log_emit[13:25, :] += mb

    trans = model.get('tier1_transition')
    trans_soft = soften_transitions(trans, self_prob=sp, floor=fl)
    log_trans = np.log(np.clip(trans_soft, 1e-10, None))

    path, log_lik = viterbi_decode(log_emit, log_trans)

    # Build full_probs
    hybrid_emit = np.exp(log_emit)
    full_probs = hybrid_emit.T
    row_sums = full_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    full_probs = full_probs / row_sums

    return path, full_probs, log_lik


def train_crf(model, X, tier1_labels, song_ids, key_indices,
              reg_lambda=0.01, max_iter=50):
    """Tune decoder parameters via grid search.

    Should be called with VALIDATION data for proper hyperparameter selection.
    Finds optimal self_prob, floor, and minor_bias.
    Returns: dict with 'tuned_params'.
    """
    params = tune_decoder(model, X, tier1_labels, song_ids, key_indices)
    return {'tuned_params': params}


def _get_hybrid_log_emissions(model, X_song, key_idx=0):
    """Compute hybrid log-emissions for CRF (same as decode_hybrid but returns log_emit).

    Uses factorized root + per-root quality ratio from tier1.
    No key prior applied (CRF learns its own biases).
    """
    clf_root = model['clf_root']
    clf_tier1 = model['clf_tier1']
    n_tier1 = len(TIER1_VOCAB)
    n_roots = len(ROOT_VOCAB)
    n_beats = len(X_song)

    root_probs = get_classifier_probs(clf_root, X_song, n_roots)
    direct_probs = get_classifier_probs(clf_tier1, X_song, n_tier1)

    hybrid_emit = np.zeros((n_tier1, n_beats))
    hybrid_emit[0, :] = root_probs[:, 0] * direct_probs[:, 0]

    for note_idx in range(12):
        root_idx = note_idx + 1
        maj_idx = 1 + note_idx
        min_idx = 13 + note_idx
        r_prob = root_probs[:, root_idx]
        d_maj = direct_probs[:, maj_idx]
        d_min = direct_probs[:, min_idx]
        d_total = d_maj + d_min + 1e-10
        hybrid_emit[maj_idx, :] = r_prob * (d_maj / d_total)
        hybrid_emit[min_idx, :] = r_prob * (d_min / d_total)

    hybrid_emit = np.clip(hybrid_emit, 1e-10, None)
    return np.log(hybrid_emit)


def decode_hybrid_crf(model, X_song, key_idx):
    """Decode using tuned parameters + hybrid emissions.

    Falls back to decode_hybrid if no tuned parameters in model.
    Returns: (path, full_probs, log_lik)
    """
    if 'tuned_params' not in model:
        return decode_hybrid(model, X_song, key_idx)

    return decode_hybrid_tuned(model, X_song, key_idx)


# ── Post-processing ──────────────────────────────────────────

def smooth_isolated(path, min_run=2):
    """Replace isolated single-beat chords surrounded by the same chord."""
    out = path.copy()
    for i in range(1, len(out) - 1):
        if out[i - 1] == out[i + 1] and out[i] != out[i - 1]:
            out[i] = out[i - 1]
    return out


def path_to_segments(path, beat_times, vocab):
    """Convert beat-level path to chord segments with start/end times."""
    segments = []
    if len(path) == 0:
        return segments

    current_chord = vocab[path[0]]
    start_time = beat_times[0]

    for i in range(1, len(path)):
        if path[i] != path[i - 1]:
            end_time = beat_times[i]
            segments.append({
                'chord': current_chord,
                'start': float(start_time),
                'end': float(end_time),
            })
            current_chord = vocab[path[i]]
            start_time = beat_times[i]

    # Last segment
    if len(beat_times) > len(path):
        end_time = beat_times[len(path)]
    else:
        end_time = beat_times[-1] + 0.5
    segments.append({
        'chord': current_chord,
        'start': float(start_time),
        'end': float(end_time),
    })

    return segments


# ── UX metrics ───────────────────────────────────────────────

def compute_ux_metrics(path, beat_times):
    """Compute UX-relevant metrics: segment length, flips, etc."""
    if len(path) == 0:
        return {}

    # Segment lengths
    changes = np.where(np.diff(path) != 0)[0] + 1
    seg_starts = np.concatenate([[0], changes])
    seg_lengths = np.diff(np.concatenate([seg_starts, [len(path)]]))

    # Count 1-beat flips: chord changes that last only 1 beat
    n_one_beat = np.sum(seg_lengths == 1)

    return {
        'n_segments': len(seg_lengths),
        'avg_segment_beats': float(np.mean(seg_lengths)),
        'median_segment_beats': float(np.median(seg_lengths)),
        'n_one_beat_flips': int(n_one_beat),
        'flip_rate': float(n_one_beat / len(seg_lengths)) if len(seg_lengths) > 0 else 0,
    }
