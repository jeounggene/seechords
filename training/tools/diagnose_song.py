#!/usr/bin/env python3
"""Diagnose chord detection for a specific song.
Shows key detection, per-beat emission probabilities, and final Viterbi path.

Usage:
    python diagnose_song.py <audio_path>
"""
import sys, os, pickle
import numpy as np

# Add server dir to path so we can reuse analyze_chords machinery
_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(_TRAINING_ROOT, '..', 'server')
sys.path.insert(0, SERVER_DIR)

from analyze_chords import (
    SR, FRAME_SIZE, HOP_SIZE,
    _smooth_hpcp, _sync_hpcp_to_beats, _diatonic_set, _secondary_dominants,
    _load_trained_model,
)
from essentia.standard import (
    MonoLoader, EqualLoudness, HighPass, RhythmExtractor2013,
    Windowing, Spectrum, SpectralPeaks, HPCP, Flatness, Key, FrameGenerator,
)

def diagnose(audio_path):
    audio = MonoLoader(filename=audio_path, sampleRate=SR)()
    audio = EqualLoudness(sampleRate=SR)(audio)
    audio = HighPass(cutoffFrequency=100, sampleRate=SR)(audio)

    audio_raw = MonoLoader(filename=audio_path, sampleRate=SR)()
    rhythm = RhythmExtractor2013(method='multifeature')
    bpm, beats, beats_confidence, _, beats_intervals = rhythm(audio_raw)
    beat_times = beats.tolist()

    if len(beat_times) > 1:
        filtered = [beat_times[0]]
        for bt in beat_times[1:]:
            if bt - filtered[-1] >= 0.15:
                filtered.append(bt)
        beat_times = filtered

    windowing = Windowing(type='blackmanharris62', size=FRAME_SIZE)
    spectrum_algo = Spectrum(size=FRAME_SIZE)
    peaks = SpectralPeaks(orderBy='magnitude', magnitudeThreshold=0.0001,
                          maxPeaks=40, minFrequency=80, maxFrequency=4000, sampleRate=SR)
    hpcp_algo = HPCP(size=12, referenceFrequency=440, harmonics=4,
                     bandPreset=False, minFrequency=80, maxFrequency=4000,
                     weightType='cosine', nonLinear=True, windowSize=1.0, sampleRate=SR)
    flatness_algo = Flatness()

    hpcps_12 = []
    hpcps_native = []
    flatness_weights = []
    for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=True):
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

    avg_hpcp_native = np.mean(np.array(hpcps_native), axis=0)
    key_algo = Key(profileType='bgate')
    key_name, scale, key_strength, _ = key_algo(avg_hpcp_native)
    key_str = key_name if scale == 'major' else key_name + 'm'

    _ENHARMONIC_TO_SHARP = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
    kn = _ENHARMONIC_TO_SHARP.get(key_name, key_name)
    sharp_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = sharp_notes.index(kn) if kn in sharp_notes else 0
    if scale == 'minor':
        key_idx = (key_idx + 3) % 12

    print(f"Key detected: {key_str} (key_idx={key_idx})")
    print(f"BPM: {bpm:.1f}")
    print(f"Beats: {len(beat_times)}")
    print()

    beat_frames = [int(round(bt * SR / HOP_SIZE)) for bt in beat_times]
    beat_chroma = _sync_hpcp_to_beats(hpcps_12, beat_frames, n_frames, flatness_weights)

    # Show average chroma profile
    avg_chroma = beat_chroma.mean(axis=1)
    NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    print("Average beat chroma profile:")
    for i, n in enumerate(NOTES):
        bar = '#' * int(avg_chroma[i] * 50)
        print(f"  {n:>3s}: {avg_chroma[i]:.4f} {bar}")
    print()

    model = _load_trained_model()
    if model is None:
        print("ERROR: No trained model found")
        return

    clf = model['classifier']
    trans = model['transition_probs']
    key_priors = model['key_priors']
    vocab = model['vocab']
    n_classes = len(vocab)

    n_beats = beat_chroma.shape[1]
    X = beat_chroma.T
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    # Classifier takes HPCP only (12 dims) — no key features
    probs = clf.predict_proba(X_norm)
    full_probs = np.full((n_beats, n_classes), 1e-10)
    for ci, cls in enumerate(clf.classes_):
        full_probs[:, cls] = probs[:, ci]

    # Show key priors for this key
    print(f"Key priors for key_idx={key_idx}:")
    kp = key_priors[key_idx]
    top_kp = np.argsort(kp)[-10:][::-1]
    for idx in top_kp:
        print(f"  {vocab[idx]}: {kp[idx]:.4f}")
    print()

    # Show raw classifier outputs for first 20 beats
    print("Raw classifier top-3 predictions per beat (first 40 beats):")
    for b in range(min(40, n_beats)):
        top3 = np.argsort(full_probs[b])[-3:][::-1]
        t = beat_times[b]
        parts = [f"{vocab[idx]}:{full_probs[b,idx]:.3f}" for idx in top3]
        print(f"  beat {b:3d} ({t:6.2f}s): {', '.join(parts)}")
    print()

    # Show what Bb specifically looks like across all beats
    bb_idx = vocab.index('Bb') if 'Bb' in vocab else None
    if bb_idx is not None:
        bb_probs = full_probs[:, bb_idx]
        print(f"Bb classifier probability: min={bb_probs.min():.4f}, max={bb_probs.max():.4f}, mean={bb_probs.mean():.4f}")
        # Count how many beats Bb is the top prediction
        top_preds = np.argmax(full_probs, axis=1)
        bb_top = np.sum(top_preds == bb_idx)
        print(f"Bb is top classifier prediction: {bb_top}/{n_beats} beats ({100*bb_top/n_beats:.1f}%)")
    print()

    # Show emission log-probs after key prior bias
    log_emit = np.log(np.clip(full_probs.T, 1e-10, None))
    log_key_prior = np.log(np.clip(key_priors[key_idx], 1e-10, None))
    log_emit_biased = log_emit + log_key_prior[:, np.newaxis] * 0.2

    print("Top-3 after key prior bias (first 40 beats):")
    for b in range(min(40, n_beats)):
        col = log_emit_biased[:, b]
        top3 = np.argsort(col)[-3:][::-1]
        t = beat_times[b]
        parts = [f"{vocab[idx]}:{col[idx]:.2f}" for idx in top3]
        print(f"  beat {b:3d} ({t:6.2f}s): {', '.join(parts)}")
    print()

    # Now run full Viterbi with the same logic as analyze_chords.py (multi-key)
    from analyze_chords import _model_viterbi_decode
    final_path, best_key = _model_viterbi_decode(model, beat_chroma, key_idx)

    _IDX_TO_NOTE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    print(f"Best key from multi-key Viterbi: {_IDX_TO_NOTE[best_key]} (idx={best_key})")
    print(f"(Essentia detected: key_idx={key_idx})")
    print()

    print("Final Viterbi path (first 60 beats):")
    for b in range(min(60, len(final_path))):
        t = beat_times[b]
        print(f"  beat {b:3d} ({t:6.2f}s): {final_path[b]}")
    print()

    # Chord distribution in final path
    from collections import Counter
    counts = Counter(final_path)
    print("Chord distribution in final path:")
    for chord, cnt in counts.most_common():
        print(f"  {chord}: {cnt} beats ({100*cnt/len(final_path):.1f}%)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_song.py <audio_path>")
        sys.exit(1)
    diagnose(sys.argv[1])
