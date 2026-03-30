#!/usr/bin/env python3
"""Test classifier predictions with each of the 12 key hypotheses."""
import sys, os, pickle
import numpy as np
from collections import Counter

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_TRAINING_ROOT, '..', 'server'))
from analyze_chords import (
    SR, FRAME_SIZE, HOP_SIZE,
    _smooth_hpcp, _sync_hpcp_to_beats, _load_trained_model,
)
from essentia.standard import (
    MonoLoader, EqualLoudness, HighPass, RhythmExtractor2013,
    Windowing, Spectrum, SpectralPeaks, HPCP, Flatness, FrameGenerator,
)

audio_path = sys.argv[1]

audio = MonoLoader(filename=audio_path, sampleRate=SR)()
audio = EqualLoudness(sampleRate=SR)(audio)
audio = HighPass(cutoffFrequency=100, sampleRate=SR)(audio)
audio_raw = MonoLoader(filename=audio_path, sampleRate=SR)()
rhythm = RhythmExtractor2013(method='multifeature')
bpm, beats, *_ = rhythm(audio_raw)
beat_times = beats.tolist()
if len(beat_times) > 1:
    filtered = [beat_times[0]]
    for bt in beat_times[1:]:
        if bt - filtered[-1] >= 0.15:
            filtered.append(bt)
    beat_times = filtered

windowing = Windowing(type='blackmanharris62', size=FRAME_SIZE)
spectrum_algo = Spectrum(size=FRAME_SIZE)
peaks = SpectralPeaks(orderBy='magnitude', magnitudeThreshold=0.0001, maxPeaks=40,
                      minFrequency=80, maxFrequency=4000, sampleRate=SR)
hpcp_algo = HPCP(size=12, referenceFrequency=440, harmonics=4, bandPreset=False,
                 minFrequency=80, maxFrequency=4000, weightType='cosine',
                 nonLinear=True, windowSize=1.0, sampleRate=SR)
flatness_algo = Flatness()

hpcps_12 = []
flatness_weights = []
for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=True):
    spec = spectrum_algo(windowing(frame))
    freqs, mags = peaks(spec)
    h = hpcp_algo(freqs, mags)
    hpcps_12.append(np.roll(h, -3))
    sf = flatness_algo(spec)
    flatness_weights.append(max(0.2, 1.0 - sf * 2.0))

hpcps_12 = np.array(hpcps_12)
flatness_weights = np.array(flatness_weights)
n_frames = len(hpcps_12)
hpcps_12 = _smooth_hpcp(hpcps_12, window=5)
beat_frames = [int(round(bt * SR / HOP_SIZE)) for bt in beat_times]
beat_chroma = _sync_hpcp_to_beats(hpcps_12, beat_frames, n_frames, flatness_weights)

model = _load_trained_model()
vocab = model['vocab']
clf = model['classifier']
key_priors = model['key_priors']
n_classes = len(vocab)
n_beats = beat_chroma.shape[1]
X_norm = beat_chroma.T / np.maximum(np.linalg.norm(beat_chroma.T, axis=1, keepdims=True), 1e-10)

NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

print(f"Beats: {n_beats}")
print()

# Test each key: raw classifier prediction distribution
for ki in range(12):
    key_feat = np.tile(key_priors[ki], (n_beats, 1))
    X_aug = np.hstack([X_norm, key_feat])
    preds = clf.predict(X_aug)
    counts = Counter([vocab[p] for p in preds])
    top5 = counts.most_common(5)
    top5_str = ', '.join(f'{ch}:{cnt}' for ch, cnt in top5)
    print(f'Key={NOTES[ki]:>3s} (idx={ki:2d}): {top5_str}')

print()

# Also test with NO key features (zeros)
key_feat_zero = np.zeros((n_beats, n_classes))
X_aug_zero = np.hstack([X_norm, key_feat_zero])
preds_zero = clf.predict(X_aug_zero)
counts_zero = Counter([vocab[p] for p in preds_zero])
top5 = counts_zero.most_common(5)
top5_str = ', '.join(f'{ch}:{cnt}' for ch, cnt in top5)
print(f'Key=NONE (zeros): {top5_str}')

# Also test with UNIFORM key features
key_feat_uniform = np.full((n_beats, n_classes), 1.0 / n_classes)
X_aug_uniform = np.hstack([X_norm, key_feat_uniform])
preds_uniform = clf.predict(X_aug_uniform)
counts_uniform = Counter([vocab[p] for p in preds_uniform])
top5 = counts_uniform.most_common(5)
top5_str = ', '.join(f'{ch}:{cnt}' for ch, cnt in top5)
print(f'Key=UNIFORM:      {top5_str}')
