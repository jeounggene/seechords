#!/usr/bin/env python3
"""Quick inspection of the trained model.

Usage (from training/ root):
    python tools/inspect_model.py [models/chord_model.pkl]
"""
import os
import sys
import pickle
import numpy as np

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_TRAINING_ROOT, 'models', 'chord_model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print('Keys:', list(model.keys()))
print('Vocab:', model['vocab'])
print('Transition shape:', model['transition_probs'].shape)
print()

vocab = model['vocab']
trans = model['transition_probs']
ci = {v: i for i, v in enumerate(vocab)}

# Check transitions
for src in ['C', 'G', 'Am', 'F']:
    if src not in ci:
        continue
    print(f'Transitions from {src}:')
    row = trans[ci[src]]
    top5 = np.argsort(row)[-5:][::-1]
    for idx in top5:
        print(f'  {src} -> {vocab[idx]}: {row[idx]:.4f}')
    print()

# Key priors for C major (index 0)
kp = model['key_priors']
print('Key priors for C major (key_idx=0):')
top = np.argsort(kp[0])[-10:][::-1]
for idx in top:
    print(f'  {vocab[idx]}: {kp[0, idx]:.4f}')
print()

# Check classifier classes
clf = model['classifier']
print(f'Classifier classes ({len(clf.classes_)}): {[vocab[c] for c in clf.classes_]}')
print()

# Quick test: what does the classifier predict for a pure C major chroma?
c_chroma = np.zeros(12)
c_chroma[0] = 1.0  # C
c_chroma[4] = 0.8  # E
c_chroma[7] = 0.8  # G
norm = np.linalg.norm(c_chroma)
c_chroma /= norm
x = np.hstack([c_chroma, kp[0]])  # key of C
probs = clf.predict_proba(x.reshape(1, -1))[0]
full_probs = np.zeros(len(vocab))
for ci2, cls in enumerate(clf.classes_):
    full_probs[cls] = probs[ci2]
top5 = np.argsort(full_probs)[-5:][::-1]
print('Classifier output for pure C major chroma (key=C):')
for idx in top5:
    print(f'  {vocab[idx]}: {full_probs[idx]:.4f}')
print()

# Pure Am chroma
am_chroma = np.zeros(12)
am_chroma[9] = 1.0   # A
am_chroma[0] = 0.8   # C
am_chroma[4] = 0.8   # E
norm = np.linalg.norm(am_chroma)
am_chroma /= norm
x = np.hstack([am_chroma, kp[0]])
probs = clf.predict_proba(x.reshape(1, -1))[0]
full_probs = np.zeros(len(vocab))
for ci2, cls in enumerate(clf.classes_):
    full_probs[cls] = probs[ci2]
top5 = np.argsort(full_probs)[-5:][::-1]
print('Classifier output for pure Am chroma (key=C):')
for idx in top5:
    print(f'  {vocab[idx]}: {full_probs[idx]:.4f}')
print()

# Pure F chroma
f_chroma = np.zeros(12)
f_chroma[5] = 1.0   # F
f_chroma[9] = 0.8   # A
f_chroma[0] = 0.8   # C
norm = np.linalg.norm(f_chroma)
f_chroma /= norm
x = np.hstack([f_chroma, kp[0]])
probs = clf.predict_proba(x.reshape(1, -1))[0]
full_probs = np.zeros(len(vocab))
for ci2, cls in enumerate(clf.classes_):
    full_probs[cls] = probs[ci2]
top5 = np.argsort(full_probs)[-5:][::-1]
print('Classifier output for pure F chroma (key=C):')
for idx in top5:
    print(f'  {vocab[idx]}: {full_probs[idx]:.4f}')
