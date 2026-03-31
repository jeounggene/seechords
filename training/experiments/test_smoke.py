#!/usr/bin/env python3
"""Quick smoke test for Transformer modules."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v2.transformer_model import ChordTransformer
from v2.transformer_data import load_data, split_songs, ChordWindowDataset, compute_quality3_weights, get_song_data
from v2.train_transformer import emissions_from_probs, decode_song, evaluate_on_songs
import torch
import numpy as np

# Quick model test
model = ChordTransformer()
x = torch.randn(2, 64, 24)
r, q = model(x)
print(f'root_logits: {r.shape}, quality_logits: {q.shape}')

rp, qp = model.predict_probs(x)
print(f'root_probs: {rp.shape}, quality_probs: {qp.shape}')
print(f'root sums: {rp[0,0].sum():.4f}, quality sums: {qp[0,0].sum():.4f}')

# Load data
data = load_data('data/features_v2.npz', gold_only=True)
train_set, val_set, test_set = split_songs(data)
print(f'Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')

# Dataset test
ds = ChordWindowDataset(data, train_set, window=128, stride=64)
print(f'Windows: {len(ds)}')
sample = ds[0]
print(f'Sample: features={sample["features"].shape}, root={sample["root_labels"].shape}, pad={sample["padding_mask"].sum()} padded')

# Quality weights
w = compute_quality3_weights(data, train_set)
print(f'Quality weights: {w}')

# Emission conversion test
rp_np = np.random.dirichlet(np.ones(13), size=10)
qp_np = np.random.dirichlet(np.ones(3), size=10)
log_emit = emissions_from_probs(rp_np, qp_np)
print(f'Emissions shape: {log_emit.shape}')

print('\nAll imports and basic operations OK!')
