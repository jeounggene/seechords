#!/usr/bin/env python3
"""Overfit sanity check: train on 3 songs, verify loss drops and WCSR rises."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from v2.transformer_model import ChordTransformer
from v2.transformer_data import load_data, split_songs, ChordWindowDataset, get_song_data
from v2.train_transformer import evaluate_on_songs

# Load data
data = load_data('data/features_v2.npz', gold_only=True)
train_set, val_set, test_set = split_songs(data)

# Use just 3 training songs
overfit_songs = sorted(train_set)[:3]
overfit_set = set(overfit_songs)
print(f"Overfit songs: {overfit_songs}")
for sid in overfit_songs:
    mask = data['song_ids'] == sid
    n = mask.sum()
    name = data['provenance'][sid] if sid < len(data['provenance']) else '?'
    print(f"  Song {sid}: {n} beats")

# Dataset (window covering entire short songs)
ds = ChordWindowDataset(data, overfit_set, window=512, stride=256)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
print(f"Windows: {len(ds)}")

# Songs for eval
songs = get_song_data(data, overfit_set)

# Transition matrix from overfit songs
n_states = 25
counts = np.ones((n_states, n_states), dtype=np.float64)
for sid in overfit_songs:
    mask = data['song_ids'] == sid
    t1 = data['tier1_labels'][mask]
    for i in range(len(t1) - 1):
        counts[t1[i], t1[i+1]] += 1
tier1_trans = counts / counts.sum(axis=1, keepdims=True)

# Model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = ChordTransformer(input_dim=24).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0)
root_crit = nn.CrossEntropyLoss(ignore_index=-1)
qual_crit = nn.CrossEntropyLoss(ignore_index=-1)

print(f"\nDevice: {device}")
print(f"Training 200 epochs to overfit on 3 songs...\n")

for epoch in range(1, 201):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        feat = batch['features'].to(device)
        rl = batch['root_labels'].to(device)
        ql = batch['quality3_labels'].to(device)
        pad = batch['padding_mask'].to(device)
        rl = rl.clone(); rl[pad] = -1
        ql = ql.clone(); ql[pad] = -1

        rout, qout = model(feat, src_key_padding_mask=pad)
        B, W, _ = rout.shape
        loss_r = root_crit(rout.reshape(B*W, -1), rl.reshape(B*W))
        loss_q = qual_crit(qout.reshape(B*W, -1), ql.reshape(B*W))
        loss = loss_r + 1.5 * loss_q

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1

    if epoch % 10 == 0 or epoch == 1:
        avg_loss = total_loss / max(n, 1)
        metrics = evaluate_on_songs(model, songs, tier1_trans, device)
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | WCSR={metrics['wcsr']:.3f} "
              f"maj={metrics['major_acc']:.3f} min={metrics['minor_acc']:.3f} "
              f"root={metrics['root_acc']:.3f} flip={metrics['flip_rate']:.3f}")

print("\nOverfit check complete. WCSR should approach 1.0 and loss should be near 0.")
