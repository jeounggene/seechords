#!/usr/bin/env python3
"""Sweep decoder self_prob on the best augmented Transformer model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from v2.transformer_model import ChordTransformer
from v2.transformer_data import load_data, split_songs, get_song_data
from v2.train_transformer import evaluate_on_songs
from v2.decode import soften_transitions, viterbi_decode, smooth_isolated

def evaluate_with_self_prob(model, songs, tier1_trans, device, self_prob):
    """Like evaluate_on_songs but with custom self_prob."""
    model.eval()
    all_true = []
    all_pred = []
    song_flips = []

    with torch.no_grad():
        for song in songs:
            X = torch.from_numpy(song['features']).unsqueeze(0).to(device)
            root_probs, quality_probs = model.predict_probs(X)
            rp = root_probs[0].cpu().numpy()
            qp = quality_probs[0].cpu().numpy()

            # Build emissions
            n_beats = rp.shape[0]
            emit = np.zeros((25, n_beats), dtype=np.float64)
            emit[0, :] = rp[:, 0] * qp[:, 0]
            for ni in range(12):
                emit[1 + ni, :] = rp[:, ni + 1] * qp[:, 1]
                emit[13 + ni, :] = rp[:, ni + 1] * qp[:, 2]
            log_emit = np.log(np.clip(emit, 1e-10, None))

            trans_soft = soften_transitions(tier1_trans, self_prob=self_prob, floor=0.015)
            log_trans = np.log(np.clip(trans_soft, 1e-10, None))
            path, _ = viterbi_decode(log_emit, log_trans)
            path = smooth_isolated(path)

            all_true.append(song['tier1_labels'])
            all_pred.append(path)
            n_flips = np.sum(path[1:] != path[:-1]) if len(path) > 1 else 0
            song_flips.append(n_flips / max(len(path) - 1, 1))

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    wcsr = np.mean(all_true == all_pred)
    maj_mask = (all_true >= 1) & (all_true <= 12)
    min_mask = (all_true >= 13) & (all_true <= 24)
    major = np.mean(all_true[maj_mask] == all_pred[maj_mask]) if maj_mask.sum() else 0
    minor = np.mean(all_true[min_mask] == all_pred[min_mask]) if min_mask.sum() else 0
    flip = np.mean(song_flips)
    return wcsr, major, minor, flip

# Load model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
ckpt = torch.load('models/chord_transformer_gold_60d.pt', map_location=device, weights_only=False)
hp = ckpt['hyperparams']
model = ChordTransformer(
    input_dim=hp['input_dim'], d_model=hp['d_model'], nhead=hp['nhead'],
    num_layers=hp['num_layers'], d_ff=hp['d_ff'], dropout=hp['dropout'],
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
tier1_trans = ckpt['tier1_transition']

# Load data
data = load_data('data/features_v2.npz', gold_only=True, feature_dim=hp['input_dim'])
_, val_set, test_set = split_songs(data)
val_songs = get_song_data(data, val_set)
test_songs = get_song_data(data, test_set)

print(f"{'self_prob':>10s} {'val_WCSR':>10s} {'val_maj':>10s} {'val_min':>10s} {'val_flip':>10s} | {'test_WCSR':>10s} {'test_maj':>10s} {'test_min':>10s} {'test_flip':>10s}")
print("-" * 100)

for sp in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    vw, vm, vn, vf = evaluate_with_self_prob(model, val_songs, tier1_trans, device, sp)
    tw, tm, tn, tf_ = evaluate_with_self_prob(model, test_songs, tier1_trans, device, sp)
    print(f"{sp:10.2f} {vw:10.3f} {vm:10.3f} {vn:10.3f} {vf:10.3f} | {tw:10.3f} {tm:10.3f} {tn:10.3f} {tf_:10.3f}")
