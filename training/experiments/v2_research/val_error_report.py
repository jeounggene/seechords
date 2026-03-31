#!/usr/bin/env python3
"""Per-song val metrics at a fixed post-hoc self-bias (for error analysis).

Uses the same emissions + Viterbi path as posthoc_viterbi.
"""
import sys
import os
import argparse
import numpy as np
import torch

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from v2.posthoc_viterbi import (
    load_model, extract_emissions, build_transition_matrix, evaluate_with_viterbi,
)
from v2.transformer_data import load_data, split_songs, get_song_data
from v2.decode import viterbi_decode, smooth_isolated


def evaluate_per_song(song_emissions, log_trans, use_smooth=True):
    rows = []
    for song in song_emissions:
        log_emit = song['log_emit']
        t1_true = song['tier1_labels']
        path, _ = viterbi_decode(log_emit, log_trans)
        if use_smooth:
            path = smooth_isolated(path)
        path = path[: len(t1_true)]
        acc = float(np.mean(path == t1_true))
        n_flips = int(np.sum(path[1:] != path[:-1])) if len(path) > 1 else 0
        flip = n_flips / max(len(path) - 1, 1)
        rows.append({
            'name': song['name'],
            'provenance': song['provenance'],
            'n_beats': len(t1_true),
            'acc': acc,
            'flip_rate': float(flip),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--self-bias', type=float, default=6.0)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device('mps')
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model, ckpt, model_type = load_model(args.checkpoint, device)
    hp = ckpt['hyperparams']
    gold_only = ckpt.get('gold_only', True)
    feature_dim = hp.get('input_dim', 60)
    data = load_data(args.data, gold_only=gold_only, feature_dim=feature_dim)
    split_seed = hp.get('split_seed', hp.get('seed', 42))
    _, val_set, test_set = split_songs(data, seed=split_seed)
    song_set = val_set if args.split == 'val' else test_set
    songs = get_song_data(data, song_set)

    song_emissions = extract_emissions(model, model_type, songs, device)
    log_trans = build_transition_matrix(args.self_bias, n_states=25)

    agg = evaluate_with_viterbi(song_emissions, log_trans)
    per = evaluate_per_song(song_emissions, log_trans)

    print(f"checkpoint={args.checkpoint}")
    print(f"split={args.split}  seed={split_seed}  self_bias={args.self_bias}")
    print(f"aggregate: WCSR={agg['wcsr']:.4f} maj={agg['major_acc']:.4f} min={agg['minor_acc']:.4f} "
          f"root={agg['root_acc']:.4f} flip={agg['flip_rate']:.4f}")
    print("\nPer-song (sorted by acc ascending):")
    for r in sorted(per, key=lambda x: x['acc']):
        g = 'G' if r['provenance'] == 'gold' else 'S'
        print(f"  [{g}] {r['name'][:70]:70s}  n={r['n_beats']:4d}  acc={r['acc']:.3f}  flip={r['flip_rate']:.3f}")

    high_flip = [r for r in per if r['flip_rate'] > 0.25]
    print(f"\nSongs with flip_rate > 0.25: {len(high_flip)}")
    for r in sorted(high_flip, key=lambda x: -x['flip_rate']):
        print(f"  {r['name'][:70]}  flip={r['flip_rate']:.3f}  acc={r['acc']:.3f}")


if __name__ == '__main__':
    main()
