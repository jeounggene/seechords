#!/usr/bin/env python3
"""Disagreement analysis between two Transformer+CRF checkpoints.

Primary use: compare freeze5 (champion) vs 3B (high minor recall) to understand
where 3B's extra minor predictions come from — real signal or bias.

Usage:
    python -m v2.disagreement_analysis \
        --data data/features_v2.npz \
        --model-a models/chord_transformer_crf_freeze5.pt \
        --model-b models/chord_transformer_crf_rf_reg.pt \
        --label-a freeze5 --label-b 3B
"""
import sys
import os
import argparse
import numpy as np
import torch
from collections import defaultdict, Counter

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from v2.transformer_model import ChordTransformerCRF
from v2.transformer_data import load_data, split_songs, get_song_data
from v2.decode import smooth_isolated
from shared.chord_vocab import TIER1_VOCAB


# Tier1 index helpers
def tier1_name(idx):
    return TIER1_VOCAB[idx] if 0 <= idx < len(TIER1_VOCAB) else f"?{idx}"

def is_major(idx):
    return 1 <= idx <= 12

def is_minor(idx):
    return 13 <= idx <= 24

def root_of(idx):
    """Return root index (1-12) or 0 for N."""
    if idx == 0:
        return 0
    if 1 <= idx <= 12:
        return idx
    if 13 <= idx <= 24:
        return idx - 12
    return 0

def relative_minor(maj_idx):
    """Given a major chord tier1 idx (1-12), return its relative minor tier1 idx."""
    # Relative minor is 9 semitones up (or 3 down): C -> Am, G -> Em
    if 1 <= maj_idx <= 12:
        note = maj_idx - 1  # 0-based note
        rel_note = (note + 9) % 12
        return 13 + rel_note
    return 0

def relative_major(min_idx):
    """Given a minor chord tier1 idx (13-24), return its relative major tier1 idx."""
    if 13 <= min_idx <= 24:
        note = min_idx - 13  # 0-based note
        rel_note = (note + 3) % 12
        return 1 + rel_note
    return 0


def load_model(checkpoint_path, device):
    """Load a Transformer+CRF checkpoint and return the model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def decode_songs(model, songs, device):
    """Decode all songs with CRF Viterbi + smooth_isolated.

    Returns: list of (song_info, pred_path) tuples.
    """
    results = []
    with torch.no_grad():
        for song in songs:
            X = torch.from_numpy(song['features']).unsqueeze(0).to(device)
            paths = model.decode(X)
            path = np.array(paths[0], dtype=np.int64)
            path = smooth_isolated(path)

            t1_true = song['tier1_labels']
            path = path[:len(t1_true)]
            if len(path) < len(t1_true):
                path = np.pad(path, (0, len(t1_true) - len(path)), constant_values=0)

            results.append((song, path))
    return results


def analyze_disagreements(results_a, results_b, label_a, label_b):
    """Analyze beats where model A and model B disagree.

    Focus on: B predicts minor, A predicts major (3B's "extra minor calls").
    """
    # Collect all disagreement beats
    b_minor_a_major = []  # B says minor, A says major
    b_major_a_minor = []  # B says major, A says minor
    other_disagree = []   # other disagreements

    total_beats = 0
    total_agree = 0

    for (song_a, pred_a), (song_b, pred_b) in zip(results_a, results_b):
        assert song_a['song_id'] == song_b['song_id']
        true = song_a['tier1_labels']
        name = song_a['name']
        n = len(true)
        total_beats += n

        for i in range(n):
            pa, pb = int(pred_a[i]), int(pred_b[i])
            gt = int(true[i])

            if pa == pb:
                total_agree += 1
                continue

            beat_info = {
                'song': name,
                'beat_idx': i,
                'true': gt,
                'pred_a': pa,
                'pred_b': pb,
                'true_name': tier1_name(gt),
                'pred_a_name': tier1_name(pa),
                'pred_b_name': tier1_name(pb),
                'b_correct': pb == gt,
                'a_correct': pa == gt,
            }

            if is_minor(pb) and is_major(pa):
                b_minor_a_major.append(beat_info)
            elif is_major(pb) and is_minor(pa):
                b_major_a_minor.append(beat_info)
            else:
                other_disagree.append(beat_info)

    n_disagree = len(b_minor_a_major) + len(b_major_a_minor) + len(other_disagree)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"Disagreement Analysis: {label_a} vs {label_b}")
    print(f"{'='*70}")
    print(f"Total beats:     {total_beats}")
    print(f"Agreement:       {total_agree} ({100*total_agree/total_beats:.1f}%)")
    print(f"Disagreement:    {n_disagree} ({100*n_disagree/total_beats:.1f}%)")
    print(f"  {label_b} minor / {label_a} major: {len(b_minor_a_major)}")
    print(f"  {label_b} major / {label_a} minor: {len(b_major_a_minor)}")
    print(f"  Other:                        {len(other_disagree)}")

    # ── Focus: B predicts minor, A predicts major ──
    print(f"\n{'─'*70}")
    print(f"Focus: {label_b} predicts MINOR where {label_a} predicts MAJOR")
    print(f"{'─'*70}")

    if not b_minor_a_major:
        print("  (no such disagreements)")
        return

    bm = b_minor_a_major
    n_bm = len(bm)
    b_correct = sum(1 for x in bm if x['b_correct'])
    a_correct = sum(1 for x in bm if x['a_correct'])
    neither = sum(1 for x in bm if not x['b_correct'] and not x['a_correct'])

    print(f"  Count: {n_bm} beats")
    print(f"  {label_b} correct (truth=minor): {b_correct} ({100*b_correct/n_bm:.1f}%)")
    print(f"  {label_a} correct (truth=major): {a_correct} ({100*a_correct/n_bm:.1f}%)")
    print(f"  Neither correct:                {neither} ({100*neither/n_bm:.1f}%)")

    # Bucket by true label
    print(f"\n  By true label:")
    true_counter = Counter(x['true_name'] for x in bm)
    for label, count in true_counter.most_common(20):
        # How many times was B right for this true label?
        b_right = sum(1 for x in bm if x['true_name'] == label and x['b_correct'])
        a_right = sum(1 for x in bm if x['true_name'] == label and x['a_correct'])
        print(f"    {label:6s}: {count:4d} beats  "
              f"({label_b} right: {b_right}, {label_a} right: {a_right})")

    # Bucket by confusion pair: pred_a -> pred_b (what A calls X, B calls Xm)
    print(f"\n  Confusion pairs ({label_a} -> {label_b}):")
    pair_counter = Counter((x['pred_a_name'], x['pred_b_name']) for x in bm)
    for (pa, pb), count in pair_counter.most_common(20):
        b_right = sum(1 for x in bm
                      if x['pred_a_name'] == pa and x['pred_b_name'] == pb and x['b_correct'])
        a_right = sum(1 for x in bm
                      if x['pred_a_name'] == pa and x['pred_b_name'] == pb and x['a_correct'])
        print(f"    {pa:4s} -> {pb:4s}: {count:4d} beats  "
              f"({label_b} right: {b_right}, {label_a} right: {a_right})")

    # Relative major/minor analysis
    print(f"\n  Relative major/minor pairs:")
    n_relative = 0
    n_parallel = 0
    for x in bm:
        pa, pb = x['pred_a'], x['pred_b']
        if is_major(pa) and is_minor(pb):
            if relative_minor(pa) == pb:
                n_relative += 1
            elif root_of(pa) == root_of(pb):
                n_parallel += 1
    print(f"    Same root (parallel maj/min, e.g. A vs Am): {n_parallel}")
    print(f"    Relative pair (e.g. C vs Am, G vs Em):      {n_relative}")
    print(f"    Other:                                       {n_bm - n_parallel - n_relative}")

    # Per-song breakdown
    print(f"\n  Per-song breakdown:")
    song_counter = Counter(x['song'] for x in bm)
    for song, count in song_counter.most_common(30):
        b_right = sum(1 for x in bm if x['song'] == song and x['b_correct'])
        a_right = sum(1 for x in bm if x['song'] == song and x['a_correct'])
        print(f"    {song:<55s} {count:3d} beats  "
              f"({label_b}={b_right}, {label_a}={a_right})")

    # ── Reverse: B predicts major, A predicts minor ──
    print(f"\n{'─'*70}")
    print(f"Reverse: {label_b} predicts MAJOR where {label_a} predicts MINOR")
    print(f"{'─'*70}")

    if not b_major_a_minor:
        print("  (no such disagreements)")
    else:
        bm2 = b_major_a_minor
        n_bm2 = len(bm2)
        b_correct2 = sum(1 for x in bm2 if x['b_correct'])
        a_correct2 = sum(1 for x in bm2 if x['a_correct'])
        print(f"  Count: {n_bm2} beats")
        print(f"  {label_b} correct (truth=major): {b_correct2} ({100*b_correct2/n_bm2:.1f}%)")
        print(f"  {label_a} correct (truth=minor): {a_correct2} ({100*a_correct2/n_bm2:.1f}%)")

        # Confusion pairs
        print(f"\n  Confusion pairs ({label_a} -> {label_b}):")
        pair_counter2 = Counter((x['pred_a_name'], x['pred_b_name']) for x in bm2)
        for (pa, pb), count in pair_counter2.most_common(15):
            b_right = sum(1 for x in bm2
                          if x['pred_a_name'] == pa and x['pred_b_name'] == pb and x['b_correct'])
            a_right = sum(1 for x in bm2
                          if x['pred_a_name'] == pa and x['pred_b_name'] == pb and x['a_correct'])
            print(f"    {pa:4s} -> {pb:4s}: {count:4d} beats  "
                  f"({label_b} right: {b_right}, {label_a} right: {a_right})")

    # ── Argmax emission comparison ──
    print(f"\n{'─'*70}")
    print(f"Overall accuracy comparison on disagreement beats")
    print(f"{'─'*70}")

    all_disagree = b_minor_a_major + b_major_a_minor + other_disagree
    a_wins = sum(1 for x in all_disagree if x['a_correct'] and not x['b_correct'])
    b_wins = sum(1 for x in all_disagree if x['b_correct'] and not x['a_correct'])
    both_wrong = sum(1 for x in all_disagree if not x['a_correct'] and not x['b_correct'])
    both_right = sum(1 for x in all_disagree if x['a_correct'] and x['b_correct'])
    # both_right shouldn't happen (they disagree), but check for consistency
    print(f"  {label_a} wins: {a_wins}  ({100*a_wins/len(all_disagree):.1f}%)")
    print(f"  {label_b} wins: {b_wins}  ({100*b_wins/len(all_disagree):.1f}%)")
    print(f"  Both wrong:  {both_wrong}  ({100*both_wrong/len(all_disagree):.1f}%)")
    if both_right > 0:
        print(f"  (Both right: {both_right} — should be 0 if true disagreement)")

    # ── Minor-specific accuracy (global) ──
    print(f"\n{'─'*70}")
    print(f"Global minor accuracy comparison")
    print(f"{'─'*70}")

    all_true_minor_a_right = 0
    all_true_minor_b_right = 0
    all_true_minor_total = 0

    for (song_a, pred_a), (song_b, pred_b) in zip(results_a, results_b):
        true = song_a['tier1_labels']
        for i in range(len(true)):
            gt = int(true[i])
            if is_minor(gt):
                all_true_minor_total += 1
                if int(pred_a[i]) == gt:
                    all_true_minor_a_right += 1
                if int(pred_b[i]) == gt:
                    all_true_minor_b_right += 1

    print(f"  True minor beats: {all_true_minor_total}")
    print(f"  {label_a} minor recall: {all_true_minor_a_right}/{all_true_minor_total} "
          f"= {100*all_true_minor_a_right/max(all_true_minor_total,1):.1f}%")
    print(f"  {label_b} minor recall: {all_true_minor_b_right}/{all_true_minor_total} "
          f"= {100*all_true_minor_b_right/max(all_true_minor_total,1):.1f}%")

    # Minor precision: of all beats predicted as minor, how many are truly minor?
    a_pred_minor = 0
    a_pred_minor_correct = 0
    b_pred_minor = 0
    b_pred_minor_correct = 0

    for (song_a, pred_a), (song_b, pred_b) in zip(results_a, results_b):
        true = song_a['tier1_labels']
        for i in range(len(true)):
            gt = int(true[i])
            pa = int(pred_a[i])
            pb = int(pred_b[i])
            if is_minor(pa):
                a_pred_minor += 1
                if pa == gt:
                    a_pred_minor_correct += 1
            if is_minor(pb):
                b_pred_minor += 1
                if pb == gt:
                    b_pred_minor_correct += 1

    print(f"\n  {label_a} minor predictions: {a_pred_minor} (precision: "
          f"{100*a_pred_minor_correct/max(a_pred_minor,1):.1f}%)")
    print(f"  {label_b} minor predictions: {b_pred_minor} (precision: "
          f"{100*b_pred_minor_correct/max(b_pred_minor,1):.1f}%)")

    # ── Quality confusion: what does the ground truth say for "wrong minor" calls? ──
    print(f"\n{'─'*70}")
    print(f"{label_b}'s false minor calls — what was the true chord?")
    print(f"{'─'*70}")

    false_minor_truth = Counter()
    for (song_a, pred_a), (song_b, pred_b) in zip(results_a, results_b):
        true = song_a['tier1_labels']
        for i in range(len(true)):
            pb = int(pred_b[i])
            gt = int(true[i])
            if is_minor(pb) and pb != gt:
                false_minor_truth[tier1_name(gt)] += 1

    print(f"  Total false minor predictions by {label_b}: {sum(false_minor_truth.values())}")
    for label, count in false_minor_truth.most_common(20):
        print(f"    true={label:6s}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Disagreement analysis between two CRF models')
    parser.add_argument('--data', required=True, help='Path to features_v2.npz')
    parser.add_argument('--model-a', required=True, help='Path to model A checkpoint')
    parser.add_argument('--model-b', required=True, help='Path to model B checkpoint')
    parser.add_argument('--label-a', default='A', help='Label for model A')
    parser.add_argument('--label-b', default='B', help='Label for model B')
    parser.add_argument('--split', default='test', choices=['test', 'val'],
                        help='Which split to analyze')
    parser.add_argument('--gold-only', action='store_true', help='Gold-only data')
    parser.add_argument('--feature-dim', type=int, default=60, help='Feature dim')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device('mps')
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data, gold_only=args.gold_only, feature_dim=args.feature_dim)
    train_set, val_set, test_set = split_songs(data)
    song_set = test_set if args.split == 'test' else val_set
    songs = get_song_data(data, song_set)
    print(f"  {len(songs)} {args.split} songs, "
          f"{sum(len(s['tier1_labels']) for s in songs)} beats")

    # Load models
    print(f"\nLoading {args.label_a}: {args.model_a}")
    model_a, ckpt_a = load_model(args.model_a, device)
    print(f"Loading {args.label_b}: {args.model_b}")
    model_b, ckpt_b = load_model(args.model_b, device)

    # Decode
    print(f"\nDecoding {args.split} songs with both models...")
    results_a = decode_songs(model_a, songs, device)
    results_b = decode_songs(model_b, songs, device)

    # Analyze
    analyze_disagreements(results_a, results_b, args.label_a, args.label_b)


if __name__ == '__main__':
    main()
