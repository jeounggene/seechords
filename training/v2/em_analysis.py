#!/usr/bin/env python3
"""Deep-dive analysis of freeze5's Em over-prediction.

Covers:
  1. All freeze5 Em false positives bucketed by true label, song, context
  2. Freeze5 vs 3B comparison on the Em confusion set
  3. Post-hoc Em bias penalty test via emission manipulation + Viterbi resweep

Usage:
    python -m v2.em_analysis \
        --data data/features_v2.npz \
        --model-a models/chord_transformer_crf_freeze5.pt \
        --model-b models/chord_transformer_crf_rf_reg.pt
"""
import sys
import os
import argparse
import numpy as np
import torch
from collections import Counter, defaultdict

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from v2.transformer_model import ChordTransformerCRF
from v2.transformer_data import load_data, split_songs, get_song_data
from v2.decode import viterbi_decode, smooth_isolated
from shared.chord_vocab import TIER1_VOCAB

EM_IDX = TIER1_VOCAB.index('Em')  # should be 17


def tier1_name(idx):
    return TIER1_VOCAB[idx] if 0 <= idx < len(TIER1_VOCAB) else f"?{idx}"

def is_major(idx):
    return 1 <= idx <= 12

def is_minor(idx):
    return 13 <= idx <= 24


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
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


def decode_and_collect(model, songs, device):
    """Decode all songs, return list of (song, pred_path, raw_logits)."""
    results = []
    with torch.no_grad():
        for song in songs:
            X = torch.from_numpy(song['features']).unsqueeze(0).to(device)

            # CRF decode
            paths = model.decode(X)
            path = np.array(paths[0], dtype=np.int64)
            path = smooth_isolated(path)

            # Also extract raw logits for emission analysis
            h = model._encode(X)
            logits = model.tier1_head(h)
            raw_logits = logits[0].cpu().numpy()  # (seq_len, 25)

            t1_true = song['tier1_labels']
            path = path[:len(t1_true)]
            if len(path) < len(t1_true):
                path = np.pad(path, (0, len(t1_true) - len(path)), constant_values=0)

            results.append((song, path, raw_logits))
    return results


def extract_emissions(model, songs, device):
    """Extract raw log-emissions per song for post-hoc Viterbi."""
    results = []
    with torch.no_grad():
        for song in songs:
            X = torch.from_numpy(song['features']).unsqueeze(0).to(device)
            h = model._encode(X)
            if getattr(model, 'emission_mode', 'direct') == 'hybrid':
                log_emit = model._tier1_emissions_hybrid(
                    model.root_head(h), model.tier1_head(h)
                )[0].cpu().numpy().T
            else:
                logits = model.tier1_head(h)
                log_emit = logits[0].cpu().numpy().T
            results.append({
                'log_emit': log_emit,
                'tier1_labels': song['tier1_labels'],
                'name': song['name'],
            })
    return results


def build_transition_matrix(self_bias, n_states=25):
    log_trans = np.zeros((n_states, n_states))
    log_trans[np.diag_indices(n_states)] = self_bias
    for i in range(n_states):
        row_lse = np.log(np.sum(np.exp(log_trans[i])))
        log_trans[i] -= row_lse
    return log_trans


def evaluate_with_bias(song_emissions, self_bias, em_penalty=0.0):
    """Run Viterbi with optional Em emission penalty."""
    log_trans = build_transition_matrix(self_bias)
    all_true = []
    all_pred = []
    song_flips = []

    for song in song_emissions:
        log_emit = song['log_emit'].copy()
        if em_penalty != 0.0:
            log_emit[EM_IDX, :] -= em_penalty
        t1_true = song['tier1_labels']

        path, _ = viterbi_decode(log_emit, log_trans)
        path = smooth_isolated(path)
        path = path[:len(t1_true)]

        all_true.append(t1_true)
        all_pred.append(path)
        n_flips = np.sum(path[1:] != path[:-1]) if len(path) > 1 else 0
        song_flips.append(n_flips / max(len(path) - 1, 1))

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    wcsr = np.mean(all_true == all_pred)
    maj_mask = (all_true >= 1) & (all_true <= 12)
    min_mask = (all_true >= 13) & (all_true <= 24)
    major_acc = np.mean(all_true[maj_mask] == all_pred[maj_mask]) if maj_mask.sum() > 0 else 0
    minor_acc = np.mean(all_true[min_mask] == all_pred[min_mask]) if min_mask.sum() > 0 else 0
    avg_flip = np.mean(song_flips)

    # Em-specific: false positives and recall
    em_fp = np.sum((all_pred == EM_IDX) & (all_true != EM_IDX))
    em_total_true = np.sum(all_true == EM_IDX)
    em_recall = np.sum((all_pred == EM_IDX) & (all_true == EM_IDX)) / max(em_total_true, 1)

    return {
        'wcsr': float(wcsr),
        'major_acc': float(major_acc),
        'minor_acc': float(minor_acc),
        'flip_rate': float(avg_flip),
        'em_fp': int(em_fp),
        'em_recall': float(em_recall),
    }


# ─────────────────────────────────────────────────────────────────
# Part 1: Full Em false-positive analysis
# ─────────────────────────────────────────────────────────────────

def analyze_em_false_positives(results_a, label_a):
    """Analyze every beat where model A predicts Em but ground truth is not Em."""
    fp_beats = []

    for song, pred, logits in results_a:
        true = song['tier1_labels']
        name = song['name']
        n = len(true)
        for i in range(n):
            if int(pred[i]) == EM_IDX and int(true[i]) != EM_IDX:
                prev_true = int(true[i - 1]) if i > 0 else -1
                next_true = int(true[i + 1]) if i < n - 1 else -1
                prev_pred = int(pred[i - 1]) if i > 0 else -1
                next_pred = int(pred[i + 1]) if i < n - 1 else -1

                # Emission analysis: what was the top-1 argmax?
                beat_logits = logits[i]  # (25,)
                argmax = int(np.argmax(beat_logits))
                em_rank = int(np.sum(beat_logits > beat_logits[EM_IDX]))  # 0 = Em is top
                em_margin = float(beat_logits[EM_IDX] - np.max(np.delete(beat_logits, EM_IDX)))

                fp_beats.append({
                    'song': name,
                    'beat_idx': i,
                    'true': int(true[i]),
                    'true_name': tier1_name(int(true[i])),
                    'prev_true': prev_true,
                    'next_true': next_true,
                    'prev_true_name': tier1_name(prev_true) if prev_true >= 0 else '-',
                    'next_true_name': tier1_name(next_true) if next_true >= 0 else '-',
                    'prev_pred': prev_pred,
                    'next_pred': next_pred,
                    'argmax': argmax,
                    'argmax_name': tier1_name(argmax),
                    'em_rank': em_rank,
                    'em_margin': em_margin,
                })

    print(f"\n{'='*70}")
    print(f"Part 1: ALL {label_a} Em false positives")
    print(f"{'='*70}")
    print(f"Total Em false positives: {len(fp_beats)}")

    # Also count Em true positives for context
    em_tp = 0
    em_total_true = 0
    for song, pred, _ in results_a:
        true = song['tier1_labels']
        em_total_true += np.sum(true == EM_IDX)
        em_tp += np.sum((pred == EM_IDX) & (true == EM_IDX))
    em_pred_total = em_tp + len(fp_beats)
    print(f"Em true positives:       {em_tp}")
    print(f"Em total predictions:    {em_pred_total}")
    print(f"Em precision:            {100*em_tp/max(em_pred_total,1):.1f}%")
    print(f"Em true beats in data:   {em_total_true}")
    print(f"Em recall:               {100*em_tp/max(em_total_true,1):.1f}%")

    # Bucket by true label
    print(f"\n  By true label (what was actually playing):")
    true_counter = Counter(x['true_name'] for x in fp_beats)
    for label, count in true_counter.most_common(25):
        pct = 100 * count / len(fp_beats)
        print(f"    {label:6s}: {count:4d} ({pct:5.1f}%)")

    # Bucket by song
    print(f"\n  By song:")
    song_counter = Counter(x['song'] for x in fp_beats)
    for song, count in song_counter.most_common(30):
        pct = 100 * count / len(fp_beats)
        # How many total beats in this song?
        print(f"    {song:<55s} {count:3d} ({pct:4.1f}%)")

    # Context: what chords surround Em false positives?
    print(f"\n  Previous true chord (before Em FP):")
    prev_counter = Counter(x['prev_true_name'] for x in fp_beats)
    for label, count in prev_counter.most_common(15):
        print(f"    {label:6s}: {count:4d}")

    print(f"\n  Next true chord (after Em FP):")
    next_counter = Counter(x['next_true_name'] for x in fp_beats)
    for label, count in next_counter.most_common(15):
        print(f"    {label:6s}: {count:4d}")

    # Transition context: what was the true chord *sequence* around Em FPs?
    print(f"\n  Most common true context windows (prev → true → next):")
    context_counter = Counter(
        (x['prev_true_name'], x['true_name'], x['next_true_name'])
        for x in fp_beats
    )
    for (prev, curr, nxt), count in context_counter.most_common(20):
        print(f"    {prev:5s} → {curr:5s} → {nxt:5s}: {count:3d}")

    # Emission analysis: is Em actually the argmax, or is CRF pulling it in?
    em_is_argmax = sum(1 for x in fp_beats if x['argmax'] == EM_IDX)
    em_in_top3 = sum(1 for x in fp_beats if x['em_rank'] < 3)
    print(f"\n  Emission analysis (is Em the emission argmax?):")
    print(f"    Em is argmax:    {em_is_argmax}/{len(fp_beats)} "
          f"({100*em_is_argmax/len(fp_beats):.1f}%)")
    print(f"    Em in top-3:     {em_in_top3}/{len(fp_beats)} "
          f"({100*em_in_top3/len(fp_beats):.1f}%)")
    print(f"    Avg Em margin:   {np.mean([x['em_margin'] for x in fp_beats]):.3f}")

    # When Em is NOT argmax, what is?
    not_argmax = [x for x in fp_beats if x['argmax'] != EM_IDX]
    if not_argmax:
        print(f"\n  When Em is NOT emission argmax ({len(not_argmax)} beats), "
              f"what is the argmax?")
        alt_counter = Counter(x['argmax_name'] for x in not_argmax)
        for label, count in alt_counter.most_common(15):
            # How often was this argmax actually correct?
            correct = sum(1 for x in not_argmax
                          if x['argmax_name'] == label and x['argmax'] == x['true'])
            print(f"    {label:6s}: {count:4d}  (would be correct: {correct})")

    return fp_beats


# ─────────────────────────────────────────────────────────────────
# Part 2: Freeze5 vs 3B on the Em confusion set
# ─────────────────────────────────────────────────────────────────

def compare_on_em_set(results_a, results_b, fp_beats, label_a, label_b):
    """Compare models A and B specifically on beats where A predicts Em falsely."""
    print(f"\n{'='*70}")
    print(f"Part 2: {label_a} vs {label_b} on the Em false-positive set")
    print(f"{'='*70}")

    # Build lookup: (song, beat_idx) -> fp_beat info
    fp_lookup = {(x['song'], x['beat_idx']): x for x in fp_beats}

    # Collect B's predictions on these same beats
    comparisons = []
    for (song_a, pred_a, _), (song_b, pred_b, _) in zip(results_a, results_b):
        assert song_a['song_id'] == song_b['song_id']
        name = song_a['name']
        true = song_a['tier1_labels']
        for i in range(len(true)):
            key = (name, i)
            if key in fp_lookup:
                pb = int(pred_b[i])
                gt = int(true[i])
                comparisons.append({
                    **fp_lookup[key],
                    'pred_b': pb,
                    'pred_b_name': tier1_name(pb),
                    'b_correct': pb == gt,
                })

    print(f"  {label_a} Em false positives: {len(comparisons)}")

    b_correct = sum(1 for x in comparisons if x['b_correct'])
    b_also_em = sum(1 for x in comparisons if x['pred_b'] == EM_IDX)
    b_other_minor = sum(1 for x in comparisons if is_minor(x['pred_b']) and x['pred_b'] != EM_IDX)
    b_major = sum(1 for x in comparisons if is_major(x['pred_b']))
    b_n = sum(1 for x in comparisons if x['pred_b'] == 0)

    print(f"  {label_b} also predicts Em:       {b_also_em} ({100*b_also_em/len(comparisons):.1f}%)")
    print(f"  {label_b} predicts other minor:    {b_other_minor}")
    print(f"  {label_b} predicts major:          {b_major}")
    print(f"  {label_b} predicts N:              {b_n}")
    print(f"  {label_b} correct on these beats:  {b_correct} ({100*b_correct/len(comparisons):.1f}%)")

    # What does B predict instead? Bucketed.
    print(f"\n  {label_b}'s predictions on {label_a}'s Em FP beats:")
    b_pred_counter = Counter(x['pred_b_name'] for x in comparisons)
    for label, count in b_pred_counter.most_common(15):
        correct = sum(1 for x in comparisons
                      if x['pred_b_name'] == label and x['b_correct'])
        print(f"    {label:6s}: {count:4d}  (correct: {correct})")

    # Cross-tab: true label × B's prediction (for the most common true labels)
    print(f"\n  Cross-tab: true label × {label_b} prediction (top confusions):")
    true_labels_top = [t for t, _ in Counter(x['true_name'] for x in comparisons).most_common(8)]
    for tl in true_labels_top:
        subset = [x for x in comparisons if x['true_name'] == tl]
        b_preds = Counter(x['pred_b_name'] for x in subset)
        b_right = sum(1 for x in subset if x['b_correct'])
        top3 = ', '.join(f"{p}:{c}" for p, c in b_preds.most_common(3))
        print(f"    true={tl:5s} ({len(subset):3d} beats, {label_b} correct: {b_right}) "
              f"→ {top3}")

    return comparisons


# ─────────────────────────────────────────────────────────────────
# Part 3: Post-hoc Em bias penalty test
# ─────────────────────────────────────────────────────────────────

def posthoc_em_penalty_sweep(model, songs, device, label):
    """Extract emissions, sweep Em penalty + self-bias, find improvements."""
    print(f"\n{'='*70}")
    print(f"Part 3: Post-hoc Em emission penalty sweep ({label})")
    print(f"{'='*70}")

    song_emissions = extract_emissions(model, songs, device)

    # Baseline: CRF decode (already done, but recompute metrics from Viterbi for comparison)
    # First find the baseline self-bias that matches CRF behavior
    print(f"\n  Baseline (no Em penalty) across self-biases:")
    print(f"  {'self_bias':>9s} {'em_pen':>6s} | {'WCSR':>6s} {'Major':>6s} {'Minor':>6s} "
          f"{'Flip':>6s} {'Em_FP':>5s} {'Em_Rec':>6s}")
    print(f"  {'-'*9}-{'-'*6}-+-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*5}-{'-'*6}")

    self_biases = [0, 2, 4, 6, 8, 10, 15, 20]
    em_penalties = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    best_wcsr = 0
    best_config = None

    for sb in self_biases:
        for ep in em_penalties:
            m = evaluate_with_bias(song_emissions, sb, em_penalty=ep)
            tag = ' ***' if m['wcsr'] > best_wcsr else ''
            if m['wcsr'] > best_wcsr:
                best_wcsr = m['wcsr']
                best_config = (sb, ep, m)
            if ep == 0.0 or tag:
                print(f"  {sb:9.1f} {ep:6.1f} | {m['wcsr']:6.3f} {m['major_acc']:6.3f} "
                      f"{m['minor_acc']:6.3f} {m['flip_rate']:6.3f} {m['em_fp']:5d} "
                      f"{m['em_recall']:6.3f}{tag}")

    # Print full sweep for best self-bias
    if best_config:
        best_sb = best_config[0]
        print(f"\n  Full Em penalty sweep at self_bias={best_sb}:")
        print(f"  {'em_pen':>6s} | {'WCSR':>6s} {'Major':>6s} {'Minor':>6s} "
              f"{'Flip':>6s} {'Em_FP':>5s} {'Em_Rec':>6s}")
        print(f"  {'-'*6}-+-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*5}-{'-'*6}")
        for ep in em_penalties:
            m = evaluate_with_bias(song_emissions, best_sb, em_penalty=ep)
            tag = ' ← best' if (best_sb, ep) == (best_config[0], best_config[1]) else ''
            print(f"  {ep:6.1f} | {m['wcsr']:6.3f} {m['major_acc']:6.3f} "
                  f"{m['minor_acc']:6.3f} {m['flip_rate']:6.3f} {m['em_fp']:5d} "
                  f"{m['em_recall']:6.3f}{tag}")

    print(f"\n  Best config: self_bias={best_config[0]}, em_penalty={best_config[1]}")
    print(f"  Best WCSR: {best_config[2]['wcsr']:.4f}")

    # Also try penalty on ALL minor chords to compare
    print(f"\n  Comparison: penalty on ALL minor chords (13-24) vs Em-only:")
    print(f"  {'penalty':>7s} {'target':>8s} | {'WCSR':>6s} {'Major':>6s} {'Minor':>6s} {'Flip':>6s}")
    print(f"  {'-'*7}-{'-'*8}-+-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}")
    for pen in [0.0, 1.0, 2.0]:
        for target in ['Em-only', 'all-min']:
            log_trans = build_transition_matrix(best_config[0])
            all_true = []
            all_pred = []
            sf = []
            for song in song_emissions:
                le = song['log_emit'].copy()
                if target == 'Em-only':
                    le[EM_IDX, :] -= pen
                else:
                    le[13:25, :] -= pen
                path, _ = viterbi_decode(le, log_trans)
                path = smooth_isolated(path)
                path = path[:len(song['tier1_labels'])]
                all_true.append(song['tier1_labels'])
                all_pred.append(path)
                sf.append(np.sum(path[1:] != path[:-1]) / max(len(path) - 1, 1))
            at = np.concatenate(all_true)
            ap = np.concatenate(all_pred)
            w = np.mean(at == ap)
            mj = (at[( at >= 1) & (at <= 12)] == ap[(at >= 1) & (at <= 12)]).mean()
            mn = (at[(at >= 13) & (at <= 24)] == ap[(at >= 13) & (at <= 24)]).mean()
            print(f"  {pen:7.1f} {target:>8s} | {w:6.3f} {mj:6.3f} {mn:6.3f} {np.mean(sf):6.3f}")


def main():
    parser = argparse.ArgumentParser(description='Em over-prediction deep-dive')
    parser.add_argument('--data', required=True, help='Path to features_v2.npz')
    parser.add_argument('--model-a', required=True, help='Freeze5 checkpoint')
    parser.add_argument('--model-b', required=True, help='3B checkpoint')
    parser.add_argument('--label-a', default='freeze5')
    parser.add_argument('--label-b', default='3B')
    parser.add_argument('--gold-only', action='store_true')
    parser.add_argument('--feature-dim', type=int, default=60)
    parser.add_argument('--split', default='test', choices=['test', 'val'])
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

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

    # Decode both
    print(f"\nDecoding {args.split} songs...")
    results_a = decode_and_collect(model_a, songs, device)
    results_b = decode_and_collect(model_b, songs, device)

    # Part 1
    fp_beats = analyze_em_false_positives(results_a, args.label_a)

    # Part 2
    compare_on_em_set(results_a, results_b, fp_beats, args.label_a, args.label_b)

    # Part 3
    posthoc_em_penalty_sweep(model_a, songs, device, args.label_a)


if __name__ == '__main__':
    main()
