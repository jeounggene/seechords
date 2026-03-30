#!/usr/bin/env python3
"""Em-specific calibration sweep: emission penalty, transition penalty, or both.

Sweeps 3 intervention types on val, then evaluates best config on test.

Usage:
    python -m v2.em_calibration \
        --data data/features_v2.npz \
        --model models/chord_transformer_crf_freeze5.pt
"""
import sys
import os
import argparse
import numpy as np
import torch

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from v2.transformer_model import ChordTransformerCRF
from v2.transformer_data import load_data, split_songs, get_song_data
from v2.decode import viterbi_decode, smooth_isolated
from shared.chord_vocab import TIER1_VOCAB

EM_IDX = TIER1_VOCAB.index('Em')


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


def extract_emissions(model, songs, device):
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


def build_transition_matrix(self_bias, n_states=25,
                            em_trans_into_penalty=0.0,
                            em_self_penalty=0.0):
    """Build log-transition matrix with optional Em-specific penalties.

    em_trans_into_penalty: penalty on transitions FROM any state INTO Em
    em_self_penalty: additional penalty on Em→Em self-transition
    """
    log_trans = np.zeros((n_states, n_states))
    log_trans[np.diag_indices(n_states)] = self_bias

    # Em-specific transition penalties
    if em_trans_into_penalty != 0.0:
        log_trans[:, EM_IDX] -= em_trans_into_penalty  # all states → Em
    if em_self_penalty != 0.0:
        log_trans[EM_IDX, EM_IDX] -= em_self_penalty   # Em → Em persistence

    # Log-normalize rows
    for i in range(n_states):
        row_lse = np.log(np.sum(np.exp(log_trans[i])))
        log_trans[i] -= row_lse
    return log_trans


def evaluate(song_emissions, self_bias, em_emit_pen=0.0,
             em_trans_into_pen=0.0, em_self_pen=0.0):
    """Evaluate with all three intervention types."""
    log_trans = build_transition_matrix(
        self_bias, em_trans_into_penalty=em_trans_into_pen,
        em_self_penalty=em_self_pen,
    )

    all_true = []
    all_pred = []
    song_flips = []

    for song in song_emissions:
        log_emit = song['log_emit'].copy()
        if em_emit_pen != 0.0:
            log_emit[EM_IDX, :] -= em_emit_pen
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

    em_fp = int(np.sum((all_pred == EM_IDX) & (all_true != EM_IDX)))
    em_total_true = int(np.sum(all_true == EM_IDX))
    em_tp = int(np.sum((all_pred == EM_IDX) & (all_true == EM_IDX)))
    em_recall = em_tp / max(em_total_true, 1)

    return {
        'wcsr': float(wcsr),
        'major_acc': float(major_acc),
        'minor_acc': float(minor_acc),
        'flip_rate': float(avg_flip),
        'em_fp': em_fp,
        'em_recall': float(em_recall),
    }


def print_row(label, m, highlight=False):
    tag = ' ***' if highlight else ''
    print(f"  {label:<35s} | {m['wcsr']:6.3f} {m['major_acc']:6.3f} "
          f"{m['minor_acc']:6.3f} {m['flip_rate']:6.3f} {m['em_fp']:5d} "
          f"{m['em_recall']:6.3f}{tag}")


def main():
    parser = argparse.ArgumentParser(description='Em calibration sweep')
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gold-only', action='store_true')
    parser.add_argument('--feature-dim', type=int, default=60)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device('mps')
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load
    data = load_data(args.data, gold_only=args.gold_only, feature_dim=args.feature_dim)
    train_set, val_set, test_set = split_songs(data)
    model, ckpt = load_model(args.model, device)

    val_songs = get_song_data(data, val_set)
    test_songs = get_song_data(data, test_set)

    val_emit = extract_emissions(model, val_songs, device)
    test_emit = extract_emissions(model, test_songs, device)

    header = (f"  {'config':<35s} | {'WCSR':>6s} {'Major':>6s} {'Minor':>6s} "
              f"{'Flip':>6s} {'Em_FP':>5s} {'Em_Rec':>6s}")
    sep = f"  {'-'*35}-+-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*5}-{'-'*6}"

    # ── Sweep on VAL ──
    self_biases = [2, 4, 6, 8]
    em_emit_pens = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    em_trans_pens = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    em_self_pens = [0.0, 1.0, 2.0, 3.0]

    # ── A: Emission-only ──
    print(f"\n{'='*70}")
    print(f"A. Emission penalty only (VAL)")
    print(f"{'='*70}")
    print(header)
    print(sep)

    best_a = {'wcsr': 0}
    best_a_cfg = {}
    for sb in self_biases:
        for ep in em_emit_pens:
            m = evaluate(val_emit, sb, em_emit_pen=ep)
            is_best = m['wcsr'] > best_a['wcsr']
            if is_best:
                best_a = m
                best_a_cfg = {'sb': sb, 'ep': ep}
            if ep == 0.0 or is_best:
                print_row(f"sb={sb} ep={ep:.1f}", m, is_best)

    print(f"\n  Best A: sb={best_a_cfg['sb']}, em_pen={best_a_cfg['ep']:.1f} "
          f"→ WCSR={best_a['wcsr']:.4f}")

    # ── B: Transition-into penalty only ──
    print(f"\n{'='*70}")
    print(f"B. Transition-into-Em penalty only (VAL)")
    print(f"{'='*70}")
    print(header)
    print(sep)

    best_b = {'wcsr': 0}
    best_b_cfg = {}
    for sb in self_biases:
        for tp in em_trans_pens:
            m = evaluate(val_emit, sb, em_trans_into_pen=tp)
            is_best = m['wcsr'] > best_b['wcsr']
            if is_best:
                best_b = m
                best_b_cfg = {'sb': sb, 'tp': tp}
            if tp == 0.0 or is_best:
                print_row(f"sb={sb} trans_into={tp:.1f}", m, is_best)

    print(f"\n  Best B: sb={best_b_cfg['sb']}, trans_into={best_b_cfg['tp']:.1f} "
          f"→ WCSR={best_b['wcsr']:.4f}")

    # ── C: Em self-persistence penalty only ──
    print(f"\n{'='*70}")
    print(f"C. Em self-persistence penalty only (VAL)")
    print(f"{'='*70}")
    print(header)
    print(sep)

    best_c = {'wcsr': 0}
    best_c_cfg = {}
    for sb in self_biases:
        for sp in em_self_pens:
            m = evaluate(val_emit, sb, em_self_pen=sp)
            is_best = m['wcsr'] > best_c['wcsr']
            if is_best:
                best_c = m
                best_c_cfg = {'sb': sb, 'sp': sp}
            if sp == 0.0 or is_best:
                print_row(f"sb={sb} self_pen={sp:.1f}", m, is_best)

    print(f"\n  Best C: sb={best_c_cfg['sb']}, self_pen={best_c_cfg['sp']:.1f} "
          f"→ WCSR={best_c['wcsr']:.4f}")

    # ── D: Emission + transition combined ──
    print(f"\n{'='*70}")
    print(f"D. Combined: emission + transition-into penalty (VAL)")
    print(f"{'='*70}")
    print(header)
    print(sep)

    best_d = {'wcsr': 0}
    best_d_cfg = {}
    # Search around the best emission config, add transition penalty
    sb_range = [best_a_cfg['sb'] - 2, best_a_cfg['sb'], best_a_cfg['sb'] + 2]
    sb_range = [s for s in sb_range if s >= 0]
    ep_range = [max(0, best_a_cfg['ep'] - 1.0), best_a_cfg['ep'] - 0.5,
                best_a_cfg['ep'], best_a_cfg['ep'] + 0.5]
    ep_range = [e for e in ep_range if e >= 0]

    for sb in sb_range:
        for ep in ep_range:
            for tp in em_trans_pens:
                m = evaluate(val_emit, sb, em_emit_pen=ep, em_trans_into_pen=tp)
                is_best = m['wcsr'] > best_d['wcsr']
                if is_best:
                    best_d = m
                    best_d_cfg = {'sb': sb, 'ep': ep, 'tp': tp}
                    print_row(f"sb={sb} ep={ep:.1f} tp={tp:.1f}", m, True)

    print(f"\n  Best D: sb={best_d_cfg['sb']}, em_pen={best_d_cfg['ep']:.1f}, "
          f"trans_into={best_d_cfg['tp']:.1f} → WCSR={best_d['wcsr']:.4f}")

    # ── Summary & Test ──
    print(f"\n{'='*70}")
    print(f"VAL SUMMARY")
    print(f"{'='*70}")
    configs = [
        ('A: emission only', best_a, best_a_cfg),
        ('B: transition-into only', best_b, best_b_cfg),
        ('C: self-persist only', best_c, best_c_cfg),
        ('D: emission + transition', best_d, best_d_cfg),
    ]
    for name, m, cfg in configs:
        print(f"  {name:<30s}: WCSR={m['wcsr']:.4f}  minor={m['minor_acc']:.3f}  "
              f"flip={m['flip_rate']:.3f}  em_fp={m['em_fp']}  | {cfg}")

    # Pick overall best
    best_name, best_m, best_cfg = max(configs, key=lambda x: x[1]['wcsr'])
    print(f"\n  Overall best: {best_name} → {best_cfg}")

    # ── TEST with val-selected config ──
    print(f"\n{'='*70}")
    print(f"TEST EVALUATION (val-selected config)")
    print(f"{'='*70}")
    print(header)
    print(sep)

    # Baseline on test
    for sb in [4, 6, 8]:
        m = evaluate(test_emit, sb)
        print_row(f"baseline sb={sb}", m)

    # Best from each category on test
    for name, _, cfg in configs:
        sb = cfg.get('sb', 4)
        ep = cfg.get('ep', 0.0)
        tp = cfg.get('tp', 0.0)
        sp = cfg.get('sp', 0.0)
        m = evaluate(test_emit, sb, em_emit_pen=ep,
                     em_trans_into_pen=tp, em_self_pen=sp)
        print_row(f"{name} {cfg}", m)


if __name__ == '__main__':
    main()
