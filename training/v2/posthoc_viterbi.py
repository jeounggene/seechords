#!/usr/bin/env python3
"""Post-hoc Viterbi sweep: extract emissions from trained models, decode externally.

Sweeps self-transition bias in external Viterbi to test whether flip rate
can be reduced at decode time without changing training.

Usage:
    python -m v2.posthoc_viterbi \
        --data data/features_v2.npz \
        --checkpoints models/chord_transformer_gold_60d.pt models/chord_transformer_crf_t5.pt
"""
import sys, os, argparse
import numpy as np
import torch

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from v2.transformer_model import ChordTransformer, ChordTransformerCRF
from v2.transformer_data import load_data, split_songs, get_song_data
from v2.decode import viterbi_decode, smooth_isolated


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt['hyperparams']
    model_type = ckpt.get('model_type', 'ChordTransformer')

    if model_type == 'ChordTransformerCRF':
        model = ChordTransformerCRF(
            input_dim=hp['input_dim'],
            d_model=hp['d_model'],
            nhead=hp['nhead'],
            num_layers=hp['num_layers'],
            d_ff=hp['d_ff'],
            dropout=hp['dropout'],
            crf_self_bias=hp.get('crf_self_bias', 2.0),
            emission_temp=hp.get('emission_temp', 1.0),
            em_emission_bias=hp.get('em_emission_bias', 0.0),
            emission_dropout=hp.get('emission_dropout', 0.0),
            emission_noise_std=hp.get('emission_noise_std', 0.0),
            emission_mode=hp.get('emission_mode', 'direct'),
            n_qualities=hp.get('n_qualities', 3),
            use_key_aux=hp.get('key_aux_weight', 0.0) > 0.0,
            key_condition_quality=hp.get('key_condition_quality', False),
        )
    else:
        model = ChordTransformer(
            input_dim=hp['input_dim'],
            d_model=hp['d_model'],
            nhead=hp['nhead'],
            num_layers=hp['num_layers'],
            d_ff=hp['d_ff'],
            dropout=hp['dropout'],
        )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, ckpt, model_type


def extract_emissions(model, model_type, songs, device):
    """Extract raw log-emissions for each song.

    For CRF models: tier1_head logits (before temperature scaling).
    For CE models:  log-softmax of composed root*quality probabilities.

    Returns list of (log_emissions, tier1_labels) per song.
    """
    results = []
    with torch.no_grad():
        for song in songs:
            X = torch.from_numpy(song['features']).unsqueeze(0).to(device)
            t1_true = song['tier1_labels']

            if model_type == 'ChordTransformerCRF':
                h = model._encode(X)
                if getattr(model, 'emission_mode', 'direct') == 'hybrid':
                    log_emit = model._tier1_emissions_hybrid(
                        model.root_head(h), model.tier1_head(h)
                    )[0].cpu().numpy().T
                else:
                    logits = model.tier1_head(h)
                    log_emit = logits[0].cpu().numpy().T
            else:
                # CE model: compose root and quality probs into tier1
                root_probs, quality_probs = model.predict_probs(X)
                root_p = root_probs[0].cpu().numpy()     # (seq_len, 13)
                qual_p = quality_probs[0].cpu().numpy()   # (seq_len, 3)
                n_beats = root_p.shape[0]

                # Compose into 25-state emissions
                emit = np.zeros((25, n_beats))
                emit[0, :] = root_p[:, 0] * qual_p[:, 0]  # N
                for note in range(12):
                    ri = note + 1
                    emit[1 + note, :] = root_p[:, ri] * qual_p[:, 1]   # major
                    emit[13 + note, :] = root_p[:, ri] * qual_p[:, 2]  # minor
                emit = np.clip(emit, 1e-10, None)
                log_emit = np.log(emit)

            results.append({
                'log_emit': log_emit,
                'tier1_labels': t1_true,
                'name': song['name'],
                'provenance': song['provenance'],
            })
    return results


def build_transition_matrix(self_bias, n_states=25):
    """Build log-transition matrix with uniform off-diagonal + self-transition bias.

    self_bias: additive log-space bonus for staying in same state.
    """
    # Uniform base
    log_trans = np.zeros((n_states, n_states))
    # Add self-transition bias on diagonal
    log_trans[np.diag_indices(n_states)] = self_bias
    # Log-normalize rows
    for i in range(n_states):
        row_lse = np.log(np.sum(np.exp(log_trans[i])))
        log_trans[i] -= row_lse
    return log_trans


def evaluate_with_viterbi(song_emissions, log_trans, use_smooth=True):
    """Run Viterbi + optional smooth_isolated on all songs, compute metrics."""
    all_true = []
    all_pred = []
    song_flips = []

    for song in song_emissions:
        log_emit = song['log_emit']
        t1_true = song['tier1_labels']

        path, _ = viterbi_decode(log_emit, log_trans)
        if use_smooth:
            path = smooth_isolated(path)

        # Trim to match
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

    pred_roots = np.where(all_pred == 0, 0, np.where(all_pred <= 12, all_pred, all_pred - 12))
    true_roots = np.where(all_true == 0, 0, np.where(all_true <= 12, all_true, all_true - 12))
    root_acc = np.mean(true_roots == pred_roots)

    avg_flip = np.mean(song_flips)

    return {
        'wcsr': float(wcsr),
        'major_acc': float(major_acc),
        'minor_acc': float(minor_acc),
        'root_acc': float(root_acc),
        'flip_rate': float(avg_flip),
    }


def main():
    parser = argparse.ArgumentParser(description='Post-hoc Viterbi sweep')
    parser.add_argument('--data', required=True, help='Path to features_v2.npz')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Checkpoint paths to evaluate')
    parser.add_argument('--self-biases', nargs='+', type=float,
                        default=[0, 2, 4, 6, 8, 10, 15, 20],
                        help='Self-transition biases to sweep')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device('mps')
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for ckpt_path in args.checkpoints:
        model, ckpt, model_type = load_model(ckpt_path, device)
        hp = ckpt['hyperparams']
        name = os.path.basename(ckpt_path)
        temp = hp.get('emission_temp', 1.0)

        print(f"\n{'='*80}")
        print(f"Model: {name}  (type={model_type}, temp={temp})")
        print(f"{'='*80}")

        # Load data matching training config
        gold_only = ckpt.get('gold_only', True)
        feature_dim = hp.get('input_dim', 24)
        data = load_data(args.data, gold_only=gold_only, feature_dim=feature_dim)
        _, val_set, test_set = split_songs(data)
        song_set = val_set if args.split == 'val' else test_set
        songs = get_song_data(data, song_set)

        print(f"Extracting emissions from {len(songs)} {args.split} songs...")
        song_emissions = extract_emissions(model, model_type, songs, device)

        # Also compute argmax baseline (no Viterbi)
        total_true = np.concatenate([s['tier1_labels'] for s in song_emissions])
        total_argmax = np.concatenate([np.argmax(s['log_emit'], axis=0) for s in song_emissions])
        argmax_acc = np.mean(total_true == total_argmax)
        argmax_flips = []
        for s in song_emissions:
            p = np.argmax(s['log_emit'], axis=0)
            argmax_flips.append(np.sum(p[1:] != p[:-1]) / max(len(p) - 1, 1))
        print(f"  Argmax baseline: acc={argmax_acc:.3f}, avg_flip={np.mean(argmax_flips):.3f}")

        # Sweep
        print(f"\n  {'self_bias':>10s} | {'WCSR':>6s} {'Major':>6s} {'Minor':>6s} {'Root':>6s} {'Flip':>6s}")
        print(f"  {'-'*10}-+-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}")

        for sb in args.self_biases:
            log_trans = build_transition_matrix(sb, n_states=25)
            m = evaluate_with_viterbi(song_emissions, log_trans)
            print(f"  {sb:10.1f} | {m['wcsr']:6.3f} {m['major_acc']:6.3f} "
                  f"{m['minor_acc']:6.3f} {m['root_acc']:6.3f} {m['flip_rate']:6.3f}")


if __name__ == '__main__':
    main()
