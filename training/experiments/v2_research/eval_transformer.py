#!/usr/bin/env python3
"""Evaluate a trained Transformer chord model and compare with RF baseline.

Usage:
    python -m v2.eval_transformer --checkpoint models/chord_transformer.pt --data data/features_v2.npz
"""
import sys
import os
import argparse
import numpy as np
import torch

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from v2.transformer_model import ChordTransformer, ChordTransformerCRF
from v2.transformer_data import load_data, split_songs, get_song_data
from v2.train_transformer import evaluate_on_songs, decode_song, emissions_from_probs


def load_model(checkpoint_path, device):
    """Load a trained Transformer checkpoint (CE or CRF variant)."""
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
            emission_dropout=hp.get('emission_dropout', 0.0),
            emission_noise_std=hp.get('emission_noise_std', 0.0),
            em_emission_bias=hp.get('em_emission_bias', 0.0),
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
    return model, ckpt


def print_header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_metrics(label, metrics):
    print(f"  {label:<25s} WCSR={metrics['wcsr']:.3f}  maj={metrics['major_acc']:.3f}  "
          f"min={metrics['minor_acc']:.3f}  root={metrics['root_acc']:.3f}  "
          f"flip={metrics['flip_rate']:.3f}")


def print_confusion(metrics):
    cp = metrics['confusion_pairs']
    if cp['A_total'] > 0 or cp['D_total'] > 0 or cp['C_total'] > 0:
        parts = []
        if cp['A_total'] > 0:
            parts.append(f"Am→A: {cp['Am_to_A']}/{cp['A_total']} ({cp['Am_to_A']/max(cp['A_total'],1)*100:.0f}%)")
        if cp['D_total'] > 0:
            parts.append(f"Dm→D: {cp['Dm_to_D']}/{cp['D_total']} ({cp['Dm_to_D']/max(cp['D_total'],1)*100:.0f}%)")
        if cp['C_total'] > 0:
            parts.append(f"Cm→C: {cp['Cm_to_C']}/{cp['C_total']} ({cp['Cm_to_C']/max(cp['C_total'],1)*100:.0f}%)")
        print(f"  Confusion pairs: {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer chord model')
    parser.add_argument('--checkpoint', required=True, help='Path to trained checkpoint .pt')
    parser.add_argument('--data', required=True, help='Path to features_v2.npz')
    parser.add_argument('--split', choices=['val', 'test', 'both'], default='both',
                        help='Which split(s) to evaluate')
    parser.add_argument('--per-song', action='store_true', help='Show per-song breakdown')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device('mps')
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load model
    model, ckpt = load_model(args.checkpoint, device)
    hp = ckpt['hyperparams']
    print(f"Model: d={hp['d_model']}, heads={hp['nhead']}, layers={hp['num_layers']}, "
          f"d_ff={hp['d_ff']}, dropout={hp['dropout']}")
    print(f"Trained epoch: {ckpt['best_epoch']}, val WCSR: {ckpt['best_val_wcsr']:.4f}")
    print(f"Gold only: {ckpt.get('gold_only', 'unknown')}")

    # Load data with same gold_only setting as training
    gold_only = ckpt.get('gold_only', True)
    feature_dim = hp.get('input_dim', 48)
    data = load_data(args.data, gold_only=gold_only, feature_dim=feature_dim)

    # Split
    _, val_set, test_set = split_songs(data)

    model_type = ckpt.get('model_type', 'ChordTransformer')
    is_crf = model_type == 'ChordTransformerCRF'

    if not is_crf:
        tier1_trans = ckpt['tier1_transition']

    splits_to_eval = []
    if args.split in ('val', 'both'):
        splits_to_eval.append(('Validation', val_set))
    if args.split in ('test', 'both'):
        splits_to_eval.append(('Test', test_set))

    for split_name, song_set in splits_to_eval:
        songs = get_song_data(data, song_set)

        print_header(f"{split_name} Set ({len(songs)} songs)")

        if is_crf:
            from v2.train_transformer_crf import evaluate_crf_on_songs
            metrics = evaluate_crf_on_songs(model, songs, device)
            print_metrics('Transformer+CRF', metrics)
        else:
            metrics = evaluate_on_songs(model, songs, tier1_trans, device)
            print_metrics('Transformer+Viterbi', metrics)
        print_confusion(metrics)

        # Compare with RF baseline (from checkpoint stored metrics, if test)
        if split_name == 'Test' and 'test_metrics' in ckpt:
            print(f"\n  Reference (from training):")
            tm = ckpt['test_metrics']
            print(f"  {'(saved at train time)':<25s} WCSR={tm['wcsr']:.3f}  "
                  f"maj={tm['major_acc']:.3f}  min={tm['minor_acc']:.3f}  "
                  f"root={tm['root_acc']:.3f}  flip={tm['flip_rate']:.3f}")

        if args.per_song:
            print(f"\n  {'Song':<55s} {'Beats':>5s} {'Acc':>6s} {'Flip':>6s}")
            print(f"  {'-'*55} {'-'*5} {'-'*6} {'-'*6}")
            for r in sorted(metrics['song_results'], key=lambda x: x['accuracy']):
                prov = 'G' if r['provenance'] == 'gold' else 'S'
                print(f"  [{prov}] {r['name']:<52s} {r['n_beats']:5d} {r['accuracy']:6.3f} {r['flip_rate']:6.3f}")


def evaluate_emission_quality(model, songs, device):
    """Debug helper: check raw emission quality before decoding.

    Prints frame-level root and quality accuracy (no Viterbi).
    """
    model.eval()
    all_root_true = []
    all_root_pred = []
    all_q3_true = []
    all_q3_pred = []

    with torch.no_grad():
        for song in songs:
            X = torch.from_numpy(song['features']).unsqueeze(0).to(device)
            root_probs, quality_probs = model.predict_probs(X)
            root_probs = root_probs[0].cpu().numpy()
            quality_probs = quality_probs[0].cpu().numpy()

            root_pred = np.argmax(root_probs, axis=1)
            q3_pred = np.argmax(quality_probs, axis=1)

            all_root_true.append(song['root_labels'])
            all_root_pred.append(root_pred)
            all_q3_true.append(song['quality3_labels'])
            all_q3_pred.append(q3_pred)

    all_root_true = np.concatenate(all_root_true)
    all_root_pred = np.concatenate(all_root_pred)
    all_q3_true = np.concatenate(all_q3_true)
    all_q3_pred = np.concatenate(all_q3_pred)

    root_acc = np.mean(all_root_true == all_root_pred)
    q3_acc = np.mean(all_q3_true == all_q3_pred)

    # Per-quality accuracy
    for q, label in enumerate(['N', 'maj', 'min']):
        mask = all_q3_true == q
        if mask.sum() > 0:
            acc = np.mean(all_q3_pred[mask] == q)
            print(f"  Quality '{label}': {acc:.3f} ({mask.sum()} beats)")

    print(f"  Frame-level root acc: {root_acc:.3f}")
    print(f"  Frame-level quality acc: {q3_acc:.3f}")

    return root_acc, q3_acc


if __name__ == '__main__':
    main()
