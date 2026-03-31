#!/usr/bin/env python3
"""Train Transformer chord model (Phase 1: gold-only baseline).

Usage:
    python -m v2.train_transformer --data data/features_v2.npz --out models/chord_transformer.pt
    python -m v2.train_transformer --data data/features_v2.npz --out models/chord_transformer.pt --gold-only
"""
import sys
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_ROOT)

from v2.transformer_model import ChordTransformer
from v2.transformer_data import (
    load_data, split_songs, compute_quality3_weights,
    ChordWindowDataset, get_song_data,
)
from v2.decode import viterbi_decode, soften_transitions, smooth_isolated
from shared.chord_vocab import TIER1_VOCAB


def emissions_from_probs(root_probs, quality_probs):
    """Convert Transformer root + quality probs to 25-class hybrid emissions.

    Args:
        root_probs:    (n_beats, 13) — softmax over [N, C, C#, ..., B]
        quality_probs: (n_beats, 3)  — softmax over [N, maj, min]

    Returns:
        log_emit: (25, n_beats) — log-emission matrix for Viterbi
    """
    n_beats = root_probs.shape[0]
    n_tier1 = 25
    emit = np.zeros((n_tier1, n_beats), dtype=np.float64)

    # N chord: P(root=N) * P(qual=N)
    emit[0, :] = root_probs[:, 0] * quality_probs[:, 0]

    # For each chromatic root 1-12 -> major and minor
    for note_idx in range(12):
        r_prob = root_probs[:, note_idx + 1]        # P(root = this note)
        q_maj = quality_probs[:, 1]                   # P(quality = maj)
        q_min = quality_probs[:, 2]                   # P(quality = min)
        emit[1 + note_idx, :] = r_prob * q_maj       # major chord
        emit[13 + note_idx, :] = r_prob * q_min      # minor chord

    log_emit = np.log(np.clip(emit, 1e-10, None))
    return log_emit


def decode_song(root_probs, quality_probs, tier1_transition, self_prob=0.40, floor=0.015):
    """Run Viterbi decoding on Transformer emissions for one song.

    Returns: (path, log_lik) where path is (n_beats,) int array.
    """
    log_emit = emissions_from_probs(root_probs, quality_probs)
    trans_soft = soften_transitions(tier1_transition, self_prob=self_prob, floor=floor)
    log_trans = np.log(np.clip(trans_soft, 1e-10, None))
    path, log_lik = viterbi_decode(log_emit, log_trans)
    path = smooth_isolated(path)
    return path, log_lik


def evaluate_on_songs(model, songs, tier1_transition, device):
    """Evaluate Transformer on full songs via Viterbi decoding.

    Returns dict with: wcsr, major_acc, minor_acc, root_acc, flip_rate,
                        per_song results, confusion_pairs
    """
    model.eval()
    all_true = []
    all_pred = []
    all_root_true = []
    all_root_pred = []
    song_results = []
    # Track specific confusion pairs: Am->A, Dm->D, Cm->C
    confusion_pairs = {'Am_to_A': 0, 'Dm_to_D': 0, 'Cm_to_C': 0,
                       'A_total': 0, 'D_total': 0, 'C_total': 0}

    with torch.no_grad():
        for song in songs:
            X = torch.from_numpy(song['features']).unsqueeze(0).to(device)
            root_probs, quality_probs = model.predict_probs(X)
            root_probs = root_probs[0].cpu().numpy()      # (n_beats, 13)
            quality_probs = quality_probs[0].cpu().numpy()  # (n_beats, 3)

            t1_true = song['tier1_labels']
            path, _ = decode_song(root_probs, quality_probs, tier1_transition)

            # Beat-level metrics
            correct = np.sum(path == t1_true)
            total = len(t1_true)
            acc = correct / max(total, 1)

            # Root accuracy (from predicted tier1 path)
            pred_roots = np.where(path == 0, 0, np.where(path <= 12, path, path - 12))
            true_roots = song['root_labels']

            # Flip rate
            n_flips = np.sum(path[1:] != path[:-1]) if len(path) > 1 else 0
            flip_rate = n_flips / max(len(path) - 1, 1)

            all_true.append(t1_true)
            all_pred.append(path)
            all_root_true.append(true_roots)
            all_root_pred.append(pred_roots)

            # Confusion pairs: Am(22)->A(10), Dm(15)->D(3), Cm(13)->C(1)
            am_mask = t1_true == 22  # Am
            if am_mask.sum() > 0:
                confusion_pairs['A_total'] += int(am_mask.sum())
                confusion_pairs['Am_to_A'] += int(np.sum(path[am_mask] == 10))
            dm_mask = t1_true == 15  # Dm
            if dm_mask.sum() > 0:
                confusion_pairs['D_total'] += int(dm_mask.sum())
                confusion_pairs['Dm_to_D'] += int(np.sum(path[dm_mask] == 3))
            cm_mask = t1_true == 13  # Cm
            if cm_mask.sum() > 0:
                confusion_pairs['C_total'] += int(cm_mask.sum())
                confusion_pairs['Cm_to_C'] += int(np.sum(path[cm_mask] == 1))

            song_results.append({
                'name': song['name'],
                'provenance': song['provenance'],
                'n_beats': total,
                'accuracy': float(acc),
                'flip_rate': float(flip_rate),
            })

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_root_true = np.concatenate(all_root_true)
    all_root_pred = np.concatenate(all_root_pred)

    wcsr = np.sum(all_true == all_pred) / max(len(all_true), 1)

    maj_mask = (all_true >= 1) & (all_true <= 12)
    min_mask = (all_true >= 13) & (all_true <= 24)

    major_acc = np.sum(all_true[maj_mask] == all_pred[maj_mask]) / max(maj_mask.sum(), 1) if maj_mask.sum() > 0 else 0
    minor_acc = np.sum(all_true[min_mask] == all_pred[min_mask]) / max(min_mask.sum(), 1) if min_mask.sum() > 0 else 0
    root_acc = np.sum(all_root_true == all_root_pred) / max(len(all_root_true), 1)

    avg_flip = np.mean([r['flip_rate'] for r in song_results])

    return {
        'wcsr': float(wcsr),
        'major_acc': float(major_acc),
        'minor_acc': float(minor_acc),
        'root_acc': float(root_acc),
        'flip_rate': float(avg_flip),
        'confusion_pairs': confusion_pairs,
        'song_results': song_results,
    }


def train(args):
    # ── Device ──
    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device('mps')
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── Load data ──
    print(f"\n1. Loading data from {args.data}...")
    data = load_data(args.data, gold_only=args.gold_only, feature_dim=args.feature_dim)
    n_songs = len(data['provenance'])
    n_beats = len(data['X'])
    print(f"   {n_beats} beats from {n_songs} songs")
    print(f"   Feature dim: {data['X'].shape[1]}")

    # ── Split ──
    print("\n2. Splitting data...")
    train_set, val_set, test_set = split_songs(data)
    train_beats = np.sum([s in train_set for s in data['song_ids']])
    val_beats = np.sum([s in val_set for s in data['song_ids']])
    test_beats = np.sum([s in test_set for s in data['song_ids']])
    print(f"   Train: {len(train_set)} songs, {train_beats} beats")
    print(f"   Val:   {len(val_set)} songs, {val_beats} beats")
    print(f"   Test:  {len(test_set)} songs, {test_beats} beats")

    # ── Datasets ──
    print("\n3. Building datasets...")
    augment_shifts = list(range(12)) if args.augment else None
    train_ds = ChordWindowDataset(data, train_set, window=args.window, stride=args.stride,
                                  augment_shifts=augment_shifts)
    print(f"   Train windows: {len(train_ds)}" +
          (f" (12x augmented)" if args.augment else ""))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)

    val_songs = get_song_data(data, val_set)
    test_songs = get_song_data(data, test_set)
    print(f"   Val songs: {len(val_songs)}, Test songs: {len(test_songs)}")

    # ── Build transition matrix for decoding ──
    # Learn from training data (simple bigram counts)
    print("\n4. Building transition matrix from training data...")
    tier1_trans = _learn_transitions(data, train_set, n_states=25)

    # ── Class weights ──
    q3_weights = compute_quality3_weights(data, train_set).to(device)
    print(f"   Quality3 class weights: N={q3_weights[0]:.2f}, maj={q3_weights[1]:.2f}, min={q3_weights[2]:.2f}")

    # ── Model ──
    print(f"\n5. Building model...")
    model = ChordTransformer(
        input_dim=data['X'].shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # ── Optimizer / scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )

    root_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    quality_criterion = nn.CrossEntropyLoss(weight=q3_weights, ignore_index=-1)

    # ── Training loop ──
    print(f"\n6. Training ({args.epochs} epochs max, patience={args.patience})...")
    best_val_wcsr = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_root_loss = 0.0
        total_qual_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            features = batch['features'].to(device)           # (B, W, 24)
            root_labels = batch['root_labels'].to(device)     # (B, W)
            q3_labels = batch['quality3_labels'].to(device)   # (B, W)
            pad_mask = batch['padding_mask'].to(device)       # (B, W)

            # Mask labels at padded positions
            root_labels = root_labels.clone()
            q3_labels = q3_labels.clone()
            root_labels[pad_mask] = -1
            q3_labels[pad_mask] = -1

            root_logits, qual_logits = model(features, src_key_padding_mask=pad_mask)

            # Flatten for loss: (B*W, C)
            B, W, _ = root_logits.shape
            r_loss = root_criterion(root_logits.reshape(B * W, -1), root_labels.reshape(B * W))
            q_loss = quality_criterion(qual_logits.reshape(B * W, -1), q3_labels.reshape(B * W))

            loss = args.root_weight * r_loss + args.quality_weight * q_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_root_loss += r_loss.item()
            total_qual_loss += q_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_rl = total_root_loss / max(n_batches, 1)
        avg_ql = total_qual_loss / max(n_batches, 1)
        dt = time.time() - t0

        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # ── Periodic evaluation ──
        if epoch % args.eval_every == 0 or epoch == 1:
            val_metrics = evaluate_on_songs(model, val_songs, tier1_trans, device)
            vw = val_metrics['wcsr']
            vm = val_metrics['major_acc']
            vn = val_metrics['minor_acc']
            vr = val_metrics['root_acc']
            vf = val_metrics['flip_rate']
            cp = val_metrics['confusion_pairs']

            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} (r={avg_rl:.4f} q={avg_ql:.4f}) | "
                  f"lr={current_lr:.1e} | {dt:.1f}s")
            print(f"           | val WCSR={vw:.3f}  maj={vm:.3f}  min={vn:.3f}  "
                  f"root={vr:.3f}  flip={vf:.3f}")
            if cp['A_total'] > 0 or cp['D_total'] > 0 or cp['C_total'] > 0:
                print(f"           | Am→A: {cp['Am_to_A']}/{cp['A_total']}  "
                      f"Dm→D: {cp['Dm_to_D']}/{cp['D_total']}  "
                      f"Cm→C: {cp['Cm_to_C']}/{cp['C_total']}")

            # Early stopping on val WCSR
            if vw > best_val_wcsr:
                best_val_wcsr = vw
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += args.eval_every
        else:
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} (r={avg_rl:.4f} q={avg_ql:.4f}) | "
                      f"lr={current_lr:.1e} | {dt:.1f}s")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (best val WCSR={best_val_wcsr:.4f} at epoch {best_epoch})")
            break

    # ── Restore best and save ──
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n7. Final evaluation on test set (model from epoch {best_epoch})...")
    test_metrics = evaluate_on_songs(model, test_songs, tier1_trans, device)
    tw = test_metrics['wcsr']
    tm = test_metrics['major_acc']
    tn = test_metrics['minor_acc']
    tr = test_metrics['root_acc']
    tf_ = test_metrics['flip_rate']
    cp = test_metrics['confusion_pairs']

    print(f"\n{'='*60}")
    print(f"Test Results (Transformer + Viterbi decoder)")
    print(f"{'-'*60}")
    print(f"  WCSR:       {tw:.3f}")
    print(f"  Major:      {tm:.3f}")
    print(f"  Minor:      {tn:.3f}")
    print(f"  Root:       {tr:.3f}")
    print(f"  Flip rate:  {tf_:.3f}")
    print(f"{'-'*60}")
    print(f"  Am→A: {cp['Am_to_A']}/{cp['A_total']}  "
          f"Dm→D: {cp['Dm_to_D']}/{cp['D_total']}  "
          f"Cm→C: {cp['Cm_to_C']}/{cp['C_total']}")
    print(f"{'='*60}")

    # Per-song breakdown
    print(f"\nPer-song breakdown (sorted by accuracy):")
    for r in sorted(test_metrics['song_results'], key=lambda x: x['accuracy']):
        prov = 'G' if r['provenance'] == 'gold' else 'S'
        print(f"  [{prov}] {r['name']:<55s} {r['n_beats']:4d} beats  "
              f"acc={r['accuracy']:.3f}  flip={r['flip_rate']:.3f}")

    # ── Save checkpoint ──
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'hyperparams': {
            'input_dim': data['X'].shape[1],
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'd_ff': args.d_ff,
            'dropout': args.dropout,
            'n_roots': 13,
            'n_qualities': 3,
        },
        'tier1_transition': tier1_trans,
        'best_epoch': best_epoch,
        'best_val_wcsr': best_val_wcsr,
        'test_metrics': {
            'wcsr': tw, 'major_acc': tm, 'minor_acc': tn,
            'root_acc': tr, 'flip_rate': tf_,
        },
        'train_songs': sorted(train_set),
        'val_songs': sorted(val_set),
        'test_songs': sorted(test_set),
        'gold_only': args.gold_only,
    }
    torch.save(checkpoint, args.out)
    print(f"\nModel saved to {args.out}")
    print(f"File size: {os.path.getsize(args.out) / 1024:.0f} KB")


def _learn_transitions(data, train_set, n_states=25):
    """Count tier1 label bigrams from training songs to build transition matrix."""
    counts = np.ones((n_states, n_states), dtype=np.float64)  # Laplace smoothing
    for sid in sorted(train_set):
        mask = data['song_ids'] == sid
        t1 = data['tier1_labels'][mask]
        for i in range(len(t1) - 1):
            counts[t1[i], t1[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def main():
    parser = argparse.ArgumentParser(description='Train Transformer chord model')
    parser.add_argument('--data', required=True, help='Path to features_v2.npz')
    parser.add_argument('--out', default='models/chord_transformer.pt', help='Output checkpoint path')

    # Data
    parser.add_argument('--gold-only', action='store_true', help='Train on gold data only')
    parser.add_argument('--feature-dim', type=int, default=24, choices=[24, 48, 60],
                        help='Input feature dim: 24 (HPCP+bass), 48 (context+bass), 60 (all)')
    parser.add_argument('--window', type=int, default=128, help='Window length in beats')
    parser.add_argument('--stride', type=int, default=64, help='Stride between windows')
    parser.add_argument('--augment', action='store_true',
                        help='Enable pitch-shift augmentation (12 transpositions)')

    # Architecture
    parser.add_argument('--d-model', type=int, default=128, help='Transformer model dim')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--d-ff', type=int, default=256, help='Feedforward dim')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--root-weight', type=float, default=1.0, help='Root loss weight')
    parser.add_argument('--quality-weight', type=float, default=1.5, help='Quality loss weight')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (in epochs)')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
