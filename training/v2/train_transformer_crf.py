#!/usr/bin/env python3
"""Train Transformer + CRF chord model.

Primary loss : CRF negative log-likelihood on tier1 (25-class) sequences.
Auxiliary loss: root + quality CE for per-class discrimination.

Usage:
    python -m v2.train_transformer_crf \
        --data data/features_v2.npz --out models/chord_transformer_crf.pt \
        --augment --feature-dim 48

    Gold+silver (RF parity): omit --gold-only; sample weights on by default.
    Hybrid emissions (RF-style): add --hybrid-emissions.
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

from v2.transformer_model import ChordTransformerCRF
from v2.transformer_data import (
    load_data, split_songs, compute_quality3_weights, compute_quality7_weights,
    ChordWindowDataset, get_song_data,
)
from v2.decode import smooth_isolated


def _masked_weighted_ce(logits, targets, beat_weights, pad_mask, criterion_none):
    """Mean CE over valid positions, weighted by beat_weights (0 on padding)."""
    b, w, c = logits.shape
    logits_f = logits.reshape(b * w, c)
    targets_f = targets.reshape(b * w)
    bw = beat_weights.reshape(b * w).float()
    valid = (~pad_mask).reshape(b * w)
    per = criterion_none(logits_f, targets_f)
    mask_f = valid.float() * bw
    return (per * mask_f).sum() / mask_f.sum().clamp(min=1e-8)


# ──────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────

def evaluate_crf_on_songs(model, songs, device):
    """Evaluate CRF-decoded Transformer on full songs.

    Returns dict with: wcsr, major_acc, minor_acc, root_acc, flip_rate,
                        per_song results, confusion_pairs
    """
    model.eval()
    all_true = []
    all_pred = []
    song_results = []
    confusion_pairs = {'Am_to_A': 0, 'Dm_to_D': 0, 'Cm_to_C': 0,
                       'A_total': 0, 'D_total': 0, 'C_total': 0}

    with torch.no_grad():
        for song in songs:
            X = torch.from_numpy(song['features']).unsqueeze(0).to(device)
            paths = model.decode(X)  # CRF Viterbi
            path = np.array(paths[0], dtype=np.int64)
            path = smooth_isolated(path)

            t1_true = song['tier1_labels']
            total = len(t1_true)

            # Trim/pad to match (should be same length, but be safe)
            path = path[:total]
            if len(path) < total:
                path = np.pad(path, (0, total - len(path)), constant_values=0)

            correct = np.sum(path == t1_true)
            acc = correct / max(total, 1)

            n_flips = np.sum(path[1:] != path[:-1]) if len(path) > 1 else 0
            flip_rate = n_flips / max(len(path) - 1, 1)

            all_true.append(t1_true)
            all_pred.append(path)

            # Confusion pairs
            am_mask = t1_true == 22
            if am_mask.sum() > 0:
                confusion_pairs['A_total'] += int(am_mask.sum())
                confusion_pairs['Am_to_A'] += int(np.sum(path[am_mask] == 10))
            dm_mask = t1_true == 15
            if dm_mask.sum() > 0:
                confusion_pairs['D_total'] += int(dm_mask.sum())
                confusion_pairs['Dm_to_D'] += int(np.sum(path[dm_mask] == 3))
            cm_mask = t1_true == 13
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

    wcsr = np.sum(all_true == all_pred) / max(len(all_true), 1)
    maj_mask = (all_true >= 1) & (all_true <= 12)
    min_mask = (all_true >= 13) & (all_true <= 24)
    major_acc = np.sum(all_true[maj_mask] == all_pred[maj_mask]) / max(maj_mask.sum(), 1) if maj_mask.sum() > 0 else 0
    minor_acc = np.sum(all_true[min_mask] == all_pred[min_mask]) / max(min_mask.sum(), 1) if min_mask.sum() > 0 else 0

    pred_roots = np.where(all_pred == 0, 0, np.where(all_pred <= 12, all_pred, all_pred - 12))
    true_roots = np.where(all_true == 0, 0, np.where(all_true <= 12, all_true, all_true - 12))
    root_acc = np.sum(true_roots == pred_roots) / max(len(true_roots), 1)

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


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

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
    use_sample_weights = not args.no_sample_weights
    print(f"   Sample weights (CRF + aux): {'on' if use_sample_weights else 'off'}")

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
          (" (12x augmented)" if args.augment else ""))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)

    val_songs = get_song_data(data, val_set)
    test_songs = get_song_data(data, test_set)
    print(f"   Val songs: {len(val_songs)}, Test songs: {len(test_songs)}")

    # ── Class weights for auxiliary loss ──
    q3_weights = compute_quality3_weights(data, train_set).to(device)
    q7_weights = compute_quality7_weights(data, train_set).to(device)
    print(f"\n4. Quality aux: n_qualities={args.n_qualities}")
    if args.n_qualities == 3:
        print(f"   Quality3 class weights: N={q3_weights[0]:.2f}, maj={q3_weights[1]:.2f}, min={q3_weights[2]:.2f}")
    else:
        print(f"   Quality7 class weights: {q7_weights.cpu().numpy().round(3)}")

    emission_mode = 'hybrid' if args.hybrid_emissions else 'direct'
    use_key_aux = args.key_aux_weight > 0.0

    # ── Model ──
    print(f"\n5. Building model (Transformer + CRF)...")
    print(f"   Emission mode: {emission_mode}")
    if args.key_condition_quality:
        print("   Key-conditioned quality logits (embedding 12 → n_qualities)")
    if use_key_aux:
        print(f"   Key auxiliary CE weight: {args.key_aux_weight}")
    model = ChordTransformerCRF(
        input_dim=data['X'].shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        crf_self_bias=args.crf_self_bias,
        emission_temp=args.emission_temp,
        emission_dropout=args.emission_dropout,
        emission_noise_std=args.emission_noise_std,
        em_emission_bias=args.em_emission_bias,
        emission_mode=emission_mode,
        n_qualities=args.n_qualities,
        use_key_aux=use_key_aux,
        key_condition_quality=args.key_condition_quality,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    crf_params = sum(p.numel() for n, p in model.named_parameters() if 'crf' in n)
    print(f"   Total parameters: {n_params:,} (CRF: {crf_params:,})")
    if args.emission_temp != 1.0:
        print(f"   Emission temperature: {args.emission_temp}")
    if args.emission_dropout > 0.0:
        print(f"   Emission dropout (train only): p={args.emission_dropout}")
    if args.emission_noise_std > 0.0:
        print(f"   Emission noise (train only): std={args.emission_noise_std}")
    if args.em_emission_bias != 0.0:
        print(f"   Em emission bias: {args.em_emission_bias}")

    # ── Optional: initialize CRF transitions from RF model ──
    if args.init_transitions_from:
        import pickle
        with open(args.init_transitions_from, 'rb') as f:
            rf_model = pickle.load(f)
        rf_trans = rf_model['tier1_transition']  # (25, 25) probability matrix
        log_trans = np.log(np.clip(rf_trans, 1e-8, None))
        # Shift so diagonal mean matches the requested crf_self_bias
        diag_mean = np.diag(log_trans).mean()
        log_trans_shifted = log_trans + (args.crf_self_bias - diag_mean)
        with torch.no_grad():
            model.crf.transitions.copy_(torch.from_numpy(log_trans_shifted.astype(np.float32)))
        print(f"   CRF transitions initialized from {args.init_transitions_from}")
        print(f"   RF diag log-mean={diag_mean:.2f}, shifted to match self_bias={args.crf_self_bias:.1f}")

    # Print initial CRF self-transition stats
    with torch.no_grad():
        self_trans = model.crf.transitions.diagonal().mean().item()
        off_trans = (model.crf.transitions.sum() - model.crf.transitions.diagonal().sum()).item() / (25*24)
        print(f"   CRF init: self_trans={self_trans:.2f}, off_trans={off_trans:.3f}")

    # ── Optimizer / scheduler ──
    crf_lr = args.crf_lr if args.crf_lr is not None else args.lr
    encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
    crf_params = list(model.crf.parameters())
    crf_param_ids = {id(p) for p in crf_params}
    encoder_params = [p for p in model.parameters() if id(p) not in crf_param_ids]
    param_groups = [
        {'params': encoder_params, 'lr': args.lr},
        {'params': crf_params, 'lr': crf_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5,
    )
    if crf_lr != args.lr:
        print(f"   Dual LR: encoder={args.lr:.1e}, CRF={crf_lr:.1e}")

    # ── Frozen encoder warmup ──
    freeze_epochs = args.freeze_encoder_epochs
    encoder_frozen = False
    if freeze_epochs > 0:
        # Freeze all non-CRF, non-head params
        trainable_sub = (
            'crf', 'tier1_head', 'root_head', 'quality_head',
            'key_head', 'key_quality_emb',
        )
        for name, param in model.named_parameters():
            if not any(s in name for s in trainable_sub):
                param.requires_grad = False
        encoder_frozen = True
        # Set encoder LR to 0 during warmup (optimizer still tracks them)
        optimizer.param_groups[0]['lr'] = 0.0
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"   Frozen encoder warmup: {freeze_epochs} epochs (frozen={n_frozen}, trainable={n_trainable})")
        print(f"   After warmup: encoder LR={encoder_lr:.1e}")

    root_criterion = nn.CrossEntropyLoss(
        ignore_index=-1,
        label_smoothing=args.aux_label_smoothing,
    )
    q_w = q3_weights if args.n_qualities == 3 else q7_weights
    quality_criterion = nn.CrossEntropyLoss(
        weight=q_w,
        ignore_index=-1,
        label_smoothing=args.aux_label_smoothing,
    )
    quality_criterion_none = nn.CrossEntropyLoss(
        weight=q_w,
        ignore_index=-1,
        label_smoothing=args.aux_label_smoothing,
        reduction='none',
    )
    root_criterion_none = nn.CrossEntropyLoss(
        ignore_index=-1,
        label_smoothing=args.aux_label_smoothing,
        reduction='none',
    )
    key_criterion = nn.CrossEntropyLoss()
    tier1_criterion = nn.CrossEntropyLoss(ignore_index=-1) if args.tier1_ce_weight > 0 else None
    tier1_criterion_none = (
        nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        if args.tier1_ce_weight > 0 and use_sample_weights
        else None
    )

    # ── Training loop ──
    print(f"\n6. Training ({args.epochs} epochs max, patience={args.patience})...")
    print(f"   CRF weight: {args.crf_weight}, aux weight: {args.aux_weight}")
    if args.tier1_ce_weight > 0.0:
        print(f"   Tier1 CE weight: {args.tier1_ce_weight}")
    if args.aux_label_smoothing > 0.0:
        print(f"   Aux label smoothing: {args.aux_label_smoothing}")
    if args.encoder_grad_scale != 1.0:
        print(f"   Encoder gradient scale: {args.encoder_grad_scale}")
    if args.crf_self_reg > 0.0:
        print(f"   CRF self-transition regularization: lambda={args.crf_self_reg}, target={args.crf_self_target}")
    print(f"   Checkpoint selection: WCSR - {args.flip_penalty} * max(0, flip - {args.target_flip})")
    best_val_score = -1.0
    best_val_wcsr = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # ── Unfreeze encoder after warmup ──
        if encoder_frozen and epoch > freeze_epochs:
            for name, param in model.named_parameters():
                param.requires_grad = True
            optimizer.param_groups[0]['lr'] = encoder_lr
            encoder_frozen = False
            print(f"  >>> Encoder unfrozen at epoch {epoch}, encoder LR={encoder_lr:.1e}")

        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_crf_loss = 0.0
        total_aux_loss = 0.0
        total_reg_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            features = batch['features'].to(device)
            tier1_labels = batch['tier1_labels'].to(device)
            root_labels = batch['root_labels'].to(device)
            pad_mask = batch['padding_mask'].to(device)
            beat_w = batch['beat_weights'].to(device)
            if not use_sample_weights:
                beat_w = torch.ones_like(beat_w)
            seq_w = batch['sample_weight'].float().to(device) if use_sample_weights else None
            key_idx_batch = batch['key_idx'].to(device).long().clamp(0, 11)

            q_ce_labels = (
                batch['quality3_labels'].to(device)
                if args.n_qualities == 3
                else batch['quality_labels'].to(device)
            )

            # CRF needs valid (non-negative) tags at padded positions;
            # pad with 0 (N chord) — CRF mask will ignore them
            tier1_crf = tier1_labels.clone()
            tier1_crf[pad_mask] = 0

            root_ce = root_labels.clone()
            q_ce = q_ce_labels.clone()
            root_ce[pad_mask] = -1
            q_ce[pad_mask] = -1

            k_for_model = key_idx_batch if model.key_quality_emb is not None else None
            crf_loss, root_logits, quality_logits, tier1_logits, key_logits = model(
                features, tier1_crf, pad_mask=pad_mask,
                key_idx=k_for_model, seq_weights=seq_w,
            )

            B, W, _ = root_logits.shape
            if use_sample_weights:
                r_loss = _masked_weighted_ce(
                    root_logits, root_ce, beat_w, pad_mask, root_criterion_none)
                q_loss = _masked_weighted_ce(
                    quality_logits, q_ce, beat_w, pad_mask, quality_criterion_none)
            else:
                r_loss = root_criterion(
                    root_logits.reshape(B * W, -1), root_ce.reshape(B * W))
                q_loss = quality_criterion(
                    quality_logits.reshape(B * W, -1), q_ce.reshape(B * W))

            key_loss = torch.tensor(0.0, device=device)
            if key_logits is not None and args.key_aux_weight > 0.0:
                key_loss = key_criterion(key_logits, key_idx_batch)

            aux_loss = r_loss + args.quality_weight * q_loss + args.key_aux_weight * key_loss

            tier1_ce_loss = torch.tensor(0.0, device=device)
            if tier1_criterion is not None:
                tier1_ce = tier1_labels.clone()
                tier1_ce[pad_mask] = -1
                if use_sample_weights:
                    tier1_ce_loss = _masked_weighted_ce(
                        tier1_logits, tier1_ce, beat_w, pad_mask, tier1_criterion_none)
                else:
                    tier1_ce_loss = tier1_criterion(
                        tier1_logits.reshape(B * W, -1), tier1_ce.reshape(B * W))

            reg_loss = torch.tensor(0.0, device=device)
            if args.crf_self_reg > 0.0:
                diag = model.crf.transitions.diagonal()
                reg_loss = args.crf_self_reg * torch.mean((diag - args.crf_self_target) ** 2)

            loss = (args.crf_weight * crf_loss + args.aux_weight * aux_loss
                    + args.tier1_ce_weight * tier1_ce_loss) + reg_loss

            optimizer.zero_grad()
            loss.backward()
            if args.encoder_grad_scale != 1.0:
                for p in encoder_params:
                    if p.grad is not None:
                        p.grad.mul_(args.encoder_grad_scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_crf_loss += crf_loss.item()
            total_aux_loss += aux_loss.item()
            total_reg_loss += reg_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_crf = total_crf_loss / max(n_batches, 1)
        avg_aux = total_aux_loss / max(n_batches, 1)
        avg_reg = total_reg_loss / max(n_batches, 1)
        dt = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']
        crf_current_lr = optimizer.param_groups[1]['lr']

        # ── Periodic evaluation (CRF decode, no external Viterbi) ──
        if epoch % args.eval_every == 0 or epoch == 1:
            val_metrics = evaluate_crf_on_songs(model, val_songs, device)
            vw = val_metrics['wcsr']
            vm = val_metrics['major_acc']
            vn = val_metrics['minor_acc']
            vr = val_metrics['root_acc']
            vf = val_metrics['flip_rate']
            cp = val_metrics['confusion_pairs']

            lr_str = f"lr={current_lr:.1e}"
            if crf_current_lr != current_lr:
                lr_str += f"/crf={crf_current_lr:.1e}"
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} (crf={avg_crf:.4f} aux={avg_aux:.4f} reg={avg_reg:.4f}) | "
                  f"{lr_str} | {dt:.1f}s")
            print(f"           | val WCSR={vw:.3f}  maj={vm:.3f}  min={vn:.3f}  "
                  f"root={vr:.3f}  flip={vf:.3f}")
            if cp['A_total'] > 0 or cp['D_total'] > 0 or cp['C_total'] > 0:
                print(f"           | Am→A: {cp['Am_to_A']}/{cp['A_total']}  "
                      f"Dm→D: {cp['Dm_to_D']}/{cp['D_total']}  "
                      f"Cm→C: {cp['Cm_to_C']}/{cp['C_total']}")

            val_score = vw - args.flip_penalty * max(0, vf - args.target_flip)
            print(f"           | score={val_score:.3f}  (best={best_val_score:.3f})")

            scheduler.step(vw)

            if val_score > best_val_score:
                best_val_score = val_score
                best_val_wcsr = vw
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += args.eval_every
        else:
            if epoch % 10 == 0:
                lr_str = f"lr={current_lr:.1e}"
                if crf_current_lr != current_lr:
                    lr_str += f"/crf={crf_current_lr:.1e}"
                print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} (crf={avg_crf:.4f} aux={avg_aux:.4f} reg={avg_reg:.4f}) | "
                      f"{lr_str} | {dt:.1f}s")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (best score={best_val_score:.4f}, WCSR={best_val_wcsr:.4f} at epoch {best_epoch})")
            break

    # ── Restore best and save ──
    if best_state is not None:
        model.load_state_dict(best_state)

    # Print learned CRF transition stats
    with torch.no_grad():
        self_trans = model.crf.transitions.diagonal().mean().item()
        off_trans = (model.crf.transitions.sum() - model.crf.transitions.diagonal().sum()).item() / (25*24)
        print(f"\n   Learned CRF: self_trans={self_trans:.2f}, off_trans={off_trans:.3f}")

    print(f"\n7. Final evaluation on test set (model from epoch {best_epoch})...")
    test_metrics = evaluate_crf_on_songs(model, test_songs, device)
    tw = test_metrics['wcsr']
    tm = test_metrics['major_acc']
    tn = test_metrics['minor_acc']
    tr = test_metrics['root_acc']
    tf_ = test_metrics['flip_rate']
    cp = test_metrics['confusion_pairs']

    print(f"\n{'='*60}")
    print(f"Test Results (Transformer + CRF decode)")
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
        'model_type': 'ChordTransformerCRF',
        'hyperparams': {
            'input_dim': data['X'].shape[1],
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'd_ff': args.d_ff,
            'dropout': args.dropout,
            'crf_self_bias': args.crf_self_bias,
            'emission_temp': args.emission_temp,
            'emission_dropout': args.emission_dropout,
            'emission_noise_std': args.emission_noise_std,
            'em_emission_bias': args.em_emission_bias,
            'emission_mode': emission_mode,
            'tier1_ce_weight': args.tier1_ce_weight,
            'aux_label_smoothing': args.aux_label_smoothing,
            'encoder_grad_scale': args.encoder_grad_scale,
            'crf_self_reg': args.crf_self_reg,
            'crf_self_target': args.crf_self_target,
            'crf_lr': crf_lr,
            'encoder_lr': encoder_lr,
            'freeze_encoder_epochs': freeze_epochs,
            'n_tier1': 25,
            'n_roots': 13,
            'n_qualities': args.n_qualities,
            'use_sample_weights': use_sample_weights,
            'key_condition_quality': args.key_condition_quality,
            'key_aux_weight': args.key_aux_weight,
        },
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


def main():
    parser = argparse.ArgumentParser(description='Train Transformer + CRF chord model')
    parser.add_argument('--data', required=True, help='Path to features_v2.npz')
    parser.add_argument('--out', default='models/chord_transformer_crf.pt', help='Output checkpoint path')
    parser.add_argument('--init-transitions-from', default=None,
                        help='Path to RF .pkl model to initialize CRF transitions from learned tier1 matrix')

    # Data
    parser.add_argument('--gold-only', action='store_true', help='Train on gold data only')
    parser.add_argument('--no-sample-weights', action='store_true',
                        help='Disable per-beat sample weights (CRF + aux); default uses npz weights like RF')
    parser.add_argument('--hybrid-emissions', action='store_true',
                        help='CRF emissions = factorized root × tier1 maj/min split (decode_hybrid-style)')
    parser.add_argument('--n-qualities', type=int, default=3, choices=[3, 7],
                        help='Aux quality head classes: 3 (N/maj/min) or 7 (full v2)')
    parser.add_argument('--key-condition-quality', action='store_true',
                        help='Add key embedding (12 roots) to quality logits (uses song key from npz)')
    parser.add_argument('--key-aux-weight', type=float, default=0.0,
                        help='If >0, add song-level key classifier (12-way) with this CE weight')
    parser.add_argument('--feature-dim', type=int, default=48, choices=[24, 48, 60, 72],
                        help='Input feature dim: 48 = RF parity (context+bass); 60 = +raw HPCP; 72 adds 3rd-ratio')
    parser.add_argument('--window', type=int, default=128, help='Window length in beats')
    parser.add_argument('--stride', type=int, default=64, help='Stride between windows')
    parser.add_argument('--augment', action='store_true',
                        help='Enable pitch-shift augmentation (12 transpositions)')

    # Architecture
    parser.add_argument('--d-model', type=int, default=128, help='Transformer model dim')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--d-ff', type=int, default=256, help='Feedforward dim')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--crf-self-bias', type=float, default=2.0,
                        help='CRF initial self-transition bias')
    parser.add_argument('--emission-temp', type=float, default=1.0,
                        help='Temperature to divide emissions before CRF (>1 flattens, keeps transitions stronger)')
    parser.add_argument('--emission-dropout', type=float, default=0.0,
                        help='Dropout on tier1 emissions before CRF during training')
    parser.add_argument('--emission-noise-std', type=float, default=0.0,
                        help='Gaussian noise std added to tier1 emissions before CRF during training')
    parser.add_argument('--em-emission-bias', type=float, default=0.0,
                        help='Fixed negative bias on Em emission before CRF (calibration for Em over-prediction)')
    parser.add_argument('--tier1-ce-weight', type=float, default=0.0,
                        help='Weight for per-beat tier1 CE auxiliary loss on raw emissions')

    # Training
    parser.add_argument('--epochs', type=int, default=150, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (encoder)')
    parser.add_argument('--crf-lr', type=float, default=None,
                        help='CRF learning rate (default: same as --lr)')
    parser.add_argument('--encoder-lr', type=float, default=None,
                        help='Encoder LR after warmup unfreeze (default: same as --lr)')
    parser.add_argument('--freeze-encoder-epochs', type=int, default=0,
                        help='Freeze encoder for N epochs, train only CRF/heads')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--crf-weight', type=float, default=1.0, help='CRF loss weight')
    parser.add_argument('--aux-weight', type=float, default=0.3, help='Auxiliary CE loss weight')
    parser.add_argument('--quality-weight', type=float, default=2.0,
                        help='Quality CE weight within aux loss')
    parser.add_argument('--aux-label-smoothing', type=float, default=0.0,
                        help='Label smoothing for root/quality auxiliary CE losses')
    parser.add_argument('--encoder-grad-scale', type=float, default=1.0,
                        help='Multiply encoder gradients by this factor before optimizer step')
    parser.add_argument('--crf-self-reg', type=float, default=0.0,
                        help='L2 penalty weight encouraging CRF self-transitions toward target')
    parser.add_argument('--crf-self-target', type=float, default=2.0,
                        help='Target value for CRF self-transition regularization')
    parser.add_argument('--flip-penalty', type=float, default=2.0,
                        help='Penalty weight for flip rate exceeding target in checkpoint selection')
    parser.add_argument('--target-flip', type=float, default=0.05,
                        help='Target flip rate; excess penalized during checkpoint selection')
    parser.add_argument('--patience', type=int, default=40, help='Early stopping patience (epochs)')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
