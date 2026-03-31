#!/usr/bin/env python3
"""Evaluate BTC chord model vs Freeze5 Transformer+CRF on test songs.

Runs both models through their full production inference pipelines
and compares against ground-truth chord annotations using beat-level WCSR.

Usage:
    python -m v2.eval_btc_vs_freeze5 \\
        --config data/weights.json \\
        --checkpoint-btc ../server/btc_model/btc_model_best.pth \\
        --checkpoint-freeze5 models/chord_transformer_crf_freeze5.pt
"""
import sys
import os
import argparse
import time
import json
import numpy as np

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SERVER_ROOT = os.path.join(os.path.dirname(_TRAINING_ROOT), 'server')
sys.path.insert(0, _TRAINING_ROOT)
sys.path.insert(0, _SERVER_ROOT)

from shared.chord_vocab import TIER1_VOCAB, TIER1_TO_IDX, parse_chord_label
from v1.prepare_data import (
    parse_lab_file, parse_beat_file, find_isophonics_songs,
    find_flat_pairs, extract_hpcp, sync_to_beats, detect_key, key_to_index,
    get_chord_at_time, SR, HOP_SIZE,
)


# ── Freeze5 helpers (mirror server production pipeline) ──────

def _build_context_features(beat_chroma_cols, radius=2):
    """Concatenate (2*radius+1) beats of 12-dim HPCP context."""
    bc = beat_chroma_cols.T
    n_beats = len(bc)
    width = 2 * radius + 1
    feat_dim = width * 12
    X = np.zeros((n_beats, feat_dim), dtype=np.float32)
    for i in range(n_beats):
        for k, offset in enumerate(range(-radius, radius + 1)):
            j = max(0, min(n_beats - 1, i + offset))
            X[i, k * 12:(k + 1) * 12] = bc[j]
    return X


def _freeze5_build_features(beat_chroma_cols):
    """Build 60-dim features from (12, n_beats) beat chroma."""
    bc = beat_chroma_cols.T.astype(np.float32)
    ctx60 = _build_context_features(beat_chroma_cols, radius=2)
    x48 = ctx60[:, :48]
    x = np.hstack([bc, x48])
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


_FREEZE5_MODEL = None
_FREEZE5_DEVICE = None


def load_freeze5_model(checkpoint_path, device):
    """Load freeze5 Transformer+CRF model."""
    import torch
    from v2.transformer_model import ChordTransformerCRF

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
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def _freeze5_get_emissions(model, x_np, device):
    """Compute CRF emissions. Returns (emissions_tensor, softmax_probs_np)."""
    import torch
    with torch.no_grad():
        xt = torch.from_numpy(x_np).unsqueeze(0).to(device)
        h = model._encode(xt)
        tier1_logits = model.tier1_head(h)
        root_logits = model.root_head(h)

        if model.emission_mode == 'hybrid':
            emissions = model._tier1_emissions_hybrid(root_logits, tier1_logits)
        else:
            emissions = model._tier1_emissions_direct(tier1_logits)

        probs = torch.softmax(tier1_logits[0], dim=-1).cpu().numpy()
    return emissions, probs


def _build_key_bias_logits(key_idx, vocab, bias_strength=0.5):
    """Build per-class emission bias for a given key hypothesis."""
    import torch
    DIATONIC_OFFSETS = [0, 2, 4, 5, 7, 9, 11]
    bias = torch.zeros(len(vocab), dtype=torch.float32)
    for i, name in enumerate(vocab):
        if i == 0:
            continue
        if name.endswith('m'):
            root = name[:-1]
        else:
            root = name
        NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        if root in NOTES:
            root_idx = NOTES.index(root)
            offset = (root_idx - key_idx) % 12
            if offset in DIATONIC_OFFSETS:
                bias[i] = bias_strength
    return bias


def _freeze5_viterbi_with_bias(model, emissions, key_bias_logits):
    """Run CRF Viterbi on emissions + per-class key bias."""
    import torch
    with torch.no_grad():
        biased = emissions.clone()
        if key_bias_logits is not None:
            biased = biased + key_bias_logits.unsqueeze(0).unsqueeze(0).to(emissions.device)

        paths = model.crf.decode(biased)
        path = np.array(paths[0], dtype=np.int64)

        # Manually compute Viterbi log-score of the best path
        e = biased[0]
        crf = model.crf
        v = crf.start_transitions + e[0]
        for t in range(1, e.shape[0]):
            scores = v.unsqueeze(1) + crf.transitions
            best_scores, _ = scores.max(dim=0)
            v = best_scores + e[t]
        v = v + crf.end_transitions
        log_score = float(v.max())

    return path, log_score


def run_freeze5_full(model, beat_chroma_cols, key_idx, beat_times, device):
    """Full freeze5 production inference: 12-key search + smooth + long-run breaking."""
    from v2.decode import smooth_isolated

    x = _freeze5_build_features(beat_chroma_cols)
    emissions, emission_probs = _freeze5_get_emissions(model, x, device)

    best_score = -np.inf
    best_path = None
    best_key = key_idx

    for ki in range(12):
        key_bias = _build_key_bias_logits(ki, TIER1_VOCAB, bias_strength=0.5).to(device)
        path, score = _freeze5_viterbi_with_bias(model, emissions, key_bias)
        if score > best_score:
            best_score = score
            best_path = path.copy()
            best_key = ki

    best_path = smooth_isolated(best_path)
    return [TIER1_VOCAB[int(i)] for i in best_path]


# ── BTC helpers ──────────────────────────────────────────────

def load_btc_model(checkpoint_path, device):
    """Load BTC chord model + normalization stats."""
    import torch
    from btc_model.btc_model import BTC_model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    norm = ckpt.get('normalization', {})
    if isinstance(norm, dict) and 'mean' in norm:
        mean = float(norm['mean'])
        std = float(norm['std'])
    else:
        mean_val = ckpt.get('mean')
        std_val = ckpt.get('std')
        mean = float(mean_val) if mean_val is not None else -2.37
        std = float(std_val) if std_val is not None else 1.96

    model = BTC_model()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, mean, std


def extract_cqt(audio_path, sr=22050, hop_length=2048, n_bins=144,
                bins_per_octave=24):
    """Extract CQT spectrogram (log-magnitude, matching ChordMini)."""
    import librosa
    y, _ = librosa.load(audio_path, sr=sr)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length,
                       n_bins=n_bins, bins_per_octave=bins_per_octave,
                       fmin=librosa.note_to_hz('C1'))
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)


def run_btc_inference(btc_model, cqt, mean, std, device):
    """Run BTC on CQT with overlapping chunks. Returns frame-level predictions."""
    import torch
    cqt_norm = (cqt - mean) / max(std, 1e-6)
    n_frames = cqt_norm.shape[0]
    if n_frames == 0:
        return np.array([], dtype=np.int32)

    seq_len, stride = 108, 54
    logit_sum = np.zeros((n_frames, 170), dtype=np.float32)
    logit_count = np.zeros(n_frames, dtype=np.float32)

    with torch.no_grad():
        pos = 0
        while pos < n_frames:
            end = min(pos + seq_len, n_frames)
            chunk = cqt_norm[pos:end]
            actual_len = chunk.shape[0]
            if actual_len < seq_len:
                pad = np.zeros((seq_len - actual_len, 144), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
            x = torch.from_numpy(chunk).unsqueeze(0).to(device)
            out = btc_model(x)
            logits = out[0, :actual_len].cpu().numpy()
            logit_sum[pos:pos + actual_len] += logits
            logit_count[pos:pos + actual_len] += 1.0
            pos += stride
            if pos >= n_frames:
                break

    logit_count[logit_count == 0] = 1.0
    avg_logits = logit_sum / logit_count[:, np.newaxis]
    return avg_logits.argmax(axis=1).astype(np.int32)


def btc_beat_sync(frame_preds, beat_times, hop_dur=2048 / 22050.0):
    """Map frame-level BTC predictions to beat-level via majority vote."""
    from btc_model.vocab import btc_idx_to_tier1
    n_frames = len(frame_preds)
    beat_chords = []
    for bi in range(len(beat_times)):
        t_start = beat_times[bi]
        t_end = beat_times[bi + 1] if bi + 1 < len(beat_times) else t_start + 0.5
        f_start = max(0, int(round(t_start / hop_dur)))
        f_end = min(n_frames, int(round(t_end / hop_dur)))
        if f_end <= f_start:
            f_end = f_start + 1
        if f_start >= n_frames:
            beat_chords.append('N')
            continue
        segment = frame_preds[f_start:min(f_end, n_frames)]
        if len(segment) == 0:
            beat_chords.append('N')
            continue
        counts = np.bincount(segment, minlength=170)
        winner = int(counts.argmax())
        beat_chords.append(btc_idx_to_tier1(winner))
    return beat_chords


# ── Ground truth + metrics ───────────────────────────────────

def get_ground_truth_at_beats(chord_annotations, beat_times):
    """Get TIER1 ground-truth chord label at each beat (midpoint lookup)."""
    gt_labels = []
    for bi in range(len(beat_times)):
        if bi + 1 < len(beat_times):
            mid = (beat_times[bi] + beat_times[bi + 1]) / 2.0
        else:
            mid = beat_times[bi] + 0.25
        raw_label = get_chord_at_time(chord_annotations, mid)
        parsed = parse_chord_label(raw_label, tier=1)
        gt_labels.append(parsed)
    return gt_labels


def compute_metrics(gt_names, pred_names):
    """Compute WCSR, major/minor/root accuracy from chord name lists."""
    gt_idx = np.array([TIER1_TO_IDX.get(n, 0) for n in gt_names], dtype=np.int64)
    pred_idx = np.array([TIER1_TO_IDX.get(n, 0) for n in pred_names], dtype=np.int64)

    total = len(gt_idx)
    if total == 0:
        return {'wcsr': 0, 'major_acc': 0, 'minor_acc': 0, 'root_acc': 0, 'n_beats': 0}

    wcsr = np.sum(gt_idx == pred_idx) / total

    maj_mask = (gt_idx >= 1) & (gt_idx <= 12)
    min_mask = (gt_idx >= 13) & (gt_idx <= 24)
    major_acc = (np.sum(gt_idx[maj_mask] == pred_idx[maj_mask]) / max(maj_mask.sum(), 1)
                 if maj_mask.sum() > 0 else 0)
    minor_acc = (np.sum(gt_idx[min_mask] == pred_idx[min_mask]) / max(min_mask.sum(), 1)
                 if min_mask.sum() > 0 else 0)

    pred_roots = np.where(pred_idx == 0, 0, np.where(pred_idx <= 12, pred_idx, pred_idx - 12))
    true_roots = np.where(gt_idx == 0, 0, np.where(gt_idx <= 12, gt_idx, gt_idx - 12))
    root_acc = np.sum(true_roots == pred_roots) / total

    return {
        'wcsr': float(wcsr),
        'major_acc': float(major_acc),
        'minor_acc': float(minor_acc),
        'root_acc': float(root_acc),
        'n_beats': total,
    }


# ── Song-level evaluation ────────────────────────────────────

def evaluate_song(song, btc_model, btc_mean, btc_std, freeze5_model, device):
    """Evaluate both models on one song. Returns (btc_metrics, f5_metrics)."""
    chord_annotations = parse_lab_file(song['chords'])

    if song.get('beats'):
        beat_times = parse_beat_file(song['beats'])
    else:
        from v1.prepare_data import detect_beats
        beat_times = detect_beats(song['audio'])

    if len(beat_times) < 2:
        return None, None

    gt_names = get_ground_truth_at_beats(chord_annotations, beat_times)

    # ── BTC ──
    btc_metrics = None
    if btc_model is not None:
        try:
            cqt = extract_cqt(song['audio'])
            frame_preds = run_btc_inference(btc_model, cqt, btc_mean, btc_std, device)
            btc_names = btc_beat_sync(frame_preds, beat_times)
            n = len(gt_names)
            btc_names = (btc_names[:n] + ['N'] * max(0, n - len(btc_names)))[:n]
            btc_metrics = compute_metrics(gt_names, btc_names)
        except Exception as e:
            print(f"    BTC error: {e}")

    # ── Freeze5 (full production pipeline) ──
    freeze5_metrics = None
    if freeze5_model is not None:
        try:
            hpcps, _, flatness_weights = extract_hpcp(song['audio'])
            beat_chroma = sync_to_beats(hpcps, beat_times, flatness_weights)
            beat_chroma_cols = beat_chroma.T  # (12, n_beats) for server-style functions
            if beat_chroma_cols.shape[1] > 0:
                key_idx = detect_key(hpcps)
                f5_names = run_freeze5_full(
                    freeze5_model, beat_chroma_cols, key_idx, beat_times, device)
                n = len(gt_names)
                f5_names = (f5_names[:n] + ['N'] * max(0, n - len(f5_names)))[:n]
                freeze5_metrics = compute_metrics(gt_names, f5_names)
        except Exception as e:
            print(f"    Freeze5 error: {e}")

    return btc_metrics, freeze5_metrics


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BTC vs Freeze5 chord accuracy on test songs')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to weights.json dataset config')
    parser.add_argument('--isophonics', type=str, action='append', default=None,
                        help='Isophonics annotations root (repeatable)')
    parser.add_argument('--audio-dir', type=str, action='append', default=None,
                        help='Audio directory (repeatable, pairs with --isophonics)')
    parser.add_argument('--checkpoint-btc', type=str, default=None,
                        help='Path to BTC model checkpoint')
    parser.add_argument('--checkpoint-freeze5', type=str, default=None,
                        help='Path to Freeze5 model checkpoint')
    parser.add_argument('--max-songs', type=int, default=None,
                        help='Limit number of songs to evaluate')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated dataset names to evaluate (default: all)')
    parser.add_argument('--per-song', action='store_true',
                        help='Show per-song breakdown')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    args = parser.parse_args()

    import torch

    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device('mps')
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── Load models ──
    btc_model, btc_mean, btc_std = None, 0, 1
    if args.checkpoint_btc:
        print(f"Loading BTC model from {args.checkpoint_btc}...")
        btc_model, btc_mean, btc_std = load_btc_model(args.checkpoint_btc, device)
        print(f"  Normalization: mean={btc_mean:.4f}, std={btc_std:.4f}")

    freeze5_model = None
    if args.checkpoint_freeze5:
        print(f"Loading Freeze5 model from {args.checkpoint_freeze5}...")
        freeze5_model = load_freeze5_model(args.checkpoint_freeze5, device)

    if btc_model is None and freeze5_model is None:
        print("ERROR: provide at least one of --checkpoint-btc or --checkpoint-freeze5")
        sys.exit(1)

    # ── Discover songs ──
    all_songs = []
    filter_datasets = args.datasets.split(',') if args.datasets else None

    if args.config:
        base_dir = _TRAINING_ROOT
        with open(args.config) as f:
            cfg = json.load(f)
        for name, ds in cfg['datasets'].items():
            if filter_datasets and name not in filter_datasets:
                continue
            if ds['type'] == 'isophonics':
                ann_root = os.path.join(base_dir, ds['annotations'])
                aud_dir = os.path.join(base_dir, ds['audio_dir'])
                found = find_isophonics_songs(ann_root, aud_dir)
                for s in found:
                    s['provenance'] = 'gold' if s.get('beats') else 'silver'
                    s['dataset'] = name
                print(f"  {name}: {len(found)} songs")
                all_songs.extend(found)
            elif ds['type'] == 'flat':
                aud_dir = os.path.join(base_dir, ds['audio_dir'])
                lab_dir = os.path.join(base_dir, ds['labels_dir'])
                if os.path.isdir(aud_dir) and os.path.isdir(lab_dir):
                    found = find_flat_pairs(aud_dir, lab_dir)
                    for s in found:
                        s['provenance'] = name
                        s['dataset'] = name
                    print(f"  {name}: {len(found)} songs")
                    all_songs.extend(found)
    elif args.isophonics:
        audio_dirs = args.audio_dir or []
        while len(audio_dirs) < len(args.isophonics):
            audio_dirs.append(audio_dirs[-1] if audio_dirs else '.')
        for iso_root, aud_dir in zip(args.isophonics, audio_dirs):
            found = find_isophonics_songs(iso_root, aud_dir)
            for s in found:
                s['provenance'] = 'gold' if s.get('beats') else 'silver'
                s['dataset'] = 'isophonics'
            all_songs.extend(found)
    else:
        print("ERROR: provide --config or --isophonics")
        sys.exit(1)

    if args.max_songs and len(all_songs) > args.max_songs:
        all_songs = all_songs[:args.max_songs]

    print(f"\nEvaluating {len(all_songs)} songs...")
    print(f"{'='*80}")

    # ── Evaluate ──
    btc_results = []
    f5_results = []

    for i, song in enumerate(all_songs):
        ds = song.get('dataset', '?')
        prov = song.get('provenance', '?')
        print(f"  [{i+1}/{len(all_songs)}] [{ds}/{prov}] {song['stem']}...",
              end=' ', flush=True)

        t0 = time.time()
        bm, fm = evaluate_song(
            song, btc_model, btc_mean, btc_std, freeze5_model, device)
        dt = time.time() - t0

        parts = []
        if bm:
            btc_results.append({**bm, 'name': song['stem'], 'dataset': ds, 'prov': prov})
            parts.append(f"BTC={bm['wcsr']:.3f}")
        if fm:
            f5_results.append({**fm, 'name': song['stem'], 'dataset': ds, 'prov': prov})
            parts.append(f"F5={fm['wcsr']:.3f}")
        print(f"{' | '.join(parts) or 'SKIP'} ({dt:.1f}s)")

    # ── Aggregate ──
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS ({len(all_songs)} songs)")
    print(f"{'='*80}")

    for label, results in [('BTC', btc_results), ('Freeze5', f5_results)]:
        if not results:
            continue
        total_beats = sum(r['n_beats'] for r in results)
        wcsr = sum(r['wcsr'] * r['n_beats'] for r in results) / max(total_beats, 1)
        maj = sum(r['major_acc'] * r['n_beats'] for r in results) / max(total_beats, 1)
        minor = sum(r['minor_acc'] * r['n_beats'] for r in results) / max(total_beats, 1)
        root = sum(r['root_acc'] * r['n_beats'] for r in results) / max(total_beats, 1)
        print(f"\n  {label:8s}  WCSR={wcsr:.3f}  maj={maj:.3f}  min={minor:.3f}  root={root:.3f}")
        print(f"           ({len(results)} songs, {total_beats} beats)")

        datasets = sorted(set(r['dataset'] for r in results))
        if len(datasets) > 1:
            for ds in datasets:
                ds_results = [r for r in results if r['dataset'] == ds]
                ds_beats = sum(r['n_beats'] for r in ds_results)
                ds_wcsr = sum(r['wcsr'] * r['n_beats'] for r in ds_results) / max(ds_beats, 1)
                print(f"           {ds:12s}: WCSR={ds_wcsr:.3f} ({len(ds_results)} songs, {ds_beats} beats)")

    # ── Per-song table ──
    if args.per_song:
        btc_by_name = {r['name']: r for r in btc_results}
        f5_by_name = {r['name']: r for r in f5_results}
        all_names = sorted(set(list(btc_by_name.keys()) + list(f5_by_name.keys())))
        print(f"\n  {'Song':<50s} {'BTC':>6s} {'F5':>6s} {'Diff':>6s}")
        print(f"  {'-'*50} {'-'*6} {'-'*6} {'-'*6}")
        for name in all_names:
            bw = btc_by_name.get(name, {}).get('wcsr')
            fw = f5_by_name.get(name, {}).get('wcsr')
            bstr = f"{bw:.3f}" if bw is not None else "  -  "
            fstr = f"{fw:.3f}" if fw is not None else "  -  "
            if bw is not None and fw is not None:
                dstr = f"{bw - fw:+.3f}"
            else:
                dstr = "  -  "
            print(f"  {name:<50s} {bstr:>6s} {fstr:>6s} {dstr:>6s}")

    # ── Head-to-head comparison ──
    if btc_results and f5_results:
        btc_by_name = {r['name']: r for r in btc_results}
        f5_by_name = {r['name']: r for r in f5_results}
        common = sorted(set(btc_by_name.keys()) & set(f5_by_name.keys()))

        if common:
            btc_wins, f5_wins, ties = 0, 0, 0
            diffs = []
            for name in common:
                b = btc_by_name[name]['wcsr']
                f = f5_by_name[name]['wcsr']
                diff = b - f
                diffs.append(diff)
                if abs(diff) < 0.005:
                    ties += 1
                elif diff > 0:
                    btc_wins += 1
                else:
                    f5_wins += 1
            avg_diff = np.mean(diffs)
            print(f"\n  Head-to-head ({len(common)} songs with both models):")
            print(f"    BTC wins: {btc_wins}, Freeze5 wins: {f5_wins}, Ties: {ties}")
            print(f"    Avg WCSR diff (BTC - Freeze5): {avg_diff:+.3f}")

            named_diffs = [(name, btc_by_name[name]['wcsr'], f5_by_name[name]['wcsr'])
                           for name in common]
            named_diffs.sort(key=lambda x: x[1] - x[2])

            print(f"\n  Biggest Freeze5 advantages:")
            for name, bw, fw in named_diffs[:5]:
                print(f"    {name:<50s} BTC={bw:.3f} F5={fw:.3f} diff={bw-fw:+.3f}")
            print(f"\n  Biggest BTC advantages:")
            for name, bw, fw in named_diffs[-5:]:
                print(f"    {name:<50s} BTC={bw:.3f} F5={fw:.3f} diff={bw-fw:+.3f}")


if __name__ == '__main__':
    main()
