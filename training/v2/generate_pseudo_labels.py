#!/usr/bin/env python3
"""Generate pseudo-labels for unlabeled audio using BTC teacher model.

Takes a directory of audio files, runs the pre-trained BTC model on each,
and outputs .lab files in Harte format (start_time end_time chord_label).

The BTC model (170-class) predictions are mapped to TIER1_VOCAB (25 classes)
for compatibility with our training pipeline.

Usage:
    python -m v2.generate_pseudo_labels \
        --audio-dir data/unlabeled/audio \
        --checkpoint server/btc_model/btc_model_best.pth \
        --out-dir data/pseudo_labels

    # With temporal smoothing and confidence filtering:
    python -m v2.generate_pseudo_labels \
        --audio-dir data/unlabeled/audio \
        --checkpoint server/btc_model/btc_model_best.pth \
        --out-dir data/pseudo_labels \
        --smooth-kernel 5 --min-confidence 0.4
"""
import sys
import os
import argparse
import glob
import numpy as np

_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SERVER_ROOT = os.path.join(os.path.dirname(_TRAINING_ROOT), 'server')
sys.path.insert(0, _TRAINING_ROOT)
sys.path.insert(0, _SERVER_ROOT)


def extract_cqt(audio_path, sr=22050, hop_length=2048, n_bins=144,
                bins_per_octave=24):
    """Extract CQT spectrogram (log-magnitude, matching ChordMini)."""
    import librosa
    y, _ = librosa.load(audio_path, sr=sr)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length,
                       n_bins=n_bins, bins_per_octave=bins_per_octave,
                       fmin=librosa.note_to_hz('C1'))
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)


def load_btc_model(checkpoint_path, device):
    """Load BTC model and normalization stats from checkpoint."""
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


def run_btc_inference(model, cqt, mean, std, device, smooth_kernel=0):
    """Run BTC model on CQT features, return frame-level predictions and confidences.

    Args:
        model: BTC model
        cqt: (n_frames, 144) CQT features
        mean, std: normalization parameters
        device: torch device
        smooth_kernel: temporal smoothing kernel size (0 = no smoothing)

    Returns:
        frame_preds: (n_frames,) int array of predicted class indices
        frame_confs: (n_frames,) float array of max class probabilities
    """
    import torch

    cqt_norm = (cqt - mean) / max(std, 1e-6)
    n_frames = cqt_norm.shape[0]

    if n_frames == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    seq_len = 108
    stride = 54

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
            out = model(x)
            logits = out[0, :actual_len].cpu().numpy()
            logit_sum[pos:pos + actual_len] += logits
            logit_count[pos:pos + actual_len] += 1.0
            pos += stride
            if pos >= n_frames:
                break

    logit_count[logit_count == 0] = 1.0
    avg_logits = logit_sum / logit_count[:, np.newaxis]

    # Optional temporal smoothing on logits
    if smooth_kernel > 0:
        from scipy.ndimage import uniform_filter1d
        avg_logits = uniform_filter1d(avg_logits, size=smooth_kernel, axis=0)

    # Softmax for confidence scores
    exp_logits = np.exp(avg_logits - avg_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    frame_preds = probs.argmax(axis=1)
    frame_confs = probs.max(axis=1)

    # Simple temporal smoothing: replace isolated predictions
    if len(frame_preds) >= 3:
        smoothed = frame_preds.copy()
        for i in range(1, len(smoothed) - 1):
            if smoothed[i - 1] == smoothed[i + 1] and smoothed[i] != smoothed[i - 1]:
                smoothed[i] = smoothed[i - 1]
        frame_preds = smoothed

    return frame_preds.astype(np.int32), frame_confs.astype(np.float32)


def predictions_to_lab(frame_preds, frame_confs, hop_dur, idx_to_chord_fn,
                       min_confidence=0.0, min_duration=0.0):
    """Convert frame-level predictions to .lab format segments.

    Args:
        frame_preds: (n_frames,) int predictions
        frame_confs: (n_frames,) float confidence scores
        hop_dur: duration of each frame in seconds
        idx_to_chord_fn: function mapping class index to chord label string
        min_confidence: discard segments below this confidence
        min_duration: minimum segment duration in seconds

    Returns:
        list of (start_time, end_time, chord_label) tuples
    """
    if len(frame_preds) == 0:
        return []

    segments = []
    current_pred = frame_preds[0]
    current_start = 0
    conf_accum = [frame_confs[0]]

    for i in range(1, len(frame_preds)):
        if frame_preds[i] != current_pred:
            t_start = current_start * hop_dur
            t_end = i * hop_dur
            avg_conf = np.mean(conf_accum)
            if avg_conf >= min_confidence and (t_end - t_start) >= min_duration:
                label = idx_to_chord_fn(int(current_pred))
                segments.append((t_start, t_end, label))
            current_pred = frame_preds[i]
            current_start = i
            conf_accum = [frame_confs[i]]
        else:
            conf_accum.append(frame_confs[i])

    # Final segment
    t_start = current_start * hop_dur
    t_end = len(frame_preds) * hop_dur
    avg_conf = np.mean(conf_accum)
    if avg_conf >= min_confidence and (t_end - t_start) >= min_duration:
        label = idx_to_chord_fn(int(current_pred))
        segments.append((t_start, t_end, label))

    return segments


def write_lab_file(segments, output_path):
    """Write segments to .lab file in Harte format."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for start, end, label in segments:
            f.write(f"{start:.6f} {end:.6f} {label}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate pseudo-labels using BTC teacher model')
    parser.add_argument('--audio-dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to BTC model checkpoint')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output directory for .lab files')
    parser.add_argument('--vocab', type=str, default='tier1',
                        choices=['tier1', 'display'],
                        help='Output vocabulary: tier1 (25-class) or display (richer)')
    parser.add_argument('--smooth-kernel', type=int, default=5,
                        help='Temporal smoothing kernel size (0=off)')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Minimum confidence to include a segment')
    parser.add_argument('--min-duration', type=float, default=0.1,
                        help='Minimum segment duration in seconds')
    parser.add_argument('--extensions', type=str, default='mp3,wav,flac,m4a,ogg',
                        help='Audio file extensions to process (comma-separated)')
    args = parser.parse_args()

    import torch
    from btc_model.vocab import btc_idx_to_tier1, btc_idx_to_display

    device = torch.device('cpu')
    print(f"Loading BTC model from {args.checkpoint}...")
    model, mean, std = load_btc_model(args.checkpoint, device)
    print(f"  Normalization: mean={mean:.4f}, std={std:.4f}")

    idx_to_chord = btc_idx_to_tier1 if args.vocab == 'tier1' else btc_idx_to_display

    # Find audio files
    extensions = args.extensions.split(',')
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, f'*.{ext}')))
    audio_files.sort()

    if not audio_files:
        print(f"No audio files found in {args.audio_dir}")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files")
    hop_dur = 2048 / 22050.0

    n_success = 0
    n_fail = 0
    for i, audio_path in enumerate(audio_files):
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(args.out_dir, f"{stem}.lab")

        print(f"  [{i+1}/{len(audio_files)}] {stem}...", end=' ', flush=True)
        try:
            cqt = extract_cqt(audio_path)
            preds, confs = run_btc_inference(model, cqt, mean, std, device,
                                             smooth_kernel=args.smooth_kernel)
            segments = predictions_to_lab(preds, confs, hop_dur, idx_to_chord,
                                          min_confidence=args.min_confidence,
                                          min_duration=args.min_duration)
            write_lab_file(segments, out_path)

            duration = len(cqt) * hop_dur
            avg_conf = float(confs.mean()) if len(confs) > 0 else 0.0
            n_chords = len(set(s[2] for s in segments if s[2] != 'N'))
            print(f"OK ({duration:.0f}s, {len(segments)} segments, "
                  f"{n_chords} unique chords, conf={avg_conf:.2f})")
            n_success += 1
        except Exception as e:
            print(f"FAIL: {e}")
            n_fail += 1

    print(f"\nDone: {n_success} succeeded, {n_fail} failed")
    print(f"Labels written to {args.out_dir}/")


if __name__ == '__main__':
    main()
