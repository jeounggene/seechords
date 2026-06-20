#!/usr/bin/env python3
"""Chord & beat analysis pipeline.

Beat detection: Beat This! transformer (ISMIR 2024).
Chord recognition: BTC transformer (170-class, CQT features).
Key estimation: chroma profile correlation.
"""
import sys
import os
import json
import numpy as np

SR = 44100
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# ── Beat This! transformer beat tracker (ISMIR 2024) ────────
_BEAT_THIS_MODEL = None


def _detect_beats_beat_this(audio_path):
    """Detect beats using Beat This! transformer model (ISMIR 2024).

    Returns (beat_times, downbeat_times) as Python lists of float seconds.
    """
    global _BEAT_THIS_MODEL
    if _BEAT_THIS_MODEL is None:
        from beat_this.inference import File2Beats
        _BEAT_THIS_MODEL = File2Beats(
            checkpoint_path="small0", device="cpu", dbn=False)
    beats, downbeats = _BEAT_THIS_MODEL(audio_path)
    return beats.tolist(), downbeats.tolist()


# ── BTC (pre-trained 170-class model) inference ─────────────
_BTC_MODEL = None
_BTC_DEVICE = None
_BTC_MEAN = None
_BTC_STD = None


def _btc_checkpoint_path():
    return os.environ.get(
        'BTC_CHECKPOINT',
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'btc_model', 'btc_model_best.pth'),
    )


def _load_btc_model():
    """Lazy-load BTC model checkpoint (CPU)."""
    global _BTC_MODEL, _BTC_DEVICE, _BTC_MEAN, _BTC_STD
    if _BTC_MODEL is not None:
        return _BTC_MODEL, _BTC_DEVICE, _BTC_MEAN, _BTC_STD
    import torch
    from btc_model.btc_model import BTC_model

    path = _btc_checkpoint_path()
    device = torch.device('cpu')
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Handle both checkpoint formats (large_voca vs best)
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    norm = ckpt.get('normalization', {})
    if isinstance(norm, dict) and 'mean' in norm:
        _BTC_MEAN = float(norm['mean'])
        _BTC_STD = float(norm['std'])
    else:
        mean_val = ckpt.get('mean')
        std_val = ckpt.get('std')
        _BTC_MEAN = float(mean_val) if mean_val is not None else -2.37
        _BTC_STD = float(std_val) if std_val is not None else 1.96

    model = BTC_model()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _BTC_MODEL = model
    _BTC_DEVICE = device
    return model, device, _BTC_MEAN, _BTC_STD


def _extract_cqt(audio_path, sr=22050, hop_length=2048, n_bins=144,
                 bins_per_octave=24):
    """Extract CQT spectrogram using librosa (log-magnitude, matching ChordMini).

    Returns (n_frames, 144) float32 array.
    """
    import librosa
    y, _ = librosa.load(audio_path, sr=sr)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length,
                       n_bins=n_bins, bins_per_octave=bins_per_octave,
                       fmin=librosa.note_to_hz('C1'))
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)


def _btc_decode_chords(audio_path, beat_times):
    """Run BTC model on CQT features matching ChordMini's BTC inference.

    - No overlap (stride = seq_len = 108), zero-padded to align with seq_len
    - No temporal smoothing (raw logits → argmax)
    - Per-beat chord via majority vote of frame predictions
    """
    import torch
    from btc_model.vocab import btc_idx_to_display

    model, device, mean, std = _load_btc_model()

    cqt = _extract_cqt(audio_path)
    cqt = (cqt - mean) / max(std, 1e-6)

    n_frames = cqt.shape[0]
    if n_frames == 0:
        return None

    seq_len = 108
    # ChordMini uses 0% overlap for BTC (stride = seq_len).
    # Pad total frames to be divisible by seq_len (ChordMini-style).
    num_pad = 0
    remainder = n_frames % seq_len
    if remainder != 0:
        num_pad = seq_len - remainder
        cqt = np.pad(cqt, ((0, num_pad), (0, 0)), mode='constant')
    padded_frames = cqt.shape[0]

    all_logits = np.zeros((padded_frames, 170), dtype=np.float32)

    with torch.no_grad():
        for pos in range(0, padded_frames, seq_len):
            chunk = cqt[pos:pos + seq_len]
            x = torch.from_numpy(chunk).unsqueeze(0).to(device)
            out = model(x)  # (1, seq_len, 170)
            all_logits[pos:pos + seq_len] = out[0].cpu().numpy()

    # Trim back to original length
    all_logits = all_logits[:n_frames]

    # Frame-level argmax (ChordMini default for BTC: no smoothing, hard vote)
    frame_preds = all_logits.argmax(axis=1).astype(np.int64)

    # Beat-sync: majority vote of frame predictions within each beat window
    hop_dur = 2048 / 22050.0  # ~0.093s per CQT frame
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
        f_end = min(f_end, n_frames)
        segment = frame_preds[f_start:f_end]
        if len(segment) == 0:
            beat_chords.append('N')
            continue
        counts = np.bincount(segment, minlength=170)
        winner = int(counts.argmax())
        beat_chords.append(btc_idx_to_display(winner))

    return beat_chords


def _gate_leading_silence_beats(audio_eq, beat_times, chord_labels, sr, rel_frac=0.08):
    """Set chord to N for beats from the start until RMS exceeds rel_frac × track peak.

    The chord model can label beats while the mix is still inaudible (silent intro,
    video lead-in). Only **leading** beats are changed so quiet verses mid-song are
    not wiped. Uses equal-loudness audio so bass in the first bar still counts.
    """
    if not beat_times or not chord_labels or len(chord_labels) != len(beat_times):
        return chord_labels
    n = len(beat_times)
    n_audio = len(audio_eq)
    rms_list = []
    for bi in range(n):
        t0 = float(beat_times[bi])
        t1 = float(beat_times[bi + 1]) if bi + 1 < n else t0 + 0.5
        i0 = int(max(0, min(n_audio - 1, round(t0 * sr))))
        i1 = int(max(i0 + 1, min(n_audio, round(t1 * sr))))
        seg = audio_eq[i0:i1]
        rms_list.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    peak = max(rms_list) if rms_list else 0.0
    if peak < 1e-10:
        return chord_labels
    rel_frac = max(0.02, min(0.5, float(rel_frac)))
    thresh = rel_frac * peak
    out = list(chord_labels)
    for bi in range(n):
        if rms_list[bi] >= thresh:
            break
        out[bi] = 'N'
    return out


def _uniformize_beat_times(beat_times, bpm):
    """Replace beat times with a fixed grid t_i = t0 + i * (60/bpm).

    Chord labels stay index-aligned; only timestamps change. Used so the client
    timeline scrolls at constant speed (equal px per beat ↔ equal seconds per beat).

    Set UNIFORM_BEAT_GRID=0 to keep detector-native irregular beat times.
    """
    if not beat_times or len(beat_times) < 2:
        return list(beat_times) if beat_times else []
    if os.environ.get('UNIFORM_BEAT_GRID', '1').lower() in ('0', 'false', 'no'):
        return list(beat_times)
    bpm = float(max(40.0, min(300.0, float(bpm))))
    interval = 60.0 / bpm
    t0 = float(beat_times[0])
    n = len(beat_times)
    return [round(t0 + i * interval, 3) for i in range(n)]


def _postprocess_and_format(final_path, beat_times, bpm, key_str, audio_path,
                            skip_silence_gate=False, downbeats=None):
    """Shared post-processing: silence gate, smoothing, merge segments."""
    if not skip_silence_gate and \
       os.environ.get('SILENCE_GATE', '1').lower() not in ('0', 'false', 'no'):
        try:
            import librosa
            y_gate, _sr = librosa.load(audio_path, sr=SR, mono=True)
            try:
                rel = float(os.environ.get('SILENCE_GATE_FRAC', '0.08'))
            except ValueError:
                rel = 0.08
            final_path = _gate_leading_silence_beats(
                y_gate, beat_times, final_path, SR, rel_frac=rel)
        except Exception:
            pass

    # Remove isolated single-beat chords surrounded by the same chord
    if len(final_path) >= 3:
        smoothed = list(final_path)
        for i in range(1, len(smoothed) - 1):
            if smoothed[i - 1] == smoothed[i + 1] and smoothed[i] != smoothed[i - 1]:
                smoothed[i] = smoothed[i - 1]
        final_path = smoothed

    beat_times = _uniformize_beat_times(beat_times, bpm)
    iv_tail = (beat_times[1] - beat_times[0]) if len(beat_times) >= 2 else 0.5

    # Merge consecutive identical chords into timed segments
    merged = []
    for i, chord_name in enumerate(final_path):
        start = beat_times[i]
        end = beat_times[i + 1] if i + 1 < len(beat_times) else start + iv_tail
        if merged and merged[-1]['chord'] == chord_name:
            merged[-1]['end'] = round(end, 3)
        else:
            merged.append({'chord': chord_name, 'start': round(start, 3), 'end': round(end, 3)})

    result = {
        'chords':     merged,
        'bpm':        round(float(bpm), 1),
        'key':        key_str,
        'beat_times': [round(b, 3) for b in beat_times],
    }
    if downbeats:
        result['downbeats'] = [round(d, 3) for d in downbeats]
    return result


def _estimate_key_from_chroma(beat_chroma_cols):
    """Estimate key from beat-level chroma using profile correlation (no Essentia)."""
    profile_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    profile_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    avg_chroma = np.mean(beat_chroma_cols, axis=1)
    if np.max(avg_chroma) < 1e-8:
        return 'C', 0
    best_score, best_key, best_scale = -1, 0, 'major'
    for shift in range(12):
        rolled = np.roll(avg_chroma, -shift)
        corr_maj = float(np.corrcoef(rolled, profile_major)[0, 1])
        corr_min = float(np.corrcoef(rolled, profile_minor)[0, 1])
        if corr_maj > best_score:
            best_score, best_key, best_scale = corr_maj, shift, 'major'
        if corr_min > best_score:
            best_score, best_key, best_scale = corr_min, shift, 'minor'
    key_str = NOTES[best_key] if best_scale == 'major' else NOTES[best_key] + 'm'
    return key_str, best_key


def analyze(audio_path):
    """Run Beat This! beat detection + BTC chord recognition."""
    print('[SeeChords] Beat This! + BTC', flush=True)
    beat_times, downbeats = _detect_beats_beat_this(audio_path)
    print(f'[SeeChords] Beat This! found {len(beat_times)} beats, {len(downbeats)} downbeats', flush=True)

    bpm = 60.0 / np.median(np.diff(beat_times)) if len(beat_times) >= 2 else 120.0

    btc_chords = _btc_decode_chords(audio_path, beat_times)
    if btc_chords is None or len(btc_chords) != len(beat_times):
        raise RuntimeError('BTC chord recognition failed')

    import librosa
    y_key, _ = librosa.load(audio_path, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y_key, sr=22050)
    key_str, _ = _estimate_key_from_chroma(chroma)

    result = _postprocess_and_format(
        btc_chords, beat_times, bpm, key_str, audio_path,
        downbeats=downbeats)
    print(f'[SeeChords] Success: {len(result["chords"])} segments', flush=True)
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: analyze_chords.py <audio_path>'}))
        sys.exit(1)

    result = analyze(sys.argv[1])
    print(json.dumps(result))
