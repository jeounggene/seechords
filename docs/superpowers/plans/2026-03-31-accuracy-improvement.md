# Chord Accuracy Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the accuracy gap with ChordMini by improving chord recognition quality without retraining, using inference-time improvements only (retraining tasks are marked as Phase 2).

**Architecture:** Three-phase approach: (1) inference pipeline improvements that require zero training, (2) ensemble with the existing teacher checkpoint, (3) optional retraining via ChordMini's 2-stage pipeline on our labeled data.

**Tech Stack:** Python, PyTorch, librosa, numpy, Beat This!, BTC transformer

---

## Current Baseline

- **Beatles (180 songs, 53,638 beats):** WCSR=86.6%, Root=88.5%, Major=88.8%, Minor=60.1%
- Checkpoint: `btc_model_best.pth` (identical to ChordMini's BTC CL student)
- Inference: 75% overlap, Gaussian logit smoothing (k=9), majority filter (k=9)
- Beat detection: Beat This! (ISMIR 2024)

## Key Finding

ChordMini uses the **same checkpoint** and their BTC inference defaults to **0% overlap** (no sliding window overlap). Their accuracy gains come from the two-stage training pipeline (pseudo-labeling + continual learning with KD), not inference tricks. SeeChords is actually doing **more** at inference time than ChordMini. The 86.6% Beatles WCSR is not directly comparable to ChordMini's 71.6% on their mixed 120-song test set (different, harder dataset).

The biggest single weakness is **minor chord accuracy at 60.1%** -- the model gets the root right 88.5% of the time but confuses major/minor quality.

---

## Phase 1: Inference Improvements (No Retraining)

### Task 1: HPSS Source Separation Before CQT

**Rationale:** Harmonic-Percussive Source Separation isolates the tonal content before CQT extraction. Drums and percussion create noise in the CQT that can confuse the model about chord quality (especially the 3rd interval that distinguishes major/minor). ChordMini doesn't do this, but it's a well-known MIR preprocessing step.

**Files:**
- Modify: `server/analyze_chords.py` — `_extract_cqt()` function (~line 502-513)

- [ ] **Step 1: Add HPSS to CQT extraction**

In `server/analyze_chords.py`, modify `_extract_cqt`:

```python
def _extract_cqt(audio_path, sr=22050, hop_length=2048, n_bins=144,
                 bins_per_octave=24):
    """Extract CQT spectrogram from harmonic component (HPSS preprocessing)."""
    import librosa
    y, _ = librosa.load(audio_path, sr=sr)
    y_harm, _ = librosa.effects.hpss(y)
    cqt = librosa.cqt(y_harm, sr=sr, hop_length=hop_length,
                       n_bins=n_bins, bins_per_octave=bins_per_octave,
                       fmin=librosa.note_to_hz('C1'))
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)
```

- [ ] **Step 2: Run eval on 20 Beatles songs to measure impact**

```bash
cd training && python -m v2.eval_btc_production --datasets beatles --max-songs 20 --cpu
```

Compare WCSR and especially minor accuracy vs baseline (86.6% / 60.1%).

- [ ] **Step 3: If WCSR drops, revert; if improves or neutral, keep**

The BTC checkpoint was trained on raw CQT (no HPSS), so HPSS may help or hurt. Measure first.

- [ ] **Step 4: Commit if keeping**

```bash
git add server/analyze_chords.py
git commit -m "feat: add HPSS preprocessing to CQT extraction for cleaner harmonic content"
```

---

### Task 2: Logit-Level Beat Aggregation — **SHIPPED**

**Rationale:** Previously we took argmax per frame, applied a majority filter on frame classes, then majority-voted across frames in each beat. That discards logit magnitudes. The new default **sums Gaussian-smoothed logits** over each beat window and takes a single argmax, so uncertain frames still contribute proportionally.

**Implementation (2026-03-31)**

| Item | Detail |
|------|--------|
| **Default behavior** | `BTC_BEAT_AGGREGATION` unset or `logit` → logit sum per beat |
| **Rollback / backup** | `BTC_BEAT_AGGREGATION=majority` → previous path (argmax → majority filter → per-beat majority vote) |
| **Aliases for legacy** | `majority`, `vote`, `legacy` all select the old path |
| **Code** | `server/analyze_chords.py`: `_btc_beat_aggregation_mode()`, `_btc_decode_chords()` |
| **Eval parity** | `training/v2/eval_btc_production.py` mirrors both modes; `--beat-aggregation logit\|majority` overrides env |

**Full A/B (180 songs, 53,638 beats, 2026-03-31):**

| Mode | WCSR | Root | Major | Minor |
|------|------|------|-------|-------|
| `majority` (legacy) | 86.6% | 88.5% | 88.8% | 60.1% |
| `logit` (default) | **86.7%** | **88.6%** | **88.9%** | 60.0% |

Logit is slightly better on overall WCSR and root/major; minor is unchanged within rounding. **Keep logit as default.**

Quick re-check commands:

```bash
cd training
python -m v2.eval_btc_production --datasets beatles --cpu --beat-aggregation logit
python -m v2.eval_btc_production --datasets beatles --cpu --beat-aggregation majority
```

**Production / Fly:** To force legacy behavior without redeploying code:

```bash
fly secrets set BTC_BEAT_AGGREGATION=majority -a seechords
```

To restore default (logit aggregation), remove the secret or set `BTC_BEAT_AGGREGATION=logit`.

**Files:**
- Modify: `server/analyze_chords.py` — `_btc_decode_chords()`, `_btc_beat_aggregation_mode()`
- Modify: `training/v2/eval_btc_production.py` — beat sync matches production; `--beat-aggregation`

- [x] Logit aggregation implemented with env-based fallback
- [x] Documentation (this section)

---

### Task 3: Confidence-Weighted Quality Correction

**Rationale:** When BTC predicts a chord class with low confidence and the second-best class shares the same root but differs only in quality (e.g., top=Am at 35%, second=A at 30%), the quality decision is uncertain. In these cases, using harmonic features (the 3rd interval energy) to break the tie can improve major/minor discrimination -- our weakest metric at 60.1%.

**Files:**
- Modify: `server/analyze_chords.py` — add `_quality_tiebreak()` helper, integrate into `_btc_decode_chords`

- [ ] **Step 1: Add quality tiebreak function**

```python
def _quality_tiebreak(logits_for_beat, threshold=0.15):
    """When top two predictions share the same root but differ in quality,
    and the confidence gap is within threshold, return the index with higher
    logit. Otherwise return the argmax as-is.
    
    This specifically targets major/minor confusion where the model is uncertain.
    """
    top2 = np.argsort(logits_for_beat)[-2:][::-1]
    first, second = int(top2[0]), int(top2[1])
    
    # Check if same root (every 14 consecutive classes share a root)
    if first // 14 != second // 14:
        return first  # different roots, no tiebreak needed
    
    # Same root -- check confidence gap
    gap = logits_for_beat[first] - logits_for_beat[second]
    total = abs(logits_for_beat[first]) + abs(logits_for_beat[second])
    if total == 0:
        return first
    
    relative_gap = gap / total
    if relative_gap < threshold:
        # Low confidence between these two qualities -- already returning top pick,
        # but log for analysis
        pass
    
    return first
```

- [ ] **Step 2: Integrate into beat aggregation loop**

Replace `winner = int(beat_logits.sum(axis=0).argmax())` with:
```python
        summed = beat_logits.sum(axis=0)
        winner = _quality_tiebreak(summed)
```

- [ ] **Step 3: Run eval, compare minor accuracy specifically**

```bash
cd training && python -m v2.eval_btc_production --datasets beatles --max-songs 20 --cpu
```

Focus on whether minor chord accuracy improves from 60.1%.

- [ ] **Step 4: Commit**

```bash
git add server/analyze_chords.py
git commit -m "feat: quality tiebreak for uncertain major/minor predictions"
```

---

## Phase 2: Teacher Ensemble (No Retraining, Uses Existing Checkpoint)

### Task 4: Ensemble Student + Teacher Logits

**Rationale:** The teacher model (`btc_model_large_voca.pt`) uses different normalization (mean=-2.228, std=1.719) and was trained differently. It makes different errors than the student. Averaging their logits before argmax is a standard ensemble technique that typically improves accuracy by 1-3%.

**Files:**
- Modify: `server/analyze_chords.py` — add teacher loading, ensemble in `_btc_decode_chords`

- [ ] **Step 1: Add teacher model loading**

```python
_BTC_TEACHER_MODEL = None

def _load_btc_teacher():
    """Lazy-load original BTC teacher checkpoint."""
    global _BTC_TEACHER_MODEL
    if _BTC_TEACHER_MODEL is not None:
        return _BTC_TEACHER_MODEL
    import torch
    from btc_model.btc_model import BTC_model
    
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'btc_model', 'btc_model_large_voca.pt')
    if not os.path.isfile(path):
        return None
    
    device = torch.device('cpu')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = BTC_model()
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    _BTC_TEACHER_MODEL = model
    return model
```

- [ ] **Step 2: Add ensemble inference path**

In `_btc_decode_chords`, after computing `avg_logits` for the student, also run the teacher:

```python
    teacher = _load_btc_teacher()
    if teacher is not None:
        teacher_mean, teacher_std = -2.2280, 1.7191
        cqt_teacher = (cqt_raw - teacher_mean) / max(teacher_std, 1e-6)
        
        teacher_logit_sum = np.zeros((n_frames, 170), dtype=np.float32)
        teacher_logit_count = np.zeros(n_frames, dtype=np.float32)
        
        with torch.no_grad():
            pos = 0
            while pos < n_frames:
                end = min(pos + seq_len, n_frames)
                chunk = cqt_teacher[pos:end]
                actual_len = chunk.shape[0]
                if actual_len < seq_len:
                    pad = np.zeros((seq_len - actual_len, 144), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=0)
                x = torch.from_numpy(chunk).unsqueeze(0).to(device)
                out = teacher(x)
                logits = out[0, :actual_len].cpu().numpy()
                teacher_logit_sum[pos:pos + actual_len] += logits
                teacher_logit_count[pos:pos + actual_len] += 1.0
                pos += stride
                if pos >= n_frames:
                    break
        
        teacher_logit_count[teacher_logit_count == 0] = 1.0
        teacher_avg = teacher_logit_sum / teacher_logit_count[:, np.newaxis]
        teacher_avg = _gaussian_smooth_logits(teacher_avg, kernel_size=9)
        
        # Weighted ensemble: student 0.6, teacher 0.4
        avg_logits = 0.6 * avg_logits + 0.4 * teacher_avg
```

- [ ] **Step 3: Run eval, compare all metrics**

```bash
cd training && python -m v2.eval_btc_production --datasets beatles --max-songs 20 --cpu
```

- [ ] **Step 4: Tune ensemble weight (try 0.5/0.5, 0.7/0.3)**

- [ ] **Step 5: Commit**

```bash
git add server/analyze_chords.py
git commit -m "feat: student+teacher ensemble for chord prediction"
```

**Note:** This doubles inference time. Add an env var `USE_ENSEMBLE=1` to enable/disable.

---

## Phase 3: Retraining (Requires Labeled Data + GPU Time)

### Task 5: Run ChordMini's 2-Stage Training Pipeline on Our Data

**Rationale:** The single biggest accuracy improvement in ChordMini comes from their two-stage training: (1) pseudo-labeling on unlabeled audio, (2) continual learning with selective KD on labeled data. We have 180 Beatles + Queen songs with gold annotations, plus Billboard silver labels.

**Files:**
- Clone: ChordMini repo (`github.com/ptnghia-j/ChordMini`)
- Prepare: Our labeled data in ChordMini's expected format
- Output: New `btc_model_seechords.pth` checkpoint

- [ ] **Step 1: Clone ChordMini and set up environment**

```bash
git clone https://github.com/ptnghia-j/ChordMini.git /tmp/ChordMini
cd /tmp/ChordMini
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 2: Prepare labeled data in ChordMini format**

ChordMini expects:
```
data/labeled/audio/*.mp3
data/labeled/chordlab/*.lab
```

Create symlinks from our Isophonics data:
```bash
mkdir -p data/labeled/audio data/labeled/chordlab
# Link Beatles audio + chord labs
# Link Queen audio + chord labs
```

- [ ] **Step 3: Run Stage 1 (pseudo-labeling) on unlabeled audio**

Use our existing audio files as unlabeled data:
```bash
python src/training_scripts/train_pseudo_labeling.py \
  --model_type BTC \
  --audio_dir data/unlabeled \
  --teacher_checkpoint checkpoints/btc_model_large_voca.pt \
  --save_dir checkpoints/pseudo_labeling_btc \
  --batch_size 256 --num_epochs 100 --learning_rate 1e-4 \
  --use_focal_loss --focal_gamma 2.0 --seed 42
```

- [ ] **Step 4: Run Stage 2 (continual learning) on labeled data**

```bash
python src/training_scripts/train_continual_learning.py \
  --model_type BTC \
  --student_checkpoint checkpoints/pseudo_labeling_btc/best_model.pth \
  --teacher_checkpoint checkpoints/btc_model_large_voca.pt \
  --audio_dir data/labeled/audio \
  --label_dir data/labeled/chordlab \
  --save_dir checkpoints/continual_learning_btc \
  --batch_size 128 --num_epochs 50 --learning_rate 1e-5 \
  --selective_kd --kd_confidence_threshold 0.9 \
  --use_focal_loss --focal_gamma 2.0 --seed 42
```

- [ ] **Step 5: Evaluate new checkpoint and swap if better**

```bash
cp checkpoints/continual_learning_btc/single_split/best_model.pth \
   /path/to/seechords/server/btc_model/btc_model_best.pth
cd /path/to/seechords/training
python -m v2.eval_btc_production --datasets beatles --cpu
```

- [ ] **Step 6: Commit new checkpoint if improved**

---

## Measurement Plan

After each task, run the full eval and record:

| Task | WCSR | Root | Major | Minor | Notes |
|------|------|------|-------|-------|-------|
| Baseline | 86.6% | 88.5% | 88.8% | 60.1% | Beatles 180, **majority** beat sync (pre-Task 2) |
| +HPSS | ? | ? | ? | ? | Task 1 |
| +Logit agg | **86.7%** | **88.6%** | **88.9%** | **60.0%** | Task 2 default; **full 180 songs** (2026-03-31): vs majority below |
| Majority (legacy) | 86.6% | 88.5% | 88.8% | 60.1% | `BTC_BEAT_AGGREGATION=majority`, same 53,638 beats |

**Full Beatles A/B (180 songs, Beat This! beats, tier-1 WCSR):** Logit wins WCSR / root / major by **+0.1 percentage point** each; minor is **flat** (60.0% vs 60.1%, rounding). Safe to keep **logit** as default; legacy remains one env var away.
| +Quality tiebreak | ? | ? | ? | ? | Task 3 |
| +Ensemble | ? | ? | ? | ? | Task 4 |
| +Retrain | ? | ? | ? | ? | Task 5 |

Target: WCSR >= 90%, Minor >= 75%
