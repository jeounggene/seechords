# SeeChords Training: Session Continuation Summary

**Session Focus:** Enable controlled flip-rate reduction experiments via new training knobs.

**Completed:** ✅ Implemented 5 new opt-in CLI flags for emission stabilization research.

---

## What's Been Done

### 1. Code Enhancements

**File: `training/v2/transformer_model.py`**
- Added `emission_dropout` parameter to `ChordTransformerCRF.__init__()`
- Added `emission_noise_std` parameter to `ChordTransformerCRF.__init__()`
- Wired into forward pass (applied before temperature scaling, only during training)
- No breaking changes; defaults to 0.0 (disabled)

**File: `training/v2/train_transformer_crf.py`**
- New CLI args: `--emission-dropout`, `--emission-noise-std`, `--aux-label-smoothing`, `--encoder-grad-scale`, `--crf-self-reg`, `--crf-self-target`
- Integrated aux label smoothing into root/quality CE losses via PyTorch's built-in
- Encoder gradient scaling applied after backward, before optimizer step
- CRF self-transition regularization: L2 penalty on `(diagonal - target)^2`
- All hyperparams stored in checkpoint metadata for reproducibility
- Improved logging during training to show new losses and features

### 2. Training Script Validation

- ✅ New flags appear in `--help`
- ✅ No syntax errors
- ✅ Backward compatible (all defaults preserve existing behavior)

### 3. Documentation

- **`training/NEXT_EXPERIMENTS.md`** — Ranked list of 5 follow-up experiments with copy-paste commands
- **Session memory** (`/memories/session/seechords_continuation_plan.md`) — Full implementation notes and experiment rationale

---

## Key Findings from Original Summary

| Aspect | Status |
|--------|--------|
| WCSR | ✅ 0.644 (post-hoc) — exceeds target 0.55 |
| Minor Accuracy | ✅ 0.449 — exceeds target 0.40, +27% vs RF |
| Flip Rate | ⚠️ 0.209 (post-hoc) or 0.131 (bias=20) — vs RF 0.050 |
| **Bottleneck** | Flip rate; emission instability prevents CRF from stabilizing |

---

## Ready-to-Run Experiments

### Quickest Win: Higher Self-Bias
```bash
cd training
python -m v2.train_transformer_crf \
  --data data/features_v2.npz --out models/chord_transformer_crf_bias5.pt \
  --gold-only --augment --feature-dim 60 \
  --emission-temp 5.0 --aux-weight 0.25 --freeze-encoder-epochs 5 \
  --encoder-lr 5e-5 --crf-self-bias 5.0

# Then evaluate
python -m v2.posthoc_viterbi --data data/features_v2.npz \
  --checkpoints models/chord_transformer_crf_bias5.pt \
  --self-biases 0 2 4 6 8 10 15 20 --split test
```

### Most Likely to Reduce Flip: Emission Dropout + Smoothing
```bash
python -m v2.train_transformer_crf \
  --data data/features_v2.npz --out models/chord_transformer_crf_dropout_smooth.pt \
  --gold-only --augment --feature-dim 60 \
  --emission-temp 5.0 --aux-weight 0.25 --freeze-encoder-epochs 5 \
  --encoder-lr 5e-5 --emission-dropout 0.1 --aux-label-smoothing 0.1

# Evaluate
python -m v2.posthoc_viterbi --data data/features_v2.npz \
  --checkpoints models/chord_transformer_crf_dropout_smooth.pt \
  --self-biases 0 2 4 6 8 10 15 20 --split test
```

Full details and all 5 experiments in `training/NEXT_EXPERIMENTS.md`.

---

## How to Use

1. **Activate environment:**
   ```bash
   cd /Users/genej/projects/seechords/training
   source ../.venv/bin/activate
   export PYTHONUNBUFFERED=1
   ```

2. **Pick an experiment from `NEXT_EXPERIMENTS.md`**, copy the command, run it (≈30–60 min on MPS)

3. **Evaluate with post-hoc Viterbi sweep** for flip/WCSR tradeoff

4. **Log best metrics** to compare across experiments

---

## Checkpoints

- **Current best:** `models/chord_transformer_crf_freeze5.pt` (WCSR=0.630 CRF, 0.644 post-hoc)
- **Test metrics saved:** Inside each `.pt` file under `checkpoint['test_metrics']`
- **Hyperparams saved:** Under `checkpoint['hyperparams']` for full reproducibility

---

## What Each New Flag Does

| Flag | Purpose | Default | Expected Effect |
|------|---------|---------|-----------------|
| `--emission-dropout` | Dropout on tier1 logits before CRF (train only) | 0.0 | Reduces emission peakiness; CRF learns transitions better |
| `--emission-noise-std` | Gaussian noise on emissions (train only) | 0.0 | Regularizes overconfidence; softer predictions |
| `--aux-label-smoothing` | Label smoothing on root/quality CE | 0.0 | Softer targets → less peaky emission dist. |
| `--encoder-grad-scale` | Scale encoder gradients by factor | 1.0 | Decouple encoder/CRF learning rates (gradient-level) |
| `--crf-self-reg` | L2 penalty on CRF diagonal toward target | 0.0 | Explicit encouragement of self-transitions |

---

## Theoretical Rationale

**Core problem:** Transformer emits sharp, frame-local predictions. CRF transitions can't stabilize them during training.

**Why flip happens:** Emissions spike on frame t in state A, frame t+1 in state B → CRF forced to transition by emission, not learned state prior.

**Solutions tested:**
- ✅ Frozen warmup (5 epochs) — works; best current model
- ✅ τ=5 emission temperature — essential
- ✅ aux=0.25 — sweetspot for encoder supervision

**New levers:**
1. **Dropout** — Force CRF to learn, not memorize emissions
2. **Smoothing** — Soften targets so emissions aren't so peaky
3. **Noise** — Regularize overconfident predictions
4. **Gradient scaling** — Decouple encoder/CRF convergence rates
5. **Self-reg** — Explicit diagonal penalty to prevent unlearning

---

## Success Criteria

- **Original spec:** WCSR ≥ 0.55, Minor ≥ 0.40, Flip ≤ 0.055
- **Currently met:** WCSR=0.644 ✅, Minor=0.449 ✅, Flip=0.209 ❌
- **Any experiment achieving Flip < 0.15 at WCSR > 0.62 is a win**

---

## Troubleshooting

**Q: ModuleNotFoundError when running?**
A: Run `pip install -r requirements.txt` from `training/` or let env setup handle it.

**Q: Training is slow?**
A: Normal on CPU. On MPS (Apple Silicon), ≈2–8 min/epoch depending on batch size. Use `--cpu` to force CPU if MPS hangs.

**Q: Post-hoc eval says "no improvement"?**
A: Check if the bias range is sufficient. Default sweeps 0–20; if flip is still high at 20, try `--self-biases 0 5 10 20 30 50 100`.

**Q: Which checkpoint should I evaluate?**
A: `models/chord_transformer_crf_freeze5.pt` is the standing best. Compare new experiments against it.

---

## Files Modified / Created

- ✅ `training/v2/transformer_model.py` — Added dropout/noise to emissions
- ✅ `training/v2/train_transformer_crf.py` — Added 5 new CLI flags + loss integrations
- ✅ `training/NEXT_EXPERIMENTS.md` — Experiment guide (NEW)
- ✅ `/memories/session/seechords_continuation_plan.md` — Detailed plan (NEW)

---

**Ready to experiment!** Pick any command from `NEXT_EXPERIMENTS.md` and run. All infrastructure is in place; defaults preserve existing behavior.
