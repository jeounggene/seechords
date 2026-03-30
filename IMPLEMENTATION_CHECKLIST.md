# SeeChords Training: Implementation Checklist ✅

**Session Goal:** Enable controlled flip-rate reduction experiments via new training knobs, starting from WCSR=0.644, Minor=0.449, Flip=0.209.

---

## Code Changes

- [x] **`training/v2/transformer_model.py`**
  - Added `emission_dropout` parameter to `ChordTransformerCRF.__init__()`
  - Added `emission_noise_std` parameter to `ChordTransformerCRF.__init__()`
  - Integrated both into forward pass (before temperature scaling, training mode only)
  - Emission dropout module initialized correctly
  - No breaking changes; defaults=0.0

- [x] **`training/v2/train_transformer_crf.py`**
  - Added 5 new CLI arguments: `--emission-dropout`, `--emission-noise-std`, `--aux-label-smoothing`, `--encoder-grad-scale`, `--crf-self-reg`, `--crf-self-target`
  - Integrated aux label smoothing into root/quality CE losses
  - Implemented encoder gradient scaling after backward pass
  - Added CRF self-transition regularization (L2 penalty on diagonal)
  - Updated logging to show new losses/features
  - All hyperparams saved in checkpoint metadata
  - No breaking changes; all defaults preserve existing behavior

---

## Documentation

- [x] **`training/NEXT_EXPERIMENTS.md`** (NEW)
  - 5 ranked experiments with full copy-paste commands
  - Detailed rationale for each approach
  - Expected impact and success criteria
  - Evaluation protocol with post-hoc Viterbi

- [x] **`CONTINUATION_SUMMARY.md`** (NEW)
  - Project-level overview of session work
  - Summary of findings and current state
  - New features explained
  - Troubleshooting guide
  - Links to key checkpoints

- [x] **`training/RUN_EXPERIMENTS.sh`** (NEW)
  - Shell script with all 5 experiment commands
  - Quick-reference format
  - Environment setup notes

- [x] **Session Memory** (`/memories/session/seechords_continuation_plan.md`) (NEW)
  - Full implementation notes
  - Experiment details and rationale
  - Success criteria
  - Validation status

---

## Validation

- [x] **Syntax & Imports**
  - ✅ No errors in modified Python files
  - ✅ Model loads correctly with new parameters
  - ✅ All 5 new CLI flags registered and accessible

- [x] **Runtime Tests**
  - ✅ Model instantiation with new params successful
  - ✅ Forward pass works with dropout/noise enabled
  - ✅ Backward pass works (gradient scaling tested)
  - ✅ Inference (eval mode) works; dropout/noise disabled
  - ✅ CRF decode path length correct

- [x] **Backward Compatibility**
  - ✅ All new parameters have sensible defaults (e.g., 0.0 = disabled)
  - ✅ Existing checkpoints still load
  - ✅ Default training runs use no new features (unless flags passed)

---

## Files Modified or Created

| File | Status | Purpose |
|------|--------|---------|
| `training/v2/transformer_model.py` | ✅ Modified | Added emission dropout/noise |
| `training/v2/train_transformer_crf.py` | ✅ Modified | Added 5 new CLI flags + loss integrations |
| `training/NEXT_EXPERIMENTS.md` | ✅ New | Experiment guide + commands |
| `CONTINUATION_SUMMARY.md` | ✅ New | Project overview |
| `training/RUN_EXPERIMENTS.sh` | ✅ New | Quick-start shell script |
| `/memories/session/seechords_continuation_plan.md` | ✅ New | Detailed plan |
| `training/test_new_features.py` | ✅ New | Sanity check script |

---

## Ready-to-Run Experiments

**Quickest:** `--crf-self-bias 5.0`   
(Reduces flip from initialization)

**Most Likely to Help:** `--emission-dropout 0.1 --aux-label-smoothing 0.1`   
(Stabilizes learning + forces CRF to learn transitions)

**See `training/NEXT_EXPERIMENTS.md` for all 5 with full commands.**

---

## How to Use

### Setup
```bash
cd /Users/genej/projects/seechords/training
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1
```

### Run Any Experiment
```bash
# Example: Higher self-bias
python -m v2.train_transformer_crf \
  --data data/features_v2.npz --out models/chord_transformer_crf_bias5.pt \
  --gold-only --augment --feature-dim 60 \
  --emission-temp 5.0 --aux-weight 0.25 \
  --freeze-encoder-epochs 5 --encoder-lr 5e-5 \
  --crf-self-bias 5.0
```

### Evaluate Results
```bash
python -m v2.posthoc_viterbi \
  --data data/features_v2.npz \
  --checkpoints models/chord_transformer_crf_bias5.pt \
  --self-biases 0 2 4 6 8 10 15 20 \
  --split test
```

---

## Current Status

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| WCSR | 0.644 (post-hoc) | ≥ 0.55 | ✅ Achieved |
| Minor Accuracy | 0.449 | ≥ 0.40 | ✅ Achieved |
| Flip Rate | 0.209 (post-hoc) | ≤ 0.055 | ⚠️ Bottleneck |

**Remaining Work:** Run experiments from `NEXT_EXPERIMENTS.md` to reduce flip rate.

**Expected Impact:** Any single experiment achieving Flip < 0.15 at WCSR > 0.62 = meaningful win.

---

## Key Findings

✅ **Frozen warmup (5 epochs) works** — Encoder frozen for 5 epochs, CRF + heads learn transitions first  
✅ **τ=5 emission temperature essential** — Keeps transitions competitive with emissions  
✅ **aux=0.25 is the sweet spot** — Balances encoder supervision vs. CRF learning  
✅ **Post-hoc Viterbi recovers 1–2% WCSR** — External self-bias tuning provides flip control  
✅ **Minor accuracy is CRF's strength** — +27% vs. RF (0.449 vs. 0.355)  

⚠️ **Flip rate is the remaining gap** — Emissions still too unstable to prevent rapid switching

---

## Next Steps (After Experiments)

1. **If flip improves significantly** (e.g., < 0.12):
   - Combine best techniques
   - Run longer training (extend patience)
   - Consider ensemble with RF for production

2. **If flip still high**:
   - Add more diverse training data (RWC, Billboard)
   - Try learned CRF initialization from RF model
   - Consider separate root/quality decoders

3. **If WCSR drops below 0.62**:
   - Increase aux-weight or use curriculum learning
   - Reduce regularization strength

---

## Success Criteria

✅ All code is implemented and validated  
✅ All CLI flags are registered and working  
✅ No breaking changes to existing functionality  
✅ Comprehensive documentation provided  
✅ Ready for experimental runs  

**Next phase:** Execute experiments from `NEXT_EXPERIMENTS.md` to find flip-reducing approach.

---

## Support / Troubleshooting

**Q: Model won't load?**  
A: Ensure checkpoint path is correct. Use `weights_only=False` when loading old checkpoints.

**Q: Training is very slow?**  
A: Normal on CPU. On Apple Silicon (MPS), expect 2–8 min/epoch. If MPS hangs, use `--cpu`.

**Q: Post-hoc eval shows no improvement?**  
A: Try extending bias sweep range (e.g., `--self-biases 0 5 10 20 50 100`).

**Q: Help!**  
A: Check `CONTINUATION_SUMMARY.md` or `NEXT_EXPERIMENTS.md` for detailed guidance.

---

**Session Complete.** Ready to run experiments! 🎸
