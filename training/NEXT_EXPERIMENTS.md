# SeeChords CRF Training: Next Experiments

**Status:** Emit stabilization via tuning is the core bottleneck. Flip rate (0.209 with post-hoc bias) is the remaining gap to match RF's 0.050.

**Best Checkpoint:** `models/chord_transformer_crf_freeze5.pt` (WCSR=0.630, Minor=0.429, Flip=0.170 CRF-decode)

## New Features Available (as of this session)

Five opt-in flags were added to the training script. All default to values that preserve existing behavior:

- `--emission-dropout FLOAT` — Apply dropout to tier1 logits before CRF during training (default 0.0)
- `--emission-noise-std FLOAT` — Add Gaussian noise to tier1 logits before CRF (default 0.0)
- `--aux-label-smoothing FLOAT` — Label smoothing on root/quality auxiliary CE losses (default 0.0)
- `--encoder-grad-scale FLOAT` — Multiply encoder gradients by this factor (default 1.0)
- `--crf-self-reg FLOAT` — L2 penalty on CRF self-transitions, toward `--crf-self-target` (default 0.0, target 2.0)

## Ranked Experiments (by expected flip reduction)

### 1. Higher Initial CRF Self-Bias → Less Flipping at Start
The CRF transitions are initialized with `--crf-self-bias` (current: 2.0). Raising it means stronger "stay in current chord" preference from epoch 1. May directly reduce flip without post-hoc tuning.

```bash
cd training
PYTHONUNBUFFERED=1 python -m v2.train_transformer_crf \
  --data data/features_v2.npz --out models/chord_transformer_crf_bias5.pt \
  --gold-only --augment --feature-dim 60 \
  --emission-temp 5.0 --aux-weight 0.25 --freeze-encoder-epochs 5 \
  --encoder-lr 5e-5 --crf-self-bias 5.0
```
Then evaluate:
```bash
PYTHONUNBUFFERED=1 python -m v2.posthoc_viterbi \
  --data data/features_v2.npz --checkpoints models/chord_transformer_crf_bias5.pt \
  --self-biases 0 2 4 6 8 10 15 20 --split test
```

### 2. Longer Freeze-Warmup + Faster CRF Learning
Previous run failed with `freeze=10` + halved encoder LR. Try freeze=10 at **full** encoder LR (1e-4) + faster CRF LR (5e-4) to give transitions more time to establish.

```bash
PYTHONUNBUFFERED=1 python -m v2.train_transformer_crf \
  --data data/features_v2.npz --out models/chord_transformer_crf_freeze10_v2.pt \
  --gold-only --augment --feature-dim 60 \
  --emission-temp 5.0 --aux-weight 0.25 --freeze-encoder-epochs 10 \
  --encoder-lr 1e-4 --crf-lr 5e-4
```

### 3. Aux Label Smoothing → Softer Emissions
Softens the root/quality targets, producing less peaky emission distributions. The CRF can then learn transitions more effectively.

```bash
PYTHONUNBUFFERED=1 python -m v2.train_transformer_crf \
  --data data/features_v2.npz --out models/chord_transformer_crf_smooth01.pt \
  --gold-only --augment --feature-dim 60 \
  --emission-temp 5.0 --aux-weight 0.25 --freeze-encoder-epochs 5 \
  --encoder-lr 5e-5 --aux-label-smoothing 0.1
```

### 4. Emission Dropout + Aux Smoothing → Combined Stabilization
Combines two mechanisms: dropout forces CRF to learn transitions, smoothing softens emission targets.

```bash
PYTHONUNBUFFERED=1 python -m v2.train_transformer_crf \
  --data data/features_v2.npz --out models/chord_transformer_crf_dropout_smooth.pt \
  --gold-only --augment --feature-dim 60 \
  --emission-temp 5.0 --aux-weight 0.25 --freeze-encoder-epochs 5 \
  --encoder-lr 5e-5 --emission-dropout 0.1 --aux-label-smoothing 0.1
```

### 5. CRF Self-Transition Regularization → Encourage High Diagonal
Adds an L2 penalty on `(diag - 2.5)^2` during training, explicitly pushing learned self-transitions to stay high.

```bash
PYTHONUNBUFFERED=1 python -m v2.train_transformer_crf \
  --data data/features_v2.npz --out models/chord_transformer_crf_self_reg.pt \
  --gold-only --augment --feature-dim 60 \
  --emission-temp 5.0 --aux-weight 0.25 --freeze-encoder-epochs 5 \
  --encoder-lr 5e-5 --crf-self-reg 0.01 --crf-self-target 2.5
```

---

## Evaluation Protocol

For any new model, run:
```bash
PYTHONUNBUFFERED=1 python -m v2.posthoc_viterbi \
  --data data/features_v2.npz \
  --checkpoints models/chord_transformer_crf_<name>.pt \
  --self-biases 0 2 4 6 8 10 15 20 \
  --split test
```

Look for:
- **Best WCSR**: Often at low bias (0–6)
- **Best Flip**: High bias (10–20)
- **WCSR/Flip tradeoff**: Usually bias=6–10

---

## What NOT to Try (Already Ruled Out)

- Dual LR without warmup — no benefit
- aux_weight < 0.25 — encoder starves for supervision
- freeze=10 with halved encoder_lr — too slow to recover
- Pure CRF loss (aux=0) — encoder can't learn effectively

---

## Logs & Hyperparams

Each checkpoint saves to the `.pt` file with metadata:
```python
torch.load('models/chord_transformer_crf_<name>.pt', weights_only=False)
# → checkpoint['hyperparams'] contains all args used
# → checkpoint['best_val_wcsr'] and ['test_metrics']
```

Use for comparisons and ablations.

---

## If Flip Doesn't Improve

If none of the above reduce flip significantly, next steps (lower dev impact):
1. **More data:** Add RWC + Billboard datasets to training (not just Beatles)
2. **Dual heads with stronger regularization:** Separate root/quality decoders with explicit constraint on quality discrimination
3. **Learned CRF init:** Warm-start self_bias from pre-trained RF model

---

## Questions or Issues?

- **Model won't load:** Check that checkpoint path is correct and `weights_only=False`
- **Training loop crashes:** Verify Python env has torch, numpy, scipy installed
- **Post-hoc eval shows same flip:** Try higher bias values; sweep may not cover needed range

Good luck! 🎸
