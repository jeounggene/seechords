#!/bin/bash
cd /Users/genej/projects/chords/seechords/training
mkdir -p logs

# Frozen encoder warmup: train CRF/heads first, then unfreeze encoder
# Base config: aux=0.25, τ=5, crf_self_bias=2.0
# After unfreeze: encoder_lr=5e-5 (half of CRF/head LR)

echo "=== freeze=5, encoder_lr=5e-5 at $(date) ==="
PYTHONUNBUFFERED=1 /opt/miniconda3/bin/python -m v2.train_transformer_crf \
  --data data/features_v2.npz \
  --out models/chord_transformer_crf_freeze5.pt \
  --gold-only --augment --feature-dim 60 \
  --epochs 150 --patience 40 --eval-every 5 \
  --lr 1e-4 --encoder-lr 5e-5 --weight-decay 1e-3 \
  --crf-weight 1.0 --aux-weight 0.25 --quality-weight 2.0 \
  --crf-self-bias 2.0 --emission-temp 5.0 \
  --freeze-encoder-epochs 5 \
  > logs/crf_freeze5.log 2>&1
echo "=== freeze=5 done at $(date), exit=$? ==="

echo "=== freeze=10, encoder_lr=5e-5 at $(date) ==="
PYTHONUNBUFFERED=1 /opt/miniconda3/bin/python -m v2.train_transformer_crf \
  --data data/features_v2.npz \
  --out models/chord_transformer_crf_freeze10.pt \
  --gold-only --augment --feature-dim 60 \
  --epochs 150 --patience 40 --eval-every 5 \
  --lr 1e-4 --encoder-lr 5e-5 --weight-decay 1e-3 \
  --crf-weight 1.0 --aux-weight 0.25 --quality-weight 2.0 \
  --crf-self-bias 2.0 --emission-temp 5.0 \
  --freeze-encoder-epochs 10 \
  > logs/crf_freeze10.log 2>&1
echo "=== freeze=10 done at $(date), exit=$? ==="

echo "=== All frozen warmup done ==="
