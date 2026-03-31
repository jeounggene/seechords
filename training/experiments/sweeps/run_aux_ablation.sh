#!/bin/bash
cd /Users/genej/projects/chords/seechords/training
mkdir -p logs

# Aux loss ablation: sweep aux_weight with best settings (τ=5)
# Current baseline: aux_weight=0.3
# Testing: 0, 0.1, 0.25

echo "=== aux_weight=0 at $(date) ==="
PYTHONUNBUFFERED=1 /opt/miniconda3/bin/python -m v2.train_transformer_crf \
  --data data/features_v2.npz \
  --out models/chord_transformer_crf_aux0.pt \
  --gold-only --augment --feature-dim 60 \
  --epochs 150 --patience 40 --eval-every 5 \
  --lr 1e-4 --dropout 0.2 --weight-decay 1e-3 \
  --crf-weight 1.0 --aux-weight 0.0 --quality-weight 2.0 \
  --crf-self-bias 2.0 --emission-temp 5.0 \
  > logs/crf_aux0.log 2>&1
echo "=== aux_weight=0 done at $(date), exit=$? ==="

echo "=== aux_weight=0.1 at $(date) ==="
PYTHONUNBUFFERED=1 /opt/miniconda3/bin/python -m v2.train_transformer_crf \
  --data data/features_v2.npz \
  --out models/chord_transformer_crf_aux01.pt \
  --gold-only --augment --feature-dim 60 \
  --epochs 150 --patience 40 --eval-every 5 \
  --lr 1e-4 --dropout 0.2 --weight-decay 1e-3 \
  --crf-weight 1.0 --aux-weight 0.1 --quality-weight 2.0 \
  --crf-self-bias 2.0 --emission-temp 5.0 \
  > logs/crf_aux01.log 2>&1
echo "=== aux_weight=0.1 done at $(date), exit=$? ==="

echo "=== aux_weight=0.25 at $(date) ==="
PYTHONUNBUFFERED=1 /opt/miniconda3/bin/python -m v2.train_transformer_crf \
  --data data/features_v2.npz \
  --out models/chord_transformer_crf_aux025.pt \
  --gold-only --augment --feature-dim 60 \
  --epochs 150 --patience 40 --eval-every 5 \
  --lr 1e-4 --dropout 0.2 --weight-decay 1e-3 \
  --crf-weight 1.0 --aux-weight 0.25 --quality-weight 2.0 \
  --crf-self-bias 2.0 --emission-temp 5.0 \
  > logs/crf_aux025.log 2>&1
echo "=== aux_weight=0.25 done at $(date), exit=$? ==="

echo "=== All aux ablation done ==="
