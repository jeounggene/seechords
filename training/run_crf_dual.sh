#!/bin/bash
cd /Users/genej/projects/chords/seechords/training
mkdir -p logs

echo "=== Starting dual-LR (crf-lr=1e-3, tau=5) at $(date) ==="
PYTHONUNBUFFERED=1 /opt/miniconda3/bin/python -m v2.train_transformer_crf \
  --data data/features_v2.npz \
  --out models/chord_transformer_crf_dual.pt \
  --gold-only --augment --feature-dim 60 \
  --epochs 150 --patience 40 --eval-every 5 \
  --lr 1e-4 --crf-lr 1e-3 --dropout 0.2 --weight-decay 1e-3 \
  --crf-weight 1.0 --aux-weight 0.3 --quality-weight 2.0 \
  --crf-self-bias 2.0 --emission-temp 5.0 \
  > logs/crf_dual.log 2>&1
echo "=== dual-LR done at $(date), exit=$? ==="

echo "=== Starting dual-LR (crf-lr=1e-2, tau=5) at $(date) ==="
PYTHONUNBUFFERED=1 /opt/miniconda3/bin/python -m v2.train_transformer_crf \
  --data data/features_v2.npz \
  --out models/chord_transformer_crf_dual2.pt \
  --gold-only --augment --feature-dim 60 \
  --epochs 150 --patience 40 --eval-every 5 \
  --lr 1e-4 --crf-lr 1e-2 --dropout 0.2 --weight-decay 1e-3 \
  --crf-weight 1.0 --aux-weight 0.3 --quality-weight 2.0 \
  --crf-self-bias 2.0 --emission-temp 5.0 \
  > logs/crf_dual2.log 2>&1
echo "=== dual-LR (1e-2) done at $(date), exit=$? ==="

echo "=== All done ==="
