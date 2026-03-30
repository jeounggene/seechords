#!/bin/bash
cd /Users/genej/projects/chords/seechords/training
mkdir -p logs
PYTHONUNBUFFERED=1 /opt/miniconda3/bin/python -m v2.train_transformer_crf \
  --data data/features_v2.npz \
  --out models/chord_transformer_crf_t2.pt \
  --gold-only --augment --feature-dim 60 \
  --epochs 150 --patience 40 --eval-every 5 \
  --lr 1e-4 --dropout 0.2 --weight-decay 1e-3 \
  --crf-weight 1.0 --aux-weight 0.3 --quality-weight 2.0 \
  --crf-self-bias 2.0 --emission-temp 2.0 \
  2>&1 | tee logs/crf_t2.log
