#!/bin/bash
# SeeChords Training: Quick-Start Commands
# Use any of these to run the next round of experiments

set -e  # Exit on error

cd "$(dirname "$0")/training"
export PYTHONUNBUFFERED=1
export VENV="${VENV:-$(pwd)/../.venv/bin}"

echo "SeeChords CRF Training — Flip Reduction Experiments"
echo "======================================================"
echo ""
echo "Usage: Pick an experiment below and run it."
echo "       Each takes ~30-60 min on Apple Silicon (MPS)."
echo ""

# Check environment
if [[ ! -x "$VENV/python" ]]; then
    echo "ERROR: Virtual environment not found at $VENV"
    echo "       Update VENV variable or create .venv in parent dir"
    exit 1
fi

echo "Environment: $VENV/python"
echo ""

# Show experiments
cat << 'EOF'
EXPERIMENTS:

1. HIGHEST PRIORITY — Higher Initial CRF Self-Bias (reduces flip at init)
   $VENV/python -m v2.train_transformer_crf \
     --data data/features_v2.npz \
     --out models/chord_transformer_crf_bias5.pt \
     --gold-only --augment --feature-dim 60 \
     --emission-temp 5.0 --aux-weight 0.25 \
     --freeze-encoder-epochs 5 --encoder-lr 5e-5 \
     --crf-self-bias 5.0

2. MOST LIKELY — Emission Dropout + Aux Smoothing (stabilizes learning)
   $VENV/python -m v2.train_transformer_crf \
     --data data/features_v2.npz \
     --out models/chord_transformer_crf_dropout_smooth.pt \
     --gold-only --augment --feature-dim 60 \
     --emission-temp 5.0 --aux-weight 0.25 \
     --freeze-encoder-epochs 5 --encoder-lr 5e-5 \
     --emission-dropout 0.1 --aux-label-smoothing 0.1

3. Longer Freeze + Faster CRF Learning (transitions have more time)
   $VENV/python -m v2.train_transformer_crf \
     --data data/features_v2.npz \
     --out models/chord_transformer_crf_freeze10_v2.pt \
     --gold-only --augment --feature-dim 60 \
     --emission-temp 5.0 --aux-weight 0.25 \
     --freeze-encoder-epochs 10 \
     --encoder-lr 1e-4 --crf-lr 5e-4

4. Aux Label Smoothing Only (soften target distributions)
   $VENV/python -m v2.train_transformer_crf \
     --data data/features_v2.npz \
     --out models/chord_transformer_crf_smooth01.pt \
     --gold-only --augment --feature-dim 60 \
     --emission-temp 5.0 --aux-weight 0.25 \
     --freeze-encoder-epochs 5 --encoder-lr 5e-5 \
     --aux-label-smoothing 0.1

5. CRF Self-Transition Regularization (explicit diagonal penalty)
   $VENV/python -m v2.train_transformer_crf \
     --data data/features_v2.npz \
     --out models/chord_transformer_crf_self_reg.pt \
     --gold-only --augment --feature-dim 60 \
     --emission-temp 5.0 --aux-weight 0.25 \
     --freeze-encoder-epochs 5 --encoder-lr 5e-5 \
     --crf-self-reg 0.01 --crf-self-target 2.5

AFTER TRAINING: Evaluate with Post-Hoc Viterbi
   $VENV/python -m v2.posthoc_viterbi \
     --data data/features_v2.npz \
     --checkpoints models/chord_transformer_crf_<NAME>.pt \
     --self-biases 0 2 4 6 8 10 15 20 \
     --split test

For full details, see NEXT_EXPERIMENTS.md

EOF

echo "To run an experiment, copy the command above and execute it."
echo "Or open NEXT_EXPERIMENTS.md for detailed explanations."
echo ""
