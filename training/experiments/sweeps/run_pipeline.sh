#!/bin/bash
# Full training pipeline: prepare data → train model → evaluate
#
# Usage:
#   ./run_pipeline.sh                              # Isophonics Beatles mode (default)
#   ./run_pipeline.sh --flat                        # Flat directory mode
#   ./run_pipeline.sh --extra data/extra            # Beatles + extra chord-sheet data
#   ./run_pipeline.sh --silver                      # Beatles + silver (DP-aligned) data
#
# Isophonics mode expects:
#   data/beatles/annotations/      — Isophonics annotations
#   data/audio/ALBUM/SONG.mp3     — audio matching album/track structure
#
# Flat mode expects:
#   data/audio/   — audio files
#   data/labels/  — matching .lab files
#
# Extra data (from ingest_sheet.py):
#   data/extra/audio/  — symlinked audio files
#   data/extra/labels/ — generated .lab files

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${VENV_PYTHON:-../../ezchords/.venv312/bin/python}"

echo "=== SeeChords Training Pipeline ==="
echo "Python: $PYTHON"
echo ""

MODE="${1:-isophonics}"
EXTRA_DIR=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --flat|flat) MODE="flat" ;;
        --extra) EXTRA_DIR="$2"; shift ;;
        --silver) EXTRA_DIR="data/silver" ;;
    esac
    shift
done

if [ "$MODE" = "flat" ]; then
    # Flat mode
    if [ ! -d "data/audio" ] || [ ! -d "data/labels" ]; then
        echo "ERROR: data/audio/ and data/labels/ directories required for flat mode"
        exit 1
    fi
    echo "Mode: flat directories"
    PREPARE_ARGS="--audio-dir data/audio --lab-dir data/labels"
else
    # Isophonics mode
    ANNOT_DIR="data/beatles/annotations"
    if [ ! -d "$ANNOT_DIR" ]; then
        echo "ERROR: Isophonics annotations not found at $ANNOT_DIR"
        exit 1
    fi
    if [ ! -d "data/audio" ]; then
        echo "ERROR: data/audio/ directory not found."
        echo "Create it with album subdirectories matching the annotation structure:"
        echo "  data/audio/01_-_Please_Please_Me/01_-_I_Saw_Her_Standing_There.mp3"
        exit 1
    fi
    echo "Mode: Isophonics Beatles"
    PREPARE_ARGS="--isophonics \"$ANNOT_DIR\" --audio-dir data/audio"
fi

# Step 1: Extract features (main dataset)
echo "=== Step 1: Extracting features ==="
eval $PYTHON -m v1.prepare_data \
    $PREPARE_ARGS \
    --out data/features.npz \
    --tier 1

# Step 1b: Extract features from extra data and merge
if [ -n "$EXTRA_DIR" ] && [ -d "$EXTRA_DIR/audio" ] && [ -d "$EXTRA_DIR/labels" ]; then
    echo ""
    echo "=== Step 1b: Extracting extra features from $EXTRA_DIR ==="
    $PYTHON -m v1.prepare_data \
        --audio-dir "$EXTRA_DIR/audio" \
        --lab-dir "$EXTRA_DIR/labels" \
        --out data/features_extra.npz \
        --tier 1
    echo ""
    echo "Merging main + extra features..."
    $PYTHON -c "
import numpy as np
main = np.load('data/features.npz', allow_pickle=True)
extra = np.load('data/features_extra.npz', allow_pickle=True)
# Offset extra song_ids so they don't overlap
n_main_songs = len(main['filenames'])
extra_ids = extra['song_ids'] + n_main_songs
np.savez_compressed('data/features.npz',
    X=np.concatenate([main['X'], extra['X']]),
    y=np.concatenate([main['y'], extra['y']]),
    song_ids=np.concatenate([main['song_ids'], extra_ids]),
    beat_times=np.concatenate([main['beat_times'], extra['beat_times']]),
    filenames=np.concatenate([main['filenames'], extra['filenames']]),
    key_indices=np.concatenate([main['key_indices'], extra['key_indices']]),
)
print(f'Merged: {len(main[\"X\"])} + {len(extra[\"X\"])} = {len(main[\"X\"])+len(extra[\"X\"])} beats')
print(f'Songs: {n_main_songs} + {len(extra[\"filenames\"])} = {n_main_songs+len(extra[\"filenames\"])}')
"
fi
echo ""

# Step 2: Train model
echo "=== Step 2: Training model ==="
mkdir -p models
$PYTHON -m v1.train_model \
    --data data/features.npz \
    --out models/chord_model.pkl \
    --tier 1
echo ""

# Step 3: Evaluate
echo "=== Step 3: Evaluating ==="
$PYTHON -m v1.evaluate \
    --model models/chord_model.pkl \
    --data data/features.npz
echo ""

echo "=== Done! ==="
echo "Model saved to: $SCRIPT_DIR/models/chord_model.pkl"
echo "The seechords server will automatically use this model on next restart."
