# SeeChords Architecture

## Overview

SeeChords is a Chrome extension that detects and displays chord progressions for YouTube videos. It consists of three components:

1. **Chrome Extension** (`extension/`) -- injects a chord overlay on YouTube pages
2. **Flask Server** (`server/`) -- analyzes audio and serves cached chord data via REST API
3. **Training Pipeline** (`training/`) -- trains and evaluates chord recognition models

The server runs on [Fly.io](https://fly.io) with a single 2GB/1CPU machine that scales to zero.

## Inference Pipeline

```
Audio File
    |
    +---> Beat This! transformer (ISMIR 2024, "small0" ~8MB)
    |         |
    |         +--> beat times + downbeats
    |         |
    |     [fallback: Essentia RhythmExtractor2013]
    |
    +---> librosa CQT (144 bins, 22050 Hz, hop=2048)
    |         |
    |         +--> BTC transformer (170-class, ~34MB checkpoint)
    |         |        frame-level preds -> majority vote at beats
    |         |        -> map to display chord names
    |         |
    |     [fallback: Essentia HPCP -> freeze5 Transformer+CRF (25-class)]
    |     [fallback: hand-crafted template HMM]
    |
    +--> merged chord segments with beat times, BPM, key
```

The cascade is controlled by environment variables: `USE_BTC`, `USE_FREEZE5`, `USE_BEAT_THIS`.
First successful model wins. Template HMM is the last resort (no external files needed).

## Active Components

### Server (`server/`)

| File | Purpose |
|------|---------|
| `app.py` | Flask application: REST API, site pages, ingest workflow |
| `analyze_chords.py` | Core analysis: beat detection, chord recognition, post-processing |
| `btc_model/` | BTC transformer architecture (inference-only) and vocabulary mapping |
| `requirements.txt` | Python dependencies |
| `templates/site_home.html` | Landing page |
| `templates/ingest.html` | Ingest/annotation UI |
| `verified/` | Gold-standard chord annotations and video metadata |

### Extension (`extension/`)

| File | Purpose |
|------|---------|
| `manifest.json` | Chrome extension manifest (MV3) |
| `background.js` | Service worker: API client, upload/analyze orchestration |
| `content.js` | YouTube overlay: chord display, beat sync, transpose, diagrams |
| `overlay.css` | Overlay styling |
| `popup.html` / `popup.js` | Extension popup UI |
| `privacy-policy.html` | Privacy policy |

### Training (`training/`)

**Active pipeline (v2 Transformer+CRF):**

| File | Purpose |
|------|---------|
| `v2/train_transformer_crf.py` | Main training script |
| `v2/prepare_data.py` | Feature extraction (HPCP, CQT, pseudo-labels) |
| `v2/transformer_model.py` | `ChordTransformerCRF` model definition |
| `v2/transformer_data.py` | Dataset loading and batching |
| `v2/crf.py` | CRF layer implementation |
| `v2/decode.py` | Viterbi decoding and post-processing |
| `v2/eval_btc_vs_freeze5.py` | BTC vs freeze5 evaluation harness |
| `v2/eval_beats.py` | Beat detection evaluation (Beat This! vs Essentia) |
| `v2/generate_pseudo_labels.py` | BTC teacher -> pseudo-label .lab files |

**Shared utilities:**

| File | Purpose |
|------|---------|
| `shared/chord_vocab.py` | TIER1 (25-class) vocabulary, parsing, index mapping |
| `shared/btc_vocab.py` | BTC 170-class -> TIER1 mapping |
| `v1/prepare_data.py` | Data utilities: `parse_lab_file`, `detect_beats`, `find_isophonics_songs`, `extract_hpcp` |
| `tools/align_chords.py` | DP chord alignment (used by server `/api/align`) |
| `tools/lyric_align.py` | Whisper-based lyric alignment |
| `tools/download_*.py` | Dataset download scripts |

**Experiments (archived):**

| Directory | Contents |
|-----------|----------|
| `experiments/v1_legacy/` | Original RF training (`train_model.py`, `evaluate.py`) |
| `experiments/v2_research/` | Research scripts: emission analysis, calibration, older transformer |
| `experiments/sweeps/` | Hyperparameter sweep shell scripts |
| `experiments/` | One-off test scripts |

## Environment Variables

### Model selection (set in Dockerfile)

| Variable | Default | Purpose |
|----------|---------|---------|
| `USE_BTC` | `1` | Enable BTC chord model (primary) |
| `USE_FREEZE5` | `1` | Enable freeze5 Transformer+CRF (fallback) |
| `USE_BEAT_THIS` | `1` | Enable Beat This! beat tracker |

### Checkpoint overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `BTC_CHECKPOINT` | `<server>/btc_model/btc_model_best.pth` | Path to BTC weights |
| `FREEZE5_CHECKPOINT` | `<models>/chord_transformer_crf_freeze5.pt` | Path to freeze5 weights |
| `SEECHORDS_MODEL_DIR` | `<training>/models` | Base directory for model checkpoints |

### Server configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DB_PATH` | `/data/seechords.db` | SQLite database path |
| `UPLOAD_DIR` | `/data/uploads` | Temporary audio upload directory |
| `FLASK_ENV` | `production` | Flask environment |
| `SEECHORDS_CHROME_STORE_URL` | (hardcoded) | Chrome Web Store link for landing page |
| `SEECHORDS_DONATION_URL` | (none) | Optional tip jar URL |

### Tuning

| Variable | Default | Purpose |
|----------|---------|---------|
| `BEAT_OFFSET_MS` | `50` | Beat offset in ms (only applies to Essentia beats) |
| `SILENCE_GATE` | `1` | Gate leading silence beats to N |
| `SILENCE_GATE_FRAC` | `0.08` | RMS fraction threshold for silence gate |

## Deployment

### Prerequisites

- [Fly CLI](https://fly.io/docs/hands-on/install-flyctl/) installed
- Fly.io account with `seechords` app and `seechords_data` volume in `sjc` region

### Build and deploy

```bash
cd server
fly deploy
```

This uses `fly.toml` which references `../Dockerfile` with the repo root as build context.

The Docker build:
1. Installs Python deps + PyTorch (CPU) + beat_this
2. Copies server code, training v2/shared modules, freeze5 checkpoint
3. Downloads BTC checkpoint from GitHub at build time
4. Pre-caches Beat This! small0 checkpoint at build time
5. Runs gunicorn on port 8080 with 600s timeout

### Memory budget (2GB VM)

| Component | Estimate |
|-----------|----------|
| PyTorch | ~200MB |
| Essentia | ~100MB |
| librosa + CQT | ~50MB |
| BTC model | ~6MB |
| Beat This! small0 | ~8MB |
| Peak working memory | ~50MB |
| **Total** | **~414MB** |

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Landing page |
| `GET` | `/privacy` | Privacy policy |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/chords/<video_id>` | Cached chord data |
| `POST` | `/api/analyze` | Upload audio for analysis |
| `POST` | `/api/analyze-youtube` | Analyze YouTube video by URL |
| `GET` | `/api/status/<job_id>` | Analysis job status |
| `POST` | `/api/compare` | Compare predictions vs reference |
| `POST` | `/api/align` | DP alignment of chord sheet |
| `GET` | `/ingest` | Annotation ingest UI |

## Evaluation Results (March 2026)

### Chord accuracy (200 songs: 180 Beatles + 20 Queen)

| Model | WCSR | Major | Minor | Root |
|-------|------|-------|-------|------|
| **BTC** | **0.839** | 0.854 | 0.587 | 0.858 |
| Freeze5 | 0.453 | 0.501 | 0.191 | 0.497 |

### Beat detection (179 Beatles songs, gold beats, 70ms tolerance)

| Tracker | Mean F1 | Songs F1 >= 0.9 |
|---------|---------|-----------------|
| **Beat This! (small0)** | **0.533** | **50/179** |
| Essentia | 0.456 | 30/179 |

Downstream chord accuracy (BTC model):
- Gold beats: 0.864 WCSR
- Beat This! beats: 0.865 WCSR
- Essentia beats: 0.851 WCSR
