# SeeChords Training Pipeline

Train chord recognition models from annotated audio data.

## Directory Structure

```
training/
├── shared/                    # Shared utilities
│   └── chord_vocab.py         #   Tier-1/2 chord label vocabulary & normalization
├── v1/                        # V1 pipeline: 25-class flat classifier + Viterbi
│   ├── prepare_data.py        #   Extract beat-sync 12-dim HPCP features
│   ├── train_model.py         #   Train Random Forest + learn HMM transitions
│   └── evaluate.py            #   Evaluate with Viterbi decoding, WCSR metrics
├── v2/                        # V2 pipeline: factorized root+quality + combined Viterbi
│   ├── chord_schema.py        #   Root (13) + quality (7) vocabulary & parsing
│   ├── prepare_data.py        #   Extract 36-dim contextual HPCP + factorized labels
│   ├── train.py               #   Train root, quality, and tier-1 classifiers
│   ├── decode.py              #   Viterbi: direct, factorized, and hybrid decoders
│   └── evaluate.py            #   Evaluate all 3 decode modes, per-slice metrics
├── tools/                     # Diagnostic & utility scripts
│   ├── download_beatles.py    #   Download Beatles audio from YouTube
│   ├── download_queen.py      #   Download Queen audio from YouTube
│   ├── download_billboard.py   #   Download McGill Billboard annotations + audio
│   ├── download_songs.py      #   Download songs from YouTube (playlist/single)
│   ├── align_chords.py        #   DP alignment of chord sheets to beat predictions
│   ├── compare_chords.py      #   Compare predictions vs reference chord sheet
│   ├── diagnose_song.py       #   Per-beat emission/Viterbi diagnosis
│   ├── ingest_sheet.py        #   Convert chord sheet + audio to .lab for training
│   ├── inspect_model.py       #   Dump model vocab, transitions, test predictions
│   ├── make_annotations.py    #   Create .lab files from chord sheets + beat detection
│   └── test_keys.py           #   Test classifier across 12 key hypotheses
├── data/
│   ├── weights.json           #   Dataset weights config
│   ├── beatles/               #   Beatles training data
│   │   ├── annotations/       #     Isophonics annotations (chordlab, beat, keylab, etc.)
│   │   └── audio/             #     Album subdirs with .mp3 files
│   ├── queen/                 #   Queen training data
│   │   ├── annotations/       #     Isophonics annotations (chordlab, keylab, etc.)
│   │   └── audio/             #     Album subdirs with .mp3 files
│   ├── verified/              #   Human-verified data
│   │   ├── audio/             #     .wav files
│   │   ├── labels/            #     .lab chord annotations
│   │   └── video_map.json     #     YouTube video metadata
│   ├── billboard/             #   McGill Billboard dataset (890 songs)
│   │   ├── audio/             #     .wav files (downloaded from YouTube)
│   │   ├── labels/            #     .lab chord annotations (from McGill)
│   │   ├── metadata.csv       #     Song index (artist, title, chart date)
│   │   └── video_map.json     #     YouTube video metadata
│   └── features_v2.npz        #   Extracted features
├── models/
│   └── chord_model_v2.pkl     #   Trained model (production)
└── README.md
```

## Training Pipeline

Factorized chord prediction: root (13 classes) and quality
(7 classes: N, maj, min, dom7, maj7, min7, other), with 36-dim contextual
features (previous + current + next beat HPCP) and support for gold/silver
sample weighting.

Three decoding strategies are available:

| Decoder | Description | Default? |
|---------|-------------|----------|
| **hybrid** | Factorized root + direct quality | **Yes** (production) |
| direct | Flat 25-class Tier-1 Viterbi | Safe fallback |
| factorized | Root × Quality → Tier-1 Viterbi | Experimental |

The **hybrid** decoder gets the best of both worlds:
- **Root** from the factorized classifier (best root accuracy, lowest flip rate)
- **Quality** (major/minor) from the direct 25-class classifier (preserves minors)

The environment variable `CHORD_DECODE_MODE` can override the default:
```bash
CHORD_DECODE_MODE=direct python app.py     # Use direct decoding
CHORD_DECODE_MODE=factorized python app.py # Use factorized (experimental)
```

### Step 1: Prepare features

```bash
cd training

# Using weights.json config (recommended — auto-discovers all datasets)
python -m v2.prepare_data --config data/weights.json --out data/features_v2.npz

# Or manually specify sources (each --isophonics pairs with an --audio-dir):
python -m v2.prepare_data \
    --isophonics "data/beatles/annotations" --audio-dir data/beatles/audio --weight 1.0 \
    --isophonics "data/queen/annotations"       --audio-dir data/queen/audio   --weight 0.8 \
    --out data/features_v2.npz
```

### Step 2: Train

```bash
# Default: 36-dim context features with sample weighting
python -m v2.train --data data/features_v2.npz --out models/chord_model_v2.pkl

# Ablation: 12-dim features (v1 baseline with v2 classifiers)
python -m v2.train --data data/features_v2.npz --feature-dim 12 --out models/chord_model_v2_12d.pkl

# Ablation: no sample weighting
python -m v2.train --data data/features_v2.npz --no-weights --out models/chord_model_v2_nowt.pkl
```

### Step 3: Evaluate

```bash
python -m v2.evaluate --model models/chord_model_v2.pkl --data data/features_v2.npz
```

Reports direct vs factorized vs hybrid accuracy, WCSR, major/minor/N breakdown,
gold vs silver slices, UX metrics, per-song results, and confusion matrices.

## Experiment Results

### Training accuracy (validation set, no Viterbi)

| # | Features | Weights | Tier-1 Val | Root Val | Quality Val |
|---|----------|---------|------------|----------|-------------|
| 1 | 12-dim   | No      | 0.508      | 0.568    | 0.695       |
| 2 | 36-dim   | Yes     | 0.580      | 0.636    | 0.703       |
| 4 | 36-dim   | No      | 0.582      | 0.638    | 0.703       |
| 5 | 12-dim   | Yes     | 0.504      | 0.571    | 0.698       |

### Viterbi evaluation (200 songs, 62,687 beats, full dataset)

| # | Features | Weights | Direct | Factorized | **Hybrid** | Hybrid Maj | Hybrid Min | Flip Rate |
|---|----------|---------|--------|------------|-----------|------------|------------|-----------|
| 1 | 12-dim   | No      | 0.757  | 0.829      | **0.837** | 0.866      | 0.721      | ≈0.03     |
| 2 | 36-dim   | Yes     | 0.752  | 0.825      | **0.830** | 0.862      | 0.710      | ≈0.03     |
| 4 | 36-dim   | No      | 0.752  | 0.830      | **0.835** | 0.864      | 0.720      | ≈0.03     |
| 5 | 12-dim   | Yes     | 0.758  | 0.826      | **0.832** | 0.863      | 0.717      | ≈0.03     |

**Winner**: Exp 1 (12-dim, no weights) with hybrid decode → **WCSR 0.837**

### Key findings

1. **Hybrid beats everything**: Combines factorized root accuracy (0.886) with
   direct quality discrimination (minor recall 0.72 vs factorized's 0.56).

2. **v2 massively beats v1**: 0.837 hybrid vs 0.683 v1 = +22.5% relative improvement.
   Driven by better RF hyperparams (300 trees, min_samples_leaf=2) and Viterbi tuning.

3. **Feature dimension barely matters**: 12-dim vs 36-dim is within noise after Viterbi.
   Context features help training accuracy but Viterbi absorbs the temporal info.

4. **Sample weighting doesn't help much**: Weighted vs unweighted ~1% difference.

5. **Quality classifier is the bottleneck**: 70% accuracy but almost entirely from
   predicting majority class (maj). Minor recall ≈5%, dom7/maj7/min7/other ≈0%.
   The 4:1 major:minor imbalance and shared HPCP pitch classes (C major = C,E,G
   vs A minor = A,C,E) make quality inherently hard with chroma alone.

6. **Factorized has a bad failure mode**: Am→A (32.7%), Em→E (26%). Too damaging
   for user trust. Hybrid fixes this by taking quality from direct classifier.

### Canonical model

`models/chord_model_v2.pkl` = Exp 1 (12-dim, no weights, hybrid WCSR 0.837)

Chosen because:
- Highest hybrid WCSR (0.837)
- Simplest features (12-dim, no context construction needed at inference)
- No weight tuning dependency

## Quality Head Improvement Plan

The quality classifier is the single biggest bottleneck. Current state:

| Quality | % of data | Recall |
|---------|-----------|--------|
| maj     | 66.2%     | ~99%   |
| min     | 16.9%     | ~5%    |
| dom7    | 8.6%      | ~0%    |
| N       | 2.7%      | varies |
| min7    | 2.6%      | ~0%    |
| other   | 2.1%      | ~0%    |
| maj7    | 1.0%      | ~0%    |

### Why it fails

1. **Class imbalance**: 4:1 major:minor ratio means predicting "maj" always
   gets 66% accuracy "for free"

2. **Feature overlap**: HPCP chroma can't distinguish relative major/minor pairs.
   C major (C,E,G) and A minor (A,C,E) share 2 of 3 pitch classes.
   The distinguishing note (G vs A) is often weak in the mix.

3. **Extended qualities are too rare**: dom7/maj7/min7/other are 2-9% each.
   Not enough training signal for the classifier to learn them.

### Planned approaches (in priority order)

1. **Binary quality head**: Collapse to 3 classes (N, major-family, minor-family).
   This is the actual decision boundary that matters. dom7→major, min7→minor.
   Should dramatically improve minor recall.

2. **Harmonic difference features**: Instead of raw HPCP, compute features that
   emphasize the major/minor distinction:
   - Third interval energy ratio: `HPCP[root+4] / HPCP[root+3]` (major 3rd vs minor 3rd)
   - This requires knowing the root first → two-stage: root → quality with root-relative features

3. **Root-relative chroma rotation**: Rotate HPCP so root = bin 0 before quality
   classification. This normalizes across keys and makes quality patterns uniform.
   E.g., all major chords become [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] regardless of key.

4. **Ensemble with template matching**: Use HPCP templates (major vs minor triad
   profiles) as features alongside classifier output. Template cosine similarity
   to `[1,0,0,0,1,0,0,1,0,0,0,0]` (major) vs `[1,0,0,1,0,0,0,1,0,0,0,0]` (minor)
   gives a direct discriminative signal.

5. **SMOTE or focal loss**: Oversample minority classes or use focal loss to
   focus gradient on hard examples. May help with extended qualities.

## Vocabulary

### V1 (Tier 1) — 25 classes
- N (no chord)
- 12 major: C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B
- 12 minor: Cm, C#m, Dm, Ebm, Em, Fm, F#m, Gm, Abm, Am, Bbm, Bm

### V2 — Factorized
- **Root** (13): N, C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B
- **Quality** (7): N, maj, min, dom7, maj7, min7, other

Extended chord types are mapped: dom7→dom7, maj7→maj7, min7→min7,
dim/aug/sus→other. Confidence-based simplification can collapse uncertain
extended qualities to their parent triad (dom7→maj, min7→min).

## Training Data

| Dataset | Songs | Beats | Gold/Silver |
|---------|-------|-------|-------------|
| Beatles (Isophonics) | 180 | ~54K | Gold (beat + key annotations) |
| Queen (Isophonics) | 20 | ~8K | Silver (key annotations only, auto beats) |
| **Total** | **200** | **~62K** | |

## Tools

### Download audio
```bash
python tools/download_beatles.py --missing          # List songs without audio
python tools/download_beatles.py --playlist --album "01_-_Please_Please_Me" URL
python tools/download_queen.py --missing
python tools/download_queen.py --playlist --album "Greatest Hits I" URL
```

### Compare with chord sheets
```bash
python tools/compare_chords.py audio.mp3 chords.txt     # Compare predictions vs sheet
python tools/compare_chords.py audio.mp3 --paste         # Paste chords interactively
```

### Diagnose specific songs
```bash
python tools/diagnose_song.py audio.mp3       # Per-beat emission analysis
python tools/test_keys.py audio.mp3           # Test all 12 key hypotheses
python tools/inspect_model.py                 # Dump model internals
```

### Create training annotations
```bash
python tools/make_annotations.py --audio song.mp3 --chords chords.txt --out song.lab
python tools/ingest_sheet.py audio.mp3 chords.txt --out-dir data/silver
```

## Deploy

The server auto-loads the best available model:
1. `training/models/chord_model_v2.pkl` (v2, hybrid decode — default)
2. `training/models/chord_model.pkl` (v1 fallback)

```bash
cd ../server && python app.py
```

Override decode mode with environment variable:
```bash
CHORD_DECODE_MODE=direct python app.py       # Safe fallback
CHORD_DECODE_MODE=factorized python app.py   # Experimental
CHORD_DECODE_MODE=hybrid python app.py       # Default (best)
```
