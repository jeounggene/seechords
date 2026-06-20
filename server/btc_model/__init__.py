"""ChordMini BTC (Bi-directional Transformer for Chords) inference-only package.

Runs ChordMini's pre-trained BTC model — a 170-class chord recognition model
operating on CQT features. Model, weights, and vocabulary are from:

  ChordMini — https://github.com/ptnghia-j/ChordMini (MIT license)
  Phan, Jin, Liu & Dong (2026), "Enhancing Automatic Chord Recognition via
  Pseudo-Labeling and Knowledge Distillation", arXiv:2602.19778.

See CREDITS.md for the full citation.

Model imports are deferred to avoid importing torch at package level.
Use: from btc_model.btc_model import BTC_model
     from btc_model.vocab import idx2voca_chord, btc_idx_to_tier1
"""
