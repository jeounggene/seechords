"""BTC (Bi-directional Transformer for Chords) inference-only package.

Adapted from ptnghia-j/ChordMini (MIT license).
Provides a pre-trained 170-class chord recognition model operating on CQT features.

Model imports are deferred to avoid importing torch at package level.
Use: from btc_model.btc_model import BTC_model
     from btc_model.vocab import idx2voca_chord, btc_idx_to_tier1
"""
