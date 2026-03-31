"""BTC 170-class chord vocabulary and mapping to SeeChords TIER1_VOCAB.

Utilities for converting between BTC model output indices and our
training pipeline's chord representations.
"""
from .chord_vocab import TIER1_VOCAB, TIER1_TO_IDX, NOTES

PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

_ENHARMONIC_TO_SEECHORDS = {
    'D#': 'Eb', 'G#': 'Ab', 'A#': 'Bb',
}

BTC_QUALITY_LIST = [
    'min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7',
    'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4',
]


def btc_idx_to_label():
    """Map BTC index (0-169) to BTC label string.

    Layout: 12 roots x 14 qualities = 168, plus X=168, N=169.
    """
    mapping = {}
    for root_idx in range(12):
        root = PITCH_CLASS[root_idx]
        root = _ENHARMONIC_TO_SEECHORDS.get(root, root)
        for qi, quality in enumerate(BTC_QUALITY_LIST):
            idx = root_idx * 14 + qi
            if quality == 'maj':
                mapping[idx] = root
            else:
                mapping[idx] = f"{root}:{quality}"
    mapping[168] = "X"
    mapping[169] = "N"
    return mapping


_BTC_IDX_TO_LABEL = btc_idx_to_label()

# Map BTC quality → TIER1 suffix (major/minor collapse)
_QUALITY_TO_TIER1_SUFFIX = {
    'maj': '', 'min': 'm',
    '7': '', 'maj7': '', 'min7': 'm',
    'dim': 'm', 'aug': '', 'dim7': 'm', 'hdim7': 'm',
    'minmaj7': 'm', 'maj6': '', 'min6': 'm',
    'sus2': '', 'sus4': '', 'aug7': '',
}


def btc_idx_to_tier1_idx(btc_idx):
    """Convert BTC class index to TIER1 class index.

    Maps 170-class BTC predictions to our 25-class vocabulary.
    """
    label = _BTC_IDX_TO_LABEL.get(btc_idx, 'N')
    if label in ('N', 'X'):
        return TIER1_TO_IDX['N']

    if ':' in label:
        root, quality = label.split(':', 1)
    else:
        root = label
        quality = 'maj'

    suffix = _QUALITY_TO_TIER1_SUFFIX.get(quality, '')
    tier1_name = root + suffix

    if tier1_name in TIER1_TO_IDX:
        return TIER1_TO_IDX[tier1_name]
    return TIER1_TO_IDX['N']


def btc_idx_to_tier1_name(btc_idx):
    """Convert BTC class index to TIER1 chord name string."""
    idx = btc_idx_to_tier1_idx(btc_idx)
    return TIER1_VOCAB[idx]


# Pre-compute full mapping table for fast vectorized lookups
BTC_TO_TIER1_TABLE = [btc_idx_to_tier1_idx(i) for i in range(170)]
