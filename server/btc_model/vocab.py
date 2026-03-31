"""BTC 170-class chord vocabulary and mapping to SeeChords display names.

Adapted from ptnghia-j/ChordMini (MIT license).
"""

PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

PREFERRED_SPELLING = {
    'D#': 'Eb', 'G#': 'Ab', 'A#': 'Bb',
}

QUALITY_LIST = [
    'min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7',
    'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4',
]


def idx2voca_chord():
    """Map chord index (0-169) to chord label string for BTC large vocabulary.

    Layout: 12 roots x 14 qualities = 168, plus X=168, N=169.
    """
    mapping = {}
    for root_idx in range(12):
        root = PITCH_CLASS[root_idx]
        root = PREFERRED_SPELLING.get(root, root)
        for qi, quality in enumerate(QUALITY_LIST):
            idx = root_idx * 14 + qi
            if quality == 'maj':
                mapping[idx] = root
            else:
                mapping[idx] = f"{root}:{quality}"
    mapping[168] = "X"
    mapping[169] = "N"
    return mapping


_BTC_VOCAB = idx2voca_chord()

_ENHARMONIC = {
    'D#': 'Eb', 'G#': 'Ab', 'A#': 'Bb',
}

_SEECHORDS_NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

_QUALITY_TO_DISPLAY = {
    'maj': '',
    'min': 'm',
    '7': '7',
    'maj7': 'maj7',
    'min7': 'm7',
    'dim': 'dim',
    'aug': 'aug',
    'sus2': 'sus2',
    'sus4': 'sus4',
    'dim7': 'dim',
    'hdim7': 'm7',
    'minmaj7': 'm',
    'maj6': '',
    'min6': 'm',
    'aug7': '7',
}

_QUALITY_TO_TIER1 = {
    'maj': '', 'min': 'm',
    '7': '', 'maj7': '', 'min7': 'm',
    'dim': 'm', 'aug': '', 'dim7': 'm', 'hdim7': 'm',
    'minmaj7': 'm', 'maj6': '', 'min6': 'm',
    'sus2': '', 'sus4': '', 'aug7': '',
}


def _normalize_root(root):
    """Normalize BTC root spelling to SeeChords convention."""
    return _ENHARMONIC.get(root, root)


def btc_label_to_display(label):
    """Convert BTC label like 'C#:min7' to display name like 'C#m7'."""
    if label in ('N', 'X'):
        return 'N'
    if ':' in label:
        root, quality = label.split(':', 1)
    else:
        root = label
        quality = 'maj'
    root = _normalize_root(root)
    suffix = _QUALITY_TO_DISPLAY.get(quality, '')
    return root + suffix


def btc_label_to_tier1(label):
    """Convert BTC label to tier-1 (25-class) name like 'C#m'."""
    if label in ('N', 'X'):
        return 'N'
    if ':' in label:
        root, quality = label.split(':', 1)
    else:
        root = label
        quality = 'maj'
    root = _normalize_root(root)
    suffix = _QUALITY_TO_TIER1.get(quality, '')
    return root + suffix


def btc_idx_to_display(idx):
    """Convert BTC class index to display chord name."""
    label = _BTC_VOCAB.get(idx, 'N')
    return btc_label_to_display(label)


def btc_idx_to_tier1(idx):
    """Convert BTC class index to tier-1 chord name."""
    label = _BTC_VOCAB.get(idx, 'N')
    return btc_label_to_tier1(label)
