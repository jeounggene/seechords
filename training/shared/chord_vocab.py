"""Chord vocabulary and label normalization for training.

Defines the Tier 1 (25 classes: N + 12 major + 12 minor) and
Tier 2 (109 classes: Tier 1 + 7x12 extended qualities) chord vocabularies.

Core functions:
    parse_chord_label(label, tier=1)  -- Harte format -> vocabulary label
    label_to_idx(label, tier=1)       -- label string -> class index
    idx_to_label(idx, tier=1)         -- class index -> label string

Used by both v1 and v2 pipelines.
"""

NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# Enharmonic mappings (normalize everything to our canonical names)
_ENHARMONIC = {
    'Db': 'C#', 'D#': 'Eb', 'E#': 'F', 'Fb': 'E',
    'Gb': 'F#', 'G#': 'Ab', 'A#': 'Bb', 'B#': 'C', 'Cb': 'B',
}

# ── Tier 1: Basic vocabulary (start here) ────────────────────
TIER1_SUFFIXES = {
    'maj': '',    # major → ''
    'min': 'm',   # minor → 'm'
}

TIER1_VOCAB = ['N']  # no-chord
for note in NOTES:
    TIER1_VOCAB.append(note)        # C, C#, D, ... (major)
for note in NOTES:
    TIER1_VOCAB.append(note + 'm')  # Cm, C#m, Dm, ... (minor)
# 25 classes total

TIER1_TO_IDX = {name: i for i, name in enumerate(TIER1_VOCAB)}
TIER1_IDX_TO_NAME = {i: name for name, i in TIER1_TO_IDX.items()}

# ── Tier 2: Extended vocabulary (add later) ──────────────────
TIER2_EXTRA_SUFFIXES = ['7', 'maj7', 'min7', 'dim', 'aug', 'sus2', 'sus4']

TIER2_VOCAB = list(TIER1_VOCAB)
for suffix in TIER2_EXTRA_SUFFIXES:
    for note in NOTES:
        TIER2_VOCAB.append(note + suffix)
# 25 + 7*12 = 109 classes

TIER2_TO_IDX = {name: i for i, name in enumerate(TIER2_VOCAB)}


def _normalize_note(note_str):
    """Normalize note name to canonical form (e.g. Db → C#)."""
    return _ENHARMONIC.get(note_str, note_str)


def parse_chord_label(label, tier=1):
    """Parse a chord annotation label (Harte/MIREX format) into our vocabulary.

    Harte format examples:
        C:maj, C:min, C:7, C:maj7, C:min7, C:sus4, C/E, A:min/C#, N
        C:maj(*3), C:(1,3,5), etc.

    Returns the normalized chord name in our vocabulary, or 'N' if unmappable.
    """
    label = label.strip()

    # No-chord / silence
    if label in ('N', 'X', 'silence', ''):
        return 'N'

    # Strip bass note (slash chord → ignore inversion for now)
    if '/' in label:
        label = label.split('/')[0]

    # Split root:quality
    if ':' in label:
        root, quality = label.split(':', 1)
    else:
        # Could be just "C" (= C:maj) or "Cm" or "N"
        root = label
        quality = 'maj'
        # Check for common shorthand: Am, Dm, etc.
        if len(root) >= 2 and root[-1] == 'm' and root[-2] != '#' and root[-2] != 'b':
            root = root[:-1]
            quality = 'min'
        elif len(root) >= 3 and root[-1] == 'm':
            root = root[:-1]
            quality = 'min'

    root = _normalize_note(root)
    if root not in NOTES:
        return 'N'

    # Strip modifiers like (*3), (1,3,5), etc.
    if '(' in quality:
        quality = quality.split('(')[0]

    # Map quality to our suffix
    _Q_MAP_TIER1 = {
        'maj': '', 'min': 'm', 'major': '', 'minor': 'm',
        '': '', 'm': 'm',
        # Collapse extended chords to basic for tier 1
        '7': '', 'maj7': '', 'min7': 'm', 'min/b7': 'm',
        'dim': 'm', 'dim7': 'm', 'hdim7': 'm',
        'aug': '', 'sus2': '', 'sus4': '',
        '9': '', 'min9': 'm', 'maj9': '',
        '11': '', '13': '',
    }

    _Q_MAP_TIER2 = {
        'maj': '', 'min': 'm', 'major': '', 'minor': 'm',
        '': '', 'm': 'm',
        '7': '7', 'maj7': 'maj7', 'min7': 'min7', 'min/b7': 'min7',
        'dim': 'dim', 'dim7': 'dim', 'hdim7': 'dim',
        'aug': 'aug', 'sus2': 'sus2', 'sus4': 'sus4',
        '9': '7', 'min9': 'min7', 'maj9': 'maj7',
        '11': '7', '13': '7',
    }

    q_map = _Q_MAP_TIER1 if tier == 1 else _Q_MAP_TIER2
    suffix = q_map.get(quality)
    if suffix is None:
        # Try fuzzy matching
        ql = quality.lower()
        if 'min' in ql:
            suffix = 'm'
        elif 'dim' in ql:
            suffix = 'm' if tier == 1 else 'dim'
        elif 'aug' in ql:
            suffix = '' if tier == 1 else 'aug'
        else:
            suffix = ''

    chord_name = root + suffix
    vocab = TIER1_VOCAB if tier == 1 else TIER2_VOCAB
    if chord_name not in vocab:
        return 'N'
    return chord_name


def label_to_idx(label, tier=1):
    """Convert chord label string to class index."""
    mapping = TIER1_TO_IDX if tier == 1 else TIER2_TO_IDX
    return mapping.get(label, 0)  # 0 = N


def idx_to_label(idx, tier=1):
    """Convert class index to chord label string."""
    mapping = TIER1_IDX_TO_NAME if tier == 1 else {i: n for i, n in enumerate(TIER2_VOCAB)}
    return mapping.get(idx, 'N')


if __name__ == '__main__':
    print(f"Tier 1 vocabulary ({len(TIER1_VOCAB)} classes):")
    print(TIER1_VOCAB)
    print()
    # Test parsing
    tests = [
        'C:maj', 'A:min', 'N', 'G:7', 'D:maj7', 'F#:min7',
        'Db:min', 'C/E', 'A:min/C#', 'Bb:sus4', 'C:maj(*3)',
        'E:hdim7', 'silence', 'X',
    ]
    for t in tests:
        print(f"  {t:20s} → tier1: {parse_chord_label(t, tier=1):6s}  "
              f"tier2: {parse_chord_label(t, tier=2)}")
