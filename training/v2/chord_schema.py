"""V2 chord schema: factorized root + quality prediction.

Factorizes chord labels into two independent dimensions:
    Root (13 classes):    N, C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B
    Quality (7 classes):  N, maj, min, dom7, maj7, min7, other

Composition rules:
    root=N or quality=N  ->  N (no chord)
    root + quality        ->  chord label  (e.g. A + min -> Am)
    quality=other + low confidence  ->  simplify to parent triad

Key functions:
    parse_chord_v2(label)      -- Harte-format label -> (root_idx, quality_idx)
    compose_chord(r, q)        -- indices -> label string
    v2_to_tier1_idx(r, q)      -- v2 indices -> Tier 1 class index (0-24)
    simplify_chord(r, q, ...)  -- confidence-based simplification
"""""
import numpy as np

# ── Root vocabulary ──────────────────────────────────────────
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

ROOT_VOCAB = ['N'] + list(NOTES)  # 13 classes
ROOT_TO_IDX = {r: i for i, r in enumerate(ROOT_VOCAB)}
IDX_TO_ROOT = {i: r for r, i in ROOT_TO_IDX.items()}

# ── Quality vocabulary ───────────────────────────────────────
QUALITY_VOCAB = ['N', 'maj', 'min', 'dom7', 'maj7', 'min7', 'other']
QUALITY_TO_IDX = {q: i for i, q in enumerate(QUALITY_VOCAB)}
IDX_TO_QUALITY = {i: q for q, i in QUALITY_TO_IDX.items()}

# ── Collapsed 3-class quality vocabulary ─────────────────────
QUALITY3_VOCAB = ['N', 'maj', 'min']
QUALITY3_TO_IDX = {q: i for i, q in enumerate(QUALITY3_VOCAB)}
# Map 7-class quality index → 3-class index
QUALITY7_TO_3 = [0, 1, 2, 1, 1, 2, 1]  # N, maj, min, dom7→maj, maj7→maj, min7→min, other→maj

# ── Enharmonic normalization ────────────────────────────────
_ENHARMONIC = {
    'Db': 'C#', 'D#': 'Eb', 'E#': 'F', 'Fb': 'E',
    'Gb': 'F#', 'G#': 'Ab', 'A#': 'Bb', 'B#': 'C', 'Cb': 'B',
}


def _normalize_note(note_str):
    return _ENHARMONIC.get(note_str, note_str)


# ── Quality mapping from Harte annotation format ────────────
_QUALITY_MAP = {
    'maj': 'maj', 'min': 'min', 'major': 'maj', 'minor': 'min',
    '': 'maj', 'm': 'min',
    '7': 'dom7', 'maj7': 'maj7', 'min7': 'min7', 'min/b7': 'min7',
    'dim': 'other', 'dim7': 'other', 'hdim7': 'other',
    'aug': 'other', 'sus2': 'other', 'sus4': 'other',
    '9': 'dom7', 'min9': 'min7', 'maj9': 'maj7',
    '11': 'dom7', '13': 'dom7',
}

# Quality → parent family for simplification
QUALITY_PARENT = {
    'N': 'N',
    'maj': 'maj',
    'min': 'min',
    'dom7': 'maj',     # G7 simplifies to G
    'maj7': 'maj',     # Cmaj7 simplifies to C
    'min7': 'min',     # Am7 simplifies to Am
    'other': 'maj',    # default simplification
}

# Quality → display suffix
QUALITY_SUFFIX = {
    'N': '',
    'maj': '',
    'min': 'm',
    'dom7': '7',
    'maj7': 'maj7',
    'min7': 'm7',
    'other': '',
}

# Simplified suffix (parent family)
QUALITY_SIMPLE_SUFFIX = {
    'N': '',
    'maj': '',
    'min': 'm',
    'dom7': '',
    'maj7': '',
    'min7': 'm',
    'other': '',
}


def parse_chord_v2(label):
    """Parse a Harte-format chord label into (root_idx, quality_idx).

    Returns:
        (root_idx, quality_idx) — indices into ROOT_VOCAB and QUALITY_VOCAB
    """
    label = label.strip()

    if label in ('N', 'X', 'silence', ''):
        return 0, 0  # N, N

    # Strip bass note
    if '/' in label:
        label = label.split('/')[0]

    # Split root:quality
    if ':' in label:
        root, quality = label.split(':', 1)
    else:
        root = label
        quality = 'maj'
        if len(root) >= 2 and root[-1] == 'm' and root[-2] not in ('#', 'b'):
            root = root[:-1]
            quality = 'min'
        elif len(root) >= 3 and root[-1] == 'm':
            root = root[:-1]
            quality = 'min'

    root = _normalize_note(root)
    if root not in NOTES:
        return 0, 0

    # Strip modifiers like (*3)
    if '(' in quality:
        quality = quality.split('(')[0]

    q = _QUALITY_MAP.get(quality)
    if q is None:
        ql = quality.lower()
        if 'min' in ql:
            q = 'min'
        elif 'dim' in ql or 'aug' in ql or 'sus' in ql:
            q = 'other'
        else:
            q = 'maj'

    return ROOT_TO_IDX[root], QUALITY_TO_IDX[q]


def compose_chord(root_idx, quality_idx, simplified=False):
    """Compose a chord label from root and quality indices.

    Args:
        root_idx: index into ROOT_VOCAB
        quality_idx: index into QUALITY_VOCAB
        simplified: if True, collapse extended qualities to parent triad

    Returns:
        chord label string (e.g. 'Am', 'G7', 'Bb')
    """
    root = IDX_TO_ROOT.get(root_idx, 'N')
    quality = IDX_TO_QUALITY.get(quality_idx, 'N')

    if root == 'N' or quality == 'N':
        return 'N'

    if simplified:
        suffix = QUALITY_SIMPLE_SUFFIX[quality]
    else:
        suffix = QUALITY_SUFFIX[quality]

    return root + suffix


def v2_to_tier1_idx(root_idx, quality_idx):
    """Convert v2 root+quality to Tier 1 class index (for comparison).

    Tier 1: N=0, C=1...B=12, Cm=13...Bm=24
    """
    if root_idx == 0 or quality_idx == 0:
        return 0  # N

    # Root 1-12 maps to note index 0-11
    note_idx = root_idx - 1

    parent = QUALITY_PARENT.get(IDX_TO_QUALITY[quality_idx], 'maj')
    if parent == 'min':
        return 13 + note_idx  # minor: 13-24
    else:
        return 1 + note_idx   # major: 1-12


def tier1_to_v2(tier1_idx):
    """Convert Tier 1 class index to v2 (root_idx, quality_idx)."""
    if tier1_idx == 0:
        return 0, 0
    elif 1 <= tier1_idx <= 12:
        return tier1_idx, QUALITY_TO_IDX['maj']
    elif 13 <= tier1_idx <= 24:
        return tier1_idx - 12, QUALITY_TO_IDX['min']
    return 0, 0


# ── Confidence-based simplification ─────────────────────────

def simplify_chord(root_idx, quality_idx, root_conf, quality_conf,
                   root_threshold=0.55, quality_threshold=0.45):
    """Simplify chord based on confidence.

    If root confidence is high but quality confidence is low,
    simplify to parent triad.

    Returns:
        (chord_label, is_simplified)
    """
    if root_idx == 0 or quality_idx == 0:
        return 'N', False

    quality = IDX_TO_QUALITY[quality_idx]

    if root_conf >= root_threshold and quality_conf < quality_threshold:
        # Low quality confidence → simplify to parent
        return compose_chord(root_idx, quality_idx, simplified=True), True
    else:
        return compose_chord(root_idx, quality_idx, simplified=False), False


if __name__ == '__main__':
    print(f"Root vocab ({len(ROOT_VOCAB)}): {ROOT_VOCAB}")
    print(f"Quality vocab ({len(QUALITY_VOCAB)}): {QUALITY_VOCAB}")
    print()

    tests = [
        'C:maj', 'A:min', 'N', 'G:7', 'D:maj7', 'F#:min7',
        'Db:min', 'C/E', 'Bb:sus4', 'C:dim', 'E:aug',
        'silence', 'Ab:hdim7', 'C:9',
    ]
    for t in tests:
        ri, qi = parse_chord_v2(t)
        full = compose_chord(ri, qi)
        simp = compose_chord(ri, qi, simplified=True)
        t1 = v2_to_tier1_idx(ri, qi)
        print(f"  {t:20s} → root={ROOT_VOCAB[ri]:3s} qual={QUALITY_VOCAB[qi]:5s} "
              f"→ {full:6s} (simplified: {simp:4s}, tier1_idx={t1})")
