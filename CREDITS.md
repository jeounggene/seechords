# Credits

## Chord recognition model — ChordMini BTC

SeeChords' automatic chord recognition is powered by the **BTC** (Bi-directional
Transformer for Chords) model from **ChordMini**. We use ChordMini's pre-trained
BTC model and its 170-class chord vocabulary; the inference code under
[`server/btc_model/`](server/btc_model/) is adapted from the ChordMini repository.

In `chord_versions.source`, analyses produced by this model are tagged
`chordmini-btc-v2.1`.

- **Repository:** ChordMini — https://github.com/ptnghia-j/ChordMini
- **License:** MIT
- **Paper:** Nghia Phan, Rong Jin, Gang Liu, and Xiao Dong.
  "Enhancing Automatic Chord Recognition via Pseudo-Labeling and Knowledge
  Distillation." arXiv:2602.19778 (2026).
  https://arxiv.org/abs/2602.19778

### Citation (BibTeX)

```bibtex
@misc{phan2026enhancingautomaticchordrecognition,
      title={Enhancing Automatic Chord Recognition via Pseudo-Labeling and Knowledge Distillation},
      author={Nghia Phan and Rong Jin and Gang Liu and Xiao Dong},
      year={2026},
      eprint={2602.19778},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2602.19778}
}
```

The BTC architecture itself originates with:

- Jonggwon Park, Kyoyun Choi, Sungwook Jeon, Dokyun Kim, and Jonghun Park.
  "A Bi-directional Transformer for Musical Chord Recognition." ISMIR 2019.
  https://arxiv.org/abs/1907.02698
