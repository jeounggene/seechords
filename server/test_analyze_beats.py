"""Unit tests for beat-time handling in analyze_chords.

Run: python3 server/test_analyze_beats.py  (stdlib unittest)

Guards the timing-drift fix: the chord/beat timeline must preserve the detector's
NATIVE (irregular) beat timestamps by default. The old constant-tempo "uniform grid"
discarded real timing and drifted from the audio by ~a full beat late in a song.
"""
import os
import sys
import statistics
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_chords  # noqa: E402

# A realistic, slightly-irregular beat track (~130 BPM, real performances breathe).
NATIVE_BEATS = [0.50, 0.96, 1.43, 1.89, 2.34, 2.81, 3.29, 3.74, 4.18, 4.65]
BPM = 130.0


class UniformizeBeatTimesTests(unittest.TestCase):
    def _call(self, beats, bpm, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=False):
            if not env or 'UNIFORM_BEAT_GRID' not in env:
                os.environ.pop('UNIFORM_BEAT_GRID', None)
            return analyze_chords._uniformize_beat_times(beats, bpm)

    def test_native_times_preserved_by_default(self):
        """Default (no env): return detector-native times UNCHANGED — this is the fix."""
        out = self._call(NATIVE_BEATS, BPM)
        self.assertEqual(out, NATIVE_BEATS)
        # Real beats are irregular; a nonzero spread proves we didn't flatten them.
        diffs = [out[i + 1] - out[i] for i in range(len(out) - 1)]
        self.assertGreater(statistics.pstdev(diffs), 1e-4,
                           'native beat intervals must remain irregular')

    def test_uniform_grid_only_when_opted_in(self):
        out = self._call(NATIVE_BEATS, BPM, env={'UNIFORM_BEAT_GRID': '1'})
        diffs = [out[i + 1] - out[i] for i in range(len(out) - 1)]
        # Uniform up to 3-decimal rounding (spread is rounding noise, not real jitter).
        self.assertLess(max(diffs) - min(diffs), 0.0015,
                        'opt-in grid should be uniform up to rounding')
        self.assertAlmostEqual(diffs[0], 60.0 / BPM, places=2)
        self.assertAlmostEqual(out[0], NATIVE_BEATS[0], places=6)
        # And it must differ from the (irregular) native track.
        self.assertNotEqual(out, NATIVE_BEATS)

    def test_uniform_off_variants_return_native(self):
        for val in ('0', 'false', 'no', 'off-typo', ''):
            out = self._call(NATIVE_BEATS, BPM, env={'UNIFORM_BEAT_GRID': val})
            self.assertEqual(out, NATIVE_BEATS, f'value {val!r} should not uniformize')

    def test_short_input_returned_as_is(self):
        self.assertEqual(self._call([], BPM), [])
        self.assertEqual(self._call([1.23], BPM), [1.23])


if __name__ == '__main__':
    unittest.main(verbosity=2)
