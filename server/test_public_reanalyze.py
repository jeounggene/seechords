"""Unit tests for the public re-analyze guardrail helpers in app.py.

Run: python3 server/test_public_reanalyze.py  (stdlib unittest)

Isolated against a temp SQLite DB (app.py falls back to sqlite3 when libsql is absent).
Covers the cooldown, in-flight, and cached-WAV decisions that gate public re-analysis.
"""
import os
import sys
import time
import tempfile
import unittest

# Isolate BEFORE importing app: temp DB + temp WAV dir, no Turso.
_TMPDB = tempfile.mktemp(suffix='.db')
_TMPWAV = tempfile.mkdtemp(prefix='wavcache-')
os.environ.pop('TURSO_DATABASE_URL', None)
os.environ.pop('TURSO_AUTH_TOKEN', None)
os.environ['DB_PATH'] = _TMPDB
os.environ['WAV_CACHE_DIR'] = _TMPWAV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app  # noqa: E402

SRC = app.CURRENT_MODEL_SOURCE


def _reset_db():
    con = app._get_db()
    for t in ('chord_versions', 'jobs', 'wav_cache_backups'):
        con.execute(f'DROP TABLE IF EXISTS {t}')
    con.execute("CREATE TABLE chord_versions (version_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " video_id TEXT, title TEXT, source TEXT, analyzed_at INTEGER, is_active INTEGER)")
    con.execute("CREATE TABLE jobs (job_id TEXT, video_id TEXT, status TEXT, updated_at INTEGER)")
    con.execute("CREATE TABLE wav_cache_backups (video_id TEXT PRIMARY KEY, wav_data BLOB)")
    con.commit()
    con.close()


class ReanalyzeHelperTests(unittest.TestCase):
    def setUp(self):
        _reset_db()
        for f in os.listdir(_TMPWAV):
            os.remove(os.path.join(_TMPWAV, f))
        self.vid = 'abcDEF12345'

    def _insert_version(self, analyzed_at, source=SRC):
        con = app._get_db()
        con.execute("INSERT INTO chord_versions (video_id, title, source, analyzed_at, is_active)"
                    " VALUES (?,?,?,?,1)", (self.vid, 't', source, analyzed_at))
        con.commit()
        con.close()

    def _insert_job(self, status, updated_at):
        con = app._get_db()
        con.execute("INSERT INTO jobs (job_id, video_id, status, updated_at) VALUES (?,?,?,?)",
                    ('j-' + status, self.vid, status, updated_at))
        con.commit()
        con.close()

    # ── _has_cached_wav ────────────────────────────────────────────
    def test_has_cached_wav_false_when_none(self):
        self.assertFalse(app._has_cached_wav(self.vid))

    def test_has_cached_wav_true_from_backup_row(self):
        con = app._get_db()
        con.execute("INSERT INTO wav_cache_backups (video_id, wav_data) VALUES (?, ?)",
                    (self.vid, b'x' * 100))
        con.commit()
        con.close()
        self.assertTrue(app._has_cached_wav(self.vid))

    def test_has_cached_wav_true_from_disk(self):
        with open(os.path.join(_TMPWAV, f'{self.vid}.wav'), 'wb') as f:
            f.write(b'x' * 100)
        self.assertTrue(app._has_cached_wav(self.vid))

    # ── _reanalyze_cooldown_remaining ──────────────────────────────
    def test_cooldown_zero_when_never_analyzed(self):
        self.assertEqual(app._reanalyze_cooldown_remaining(self.vid), 0)

    def test_cooldown_positive_within_window(self):
        self._insert_version(int(time.time()) - 60)  # 1 min ago
        rem = app._reanalyze_cooldown_remaining(self.vid)
        self.assertGreater(rem, 0)
        self.assertLessEqual(rem, 600)

    def test_cooldown_zero_after_window(self):
        self._insert_version(int(time.time()) - 700)  # >10 min ago
        self.assertEqual(app._reanalyze_cooldown_remaining(self.vid), 0)

    def test_cooldown_ignores_other_sources(self):
        # A 'verified' row must not trigger the model-source cooldown.
        self._insert_version(int(time.time()) - 30, source='verified')
        self.assertEqual(app._reanalyze_cooldown_remaining(self.vid), 0)

    # ── _song_job_in_flight ────────────────────────────────────────
    def test_in_flight_false_when_no_jobs(self):
        self.assertFalse(app._song_job_in_flight(self.vid))

    def test_in_flight_true_when_processing(self):
        self._insert_job('processing', int(time.time()))
        self.assertTrue(app._song_job_in_flight(self.vid))

    def test_in_flight_true_when_pending(self):
        self._insert_job('pending', int(time.time()))
        self.assertTrue(app._song_job_in_flight(self.vid))

    def test_in_flight_false_when_done(self):
        self._insert_job('done', int(time.time()))
        self.assertFalse(app._song_job_in_flight(self.vid))

    def test_in_flight_false_when_stale(self):
        # A processing job older than the 15-min window is treated as abandoned.
        self._insert_job('processing', int(time.time()) - 3600)
        self.assertFalse(app._song_job_in_flight(self.vid))


if __name__ == '__main__':
    unittest.main(verbosity=2)
