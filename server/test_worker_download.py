"""Unit tests for worker.py YouTube-download retry + client-rotation logic.

Run with: python3 server/test_worker_download.py  (stdlib unittest, no pytest needed)

These tests mock subprocess.run so no real network/yt-dlp is invoked. They verify
that a transient YouTube "not a bot" block is recovered by re-invoking yt-dlp with a
rotated player client — the behavior the production failure needs.
"""
import os
import sys
import glob
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker  # noqa: E402

BOT_ERR = (
    "ERROR: [youtube] _Paw8ZRSlqY: Sign in to confirm you're not a bot. "
    "Use --cookies-from-browser or --cookies for the authentication."
)


class _FakeCompleted:
    def __init__(self, returncode, stderr=''):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ''


def _extractor_client(cmd):
    """Pull the youtube:player_client=... value out of a yt-dlp argv."""
    for i, a in enumerate(cmd):
        if a == '--extractor-args' and i + 1 < len(cmd):
            val = cmd[i + 1]
            if val.startswith('youtube:player_client='):
                return val.split('=', 1)[1]
    return None


def _make_fake_run(script, job_id):
    """Return (fake_run, calls). `script` is a list of (returncode, stderr) per attempt.
    On a returncode==0 attempt the fake writes /tmp/{job_id}_audio.webm so the real
    _find_downloaded_audio_file() resolves it, mirroring a genuine yt-dlp success."""
    calls = []

    def _run(cmd, **kwargs):
        idx = len(calls)
        calls.append(list(cmd))
        rc, err = script[idx]
        if rc == 0:
            with open(f'/tmp/{job_id}_audio.webm', 'wb') as f:
                f.write(b'\x00' * 8192)
        return _FakeCompleted(rc, err)

    return _run, calls


class DownloadRetryTests(unittest.TestCase):
    def setUp(self):
        self.job_id = 'testjob-abc123'
        self.ytdlp = '/usr/local/bin/yt-dlp'
        self.yt_url = 'https://www.youtube.com/watch?v=_Paw8ZRSlqY'
        self.out_template = f'/tmp/{self.job_id}_audio.%(ext)s'
        self._cleanup()
        # No real backoff sleeping in tests.
        self.no_sleep = lambda _s: None

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        for p in glob.glob(f'/tmp/{self.job_id}_audio.*'):
            try:
                os.remove(p)
            except OSError:
                pass

    def _download(self, fake_run, **kw):
        with mock.patch.object(worker.subprocess, 'run', fake_run):
            return worker._download_audio(
                self.ytdlp, self.yt_url, self.out_template, self.job_id,
                sleep=self.no_sleep, **kw,
            )

    def test_success_first_attempt_no_retry(self):
        fake_run, calls = _make_fake_run([(0, '')], self.job_id)
        audio_path, client = self._download(fake_run)
        self.assertEqual(len(calls), 1)
        self.assertTrue(audio_path.endswith('.webm'))
        self.assertTrue(os.path.exists(audio_path))
        self.assertIsNotNone(client)

    def test_bot_block_then_success_rotates_client(self):
        fake_run, calls = _make_fake_run([(1, BOT_ERR), (0, '')], self.job_id)
        audio_path, client = self._download(fake_run)
        self.assertEqual(len(calls), 2, 'should retry once after the bot block')
        self.assertTrue(os.path.exists(audio_path))
        c0, c1 = _extractor_client(calls[0]), _extractor_client(calls[1])
        self.assertNotEqual(c0, c1, 'retry must rotate to a different player client')
        self.assertEqual(client, c1, 'returned client is the one that succeeded')

    def test_all_attempts_fail_raises_after_max(self):
        fake_run, calls = _make_fake_run(
            [(1, BOT_ERR), (1, BOT_ERR), (1, BOT_ERR)], self.job_id)
        with self.assertRaises(RuntimeError) as ctx:
            self._download(fake_run, max_attempts=3)
        self.assertEqual(len(calls), 3)
        self.assertIn('attempt', str(ctx.exception).lower())

    def test_default_is_five_attempts(self):
        # No max_attempts passed → uses the hardened default (5).
        fake_run, calls = _make_fake_run([(1, BOT_ERR)] * 5, self.job_id)
        with self.assertRaises(RuntimeError):
            self._download(fake_run)
        self.assertEqual(len(calls), 5)

    def test_backoff_is_spread_and_capped(self):
        vals = [worker._backoff_seconds(i) for i in range(6)]
        self.assertEqual(vals[0], 8.0)                       # first retry not too slow
        self.assertTrue(all(b <= a for a, b in zip(vals[1:], vals)))  # non-decreasing...
        self.assertTrue(all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)))
        self.assertLessEqual(max(vals), 45.0)               # capped
        self.assertGreaterEqual(sum(vals[:4]), 100.0)       # ~2 min span across a failing job

    def test_client_ladder_is_distinct_across_attempts(self):
        fake_run, calls = _make_fake_run(
            [(1, BOT_ERR), (1, BOT_ERR), (1, BOT_ERR)], self.job_id)
        with self.assertRaises(RuntimeError):
            self._download(fake_run, max_attempts=3)
        clients = [_extractor_client(c) for c in calls]
        self.assertEqual(len(clients), 3)
        self.assertEqual(len(set(clients)), 3, f'expected 3 distinct clients, got {clients}')

    def test_env_override_pins_single_client_no_rotation(self):
        with mock.patch.dict(os.environ, {'YTDLP_YOUTUBE_PLAYER_CLIENT': 'android_vr'}):
            fake_run, calls = _make_fake_run([(1, BOT_ERR), (0, '')], self.job_id)
            self._download(fake_run)
        clients = [_extractor_client(c) for c in calls]
        self.assertEqual(clients, ['android_vr', 'android_vr'],
                         'explicit override should pin the client across retries')

    def test_auth_args_included_in_command(self):
        fake_run, calls = _make_fake_run([(0, '')], self.job_id)
        with mock.patch.object(worker.subprocess, 'run', fake_run):
            worker._download_audio(
                self.ytdlp, self.yt_url, self.out_template, self.job_id,
                sleep=self.no_sleep,
                auth_args=['--cookies', '/tmp/x.txt', '--proxy', 'http://p:1'],
            )
        cmd = calls[0]
        self.assertIn('--cookies', cmd)
        self.assertIn('--proxy', cmd)
        # Proxy/cookies must come before the URL and download flags take the file.
        self.assertEqual(cmd[cmd.index('--proxy') + 1], 'http://p:1')


class PrepareAuthTests(unittest.TestCase):
    def setUp(self):
        self.job_id = 'authjob-xyz789'
        self._cleanup()

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        for p in glob.glob(f'/tmp/{self.job_id}_cookies.txt'):
            try:
                os.remove(p)
            except OSError:
                pass

    def test_no_env_returns_empty(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('YTDLP_COOKIES_B64', None)
            os.environ.pop('YTDLP_PROXY', None)
            args, cookie_file = worker._prepare_ytdlp_auth(self.job_id)
        self.assertEqual(args, [])
        self.assertIsNone(cookie_file)

    def test_cookies_written_and_flagged(self):
        import base64
        payload = b'# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tabc\n'
        b64 = base64.b64encode(payload).decode()
        with mock.patch.dict(os.environ, {'YTDLP_COOKIES_B64': b64}, clear=False):
            os.environ.pop('YTDLP_PROXY', None)
            args, cookie_file = worker._prepare_ytdlp_auth(self.job_id)
        self.assertIn('--cookies', args)
        self.assertEqual(args[args.index('--cookies') + 1], cookie_file)
        self.assertTrue(os.path.exists(cookie_file))
        with open(cookie_file, 'rb') as f:
            self.assertEqual(f.read(), payload)

    def test_cookies_b64_tolerates_whitespace(self):
        import base64
        payload = b'cookiejar-contents'
        b64 = base64.b64encode(payload).decode()
        # Fly secrets / shell quoting can inject newlines and spaces.
        munged = b64[:4] + '\n ' + b64[4:] + '\n'
        with mock.patch.dict(os.environ, {'YTDLP_COOKIES_B64': munged}, clear=False):
            os.environ.pop('YTDLP_PROXY', None)
            args, cookie_file = worker._prepare_ytdlp_auth(self.job_id)
        self.assertIsNotNone(cookie_file)
        with open(cookie_file, 'rb') as f:
            self.assertEqual(f.read(), payload)

    def test_proxy_flagged(self):
        with mock.patch.dict(os.environ, {'YTDLP_PROXY': 'http://user:pw@host:8080'}, clear=False):
            os.environ.pop('YTDLP_COOKIES_B64', None)
            args, cookie_file = worker._prepare_ytdlp_auth(self.job_id)
        self.assertEqual(args, ['--proxy', 'http://user:pw@host:8080'])
        self.assertIsNone(cookie_file)

    def test_both_cookies_and_proxy(self):
        import base64
        b64 = base64.b64encode(b'jar').decode()
        with mock.patch.dict(os.environ, {
            'YTDLP_COOKIES_B64': b64, 'YTDLP_PROXY': 'socks5://p:1080',
        }, clear=False):
            args, cookie_file = worker._prepare_ytdlp_auth(self.job_id)
        self.assertIn('--cookies', args)
        self.assertIn('--proxy', args)
        self.assertIsNotNone(cookie_file)


if __name__ == '__main__':
    unittest.main(verbosity=2)
