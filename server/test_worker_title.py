"""Unit tests for worker.py title resolution.

Run with: python3 server/test_worker_title.py  (stdlib unittest, no pytest needed)

A title equal to the video id is a placeholder left by an earlier bot-blocked
run; it must be treated as missing and re-fetched. YouTube's oEmbed endpoint
needs no cookies and is not bot-blocked, so it is tried before yt-dlp.
"""
import io
import json
import os
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker  # noqa: E402

VID = 'Jnq9wPDoDKg'
REAL = 'Sixpence None The Richer - Kiss Me (Official Music Video)'


def _oembed_ok(url, timeout=None):
    assert VID in url and 'oembed' in url, url
    return io.BytesIO(json.dumps({'title': REAL}).encode())


def _oembed_fail(url, timeout=None):
    raise urllib.error.HTTPError(url, 403, 'Forbidden', {}, None)


class ResolveTitleTests(unittest.TestCase):
    def test_provided_title_is_kept_without_network(self):
        with mock.patch.object(worker.urllib.request, 'urlopen', side_effect=AssertionError('no network')):
            self.assertEqual(worker._resolve_title('My Song', VID, ytdlp_fallback=None), 'My Song')

    def test_missing_title_is_fetched_from_oembed(self):
        with mock.patch.object(worker.urllib.request, 'urlopen', side_effect=_oembed_ok):
            self.assertEqual(worker._resolve_title('', VID, ytdlp_fallback=None), REAL)

    def test_placeholder_title_equal_to_video_id_is_refetched(self):
        with mock.patch.object(worker.urllib.request, 'urlopen', side_effect=_oembed_ok):
            self.assertEqual(worker._resolve_title(VID, VID, ytdlp_fallback=None), REAL)

    def test_falls_back_to_ytdlp_when_oembed_fails(self):
        calls = []

        def ytdlp():
            calls.append(1)
            return 'From yt-dlp'

        with mock.patch.object(worker.urllib.request, 'urlopen', side_effect=_oembed_fail):
            self.assertEqual(worker._resolve_title('', VID, ytdlp_fallback=ytdlp), 'From yt-dlp')
        self.assertEqual(len(calls), 1)

    def test_falls_back_to_video_id_when_everything_fails(self):
        with mock.patch.object(worker.urllib.request, 'urlopen', side_effect=_oembed_fail):
            self.assertEqual(worker._resolve_title('', VID, ytdlp_fallback=lambda: ''), VID)


if __name__ == '__main__':
    unittest.main()
