#!/usr/bin/env python3
"""Ephemeral worker: download audio from YouTube, run chord analysis, write results to Turso.

Reads configuration from environment variables:
  JOB_ID              – unique job identifier
  VIDEO_ID            – YouTube video ID (11 chars)
  TITLE               – optional song title override
  TURSO_DATABASE_URL  – Turso database URL
  TURSO_AUTH_TOKEN    – Turso auth token
  YTDLP_COOKIEFILE    – optional path to Netscape cookies.txt on this machine (local dev)
  YTDLP_COOKIES_B64   – optional base64 of cookies.txt (set on API; forwarded to worker on Fly)
  YTDLP_YOUTUBE_PLAYER_CLIENT – optional override (e.g. web, android). Otherwise we use a
    ChordMini-style rotation (yt-dlp + bestaudio + multiple innertube clients, like yt-mp3-go).
  PUBLIC_APP_URL       – API base URL (default https://seechords.fly.dev) for WAV cache GET/PUT.
  WAV_CACHE_SECRET     – Bearer token shared with API; workers upload WAV after download for reuse.
  SKIP_YTDLP_DOWNLOAD  – if '1', fetch cached WAV from API instead of yt-dlp (set by API on cache hit).

Exits with code 0 on success, 1 on failure.
The Fly Machine that runs this has auto_destroy: true, so it self-terminates.
"""
import os
import sys
import json
import math
import base64
import glob
import time
import threading
import subprocess
import shutil
import traceback
import urllib.error
import urllib.request

import libsql_experimental as libsql

_YTDLP_COOKIE_PATH = None
# Set after a successful yt-dlp download so --get-title uses the same player client.
_SUCCESS_YTDLP_PLAYER_CLIENT = None

# Browser-like UA so session cookies match what YouTube expects alongside --cookies.
_YTDLP_CHROME_UA = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
)


def _decode_cookie_b64(b64: str) -> bytes:
    s = ''.join((b64 or '').split())
    pad = (-len(s)) % 4
    if pad:
        s += '=' * pad
    return base64.b64decode(s)


def _normalize_netscape_cookie_bytes(raw: bytes) -> bytes:
    """Strip BOM, normalize newlines, ensure Netscape header so yt-dlp accepts the file."""
    if not raw:
        raise ValueError('empty after decode')
    text = raw.decode('utf-8-sig')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    if not text.endswith('\n'):
        text += '\n'
    head_ok = any(
        (ln.startswith('# Netscape') or 'HTTP Cookie File' in ln[:80])
        for ln in text.split('\n')[:8]
    )
    if not head_ok:
        text = '# Netscape HTTP Cookie File\n' + text
    return text.encode('utf-8')


def _cookie_data_line_count(path: str) -> int:
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            n = 0
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if '\t' in s and len(s.split('\t')) >= 7:
                    n += 1
            return n
    except OSError:
        return -1


def _init_ytdlp_cookies():
    """Resolve cookies once: local file, or YTDLP_COOKIES_B64 written to /tmp."""
    global _YTDLP_COOKIE_PATH
    p = (os.environ.get('YTDLP_COOKIEFILE') or '').strip()
    if p and os.path.isfile(p):
        _YTDLP_COOKIE_PATH = p
        n = _cookie_data_line_count(p)
        print(
            f'[Worker] yt-dlp cookies: file {_YTDLP_COOKIE_PATH} ({n} tab-separated rows)',
            flush=True,
        )
        return
    b64 = (os.environ.get('YTDLP_COOKIES_B64') or '').strip()
    if b64:
        out = '/tmp/yt_cookies.txt'
        try:
            raw = _decode_cookie_b64(b64)
            normalized = _normalize_netscape_cookie_bytes(raw)
            with open(out, 'wb') as f:
                f.write(normalized)
            os.chmod(out, 0o600)
            _YTDLP_COOKIE_PATH = out
            n = _cookie_data_line_count(out)
            print(
                f'[Worker] yt-dlp cookies: YTDLP_COOKIES_B64 -> {out} '
                f'({len(normalized)} bytes, {n} cookie rows)',
                flush=True,
            )
            head = normalized[:120].decode('utf-8', errors='replace')
            if 'youtube' not in head.lower() and 'google' not in head.lower():
                print(
                    '[Worker] WARNING: cookie file header lines do not mention youtube/google; '
                    'ensure export includes youtube.com / google.com session cookies.',
                    flush=True,
                )
            if n == 0:
                print(
                    '[Worker] WARNING: no tab-separated cookie rows found — file may be wrong format.',
                    flush=True,
                )
        except Exception as e:
            _YTDLP_COOKIE_PATH = None
            print(f'[Worker] WARNING: YTDLP_COOKIES_B64 invalid: {e}', flush=True)
        return
    if p:
        print(
            f'[Worker] WARNING: YTDLP_COOKIEFILE={p!r} not found on this machine; '
            'configure cookies on the API app (YTDLP_COOKIEFILE path or YTDLP_COOKIES_B64).',
            flush=True,
        )
    else:
        print(
            '[Worker] No yt-dlp cookies configured (datacenter IPs often need YouTube cookies).',
            flush=True,
        )


def _ytdlp_cookie_args():
    if _YTDLP_COOKIE_PATH:
        return ['--cookies', _YTDLP_COOKIE_PATH]
    return []


def _ytdlp_browser_headers_args():
    """When using exported cookies, send Chrome-like UA / Accept-Language (matches many exports)."""
    if not _YTDLP_COOKIE_PATH:
        return []
    return [
        '--add-header',
        f'User-Agent:{_YTDLP_CHROME_UA}',
        '--add-header',
        'Accept-Language:en-US,en;q=0.9',
    ]


def _ytdlp_extractor_args_for_client(player_client: str):
    if not player_client or player_client.lower() in ('none', 'off', '-'):
        return []
    return ['--extractor-args', f'youtube:player_client={player_client}']


def _ytdlp_player_clients_to_try():
    """ChordMini / yt-mp3-go style: try several innertube clients (cookies → web-first for session)."""
    o = (os.environ.get('YTDLP_YOUTUBE_PLAYER_CLIENT') or '').strip()
    if o and o.lower() not in ('none', 'off', '-'):
        return [o]
    if _YTDLP_COOKIE_PATH:
        return ['web', 'ios', 'android', 'mediaconnect', 'android_embedded', 'tv_embedded', 'mweb']
    return ['android', 'ios', 'mediaconnect', 'android_embedded', 'tv_embedded', 'web', 'mweb']


def _cleanup_partial_downloads(job_id: str) -> None:
    """Remove any /tmp/{job_id}_audio.* from a previous client attempt (any extension)."""
    for p in glob.glob(f'/tmp/{job_id}_audio.*'):
        try:
            if not p.endswith('.wav'):
                os.remove(p)
        except OSError:
            pass


def _find_downloaded_audio_file(job_id: str) -> str | None:
    """Resolve file written by yt-dlp -o ...%(ext)s (may be m4a, webm, opus, etc.)."""
    matches = [p for p in glob.glob(f'/tmp/{job_id}_audio.*') if not p.endswith('.wav')]
    if not matches:
        return None
    return max(matches, key=lambda p: os.path.getsize(p))


def _get_db():
    return libsql.connect(
        database=os.environ['TURSO_DATABASE_URL'],
        auth_token=os.environ['TURSO_AUTH_TOKEN'],
    )


def _update_job(job_id, **kwargs):
    sets = ['updated_at = ?']
    vals = [int(time.time())]
    for col in ('status', 'progress', 'message', 'worker_id'):
        if col in kwargs:
            sets.append(f'{col} = ?')
            vals.append(kwargs[col])
    extra = {k: v for k, v in kwargs.items()
             if k not in ('status', 'progress', 'message', 'worker_id', 'video_id')}
    if extra:
        sets.append('result = ?')
        vals.append(json.dumps(extra, default=str))
    vals.append(job_id)
    con = _get_db()
    con.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", tuple(vals))
    con.commit()
    con.close()


def _cache_put(video_id, title, key, bpm, chords_data, beat_times):
    con = _get_db()
    con.execute('DELETE FROM chord_versions WHERE video_id = ?', (video_id,))
    con.execute('''
        INSERT INTO chord_versions
            (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, 'btc-v2', ?, 1)
    ''', (video_id, title, key, bpm,
          json.dumps(chords_data), json.dumps(beat_times),
          int(time.time())))
    con.commit()
    version_id = con.execute('SELECT last_insert_rowid()').fetchone()[0]
    con.close()
    return version_id


def _ramp_analyze_progress(job_id, stop_event):
    """Gradually increase progress during long ML analysis (30% → ~87%) so the bar reflects elapsed time."""
    t0 = time.monotonic()
    while not stop_event.wait(2.5):
        elapsed = time.monotonic() - t0
        p = 30 + int(58 * (1 - math.exp(-elapsed / 95.0)))
        p = min(87, p)
        try:
            _update_job(job_id, status='processing', progress=p, message='Analyzing chords…')
        except Exception:
            pass


def _heartbeat_job_message(job_id: str, stop_event: threading.Event, state: dict) -> None:
    """Refresh job.message every ~14s during long steps so the UI does not look stuck at 5%."""
    t0 = time.monotonic()
    while not stop_event.wait(14):
        if not state.get('active'):
            break
        elapsed = int(time.monotonic() - t0)
        phase = state.get('phase', 'download')
        client = state.get('client', '—')
        progress = int(state.get('progress', 5))
        if phase == 'cache':
            msg = f'Loading cached audio… ~{elapsed}s (not stuck — large files take time)'
        elif phase == 'download':
            msg = (
                f'Downloading from YouTube… ~{elapsed}s (player: {client}) — '
                'often 1–3 min; still working'
            )
        elif phase == 'ffmpeg':
            msg = f'Converting to WAV… ~{elapsed}s'
        elif phase == 'upload':
            msg = f'Saving audio for next time… ~{elapsed}s'
        else:
            msg = f'Working… ~{elapsed}s'
        try:
            _update_job(job_id, status='processing', progress=progress, message=msg)
        except Exception:
            pass


def _public_app_url() -> str:
    return (os.environ.get('PUBLIC_APP_URL') or 'https://seechords.fly.dev').strip().rstrip('/')


def _fetch_wav_cache_from_api(video_id: str, dest_path: str) -> None:
    """Download cached 44.1kHz mono WAV from API (ChordMini-style reuse)."""
    secret = (os.environ.get('WAV_CACHE_SECRET') or '').strip()
    if not secret:
        raise RuntimeError('WAV_CACHE_SECRET missing')
    url = f'{_public_app_url()}/api/internal/wav-cache/{video_id}'
    req = urllib.request.Request(url)
    req.add_header('Authorization', f'Bearer {secret}')
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'Cached WAV HTTP {e.code}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'Cached WAV fetch failed: {e!r}') from e
    if len(data) < 4096:
        raise RuntimeError('Cached WAV too small')
    with open(dest_path, 'wb') as f:
        f.write(data)
    os.chmod(dest_path, 0o644)
    print(f'[Worker] Loaded cached WAV from API ({len(data)} bytes)', flush=True)


def _upload_wav_cache_to_api(wav_path: str, video_id: str) -> None:
    """Persist WAV on API disk so re-analysis can skip YouTube download."""
    secret = (os.environ.get('WAV_CACHE_SECRET') or '').strip()
    if not secret:
        print('[Worker] WAV cache upload skipped (no WAV_CACHE_SECRET)', flush=True)
        return
    try:
        with open(wav_path, 'rb') as f:
            data = f.read()
    except OSError as e:
        print(f'[Worker] WAV cache upload read failed: {e}', flush=True)
        return
    if len(data) < 4096:
        return
    url = f'{_public_app_url()}/api/internal/wav-cache/{video_id}'
    req = urllib.request.Request(url, data=data, method='PUT')
    req.add_header('Authorization', f'Bearer {secret}')
    req.add_header('Content-Type', 'audio/wav')
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            _ = resp.read()
        print(f'[Worker] WAV cache stored on API ({len(data)} bytes)', flush=True)
    except urllib.error.HTTPError as e:
        err_body = e.read()[:300] if e.fp else b''
        print(f'[Worker] WAV cache upload HTTP {e.code}: {err_body!r}', flush=True)
    except urllib.error.URLError as e:
        print(f'[Worker] WAV cache upload failed: {e!r}', flush=True)


def main():
    global _SUCCESS_YTDLP_PLAYER_CLIENT
    _SUCCESS_YTDLP_PLAYER_CLIENT = None

    job_id = os.environ.get('JOB_ID', '')
    video_id = os.environ.get('VIDEO_ID', '')
    title = os.environ.get('TITLE', '')

    if not job_id or not video_id:
        print('ERROR: JOB_ID and VIDEO_ID are required', flush=True)
        sys.exit(1)

    print(f'[Worker] Starting job={job_id} video={video_id}', flush=True)
    skip_download = os.environ.get('SKIP_YTDLP_DOWNLOAD') == '1'

    try:
        wav_path = f'/tmp/{job_id}_audio.wav'

        if skip_download:
            print('[Worker] SKIP_YTDLP_DOWNLOAD=1 — loading WAV from API cache', flush=True)
            _update_job(job_id, status='processing', progress=5, message='Loading cached audio…')
            hb_state = {'active': True, 'phase': 'cache', 'client': '—'}
            hb_stop = threading.Event()
            hb_t = threading.Thread(
                target=_heartbeat_job_message, args=(job_id, hb_stop, hb_state), daemon=True
            )
            hb_t.start()
            try:
                _fetch_wav_cache_from_api(video_id, wav_path)
            except Exception as e:
                raise RuntimeError(f'Cached WAV unavailable: {e}') from e
            finally:
                hb_state['active'] = False
                hb_stop.set()
                hb_t.join(timeout=3.0)
            if not title:
                title = video_id
            _update_job(
                job_id, status='processing', progress=15,
                message='Cached audio ready — analyzing chords…',
            )
        else:
            _init_ytdlp_cookies()
            b64_in_env = bool((os.environ.get('YTDLP_COOKIES_B64') or '').strip())
            print(
                f'[Worker] yt-dlp cookies: env B64 present={b64_in_env}, '
                f'file ready={bool(_YTDLP_COOKIE_PATH)}',
                flush=True,
            )
            clients = _ytdlp_player_clients_to_try()
            print(f'[Worker] yt-dlp will try player_client order: {clients}', flush=True)

            yt_url = f'https://www.youtube.com/watch?v={video_id}'
            ytdlp = shutil.which('yt-dlp')
            if not ytdlp:
                raise RuntimeError('yt-dlp not found')

            out_template = f'/tmp/{job_id}_audio.%(ext)s'
            MAX_DL_RETRIES = 2
            last_dl_exc = None

            for dl_attempt in range(MAX_DL_RETRIES + 1):
                if dl_attempt > 0:
                    delay = 20 * dl_attempt
                    print(f'[Worker] Download retry {dl_attempt}/{MAX_DL_RETRIES} in {delay}s…', flush=True)
                    _update_job(
                        job_id, status='processing', progress=5,
                        message=f'Download failed — retrying in {delay}s… (attempt {dl_attempt + 1}/{MAX_DL_RETRIES + 1})',
                    )
                    time.sleep(delay)
                    _cleanup_partial_downloads(job_id)

                # Phase 1: Download audio (ChordMini / yt-mp3-go style: yt-dlp bestaudio + ffmpeg later)
                _update_job(
                    job_id,
                    status='processing',
                    progress=5,
                    message='Downloading from YouTube… (usually 1–3 min — progress updates every ~15s)',
                )
                hb_state = {'active': True, 'phase': 'download', 'client': '—'}
                hb_stop = threading.Event()
                hb_t = threading.Thread(
                    target=_heartbeat_job_message, args=(job_id, hb_stop, hb_state), daemon=True
                )
                hb_t.start()

                result = None
                last_stderr = ''
                audio_path = ''
                try:
                    for client in clients:
                        hb_state['client'] = client
                        _update_job(
                            job_id,
                            status='processing',
                            progress=5,
                            message=(
                                f'Downloading from YouTube… trying “{client}” player '
                                '(may take up to 3 min)'
                            ),
                        )
                        _cleanup_partial_downloads(job_id)
                        cmd = (
                            [ytdlp]
                            + _ytdlp_cookie_args()
                            + _ytdlp_browser_headers_args()
                            + _ytdlp_extractor_args_for_client(client)
                            + [
                                '-f', 'bestaudio/best',
                                '--no-playlist', '--no-check-certificates',
                                '--retries', '3',
                                '--fragment-retries', '3',
                                '--remote-components', 'ejs:github',
                                '-o', out_template, yt_url,
                            ]
                        )
                        print(f'[Worker] yt-dlp download try player_client={client}', flush=True)
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                        last_stderr = result.stderr or ''
                        audio_path = _find_downloaded_audio_file(job_id) or ''
                        if result.returncode == 0 and audio_path and os.path.getsize(audio_path) > 0:
                            _SUCCESS_YTDLP_PLAYER_CLIENT = client
                            print(f'[Worker] yt-dlp download ok with player_client={client} -> {audio_path}', flush=True)
                            break

                    if result is None or result.returncode != 0 or not audio_path:
                        err_msg = (
                            last_stderr[:800]
                            if last_stderr
                            else (result.stderr[:800] if result else 'yt-dlp download failed')
                        )
                        if 'sign in' in err_msg.lower() or 'not a bot' in err_msg.lower():
                            if _YTDLP_COOKIE_PATH is None:
                                err_msg += (
                                    ' — Set YTDLP_COOKIES_B64 on app `seechords`, redeploy API, and worker '
                                    'image. See yt-dlp wiki for PO token if cookies alone fail.'
                                )
                            else:
                                err_msg += (
                                    ' — Tried player clients: '
                                    + ', '.join(clients)
                                    + '. Re-export fresh cookies while logged into YouTube; some videos need '
                                    'yt-dlp PoToken plugins (see yt-dlp wiki / EJS).'
                                )
                        last_dl_exc = RuntimeError(f'Download failed: {err_msg}')
                        continue  # retry

                    # Get title from yt-dlp if not provided
                    if not title:
                        try:
                            pc = _SUCCESS_YTDLP_PLAYER_CLIENT or 'web'
                            t_result = subprocess.run(
                                [ytdlp]
                                + _ytdlp_cookie_args()
                                + _ytdlp_browser_headers_args()
                                + _ytdlp_extractor_args_for_client(pc)
                                + ['--get-title', '--no-playlist', yt_url],
                                capture_output=True, text=True, timeout=15,
                            )
                            if t_result.returncode == 0 and t_result.stdout.strip():
                                title = t_result.stdout.strip()
                            else:
                                title = video_id
                        except Exception:
                            title = video_id

                    print(f'[Worker] Downloaded audio: {os.path.getsize(audio_path)} bytes', flush=True)
                    _update_job(
                        job_id,
                        status='processing',
                        progress=15,
                        message='Download done — converting to WAV…',
                    )

                    hb_state['phase'] = 'ffmpeg'
                    hb_state['progress'] = 15
                    # Phase 2: Convert to WAV
                    subprocess.run(
                        ['ffmpeg', '-i', audio_path, '-vn', '-ar', '44100', '-ac', '1',
                         wav_path, '-y'],
                        capture_output=True, timeout=180,
                    )
                    if not os.path.exists(wav_path):
                        raise RuntimeError('FFmpeg conversion failed')

                    hb_state['phase'] = 'upload'
                    _upload_wav_cache_to_api(wav_path, video_id)
                    last_dl_exc = None  # success
                    break
                finally:
                    hb_state['active'] = False
                    hb_stop.set()
                    hb_t.join(timeout=3.0)

            if last_dl_exc is not None:
                raise last_dl_exc

        # Phase 3: Chord analysis
        _update_job(job_id, status='processing', progress=30, message='Analyzing chords…')
        print('[Worker] Starting chord analysis…', flush=True)

        from analyze_chords import analyze as _analyze_chords

        stop_ramp = threading.Event()
        ramp_thread = threading.Thread(
            target=_ramp_analyze_progress, args=(job_id, stop_ramp), daemon=True
        )
        ramp_thread.start()
        try:
            data = _analyze_chords(wav_path)
        finally:
            stop_ramp.set()
            ramp_thread.join(timeout=2.0)

        chords_data = data['chords']
        bpm_val = data['bpm']
        key_val = data['key']
        beat_times = data['beat_times']

        print(f'[Worker] Analysis complete: {len(chords_data)} segments, key={key_val}, bpm={bpm_val}', flush=True)

        # Phase 4: Store results in Turso
        _update_job(job_id, status='processing', progress=90, message='Saving results…')
        version_id = _cache_put(video_id, title, key_val, round(bpm_val, 1), chords_data, beat_times)

        _update_job(
            job_id,
            status='done',
            progress=100,
            message='Complete',
            versionId=version_id,
            videoId=video_id,
            title=title,
            chords=chords_data,
            bpm=round(bpm_val, 1),
            key=key_val,
            beat_times=beat_times,
            source='btc-v2',
        )
        print(f'[Worker] Job complete! versionId={version_id}', flush=True)

    except Exception as exc:
        traceback.print_exc()
        print(f'[Worker] Job failed: {exc}', flush=True)
        try:
            _update_job(job_id, status='error', message=f'Analysis failed: {exc}')
        except Exception:
            traceback.print_exc()
        sys.exit(1)
    finally:
        for f in glob.glob(f'/tmp/{job_id}_audio.*'):
            try:
                os.remove(f)
            except OSError:
                pass


if __name__ == '__main__':
    main()
