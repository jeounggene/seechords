#!/usr/bin/env python3
"""Ephemeral worker: download audio from YouTube, run chord analysis, write results to Turso.

YouTube auth is handled by the bgutil PO token server (started by worker-entrypoint.sh).
The yt-dlp plugin auto-detects it on localhost:4416 — no cookies needed.

Reads configuration from environment variables:
  JOB_ID              – unique job identifier
  VIDEO_ID            – YouTube video ID (11 chars)
  TITLE               – optional song title override
  TURSO_DATABASE_URL  – Turso database URL
  TURSO_AUTH_TOKEN    – Turso auth token
  YTDLP_YOUTUBE_PLAYER_CLIENT – optional override (e.g. web, android).
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
import glob
import time
import threading
import subprocess
import shutil
import traceback
import urllib.error
import urllib.request

NTFY_TOPIC = os.environ.get('NTFY_TOPIC', '')

# Chord-analysis model version tag stored in chord_versions.source.
# We run ChordMini's BTC model (ptnghia-j/ChordMini, MIT) — see CREDITS.md.
MODEL_SOURCE = 'chordmini-btc-v2.1'


def _ntfy(title: str, message: str, priority: str = 'high', tags: str = 'warning'):
    """Send a push notification via ntfy.sh. Silently ignores errors."""
    if not NTFY_TOPIC:
        return
    try:
        req = urllib.request.Request(
            f'https://ntfy.sh/{NTFY_TOPIC}',
            data=message.encode(),
            headers={'Title': title, 'Priority': priority, 'Tags': tags},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

# Ordered ladder of YouTube player-client sets to try across download attempts.
# Each attempt uses a DIFFERENT client set so a client-specific bot block can be
# bypassed by the next attempt. `default,mweb` matches what worked in isolation;
# `tv,web_safari` and `android_vr,mweb` are alternate profiles with different
# bot-detection behavior. The bgutil PO-token plugin supplies GVS tokens for all.
YTDLP_CLIENT_LADDER = ['default,mweb', 'tv,web_safari', 'android_vr,mweb']


def _client_ladder():
    """Return the client ladder to rotate through.

    An explicit YTDLP_YOUTUBE_PLAYER_CLIENT override pins a single client (no
    rotation); otherwise the full ladder is used.
    """
    o = (os.environ.get('YTDLP_YOUTUBE_PLAYER_CLIENT') or '').strip()
    if o and o.lower() not in ('none', 'off', '-'):
        return [o]
    return list(YTDLP_CLIENT_LADDER)


def _extractor_args_for(client: str):
    """Build yt-dlp extractor args pinning a specific player client."""
    return ['--extractor-args', f'youtube:player_client={client}']


def _is_bot_block(stderr: str) -> bool:
    """True if yt-dlp stderr indicates YouTube's 'not a bot' / sign-in block."""
    s = (stderr or '').lower()
    return 'sign in' in s or 'not a bot' in s


def _backoff_seconds(attempt: int) -> float:
    """Backoff between whole-extraction retries (8, 20, 32, 44s), capped at 45s.

    Spread wide on purpose: YouTube's intermittent bot-block clears on a
    minutes timescale, so a failing job's attempts should sample a ~2 min window
    rather than one ~20s burst — that recovers far more transient blocks within a
    single job. The cap keeps us from hammering the IP.
    """
    return min(45.0, 8.0 + 12.0 * attempt)


def _normalize_cookies_b64(s: str) -> str:
    """Strip whitespace/newlines so Fly secrets and shell quoting don't break base64 decode."""
    return ''.join((s or '').split())


def _prepare_ytdlp_auth(job_id: str):
    """Assemble optional yt-dlp auth args for IP-blocked datacenter downloads.

    YouTube persistently bot-blocks the shared Fly egress IP, which retrying can't
    fix. Two IP-level escapes, both optional and used if configured:
      * cookies — an authenticated session from YTDLP_COOKIES_B64 (base64 Netscape
        cookie jar), written to a per-job temp file and passed via --cookies.
      * proxy   — YTDLP_PROXY (e.g. a residential proxy URL), passed via --proxy.

    Returns (args, cookie_file); cookie_file is None if no cookies were written.
    """
    args = []
    cookie_file = None
    b64 = _normalize_cookies_b64(os.environ.get('YTDLP_COOKIES_B64') or '')
    if b64:
        try:
            import base64
            cookie_file = f'/tmp/{job_id}_cookies.txt'
            with open(cookie_file, 'wb') as f:
                f.write(base64.b64decode(b64))
            os.chmod(cookie_file, 0o600)
            args += ['--cookies', cookie_file]
            print('[Worker] Using YouTube cookies for yt-dlp', flush=True)
        except Exception as e:
            print(f'[Worker] cookie setup failed ({e}); continuing without cookies', flush=True)
            cookie_file = None
    proxy = (os.environ.get('YTDLP_PROXY') or '').strip()
    if proxy:
        args += ['--proxy', proxy]
        try:
            from urllib.parse import urlsplit
            host = urlsplit(proxy).hostname or '?'
        except Exception:
            host = '?'
        # Log the host only — never the credentials embedded in the URL.
        print(f'[Worker] Routing yt-dlp through proxy host={host}', flush=True)
    return args, cookie_file


def _download_audio(ytdlp, yt_url, out_template, job_id, hb_state=None,
                    max_attempts=5, sleep=time.sleep, auth_args=None):
    """Download bestaudio with whole-extraction retry + player-client rotation.

    YouTube returns its "Sign in to confirm you're not a bot" block as a *hard*
    extractor error, so yt-dlp's own --retries never recovers it. This re-invokes
    yt-dlp with a rotated player client and spread-out backoff. 5 attempts over a
    ~2 min window (see _backoff_seconds) sample past the intermittent block; most
    jobs still succeed on attempt 1, so only failing jobs pay the extra time.

    Rotation recovers *transient* blocks; `auth_args` (cookies and/or proxy from
    _prepare_ytdlp_auth) is what escapes a *persistent* datacenter-IP block, which
    rotation alone cannot. Both are applied together.

    Returns (audio_path, successful_client). Raises RuntimeError if every
    attempt fails.
    """
    ladder = _client_ladder()
    auth_args = list(auth_args or [])
    last_err = ''
    saw_bot_block = False
    for attempt in range(max_attempts):
        client = ladder[attempt % len(ladder)]
        if hb_state is not None:
            hb_state['client'] = client
        # Clear any partial file from a prior attempt so _find_downloaded_audio_file
        # can't resolve a stale, incomplete download.
        _cleanup_partial_downloads(job_id)
        cmd = (
            [ytdlp]
            + _extractor_args_for(client)
            + auth_args
            + [
                '-f', 'bestaudio/best',
                '--no-playlist', '--no-check-certificates',
                '--retries', '3',
                '--fragment-retries', '3',
                '--remote-components', 'ejs:github',
                '-o', out_template, yt_url,
            ]
        )
        print(f'[Worker] yt-dlp attempt {attempt + 1}/{max_attempts} '
              f'client={client} (PO token auth)', flush=True)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            stderr = result.stderr or ''
            audio_path = _find_downloaded_audio_file(job_id)
            if result.returncode == 0 and audio_path:
                print(f'[Worker] yt-dlp ok (attempt {attempt + 1}, client={client}) '
                      f'-> {audio_path}', flush=True)
                return audio_path, client
            last_err = stderr[-800:] if stderr else 'yt-dlp download failed'
            if _is_bot_block(stderr):
                saw_bot_block = True
        except subprocess.TimeoutExpired:
            last_err = 'yt-dlp timed out'
        print(f'[Worker] attempt {attempt + 1} failed (client={client}): '
              f'{last_err[-200:]}', flush=True)
        if attempt + 1 < max_attempts:
            sleep(_backoff_seconds(attempt))

    if saw_bot_block:
        _ntfy(
            'SeeChords: YouTube download blocked',
            f'Video blocked after {max_attempts} attempts. PO token may need update.',
        )
    raise RuntimeError(f'Download failed after {max_attempts} attempts: {last_err}')


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
    import libsql_experimental as libsql
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


def _cache_put(video_id, title, key, bpm, chords_data, beat_times, downbeats=None):
    con = _get_db()
    con.execute("DELETE FROM chord_versions WHERE video_id = ? AND source = ?", (video_id, MODEL_SOURCE))
    con.execute('''
        INSERT INTO chord_versions
            (video_id, title, key, bpm, chords, beat_times, downbeats, source, analyzed_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    ''', (video_id, title, key, bpm,
          json.dumps(chords_data), json.dumps(beat_times),
          json.dumps(downbeats) if downbeats else None,
          MODEL_SOURCE, int(time.time())))
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
            yt_url = f'https://www.youtube.com/watch?v={video_id}'
            ytdlp = shutil.which('yt-dlp')
            if not ytdlp:
                raise RuntimeError('yt-dlp not found')

            out_template = f'/tmp/{job_id}_audio.%(ext)s'
            # Optional cookies/proxy to escape a persistent datacenter-IP block.
            auth_args, _cookie_file = _prepare_ytdlp_auth(job_id)

            # Phase 1: Download audio
            _update_job(
                job_id,
                status='processing',
                progress=5,
                message='Downloading from YouTube… (usually 1–3 min)',
            )
            hb_state = {'active': True, 'phase': 'download', 'client': '—'}
            hb_stop = threading.Event()
            hb_t = threading.Thread(
                target=_heartbeat_job_message, args=(job_id, hb_stop, hb_state), daemon=True
            )
            hb_t.start()

            try:
                # Retry with player-client rotation: YouTube's datacenter-IP bot
                # block is intermittent and surfaces as a hard extractor error, so a
                # single yt-dlp invocation is fragile. _download_audio re-attempts
                # with a rotated player client until one gets through.
                audio_path, success_client = _download_audio(
                    ytdlp, yt_url, out_template, job_id, hb_state=hb_state,
                    auth_args=auth_args,
                )

                # Get title from yt-dlp if not provided, reusing the client that worked.
                if not title:
                    try:
                        t_result = subprocess.run(
                            [ytdlp] + _extractor_args_for(success_client) + auth_args
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
            finally:
                hb_state['active'] = False
                hb_stop.set()
                hb_t.join(timeout=3.0)

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
        downbeats = data.get('downbeats')

        print(f'[Worker] Analysis complete: {len(chords_data)} segments, key={key_val}, bpm={bpm_val}', flush=True)

        # Phase 4: Store results in Turso
        _update_job(job_id, status='processing', progress=95, message='Saving results…')
        version_id = _cache_put(video_id, title, key_val, round(bpm_val, 1), chords_data, beat_times, downbeats)

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
            source=MODEL_SOURCE,
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
        # Remove downloaded audio and the per-job cookie jar (contains session secrets).
        for f in (glob.glob(f'/tmp/{job_id}_audio.*')
                  + glob.glob(f'/tmp/{job_id}_cookies.txt')):
            try:
                os.remove(f)
            except OSError:
                pass


if __name__ == '__main__':
    main()
