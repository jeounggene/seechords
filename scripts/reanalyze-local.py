#!/usr/bin/env python3
"""Re-analyze songs by downloading audio locally, uploading to Fly, then spawning worker.

Avoids YouTube rate limits on Fly by downloading on your local machine.

Usage:
    python scripts/reanalyze-local.py
    python scripts/reanalyze-local.py --host http://localhost:5000
"""

import argparse
import getpass
import os
import subprocess
import sys
import tempfile
import time

import requests

DEFAULT_HOST = 'https://seechords.fly.dev'
MAX_WAV_BYTES = 50 * 1024 * 1024  # 50 MB — stay safely under server's 60MB limit


def login(session, host):
    password = getpass.getpass('Ingest password: ')
    resp = session.post(f'{host}/ingest/login', data={'password': password}, allow_redirects=False)
    if resp.status_code == 200 and 'totp_code' in resp.text:
        totp_code = input('MFA code: ')
        resp = session.post(f'{host}/ingest/login', data={'totp_code': totp_code}, allow_redirects=False)
        if resp.status_code not in (302, 303):
            print('MFA verification failed'); sys.exit(1)
    elif resp.status_code not in (302, 303):
        print(f'Login failed'); sys.exit(1)
    if session.get(f'{host}/api/ingest/saved-labs').status_code == 401:
        print('Login failed'); sys.exit(1)
    print('Authenticated.\n')


def get_wav_secret():
    secret = os.environ.get('WAV_CACHE_SECRET', '').strip()
    if secret:
        return secret
    try:
        r = subprocess.run(['fly', 'ssh', 'console', '-a', 'seechords', '-C', 'printenv WAV_CACHE_SECRET'],
                           capture_output=True, text=True, timeout=30)
        secret = r.stdout.strip()
        if secret:
            return secret
    except Exception:
        pass
    return input('WAV_CACHE_SECRET: ').strip()


def download_audio(video_id, wav_path):
    print(f'  Downloading audio...', end=' ', flush=True)
    try:
        r = subprocess.run(
            ['yt-dlp', '-x', '--audio-format', 'wav',
             '--postprocessor-args', 'ffmpeg:-ac 1 -ar 44100',
             '-o', wav_path, '--no-playlist',
             f'https://www.youtube.com/watch?v={video_id}'],
            capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and os.path.isfile(wav_path):
            size = os.path.getsize(wav_path)
            if size > MAX_WAV_BYTES:
                print(f'TOO LARGE ({size // (1024*1024)}MB > {MAX_WAV_BYTES // (1024*1024)}MB) — skipping')
                return False
            print(f'OK ({size // 1024}KB)')
            return True
        print(f'FAILED')
        if r.stderr:
            print(f'    {r.stderr[:200]}')
        return False
    except subprocess.TimeoutExpired:
        print('TIMEOUT'); return False


def upload_wav(host, wav_secret, video_id, wav_path):
    print(f'  Uploading to cache...', end=' ', flush=True)
    with open(wav_path, 'rb') as f:
        data = f.read()
    resp = requests.put(f'{host}/api/internal/wav-cache/{video_id}',
                        headers={'Authorization': f'Bearer {wav_secret}', 'Content-Type': 'audio/wav'},
                        data=data, timeout=180)
    if resp.status_code == 200:
        print(f'OK ({len(data) // 1024}KB)')
        return True
    print(f'FAILED ({resp.status_code})')
    return False


def wait_for_job(session, host, job_id, timeout=1800):
    """Poll job status until done/error or timeout (default 30 min)."""
    deadline = time.time() + timeout
    consecutive_errors = 0
    while time.time() < deadline:
        time.sleep(10)
        try:
            resp = session.get(f'{host}/api/status/{job_id}', timeout=30)
            if not resp.ok:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    print(f'    Giving up after {consecutive_errors} consecutive poll errors')
                    return 'error'
                continue
            consecutive_errors = 0
            job = resp.json()
            status = job.get('status', '')
            msg = job.get('message', '')
            print(f'    {status}: {msg}')
            if status in ('done', 'error'):
                return status
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= 10:
                print(f'    Giving up after {consecutive_errors} consecutive poll errors: {e}')
                return 'error'
            continue
    print(f'    Timed out after {timeout}s waiting for job {job_id}')
    return 'error'


def main():
    parser = argparse.ArgumentParser(description='Re-analyze songs with local audio download')
    parser.add_argument('--host', default=DEFAULT_HOST)
    args = parser.parse_args()
    host = args.host.rstrip('/')

    session = requests.Session()
    login(session, host)

    wav_secret = get_wav_secret()
    if not wav_secret:
        print('No WAV_CACHE_SECRET'); sys.exit(1)
    print(f'Got WAV_CACHE_SECRET.\n')

    done = 0
    errors = 0
    MAX_RETRIES = 5
    attempts = {}  # video_id -> failed-attempt count
    skipped = set()  # video_ids that hit MAX_RETRIES

    while True:
        # 1. Peek at next song needing reanalysis (excluding songs we've given up on)
        exclude_param = ','.join(sorted(skipped))
        resp = session.get(f'{host}/api/ingest/reanalyze-peek', params={'exclude': exclude_param})
        if resp.status_code != 200:
            print(f'Peek failed: {resp.status_code}'); break
        peek = resp.json()
        if peek.get('remaining', 0) == 0:
            print(f'\nAll done! {done} succeeded, {errors} errors, {len(skipped)} skipped.')
            if skipped:
                print(f'Skipped: {sorted(skipped)}')
            break

        video_id = peek['videoId']
        title = peek.get('title', video_id)
        remaining = peek['remaining']

        attempt_num = attempts.get(video_id, 0) + 1
        # Counter shows unique-song progress: (songs_finished + 1) of (songs_finished + remaining)
        finished = done + len(skipped)
        print(f'[{finished + 1}/{finished + remaining}] {title} ({video_id}) [attempt {attempt_num}/{MAX_RETRIES}]')

        def _fail():
            attempts[video_id] = attempt_num
            if attempt_num >= MAX_RETRIES:
                print(f'  Giving up on {video_id} after {MAX_RETRIES} attempts — continuing with next song')
                skipped.add(video_id)

        # 2. Download audio locally
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, f'{video_id}.wav')
            if not download_audio(video_id, wav_path):
                errors += 1
                _fail()
                continue

            # 3. Upload WAV to API cache
            if not upload_wav(host, wav_secret, video_id, wav_path):
                errors += 1
                _fail()
                continue

        # 4-6: spawn worker, wait, clean up (always clean up WAV from Turso)
        try:
            # 4. Trigger worker with skip_download (retry if busy)
            while True:
                print(f'  Spawning worker...', end=' ', flush=True)
                resp = session.post(f'{host}/api/ingest/reanalyze-next',
                                    json={'skip_download': True},
                                    headers={'Content-Type': 'application/json'})
                if resp.status_code == 429:
                    print('worker busy, waiting 30s...')
                    time.sleep(30)
                    continue
                break
            if resp.status_code != 200:
                print(f'FAILED ({resp.status_code})')
                errors += 1
                continue
            data = resp.json()
            if not data.get('queued'):
                print('nothing to analyze')
                break
            print(f'OK')

            # 5. Wait for job to complete
            print(f'  Waiting for analysis...')
            result = wait_for_job(session, host, data['jobId'])
            if result == 'done':
                done += 1
            else:
                errors += 1
        finally:
            # 6. Clean up temp WAV from Turso (always, even on error)
            try:
                dr = requests.delete(f'{host}/api/internal/wav-cache/{video_id}',
                                     headers={'Authorization': f'Bearer {wav_secret}'}, timeout=30)
                if not dr.ok:
                    print(f'  Warning: WAV cache cleanup returned {dr.status_code}')
            except Exception as e:
                print(f'  Warning: WAV cache cleanup failed: {e}')

        # Small delay between songs
        time.sleep(3)

    print(f'\nFinished: {done} succeeded, {errors} errors.')


if __name__ == '__main__':
    main()
