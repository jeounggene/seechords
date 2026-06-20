#!/usr/bin/env python3
"""Re-analyze all songs missing downbeats (or all songs) in batches of 5.

Usage:
    python scripts/reanalyze-batch.py
    python scripts/reanalyze-batch.py --all        # force re-analyze everything
    python scripts/reanalyze-batch.py --host http://localhost:5000

Authenticates via the ingest password, then calls /api/ingest/reanalyze-batch
repeatedly until remaining=0.
"""

import argparse
import getpass
import sys
import time

import requests

DEFAULT_HOST = 'https://seechords.fly.dev'


def main():
    parser = argparse.ArgumentParser(description='Batch re-analyze songs on SeeChords')
    parser.add_argument('--host', default=DEFAULT_HOST, help=f'Server URL (default: {DEFAULT_HOST})')
    parser.add_argument('--all', action='store_true', help='Re-analyze ALL songs, not just those missing downbeats')
    parser.add_argument('--batch-size', type=int, default=5, help='Songs per batch (default: 5)')
    args = parser.parse_args()

    host = args.host.rstrip('/')
    password = getpass.getpass('Ingest password: ')

    session = requests.Session()

    # Step 1: Submit password
    resp = session.post(f'{host}/ingest/login', data={'password': password}, allow_redirects=False)

    # Check if MFA is required (server returns 200 with MFA form instead of 302 redirect)
    if resp.status_code == 200 and 'totp_code' in resp.text:
        totp_code = input('MFA code: ')
        resp = session.post(f'{host}/ingest/login', data={'totp_code': totp_code}, allow_redirects=False)
        if resp.status_code not in (302, 303):
            print('MFA verification failed')
            sys.exit(1)
    elif resp.status_code not in (302, 303):
        print(f'Login failed (HTTP {resp.status_code})')
        sys.exit(1)

    # Verify auth by hitting an API endpoint
    check = session.get(f'{host}/api/ingest/saved-labs')
    if check.status_code == 401:
        print('Login failed: unauthorized')
        sys.exit(1)

    print(f'Authenticated. Starting batch re-analysis (batch size: {args.batch_size})...\n')

    batch_num = 0
    total_queued = 0

    while True:
        batch_num += 1
        payload = {'batch_size': args.batch_size}
        if args.all:
            payload['all'] = '1'

        resp = session.post(f'{host}/api/ingest/reanalyze-batch', json=payload)
        if resp.status_code != 200:
            print(f'Error (HTTP {resp.status_code}): {resp.text[:300]}')
            sys.exit(1)

        data = resp.json()
        queued = data.get('queued', 0)
        remaining = data.get('remaining', 0)
        total_queued += queued

        print(f'Batch {batch_num}: queued {queued}, remaining {remaining}')

        if remaining == 0:
            break

        # Wait a bit between batches to avoid overwhelming the worker pool
        time.sleep(3)

    print(f'\nDone! Queued {total_queued} songs for re-analysis.')


if __name__ == '__main__':
    main()
