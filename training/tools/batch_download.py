#!/usr/bin/env python3
"""Batch download Billboard audio from a JSON mapping file.

Reads /tmp/billboard_batch.json and downloads each track using download_billboard.py.
Logs results to /tmp/billboard_batch_results.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DOWNLOAD_SCRIPT = SCRIPT_DIR / "download_billboard.py"
BATCH_FILE = "/tmp/billboard_batch.json"
RESULTS_FILE = "/tmp/billboard_batch_results.json"


def main():
    with open(BATCH_FILE) as f:
        batch = json.load(f)

    # Load existing results to support resume
    results = []
    completed_ids = set()
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
            completed_ids = {r["track_id"] for r in results if r.get("status") == "ok"}

    remaining = [e for e in batch if e["track_id"] not in completed_ids]
    print(f"Batch: {len(batch)} total, {len(completed_ids)} already done, {len(remaining)} to download\n")

    for i, entry in enumerate(remaining, 1):
        tid = entry["track_id"]
        name = entry["name"]
        url = entry["url"]

        print(f"[{i}/{len(remaining)}] {tid}: {name}")
        print(f"  URL: {url}")

        # Skip playlist URLs
        if "playlist?" in url:
            print(f"  SKIPPED: playlist URL (not a single video)\n")
            results.append({"track_id": tid, "name": name, "url": url, "status": "skipped", "reason": "playlist_url"})
            _save_results(results)
            continue

        try:
            result = subprocess.run(
                [sys.executable, str(DOWNLOAD_SCRIPT), tid, "--url", url],
                capture_output=True, text=True, timeout=300
            )
            output = result.stdout + result.stderr
            print(output.strip())

            if result.returncode == 0:
                # Parse duration info from output
                duration_info = ""
                for line in output.split("\n"):
                    if "Duration check:" in line:
                        duration_info = line.strip()
                results.append({"track_id": tid, "name": name, "url": url, "status": "ok", "duration": duration_info})
            else:
                results.append({"track_id": tid, "name": name, "url": url, "status": "failed", "output": output[-500:]})

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (300s)")
            results.append({"track_id": tid, "name": name, "url": url, "status": "timeout"})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"track_id": tid, "name": name, "url": url, "status": "error", "error": str(e)})

        _save_results(results)
        print()

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    timeout = sum(1 for r in results if r["status"] == "timeout")
    print(f"\n{'='*60}")
    print(f"DONE: {ok} ok, {failed} failed, {skipped} skipped, {timeout} timeout")
    print(f"Results: {RESULTS_FILE}")


def _save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
