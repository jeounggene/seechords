#!/usr/bin/env python3
"""Download McGill Billboard chord annotations and fetch matching audio from YouTube.

The McGill Billboard dataset (v2.0) contains expert chord annotations for ~890
Billboard Hot 100 songs. Annotations are CC0 (public domain). Audio is not
included and must be sourced separately (YouTube).

Workflow:
    1. Download annotations + metadata from McGill Billboard (Dropbox)
    2. Extract .lab files to data/billboard/labels/
    3. For each song, search YouTube and download audio to data/billboard/audio/

Usage:
    # Step 1: Download annotations only (no audio)
    python download_billboard.py --fetch-annotations

    # Step 2: List available songs (with/without audio)
    python download_billboard.py --list
    python download_billboard.py --missing    # songs without audio

    # Step 3: Download audio for a specific song (by track ID)
    python download_billboard.py 0010

    # Step 4: Download audio for a batch of songs
    python download_billboard.py --batch 20    # next 20 missing songs

    # Search for songs by artist or title
    python download_billboard.py --search "billy joel"

    # Download audio with a specific YouTube URL
    python download_billboard.py 0010 --url "https://www.youtube.com/watch?v=..."
"""

import argparse
import csv
import io
import json
import lzma
import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
BILLBOARD_DIR = TRAINING_DIR / "data" / "billboard"
AUDIO_DIR = BILLBOARD_DIR / "audio"
LABELS_DIR = BILLBOARD_DIR / "labels"
METADATA_PATH = BILLBOARD_DIR / "metadata.csv"
VIDEO_MAP_PATH = BILLBOARD_DIR / "video_map.json"

# McGill Billboard download URLs (from mirdata)
METADATA_URL = "https://www.dropbox.com/s/o0olz0uwl9z9stb/billboard-2.0-index.csv?dl=1"
LAB_URL = "https://www.dropbox.com/s/t390alzrkx0c9yt/billboard-2.0.1-lab.tar.gz?dl=1"

YTDLP = "yt-dlp"
FFMPEG = "ffmpeg"


# ── SSL context (macOS often needs this for Dropbox) ─────────

def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ── Helpers ──────────────────────────────────────────────────

def sanitize_filename(text):
    """Convert text to a safe filename."""
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    text = re.sub(r'[\s]+', '_', text.strip())
    text = re.sub(r'_+', '_', text).strip('_')
    return text


def load_metadata():
    """Load billboard metadata CSV. Returns dict of track_id -> info."""
    if not METADATA_PATH.exists():
        return {}
    meta = {}
    with open(METADATA_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row['id'].strip().zfill(4)
            if row['title'].strip():
                meta[tid] = {
                    'title': row['title'].strip(),
                    'artist': row['artist'].strip(),
                    'chart_date': row.get('chart_date', ''),
                    'peak_rank': row.get('peak_rank', ''),
                }
    return meta


def load_video_map():
    if VIDEO_MAP_PATH.exists():
        with open(VIDEO_MAP_PATH) as f:
            return json.load(f)
    return {}


def save_video_map(vmap):
    VIDEO_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(VIDEO_MAP_PATH) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(vmap, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(VIDEO_MAP_PATH))


def get_lab_tracks():
    """Return sorted list of track IDs that have .lab files."""
    if not LABELS_DIR.is_dir():
        return []
    return sorted(
        f.stem for f in LABELS_DIR.iterdir() if f.suffix == '.lab'
    )


def get_audio_tracks():
    """Return set of track IDs that have audio files."""
    if not AUDIO_DIR.is_dir():
        return set()
    audio_exts = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    return {f.stem for f in AUDIO_DIR.iterdir() if f.suffix.lower() in audio_exts}


# ── Fetch annotations from McGill Billboard ─────────────────

def cmd_fetch_annotations(args):
    """Download metadata CSV and .lab annotations from McGill Billboard."""
    ctx = _ssl_ctx()

    BILLBOARD_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download metadata CSV
    print("Downloading metadata CSV...")
    data = urllib.request.urlopen(METADATA_URL, context=ctx).read()
    with open(METADATA_PATH, 'wb') as f:
        f.write(data)
    meta = load_metadata()
    print(f"  Saved {len(meta)} song entries to {METADATA_PATH.name}")

    # 2. Download and extract .lab files
    print("Downloading chord annotations (.lab files)...")
    data = urllib.request.urlopen(LAB_URL, context=ctx).read()

    # Dropbox serves this as xz-compressed despite .tar.gz extension
    try:
        decompressed = lzma.decompress(data)
        tf = tarfile.open(fileobj=io.BytesIO(decompressed), mode='r:')
    except lzma.LZMAError:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:gz')

    extracted = 0
    for member in tf.getmembers():
        if member.name.endswith('/full.lab'):
            # Extract track ID from path like "McGill-Billboard/0003/full.lab"
            parts = member.name.split('/')
            if len(parts) >= 2:
                track_id = parts[-2].zfill(4)
                f = tf.extractfile(member)
                if f:
                    content = f.read()
                    lab_path = LABELS_DIR / f"{track_id}.lab"
                    with open(lab_path, 'wb') as out:
                        out.write(content)
                    extracted += 1

    print(f"  Extracted {extracted} .lab files to {LABELS_DIR}")
    print(f"\nDone! Run 'python download_billboard.py --list' to see available songs.")


# ── List / search ────────────────────────────────────────────

def cmd_list(args):
    """List all Billboard songs with annotation/audio status."""
    meta = load_metadata()
    lab_tracks = get_lab_tracks()
    audio_tracks = get_audio_tracks()

    if not lab_tracks:
        print("No annotations found. Run: python download_billboard.py --fetch-annotations")
        return

    missing_only = args.missing
    count = 0

    print(f"\n{'─' * 70}")
    print(f"  BILLBOARD DATASET ({len(lab_tracks)} annotations, {len(audio_tracks)} audio)")
    print(f"{'─' * 70}")

    for tid in lab_tracks:
        has_audio = tid in audio_tracks
        if missing_only and has_audio:
            continue
        info = meta.get(tid, {})
        title = info.get('title', '?')
        artist = info.get('artist', '?')
        status = "✓" if has_audio else " "
        print(f"  [{status}] {tid}  {artist} — {title}")
        count += 1

    if missing_only:
        print(f"\n  {count} songs missing audio (of {len(lab_tracks)} total)")
    else:
        print(f"\n  {len(audio_tracks)}/{len(lab_tracks)} have audio")


def cmd_search(args):
    """Search Billboard songs by artist or title."""
    meta = load_metadata()
    query = args.search.lower()
    audio_tracks = get_audio_tracks()

    results = []
    for tid, info in sorted(meta.items()):
        if query in info['title'].lower() or query in info['artist'].lower():
            results.append((tid, info))

    lab_tracks = set(get_lab_tracks())
    if not results:
        print(f"No songs matching '{args.search}'")
        return

    print(f"\nFound {len(results)} matching songs:\n")
    for tid, info in results:
        has_lab = "📝" if tid in lab_tracks else "  "
        has_audio = "🎵" if tid in audio_tracks else "  "
        print(f"  {has_lab}{has_audio} {tid}  {info['artist']} — {info['title']}")
    print(f"\n  📝 = has annotations, 🎵 = has audio")


# ── Download audio ───────────────────────────────────────────

def download_and_convert(url, output_wav):
    """Download audio from YouTube URL and convert to 44100 Hz mono WAV."""
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = os.path.join(tmpdir, "audio.%(ext)s")
        cmd = [
            YTDLP, "-x", "--audio-format", "best",
            "--no-playlist", "-o", tmp_out, url,
        ]
        print(f"  Downloading audio...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR downloading: {result.stderr.strip()}")
            return False

        downloaded = [f for f in Path(tmpdir).iterdir() if f.is_file()]
        if not downloaded:
            print("  ERROR: No audio file produced")
            return False

        src = downloaded[0]
        print(f"  Converting to WAV (44100 Hz mono)...")
        conv_cmd = [
            FFMPEG, "-i", str(src),
            "-vn", "-ar", "44100", "-ac", "1",
            str(output_wav), "-y",
        ]
        result = subprocess.run(conv_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR converting: {result.stderr.strip()}")
            return False

        if not output_wav.exists():
            print("  ERROR: WAV file not created")
            return False

        size_mb = output_wav.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_wav.name} ({size_mb:.1f} MB)")
        return True


def get_lab_end_time(track_id):
    """Return the end time (seconds) of the last entry in a .lab file."""
    lab_path = LABELS_DIR / f"{track_id}.lab"
    if not lab_path.exists():
        return None
    last_end = 0.0
    with open(lab_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    last_end = max(last_end, float(parts[1]))
                except ValueError:
                    pass
    return last_end if last_end > 0 else None


def get_wav_duration(wav_path):
    """Get duration of a WAV file in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip())
    return None


def validate_duration(track_id, wav_path, tolerance=10.0):
    """Check that audio duration roughly matches the .lab annotation span.

    Returns (ok, message) tuple.
    """
    lab_end = get_lab_end_time(track_id)
    if lab_end is None:
        return True, "no lab end time to compare"
    wav_dur = get_wav_duration(wav_path)
    if wav_dur is None:
        return False, "could not read WAV duration"
    diff = abs(wav_dur - lab_end)
    if diff <= tolerance:
        return True, f"audio {wav_dur:.1f}s vs label {lab_end:.1f}s (diff {diff:.1f}s OK)"
    else:
        return False, f"MISMATCH: audio {wav_dur:.1f}s vs label {lab_end:.1f}s (diff {diff:.1f}s > {tolerance:.0f}s)"


def youtube_search_url(artist, title):
    """Build a YouTube search URL for yt-dlp."""
    query = f"{artist} {title} official audio"
    return f"ytsearch1:{query}"


def cmd_download(args):
    """Download audio for one or more Billboard tracks."""
    meta = load_metadata()
    lab_tracks = set(get_lab_tracks())
    audio_tracks = get_audio_tracks()
    vmap = load_video_map()

    # Determine which tracks to process
    track_ids = []
    if args.track_id:
        tid = args.track_id.zfill(4)
        if tid not in lab_tracks:
            print(f"ERROR: Track {tid} has no .lab annotation.")
            print("Run --fetch-annotations first, or check --list for available tracks.")
            sys.exit(1)
        track_ids = [tid]
    elif args.batch:
        # Find next N tracks missing audio
        missing = [t for t in sorted(lab_tracks) if t not in audio_tracks]
        track_ids = missing[:args.batch]
        if not track_ids:
            print("All tracks already have audio!")
            return
        print(f"Batch: downloading audio for {len(track_ids)} tracks\n")

    downloaded = 0
    for i, tid in enumerate(track_ids, 1):
        info = meta.get(tid, {})
        title = info.get('title', f'Track {tid}')
        artist = info.get('artist', 'Unknown')
        print(f"[{i}/{len(track_ids)}] {tid}: {artist} — {title}")

        if tid in audio_tracks and not args.force:
            print(f"  Already has audio — skipping (use --force to re-download)\n")
            continue

        # Determine YouTube URL
        if args.url and len(track_ids) == 1:
            url = args.url
        else:
            url = youtube_search_url(artist, title)

        output_wav = AUDIO_DIR / f"{tid}.wav"
        if download_and_convert(url, output_wav):
            ok, msg = validate_duration(tid, output_wav)
            if ok:
                print(f"  Duration check: {msg}")
            else:
                print(f"  ⚠ Duration check: {msg}")
            vmap[tid] = {
                'title': title,
                'artist': artist,
            }
            save_video_map(vmap)
            downloaded += 1
            print()
        else:
            print(f"  FAILED\n")

    print(f"Done. Downloaded: {downloaded}/{len(track_ids)}")
    print(f"Audio: {AUDIO_DIR}")
    print(f"Labels: {LABELS_DIR}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download McGill Billboard chord annotations + audio"
    )
    parser.add_argument("track_id", nargs="?",
                        help="Billboard track ID to download audio for (e.g. 0010)")
    parser.add_argument("--fetch-annotations", action="store_true",
                        help="Download chord annotations + metadata from McGill")
    parser.add_argument("--list", action="store_true",
                        help="List all songs with annotation/audio status")
    parser.add_argument("--missing", action="store_true",
                        help="List songs missing audio")
    parser.add_argument("--search", type=str,
                        help="Search songs by artist or title")
    parser.add_argument("--batch", type=int,
                        help="Download audio for next N songs missing audio")
    parser.add_argument("--url", type=str,
                        help="YouTube URL for audio (single track only)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download existing audio")

    args = parser.parse_args()

    # Check dependencies for audio download
    if args.track_id or args.batch:
        if not shutil.which(YTDLP):
            print(f"ERROR: {YTDLP} not found. Install: pip install yt-dlp")
            sys.exit(1)
        if not shutil.which(FFMPEG):
            print(f"ERROR: {FFMPEG} not found. Install: brew install ffmpeg")
            sys.exit(1)

    if args.fetch_annotations:
        cmd_fetch_annotations(args)
    elif args.list or args.missing:
        cmd_list(args)
    elif args.search:
        cmd_search(args)
    elif args.track_id or args.batch:
        cmd_download(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
