#!/usr/bin/env python3
"""Download songs from YouTube for training / verified data.

Downloads audio from a single video or a playlist, converts to 44100 Hz mono
WAV, and saves to both:
  - server/verified/audio/        (served by the extension)
  - training/data/verified/audio/  (used for training)

Usage:
    # Download a single video
    python download_songs.py "https://www.youtube.com/watch?v=VIDEO_ID"

    # Download a full playlist
    python download_songs.py --playlist "https://www.youtube.com/playlist?list=PLxxx"

    # Set a custom output name
    python download_songs.py --name "Artist - Song Title" URL

    # List all downloaded songs
    python download_songs.py --list

    # Skip first N tracks in a playlist
    python download_songs.py --playlist --skip 3 URL
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent          # training/
SEECHORDS_DIR = TRAINING_DIR.parent       # seechords/
SERVER_DIR = SEECHORDS_DIR / "server"

SERVER_VERIFIED_DIR = SERVER_DIR / "verified"
TRAINING_VERIFIED_DIR = TRAINING_DIR / "data" / "verified"

YTDLP = "yt-dlp"
FFMPEG = "ffmpeg"


# ── Helpers ───────────────────────────────────────────────────

def sanitize_filename(title):
    """Convert a video title to a safe filesystem-friendly name."""
    # Remove common YouTube noise
    title = re.sub(r"\s*\(Official\s*(Music\s*)?Video\)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(Official\s*Audio\)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(Lyric\s*Video\)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(Lyrics?\)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\[Official\s*(Music\s*)?Video\]", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\[Official\s*Audio\]", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(Audio\)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(Visualizer\)", "", title, flags=re.IGNORECASE)
    # Replace characters not safe in filenames
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces and special chars with underscores
    title = re.sub(r'[\s]+', '_', title.strip())
    # Collapse multiple underscores
    title = re.sub(r'_+', '_', title).strip('_')
    return title


def load_video_map(map_path):
    """Load an existing video_map.json or return empty dict."""
    if map_path.exists():
        with open(map_path, 'r') as f:
            return json.load(f)
    return {}


def save_video_map(map_path, data):
    """Write video_map.json atomically."""
    map_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(map_path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(map_path))


def get_video_info(url, playlist=False):
    """Fetch metadata for a URL. Returns list of entry dicts."""
    cmd = [YTDLP, "--dump-json", "--no-warnings"]
    if playlist:
        cmd.append("--flat-playlist")
    else:
        cmd.append("--no-playlist")
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"yt-dlp error: {result.stderr.strip()}")
        sys.exit(1)
    entries = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def download_and_convert(url, output_wav):
    """Download audio from YouTube URL and convert to 44100 Hz mono WAV."""
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download best audio (keep original format for quality)
        tmp_out = os.path.join(tmpdir, "audio.%(ext)s")
        cmd = [
            YTDLP,
            "-x", "--audio-format", "best",
            "--no-playlist",
            "-o", tmp_out,
            url,
        ]
        print(f"  Downloading audio...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR downloading: {result.stderr.strip()}")
            return False

        # Find the downloaded file (could be m4a, opus, webm, etc.)
        downloaded = [f for f in Path(tmpdir).iterdir() if f.is_file()]
        if not downloaded:
            print("  ERROR: No audio file produced by yt-dlp")
            return False

        src = downloaded[0]

        # Convert to 44100 Hz mono WAV with ffmpeg
        print(f"  Converting to WAV (44100 Hz mono)...")
        conv_cmd = [
            FFMPEG, "-i", str(src),
            "-vn",              # no video
            "-ar", "44100",     # 44.1 kHz sample rate
            "-ac", "1",         # mono
            str(output_wav),
            "-y",               # overwrite
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


def already_downloaded(video_id, video_map):
    """Check if a video ID is already in the video_map."""
    for stem, info in video_map.items():
        if info.get("videoId") == video_id:
            return stem
    return None


# ── Commands ──────────────────────────────────────────────────

def cmd_download(args):
    """Download one or more songs."""
    url = args.url

    # Both directories get audio + video_map updates
    output_dirs = [
        (SERVER_VERIFIED_DIR / "audio", SERVER_VERIFIED_DIR / "video_map.json"),
        (TRAINING_VERIFIED_DIR / "audio", TRAINING_VERIFIED_DIR / "video_map.json"),
    ]

    # Use server verified map as the primary for duplicate checking
    primary_map_path = output_dirs[0][1]
    video_map = load_video_map(primary_map_path)

    # Fetch info
    if args.playlist:
        print("Fetching playlist info...")
        entries = get_video_info(url, playlist=True)
        print(f"Playlist: {len(entries)} tracks\n")
        if args.skip:
            entries = entries[args.skip:]
            print(f"Skipping first {args.skip} tracks.\n")
    else:
        print("Fetching video info...")
        entries = get_video_info(url, playlist=False)

    downloaded = 0
    skipped = 0
    total = len(entries)

    for i, entry in enumerate(entries, 1):
        title = entry.get("title", "Unknown")
        video_id = entry.get("id", "")
        video_url = (
            entry.get("url")
            or entry.get("webpage_url")
            or f"https://www.youtube.com/watch?v={video_id}"
        )

        # Use custom name for single videos if provided
        if args.name and total == 1:
            safe_name = sanitize_filename(args.name)
        else:
            safe_name = sanitize_filename(title)

        print(f"[{i}/{total}] {title}")

        # Check if already downloaded
        existing = already_downloaded(video_id, video_map)
        if existing and not args.force:
            print(f"  Already downloaded as: {existing}.wav — skipping.\n")
            skipped += 1
            continue

        # Download to a temp location, then copy to both dirs
        primary_audio_dir = output_dirs[0][0]
        output_wav = primary_audio_dir / f"{safe_name}.wav"

        if output_wav.exists() and not args.force:
            print(f"  File already exists: {safe_name}.wav — skipping.\n")
            skipped += 1
            continue

        if download_and_convert(video_url, output_wav):
            # Copy to training dir as well
            training_audio_dir = output_dirs[1][0]
            training_audio_dir.mkdir(parents=True, exist_ok=True)
            training_wav = training_audio_dir / f"{safe_name}.wav"
            if not training_wav.exists():
                import shutil
                shutil.copy2(str(output_wav), str(training_wav))

            # Update both video_maps
            video_map[safe_name] = {
                "videoId": video_id,
                "title": title,
            }
            for _, map_path in output_dirs:
                vmap = load_video_map(map_path)
                vmap[safe_name] = {"videoId": video_id, "title": title}
                save_video_map(map_path, vmap)
            downloaded += 1
            print()
        else:
            print(f"  FAILED — skipping.\n")

    print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}")
    print(f"Audio saved to: server/verified/audio/ + training/data/verified/audio/")


def cmd_list(args):
    """List all downloaded songs."""
    for tier, base_dir in [
        ("server/verified", SERVER_VERIFIED_DIR),
        ("training/verified", TRAINING_VERIFIED_DIR),
    ]:
        audio_dir = base_dir / "audio"
        map_path = base_dir / "video_map.json"
        video_map = load_video_map(map_path)
        wavs = sorted(audio_dir.glob("*.wav")) if audio_dir.is_dir() else []
        print(f"\n{'─' * 60}")
        print(f"  {tier.upper()} ({len(wavs)} files)")
        print(f"{'─' * 60}")
        if not wavs:
            print("  (none)")
            continue
        for wav in wavs:
            stem = wav.stem
            info = video_map.get(stem, {})
            vid = info.get("videoId", "?")
            title = info.get("title", stem.replace("_", " "))
            size_mb = wav.stat().st_size / (1024 * 1024)
            print(f"  {title}")
            print(f"    {stem}.wav  ({size_mb:.1f} MB)  [vid:{vid}]")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download songs from YouTube for SeeChords training"
    )
    parser.add_argument("url", nargs="?", help="YouTube URL (video or playlist)")
    parser.add_argument("--name", help="Custom name for the download (single video only)")
    parser.add_argument("--playlist", action="store_true", help="Treat URL as a playlist")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N tracks in a playlist")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--list", action="store_true", help="List all downloaded songs")

    args = parser.parse_args()

    # Check dependencies
    if not args.list:
        if not shutil.which(YTDLP):
            print(f"ERROR: {YTDLP} not found. Install with: pip install yt-dlp")
            sys.exit(1)
        if not shutil.which(FFMPEG):
            print(f"ERROR: {FFMPEG} not found. Install with: brew install ffmpeg")
            sys.exit(1)

    if args.list:
        cmd_list(args)
    elif args.url:
        cmd_download(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
