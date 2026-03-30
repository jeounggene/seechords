#!/usr/bin/env python3
"""Download Beatles songs from YouTube and place them in the training audio directory.

Usage:
    # Interactive: download and fuzzy-match to annotation stems
    python download_beatles.py URL

    # Download a full album playlist (auto-matches by track order)
    python download_beatles.py --playlist --album "01_-_Please_Please_Me" URL

    # Download a playlist with interactive matching
    python download_beatles.py --playlist URL

    # List all songs still missing audio
    python download_beatles.py --missing

    # List available album directory names
    python download_beatles.py --albums

    # Skip fuzzy match, manually specify the target stem
    python download_beatles.py URL --stem "01_-_Please_Please_Me/14_-_Twist_And_Shout"
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent  # training/ root
SEECHORDS_DIR = TRAINING_DIR.parent
AUDIO_DIR = TRAINING_DIR / "data" / "beatles" / "audio"
ANNOTATIONS_ROOT = TRAINING_DIR / "data" / "beatles" / "annotations" / "chordlab" / "The Beatles"

YTDLP = "yt-dlp"


# ── Discover annotation stems ────────────────────────────────

def get_annotation_stems():
    """Return sorted list of 'ALBUM/SONG' stems from the Isophonics chord annotations."""
    stems = []
    if not ANNOTATIONS_ROOT.is_dir():
        print(f"ERROR: Annotations directory not found: {ANNOTATIONS_ROOT}")
        sys.exit(1)
    for album_dir in sorted(ANNOTATIONS_ROOT.iterdir()):
        if not album_dir.is_dir():
            continue
        for lab_file in sorted(album_dir.iterdir()):
            if lab_file.suffix == ".lab":
                stem = f"{album_dir.name}/{lab_file.stem}"
                stems.append(stem)
    return stems


def get_missing_stems():
    """Return annotation stems that don't yet have a corresponding .mp3."""
    stems = get_annotation_stems()
    missing = []
    for stem in stems:
        mp3_path = AUDIO_DIR / f"{stem}.mp3"
        if not mp3_path.exists():
            missing.append(stem)
    return missing


def get_album_dirs():
    """Return sorted list of album directory names."""
    if not ANNOTATIONS_ROOT.is_dir():
        return []
    return sorted(d.name for d in ANNOTATIONS_ROOT.iterdir() if d.is_dir())


def get_album_stems(album_dir):
    """Return sorted stems for a specific album directory."""
    stems = get_annotation_stems()
    return [s for s in stems if s.startswith(album_dir + "/")]


def fuzzy_match_album(query):
    """Find the best matching album directory name for a query string."""
    albums = get_album_dirs()
    cleaned = clean_title(query)
    scored = []
    for album in albums:
        album_clean = clean_title(re.sub(r"^\d+_-_", "", album).replace("_", " "))
        score = SequenceMatcher(None, cleaned, album_clean).ratio()
        if album_clean in cleaned or cleaned in album_clean:
            score = min(1.0, score + 0.3)
        scored.append((score, album))
    scored.sort(key=lambda x: -x[0])
    return scored


# ── Song name extraction ─────────────────────────────────────

def clean_title(title):
    """Normalise a title string for matching: lowercase, strip punctuation, collapse spaces."""
    title = title.lower()
    # Remove common YouTube noise
    for noise in ["(remastered", "(stereo", "(mono", "(official", "(lyrics",
                  "(audio", "remaster", "2009", "2015", "hd", "official video",
                  "music video", "lyric video"]:
        title = title.replace(noise, "")
    # Remove parentheticals and brackets
    title = re.sub(r"[\(\)\[\]]", " ", title)
    # Remove "the beatles" prefix
    title = re.sub(r"\bthe\s+beatles\b", "", title)
    # Strip track numbers like "01 - ", "1. ", etc.
    title = re.sub(r"^\s*\d+[\.\-\s]+", "", title)
    # Normalise punctuation and whitespace
    title = re.sub(r"[^a-z0-9 ]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def stem_to_title(stem):
    """Extract a clean song title from an annotation stem like '01_-_Album/03_-_Song_Name'."""
    song_part = stem.split("/")[-1]
    # Remove track prefix like '03_-_'
    song_part = re.sub(r"^\d+_-_", "", song_part)
    # Remove CD prefix for White Album: 'CD1_-_03_-_' → '03_-_'
    song_part = re.sub(r"^CD\d+_-_\d+_-_", "", song_part)
    # Replace underscores with spaces
    song_part = song_part.replace("_", " ")
    return song_part


def fuzzy_match(title, stems, n=5):
    """Return top-n (score, stem) matches for a title against annotation stems."""
    cleaned = clean_title(title)
    scored = []
    for stem in stems:
        stem_clean = clean_title(stem_to_title(stem))
        # SequenceMatcher ratio
        score = SequenceMatcher(None, cleaned, stem_clean).ratio()
        # Boost if the stem title appears as a substring of the input
        if stem_clean in cleaned or cleaned in stem_clean:
            score = min(1.0, score + 0.3)
        scored.append((score, stem))
    scored.sort(key=lambda x: -x[0])
    return scored[:n]


# ── yt-dlp helpers ────────────────────────────────────────────

def get_video_info(url):
    """Fetch title and other metadata for a URL (works for single videos and playlists)."""
    cmd = [YTDLP, "--dump-json", "--flat-playlist", "--no-warnings", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"yt-dlp error: {result.stderr.strip()}")
        sys.exit(1)
    entries = []
    for line in result.stdout.strip().split("\n"):
        if line:
            entries.append(json.loads(line))
    return entries


def download_audio(url, output_path):
    """Download audio from a URL and save as mp3."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = os.path.join(tmpdir, "audio.%(ext)s")
        cmd = [
            YTDLP,
            "-x", "--audio-format", "mp3", "--audio-quality", "0",
            "--no-playlist",
            "-o", tmp_out,
            url,
        ]
        print(f"  Downloading audio...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")
            return False

        # Find the downloaded mp3
        mp3s = list(Path(tmpdir).glob("*.mp3"))
        if not mp3s:
            print("  ERROR: No mp3 file produced by yt-dlp")
            return False

        # Move to final destination
        import shutil
        shutil.move(str(mp3s[0]), str(output_path))
        print(f"  Saved: {output_path.relative_to(TRAINING_DIR)}")
        return True


# ── Interactive matching ──────────────────────────────────────

def interactive_match(title, url, missing_stems):
    """Fuzzy-match a video title and let user confirm/pick the target stem."""
    matches = fuzzy_match(title, missing_stems, n=8)
    top_score, top_stem = matches[0]

    print(f"\n  Video: {title}")
    print(f"  Top match ({top_score:.0%}): {stem_to_title(top_stem)}")
    print()

    # If high confidence, ask for confirmation
    if top_score >= 0.75:
        print(f"  [1] {top_stem}  ({top_score:.0%})  ← best match")
        for i, (score, stem) in enumerate(matches[1:], 2):
            print(f"  [{i}] {stem}  ({score:.0%})")
        print(f"  [s] Skip this track")
        print(f"  [m] Enter stem manually")
        choice = input("\n  Pick [1]: ").strip() or "1"
    else:
        print("  No confident match. Top candidates:")
        for i, (score, stem) in enumerate(matches, 1):
            print(f"  [{i}] {stem}  ({score:.0%})")
        print(f"  [s] Skip this track")
        print(f"  [m] Enter stem manually")
        choice = input("\n  Pick: ").strip()

    if choice.lower() == "s":
        return None
    elif choice.lower() == "m":
        stem = input("  Enter exact stem (ALBUM/SONG): ").strip()
        return stem if stem else None
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(matches):
                return matches[idx][1]
        except ValueError:
            pass
        print("  Invalid choice, skipping.")
        return None


# ── Main ──────────────────────────────────────────────────────

def cmd_missing(args):
    """List songs that are missing audio files."""
    missing = get_missing_stems()
    if not missing:
        print("All annotation stems have audio files.")
        return
    print(f"{len(missing)} songs missing audio:\n")
    current_album = None
    for stem in missing:
        album = stem.split("/")[0]
        if album != current_album:
            current_album = album
            # Clean album name
            album_name = re.sub(r"^\d+_-_", "", album).replace("_", " ")
            print(f"\n  {album_name}:")
        song = stem_to_title(stem)
        print(f"    {song}")


def cmd_download(args):
    """Download one or more songs from a YouTube URL."""
    url = args.url
    all_stems = get_annotation_stems()

    if args.stem:
        # Direct stem specification, no fuzzy matching
        stem = args.stem
        if stem not in all_stems:
            print(f"ERROR: stem '{stem}' not found in annotations.")
            print("Use --missing to see available stems.")
            return
        output = AUDIO_DIR / f"{stem}.mp3"
        if output.exists() and not args.force:
            print(f"Already exists: {output.relative_to(TRAINING_DIR)}")
            return
        download_audio(url, output)
        return

    # Fetch video info
    print("Fetching video info...")
    entries = get_video_info(url)

    if args.playlist:
        print(f"Playlist has {len(entries)} entries.\n")

        # If --album specified, auto-match by track order within the album
        if args.album:
            album_dir = args.album
            # Try fuzzy match if exact dir doesn't exist
            if album_dir not in get_album_dirs():
                matches = fuzzy_match_album(album_dir)
                if matches and matches[0][0] > 0.5:
                    album_dir = matches[0][1]
                    album_name = re.sub(r"^\d+_-_", "", album_dir).replace("_", " ")
                    print(f"Matched album: {album_name} ({album_dir})\n")
                else:
                    print(f"ERROR: Could not match album '{args.album}'")
                    print("Use --albums to see available album names.")
                    return

            album_stems = get_album_stems(album_dir)
            skip = args.skip or 0
            if skip:
                entries = entries[skip:]
                print(f"Skipping first {skip} playlist tracks.\n")
            if len(entries) != len(album_stems):
                print(f"WARNING: Playlist has {len(entries)} tracks but album has {len(album_stems)} songs.")
                print("Will match by position up to the shorter list.\n")

            downloaded = 0
            skipped = 0
            n = min(len(entries), len(album_stems))
            for i in range(n):
                entry = entries[i]
                stem = album_stems[i]
                title = entry.get("title", "Unknown")
                video_url = entry.get("url") or entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry.get('id', '')}"

                output = AUDIO_DIR / f"{stem}.mp3"
                song_name = stem_to_title(stem)
                print(f"  [{i+1}/{n}] {title}")
                print(f"       → {song_name}")

                if output.exists() and not args.force:
                    print(f"       Already exists, skipping.\n")
                    skipped += 1
                    continue

                if download_audio(video_url, output):
                    downloaded += 1
                    print()

            print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}")
            remaining = len(get_missing_stems())
            print(f"Still missing: {remaining}/180 songs")
            return

        # No --album: interactive matching per track
        missing_stems = get_missing_stems()
        downloaded = 0
        skipped = 0
        for entry in entries:
            title = entry.get("title", "Unknown")
            video_url = entry.get("url") or entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry.get('id', '')}"

            stem = interactive_match(title, video_url, missing_stems)
            if stem is None:
                skipped += 1
                continue

            output = AUDIO_DIR / f"{stem}.mp3"
            if output.exists() and not args.force:
                print(f"  Already exists, skipping: {output.relative_to(TRAINING_DIR)}")
                skipped += 1
                continue

            if download_audio(video_url, output):
                downloaded += 1
                if stem in missing_stems:
                    missing_stems.remove(stem)

        print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}")
        remaining = len(get_missing_stems())
        print(f"Still missing: {remaining}/180 songs")

    else:
        # Single video
        missing_stems = get_missing_stems()
        if len(entries) == 0:
            print("ERROR: Could not fetch video info.")
            return
        entry = entries[0]
        title = entry.get("title", "Unknown")

        stem = interactive_match(title, url, missing_stems)
        if stem is None:
            print("Skipped.")
            return

        output = AUDIO_DIR / f"{stem}.mp3"
        if output.exists() and not args.force:
            print(f"Already exists: {output.relative_to(TRAINING_DIR)}")
            return

        download_audio(url, output)
        remaining = len(get_missing_stems())
        print(f"\nStill missing: {remaining}/180 songs")


def main():
    parser = argparse.ArgumentParser(description="Download Beatles audio for training")
    parser.add_argument("url", nargs="?", help="YouTube URL (video or playlist)")
    parser.add_argument("--stem", help="Exact annotation stem (skip fuzzy matching)")
    parser.add_argument("--album", help="Album directory name for playlist auto-matching by track order")
    parser.add_argument("--playlist", action="store_true", help="Treat URL as a playlist")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N tracks in playlist")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--missing", action="store_true", help="List songs missing audio")
    parser.add_argument("--albums", action="store_true", help="List available album directory names")

    args = parser.parse_args()

    if args.albums:
        print("Available album directories:\n")
        for album in get_album_dirs():
            album_name = re.sub(r"^\d+_-_", "", album).replace("_", " ")
            n_stems = len(get_album_stems(album))
            print(f"  {album}  ({album_name}, {n_stems} tracks)")
        return

    if args.missing:
        cmd_missing(args)
    elif args.url:
        cmd_download(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
