#!/usr/bin/env python3
"""Parse URL text file, match songs to track IDs, create batch download JSON.

Uses positional matching: both the text file and track_ids list are in the same
score-descending order. Walk both lists in parallel, using fuzzy name matching
to handle metadata discrepancies.
"""

import json
import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "billboard"


# The ordered song names as shown to the user (matching training_subset.json order).
# These are the "display names" from analyze_billboard.py that the user used to find URLs.
# Positions 25-149 (songs #26-#150).
DISPLAY_NAMES = {
    "0366": "Chicago - Feelin' Stronger Every Day",
    "1273": "Cliff Richard - We Don't Talk Anymore",
    "0553": "Michael Sembello - Maniac",
    "0889": "Steve Winwood - Don't You Know What The Night Can Do?",
    "0004": "Roberta Flack, Donny Hathaway - You've Got A Friend",
    "0856": "Hi-Five - I Like The Way (The Kissing Game)",
    "0518": "Crosby, Stills & Nash - Suite: Judy Blue Eyes",
    "0501": "Donna Summer - Last Dance",
    "1078": "Switch - There'll Never Be",
    "0511": "Bertha Tillman - Oh My Angel",
    "0811": "Jermaine Jackson - Let's Get Serious",
    "0380": "Michael Johnson - Almost Like Being In Love",
    "1197": "Stray Cats - I Won't Stand In Your Way",
    "1200": "The Doobie Brothers - Echoes Of Love",
    "0588": "Queen - We Are The Champions",
    "0708": "Daryl Hall & John Oates - Sara Smile",
    "0540": "Pat Benatar - Heartbreaker",
    "0317": "Daryl Hall & John Oates - Rich Girl",
    "0547": "Michael Johnson - Almost Like Being In Love",
    "0077": "Johnny Tillotson - Jimmy's Girl",
    "0925": "The Ritchie Family - The Best Disco In Town",
    "1127": "The 5th Dimension - One Less Bell To Answer",
    "0650": "Louis Armstrong - Hello, Dolly!",
    "0206": "Paul McCartney - Maybe I'm Amazed",
    "0834": "Oliver - Good Morning Starshine",
    "0927": "Little Anthony & The Imperials - Out Of Sight, Out Of Mind",
    "0091": "Ray Price - For The Good Times",
    "1003": "Player - Baby Come Back",
    "0102": "Squeeze - Keep Bright Dreams",
    "0705": "Gladys Knight & The Pips - Baby Don't Change Your Mind",
    "1253": "Brenda Lee - All Alone Am I",
    "0507": "Commodores - Still",
    "0326": "The 5th Dimension - (Last Night) I Didn't Get To Sleep At All",
    "0735": "Heart - Getaway",
    "0722": "Dinah Washington - Where Are You",
    "1048": "B.J. Thomas - Hooked On A Feeling",
    "0813": "Lesley Gore - California Nights",
    "1061": "Eagles - Best Of My Love",
    "0407": "Aretha Franklin - I Never Loved A Man (The Way I Love You)",
    "1161": "Cliff Richard - Daddy's Home",
    "1064": "Robert John - Sad Eyes",
    "0265": "The 5th Dimension - Go Where You Wanna Go",
    "0071": "Meat Loaf - Paradise By The Dashboard Light",
    "0049": "James Brown - Get Up (I Feel Like Being Like A) Sex Machine (Part 1)",
    "0884": "Ann Peebles - I Can't Stand The Rain",
    "0122": "Peter Gabriel - Shock The Monkey",
    "0306": "Village People - In The Navy",
    "0241": "Bing Crosby - White Christmas",
    "0427": "Steppenwolf - Born To Be Wild",
    "0205": "Gloria Gaynor - Never Can Say Goodbye",
    "0434": "Dion - Where Or When",
    "0242": "Elvis Presley - There Goes My Everything",
    "0251": "Brother Jack McDuff - Theme From Electric Surfboard",
    "0293": "Chicago - Does Anybody Really Know What Time It Is?",
    "0637": "James Brown - Think",
    "0348": "Four Tops - Standing In The Shadows Of Love",
    "0956": "Boston - Don't Look Back",
    "1192": "Four Tops - Standing In The Shadows Of Love",
    "0371": "Brenda Lee - Losing You",
    "1286": "A Taste Of Honey - Sukiyaki",
    "1276": "Milli Vanilli - Baby Don't Forget My Number",
    "0370": "Brenda Lee - Losing You",
    "1167": "Billy Idol - Rebel Yell",
    "0199": "Solomon Burke - If You Need Me",
    "0658": "Elvis Presley - Ask Me",
    "0578": "Canned Heat - Let's Work Together",
    "1082": "Alabama - Take Me Down",
    "1012": "Sting - All This Time",
    "1027": "Greg Kihn - Lucky",
    "0691": "Aaron Neville - Tell It Like It Is",
    "0192": "Dinah Washington - Unforgettable",
    "0670": "Queensryche - Silent Lucidity",
    "0633": "Elvis Presley - If You Really Love Me",
    "0421": "George Harrison - Blow Away",
    "0515": "Etta James - Fool That I Am",
    "1140": "Madonna - Oh Father",
    "1037": "Gene Pitney - Town Without Pity",
    "0267": "Johnny Tillotson - Worried Guy",
    "1170": "Blue Cheer - Summertime Blues",
    "0414": "Jefferson Starship - Count On Me",
    "1146": "Dion - Abraham, Martin And John",
    "0382": "Todd Rundgren - Can We Still Be Friends",
    "0746": "Louis Prima & Keely Smith - That Old Black Magic",
    "1221": "Chicago - Alive Again",
    "0410": "Johnny Tillotson - Jimmy's Girl",
    "1121": "Simon & Garfunkel - A Hazy Shade Of Winter",
    "0627": "Wilson Pickett - I'm In Love",
    "0528": "Tommy James - Mony Mony",
    "0543": "Billy Preston - With You I'm Born Again",
    "1289": "The La's - There She Goes",
    "1055": "Elvis Presley - My Wish Came True",
    "1114": "Glen Campbell - Wichita Lineman",
    "0097": "Bananarama - A Trick Of The Night",
    "0127": "Gino Vannelli - Hurts To Be In Love",
    "0618": "Dean Martin - Everybody Loves Somebody",
    "1145": "Tevin Campbell - Tell Me What You Want Me To Do",
    "0775": "The Kendalls - Heaven's Just A Sin Away",
    "0385": "Todd Rundgren - Can We Still Be Friends",
    "0599": "Billy Swan - I Can Help",
    "0157": "Jimi Hendrix - Freedom",
    "1149": "Count Basie - Going To Chicago Blues",
    "0442": "Grand Funk Railroad - Walk Like A Man",
    "0258": "Firehouse - Don't Treat Me Bad",
    "0349": "Tina Turner - The Best",
    "0352": "The Rolling Stones - Doo Doo Doo Doo Doo (Heartbreaker)",
    "0822": "Ray Charles - Crying Time",
    "0339": "Elvis Presley - Ask Me",
    "1169": "Jesus Jones - Right Here, Right Now",
    "1007": "Righteous Brothers - Unchained Melody",
    "0185": "James Taylor - Country Road",
    "0827": "Marty Robbins - I Walk Alone",
    "0810": "Poison - Unskinny Bop",
    "0530": "Tracie Spencer - This House",
    "0740": "Cliff Richard - Living Doll",
    "0322": "The Everly Brothers - Bird Dog",
    "0707": "Rod Bernard - This Should Go On Forever",
    "1270": "Kaoma - Lambada",
    "0396": "Jackson Browne - Running On Empty",
    "0500": "Gladys Knight & The Pips - I'm Coming Out",
    "0399": "Billy Joel - Don't Ask Me Why",
    "0787": "Johnny Cash - The Ways Of A Woman In Love",
    "1287": "Johnny Horton - Johnny Reb",
    "1072": "Fats Domino - Be My Guest",
    "0214": "Ray Charles - (Night Time Is) The Right Time",
    "0139": "Tom T. Hall - The Year That Clayton Delaney Died",
}


def normalize(s):
    """Normalize for fuzzy matching: lowercase, collapse whitespace, strip punctuation."""
    s = s.lower().strip()
    s = re.sub(r"[''`]", "'", s)  # normalize apostrophes
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    url_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/genej/Downloads/billboard_125_resolved_only.txt"

    # Load training subset (ordered by score)
    with open(DATA_DIR / "training_subset.json") as f:
        subset = json.load(f)
    track_ids = subset["track_ids"]

    # Songs #26-#150 = indices 25-149
    remaining_ids = track_ids[25:]

    # Parse the URL file
    with open(url_file) as f:
        lines = f.read().strip().split("\n")

    pairs = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("http"):
            i += 1
            continue
        song_name = line
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines):
            url = lines[i].strip()
            pairs.append((song_name, url))
            i += 1

    print(f"Parsed {len(pairs)} song-URL pairs from file")

    # Match using display names (what user saw) and track ID order
    matched = []
    unmatched_songs = []
    used_ids = set()

    for song_name, url in pairs:
        found = False
        for tid in remaining_ids:
            if tid in used_ids:
                continue
            display = DISPLAY_NAMES.get(tid, "")
            if normalize(song_name) == normalize(display):
                matched.append((tid, song_name, url))
                used_ids.add(tid)
                found = True
                break
        if not found:
            unmatched_songs.append((song_name, url))

    print(f"Matched: {len(matched)}")
    print(f"Unmatched: {len(unmatched_songs)}")

    if unmatched_songs:
        print("\nUnmatched songs:")
        for name, url in unmatched_songs:
            print(f"  {name}  ->  {url}")

    # Track IDs without URLs (skipped/unresolved)
    unmatched_ids = [tid for tid in remaining_ids if tid not in used_ids]
    print(f"\nTrack IDs without URLs: {len(unmatched_ids)}")
    for tid in unmatched_ids:
        print(f"  {tid}: {DISPLAY_NAMES.get(tid, '???')}")

    # Check for playlist URLs (not downloadable)
    playlist_entries = [(t, n, u) for t, n, u in matched if "playlist?" in u]
    if playlist_entries:
        print(f"\n⚠ Playlist URLs (may not work):")
        for tid, name, url in playlist_entries:
            print(f"  {tid}: {name}  ->  {url}")

    # Write batch file
    batch = []
    for tid, name, url in matched:
        batch.append({"track_id": tid, "name": name, "url": url})

    out = "/tmp/billboard_batch.json"
    with open(out, "w") as f:
        json.dump(batch, f, indent=2)
    print(f"\nBatch file written: {out} ({len(batch)} entries)")

if __name__ == "__main__":
    main()
