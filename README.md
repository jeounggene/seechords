# SeeChords

A Chrome extension that overlays cached chord data on YouTube videos, with first-time chord generation powered by user-uploaded matching audio.

## Architecture

```
seechords/
├── extension/          # Chrome extension (Manifest V3)
│   ├── manifest.json   # Extension config
│   ├── background.js   # Service worker - API communication
│   ├── content.js      # YouTube page injection + chord overlay
│   ├── overlay.css     # Overlay styling (EZChords dark theme)
│   ├── popup.html      # Extension popup UI
│   ├── popup.js        # Popup logic
│   └── icons/          # Extension icons
│
└── server/             # Flask backend
    ├── app.py           # API server (chord analysis + caching)
    ├── analyze_chords.py # Essentia-based analysis (subprocess)
    └── requirements.txt  # Python dependencies
```

## User Flow

1. User opens a YouTube video
2. Extension checks whether chords already exist for that video ID
3. If yes → overlay loads immediately with synced chord display
4. If no → user sees an upload prompt within the overlay
5. User uploads a matching MP3 of that song
6. Server analyzes the MP3 for chords, beats, key, BPM
7. Server deletes the uploaded audio file
8. Server stores only the derived chord/timestamp JSON
9. Extension overlays the chords on the YouTube video
10. Next time anyone opens that video → cached chords appear instantly

## Setup

### Backend

```bash
cd seechords/server
pip install -r requirements.txt
python app.py
# Server runs on http://localhost:5002
```

Requires `ffmpeg` installed on the system:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Chrome Extension

1. Open Chrome → `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `seechords/extension/` directory
5. Navigate to any YouTube video

## API Endpoints

### `GET /api/chords/:videoId`
Returns cached chord data for a YouTube video ID.

```json
{
  "videoId": "abc123xyzAB",
  "source": "user-uploaded",
  "title": "Song Title",
  "key": "C",
  "bpm": 120.0,
  "chords": [
    { "chord": "C", "start": 0.0, "end": 2.5 },
    { "chord": "G", "start": 2.5, "end": 5.1 }
  ],
  "beat_times": [0.0, 0.5, 1.0, ...]
}
```

### `POST /api/analyze`
Upload an audio file with a video ID.

Form data:
- `videoId` (required): YouTube video ID (11 chars)
- `file` (required): Audio file (mp3, wav, m4a, etc.)
- `title` (optional): Song title

Returns `{ "job_id": "...", "cached": false }` or `{ "cached": true, "data": {...} }`

### `GET /api/status/:jobId`
Poll analysis progress.

Returns `{ "status": "processing"|"done"|"error", "progress": 0-100, ... }`

## Features

- **Chord timeline**: Scrolling beat-by-beat chord blocks synced to video playback
- **Chord diagrams**: SVG guitar chord diagrams (prev / current / next)
- **Transpose**: Shift all chords up/down by semitones
- **Timing offset**: Adjust chord sync +/- 0.5s increments
- **Minimize/expand**: Toggle overlay visibility
- **SPA-aware**: Handles YouTube's single-page navigation

## Product Guardrails

- Uploaded MP3s are deleted after analysis
- No audio files are stored or served
- Only derived chord data (JSON) is persisted
- Users must confirm upload rights
- Not a YouTube ripping tool — playback stays on YouTube
