"""SeeChords backend – chord analysis API for the Chrome extension.

Endpoints:
  GET  /                         → marketing site (extension info, donate, privacy links)
  GET  /privacy                  → extension privacy policy (HTML)
  GET  /api/chords/<videoId>     → cached chord JSON or 404
  POST /api/analyze              → upload MP3 + videoId, returns job_id
  GET  /api/status/<job_id>      → poll analysis progress
  GET  /api/health               → diagnostic info
  GET/PUT /api/internal/wav-cache/<videoId> → worker-only WAV cache (Bearer WAV_CACHE_SECRET);
    WAV also mirrored to Turso table wav_cache_backups when WAV_BACKUP_TO_DB=1 (default).
  GET  /play                        → web chord player (YouTube URL + beat-synced view)
  GET  /api/youtube-stream/<videoId> → proxy YT audio stream (Range-aware, URL cached 5h)
"""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # prevent OpenMP crash (essentia + torch)
import base64
import uuid
import threading
import shutil
import re
import json
import time
import subprocess

from functools import wraps
from flask import Flask, request, jsonify, send_file, render_template, redirect, Response, stream_with_context, session
from flask_cors import CORS
import numpy as np
try:
    import libsql_experimental as libsql
except ImportError:
    import sqlite3
    libsql = sqlite3

# Fix macOS Python SSL
try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
except ImportError:
    pass

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
INGEST_PASSWORD = os.environ.get('INGEST_PASSWORD', '')

# Current analysis model version — used as source tag for chord_versions
CURRENT_MODEL_SOURCE = 'btc-v2'

# In-memory cache for resolved YouTube stream URLs (TTL 5h, evicted per-request)
_yt_stream_cache = {}   # video_id -> {'url': str, 'content_type': str, 'expires': float}
_yt_stream_lock  = threading.Lock()

TURSO_URL   = os.environ.get('TURSO_DATABASE_URL', '')
TURSO_TOKEN = os.environ.get('TURSO_AUTH_TOKEN', '')
UPLOAD_DIR           = os.environ.get('UPLOAD_DIR', os.path.join(os.path.dirname(__file__), 'uploads'))
# YouTube-derived WAVs (per video_id) for faster re-analysis without yt-dlp (ChordMini-style cache).
WAV_CACHE_DIR        = os.environ.get('WAV_CACHE_DIR', os.path.join(UPLOAD_DIR, 'wav_cache'))
TRAINING_VERIFIED_DIR = os.path.join(os.path.dirname(__file__), '..', 'training', 'data', 'verified')
SERVER_VERIFIED_DIR  = os.path.join(os.path.dirname(__file__), 'verified')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WAV_CACHE_DIR, exist_ok=True)


def _ytdlp_cookiefile():
    """Netscape-format cookies.txt for yt-dlp. Set env YTDLP_COOKIEFILE to the file path."""
    p = (os.environ.get('YTDLP_COOKIEFILE') or '').strip()
    return p if p and os.path.isfile(p) else None


def _normalize_ytdlp_cookies_b64(s: str) -> str:
    """Strip whitespace/newlines so Fly secrets and shell quoting don't break base64 decode."""
    return ''.join((s or '').split())


def _wav_cache_auth_ok() -> bool:
    secret = (os.environ.get('WAV_CACHE_SECRET') or '').strip()
    if not secret:
        return False
    auth = request.headers.get('Authorization', '')
    return auth == f'Bearer {secret}'


def _worker_youtube_cookie_env():
    """Env keys to pass to ephemeral Fly workers so yt-dlp can use cookies.

    Worker VMs do not share the API's filesystem: YTDLP_COOKIEFILE paths on the API
    are useless there unless we embed file bytes. Prefer YTDLP_COOKIES_B64 on the API
    (Fly secret), or a readable YTDLP_COOKIEFILE on the API which we re-encode here.
    """
    b64 = _normalize_ytdlp_cookies_b64(os.environ.get('YTDLP_COOKIES_B64') or '')
    if b64:
        return {'YTDLP_COOKIES_B64': b64}
    cf = _ytdlp_cookiefile()
    if cf:
        try:
            with open(cf, 'rb') as f:
                data = f.read()
            return {'YTDLP_COOKIES_B64': base64.b64encode(data).decode('ascii')}
        except OSError as e:
            print(f'[SeeChords] Cannot read YTDLP_COOKIEFILE {cf}: {e}', flush=True)
    return {}


# ─────────────────────────────────────────────
# Turso (libsql) persistent cache
# ─────────────────────────────────────────────

def _get_db():
    """Return a libsql connection to Turso (remote) or local SQLite fallback."""
    if TURSO_URL:
        con = libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)
    else:
        db_path = os.environ.get('DB_PATH', os.path.join(os.path.dirname(__file__), 'seechords.db'))
        con = libsql.connect(database=db_path)
    return con


def _init_db():
    con = _get_db()
    con.execute('''
        CREATE TABLE IF NOT EXISTS chord_versions (
            version_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id    TEXT NOT NULL,
            title       TEXT,
            key         TEXT,
            bpm         REAL,
            chords      TEXT,
            beat_times  TEXT,
            downbeats   TEXT,
            source      TEXT DEFAULT 'user-uploaded',
            analyzed_at INTEGER,
            is_active   INTEGER DEFAULT 0
        )
    ''')
    con.execute('CREATE INDEX IF NOT EXISTS idx_cv_video ON chord_versions(video_id)')
    # Migration: add downbeats column if missing
    try:
        con.execute('ALTER TABLE chord_versions ADD COLUMN downbeats TEXT')
        con.commit()
    except Exception:
        pass  # column already exists
    con.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            job_id     TEXT PRIMARY KEY,
            video_id   TEXT NOT NULL,
            status     TEXT NOT NULL DEFAULT 'pending',
            progress   INTEGER DEFAULT 0,
            message    TEXT,
            result     TEXT,
            worker_id  TEXT,
            created_at INTEGER,
            updated_at INTEGER
        )
    ''')
    con.execute('CREATE INDEX IF NOT EXISTS idx_jobs_video ON jobs(video_id)')
    con.execute('''
        CREATE TABLE IF NOT EXISTS wav_cache_backups (
            video_id   TEXT PRIMARY KEY,
            wav_data   BLOB NOT NULL,
            bytes      INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        )
    ''')
    con.commit()
    con.close()

_init_db()


_CV_COLS = ['version_id', 'video_id', 'title', 'key', 'bpm',
            'chords', 'beat_times', 'downbeats', 'source', 'analyzed_at', 'is_active']


def _row_to_dict(row, cols):
    """Convert a tuple row to a dict using column names."""
    if isinstance(row, dict):
        return row
    return dict(zip(cols, row))


def _version_row_to_dict(row):
    d = _row_to_dict(row, _CV_COLS) if not isinstance(row, dict) else row
    return {
        'versionId':  d['version_id'],
        'videoId':    d['video_id'],
        'title':      d['title'],
        'key':        d['key'],
        'bpm':        d['bpm'],
        'chords':     json.loads(d['chords']) if isinstance(d['chords'], str) else d['chords'],
        'beat_times': json.loads(d['beat_times']) if isinstance(d['beat_times'], str) else d['beat_times'],
        'downbeats':  json.loads(d['downbeats']) if d.get('downbeats') and isinstance(d['downbeats'], str) else (d.get('downbeats') or []),
        'source':     d['source'],
        'analyzedAt': d['analyzed_at'],
        'isActive':   bool(d['is_active']),
    }


# ── Isophonics → display chord conversion (Python) ────────────────
_ISO_TO_DISPLAY = {
    'maj': '', 'min': 'm', '7': '7', 'maj7': 'maj7', 'min7': 'm7',
    'dim': 'dim', 'dim7': 'dim7', 'hdim7': 'm7b5', 'aug': 'aug',
    'sus2': 'sus2', 'sus4': 'sus4', '6': '6', 'min6': 'm6', '9': '9',
    'min9': 'm9', '13': '13', '7b13': '7b13', '7add13': '7add13', '9sus4': '9sus4', 'minmaj7': 'mmaj7',
    'maj9': 'maj9', 'min11': 'm11', 'min13': 'm13', 'add9': 'add9',
    '11': '11',
}

def _iso_to_display(iso):
    """Convert Isophonics chord to display name. E.g. G:min → Gm, A:maj7 → Amaj7"""
    if not iso or iso == 'N':
        return 'N'
    bass = ''
    if '/' in iso:
        idx = iso.index('/')
        bass = iso[idx:]   # e.g. /E
        iso = iso[:idx]
    if ':' not in iso:
        return iso + bass
    root, quality = iso.split(':', 1)
    suffix = _ISO_TO_DISPLAY.get(quality, quality)
    return root + suffix + bass


def _dump_video_map(vmap, path):
    """Write video_map.json with numeric arrays collapsed to single lines.
    Also syncs to training/data/verified/video_map.json."""
    raw = json.dumps(vmap, indent=2, ensure_ascii=False)
    raw = re.sub(
        r'\[[\d\s.,\n]+\]',
        lambda m: re.sub(r'\s+', ' ', m.group(0)),
        raw,
    )
    with open(path, 'w') as f:
        f.write(raw + '\n')
    # Mirror to training directory
    training_map = os.path.join(TRAINING_VERIFIED_DIR, 'video_map.json')
    if os.path.isdir(TRAINING_VERIFIED_DIR) and training_map != os.path.abspath(path):
        with open(training_map, 'w') as f:
            f.write(raw + '\n')


# ── Import verified .lab files on startup ──────────────────────
def _import_verified():
    """On startup, import any verified .lab files that don't yet have
    a 'verified' version in chord_versions."""
    map_path = os.path.join(SERVER_VERIFIED_DIR, 'video_map.json')
    if not os.path.isfile(map_path):
        return
    with open(map_path) as f:
        vmap = json.load(f)

    label_dir = os.path.join(SERVER_VERIFIED_DIR, 'labels')
    con = _get_db()

    for stem, meta in vmap.items():
        video_id = meta['videoId']
        title = meta.get('title', stem.replace('_', ' '))
        lab_path = os.path.join(label_dir, stem + '.lab')
        if not os.path.isfile(lab_path):
            continue

        existing = con.execute(
            "SELECT version_id FROM chord_versions WHERE video_id = ? AND source = 'verified'",
            (video_id,),
        ).fetchone()
        if existing:
            continue

        segments = []
        with open(lab_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                iso_chord = parts[2]
                disp = _iso_to_display(iso_chord)
                if disp == 'N':
                    continue
                segments.append({
                    'chord': disp,
                    'start': round(float(parts[0]), 3),
                    'end': round(float(parts[1]), 3),
                })

        if not segments:
            continue

        key_val = meta.get('key')
        bpm_val = meta.get('bpm')
        beat_times_list = meta.get('beatTimes', [])
        if not key_val or not bpm_val or not beat_times_list:
            ref = con.execute(
                'SELECT key, bpm, beat_times FROM chord_versions WHERE video_id = ? ORDER BY version_id DESC LIMIT 1',
                (video_id,),
            ).fetchone()
            if ref:
                rd = _row_to_dict(ref, ['key', 'bpm', 'beat_times'])
                key_val = key_val or rd['key']
                bpm_val = bpm_val or rd['bpm']
                if not beat_times_list and rd['beat_times']:
                    beat_times_list = json.loads(rd['beat_times'])
            else:
                key_val = key_val or '?'
                bpm_val = bpm_val or 120
        if not beat_times_list:
            print(f"  WARNING: verified song '{stem}' has no beat_times (video_map empty, no DB fallback)")
        beat_times = json.dumps(beat_times_list)

        con.execute('UPDATE chord_versions SET is_active = 0 WHERE video_id = ?', (video_id,))
        con.execute('''
            INSERT INTO chord_versions
                (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 'verified', ?, 1)
        ''', (video_id, title, key_val, bpm_val,
              json.dumps(segments), beat_times, int(time.time())))

    con.commit()
    con.close()

# _import_verified() — disabled: verified labels no longer auto-ingest to production DB


def _cache_get(video_id: str):
    """Return the best version for a video: verified > current model > legacy. Excludes drafts."""
    con = _get_db()
    row = con.execute(
        '''SELECT * FROM chord_versions
           WHERE video_id = ? AND source != 'ingest-edit'
           ORDER BY
             (source = 'verified') DESC,
             (source = ?) DESC,
             version_id DESC
           LIMIT 1''',
        (video_id, CURRENT_MODEL_SOURCE),
    ).fetchone()
    con.close()
    if row is None:
        return None
    return _version_row_to_dict(row)


def _has_verified_version(video_id: str) -> bool:
    """True if this video has a human-verified chord row (re-analysis is disabled)."""
    con = _get_db()
    row = con.execute(
        "SELECT 1 FROM chord_versions WHERE video_id = ? AND source = 'verified' LIMIT 1",
        (video_id,),
    ).fetchone()
    con.close()
    return row is not None


def _cache_put(video_id, title, key, bpm, chords_data, beat_times, downbeats=None):
    con = _get_db()
    con.execute('DELETE FROM chord_versions WHERE video_id = ?', (video_id,))
    con.execute('''
        INSERT INTO chord_versions
            (video_id, title, key, bpm, chords, beat_times, downbeats, source, analyzed_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    ''', (video_id, title, key, bpm,
          json.dumps(chords_data), json.dumps(beat_times),
          json.dumps(downbeats) if downbeats else None,
          CURRENT_MODEL_SOURCE, int(time.time())))
    con.commit()
    version_id = con.execute('SELECT last_insert_rowid()').fetchone()[0]
    con.close()
    return version_id


# ─────────────────────────────────────────────
# Job store (Turso-backed, survives restarts)
# ─────────────────────────────────────────────

def _create_job(job_id: str, video_id: str, **kwargs):
    """Insert a new job row into Turso."""
    con = _get_db()
    con.execute(
        '''INSERT INTO jobs (job_id, video_id, status, progress, message, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (job_id, video_id, kwargs.get('status', 'pending'),
         kwargs.get('progress', 0), kwargs.get('message', ''),
         int(time.time()), int(time.time())),
    )
    con.commit()
    con.close()


def _set_job(job_id, **kwargs):
    """Update job fields in Turso. Stores full result as JSON blob when done."""
    sets = ['updated_at = ?']
    vals = [int(time.time())]
    for col in ('status', 'progress', 'message', 'worker_id'):
        if col in kwargs:
            sets.append(f'{col} = ?')
            vals.append(kwargs[col])
    # When job is done, pack extra fields (chords, bpm, key, etc.) into result JSON
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


def _get_job(job_id):
    """Read job from Turso and return a dict matching the old in-memory format."""
    con = _get_db()
    row = con.execute(
        'SELECT job_id, video_id, status, progress, message, result, worker_id, created_at, updated_at FROM jobs WHERE job_id = ?',
        (job_id,),
    ).fetchone()
    con.close()
    if not row:
        return {'status': 'not_found'}
    cols = ['job_id', 'video_id', 'status', 'progress', 'message', 'result', 'worker_id', 'created_at', 'updated_at']
    d = _row_to_dict(row, cols)
    out = {'status': d['status'], 'progress': d['progress'] or 0, 'message': d['message'] or ''}
    if d['result']:
        try:
            out.update(json.loads(d['result']))
        except (json.JSONDecodeError, TypeError):
            pass
    return out


# ─────────────────────────────────────────────
# Chord detection (reused from ezchords)
# ─────────────────────────────────────────────

NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

SIMPLE_INTERVALS = [
    ('',  [0, 4, 7]),
    ('m', [0, 3, 7]),
]

EXTENDED_INTERVALS = [
    ('7',    [0, 4, 7, 10]),
    ('m7',   [0, 3, 7, 10]),
    ('maj7', [0, 4, 7, 11]),
    ('sus2', [0, 2, 7]),
    ('sus4', [0, 5, 7]),
    ('dim',  [0, 3, 6]),
    ('aug',  [0, 4, 8]),
]

ALL_INTERVALS = SIMPLE_INTERVALS + EXTENDED_INTERVALS

ROOT_WEIGHT  = 1.5
FIFTH_WEIGHT = 1.2


def _build_templates(intervals_list):
    templates = {}
    for i, note in enumerate(NOTES):
        for chord_type, intervals in intervals_list:
            t = np.zeros(12)
            for k, iv in enumerate(intervals):
                if k == 0:
                    w = ROOT_WEIGHT
                elif iv in (7, 6, 8):
                    w = FIFTH_WEIGHT
                else:
                    w = 1.0
                t[(i + iv) % 12] = w
            t /= np.linalg.norm(t)
            templates[note + chord_type] = t
    return templates


SIMPLE_TEMPLATES = _build_templates(SIMPLE_INTERVALS)
SIMPLE_CHORDS    = list(SIMPLE_TEMPLATES.keys())
SIMPLE_MATRIX    = np.array([SIMPLE_TEMPLATES[c] for c in SIMPLE_CHORDS])

CHORD_TEMPLATES  = _build_templates(ALL_INTERVALS)
ALL_CHORDS       = list(CHORD_TEMPLATES.keys())
TEMPLATE_MATRIX  = np.array([CHORD_TEMPLATES[c] for c in ALL_CHORDS])

_EXTENDED_TO_SIMPLE = {}
for i, note in enumerate(NOTES):
    for suffix, _ in EXTENDED_INTERVALS:
        parent = note + ('m' if 'm' in suffix and suffix != 'maj7' else '')
        _EXTENDED_TO_SIMPLE[note + suffix] = parent

_DIATONIC_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_DIATONIC_QUALITIES = ['', 'm', 'm', '', '', 'm', 'dim']


def _diatonic_set(key_idx):
    s = set()
    for offset, quality in zip(_DIATONIC_INTERVALS, _DIATONIC_QUALITIES):
        note = NOTES[(key_idx + offset) % 12]
        s.add(note + quality)
    return s


def _viterbi_decode(template_matrix, chord_list, beat_chroma, self_prob=0.92,
                    emission_bias=None):
    n_chords = len(chord_list)
    n_beats  = beat_chroma.shape[1]

    norms = np.linalg.norm(beat_chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    sim = template_matrix @ (beat_chroma / norms)

    log_emit = np.log(np.clip(sim, 1e-10, None))
    if emission_bias is not None:
        log_emit += emission_bias

    switch_prob = (1.0 - self_prob) / max(n_chords - 1, 1)
    log_self   = np.log(self_prob)
    log_switch = np.log(switch_prob)

    viterbi = np.full((n_chords, n_beats), -np.inf)
    backptr = np.zeros((n_chords, n_beats), dtype=int)
    viterbi[:, 0] = np.log(1.0 / n_chords) + log_emit[:, 0]

    for t in range(1, n_beats):
        prev = viterbi[:, t - 1]
        for s in range(n_chords):
            candidates = prev + log_switch
            candidates[s] = prev[s] + log_self
            bp = int(np.argmax(candidates))
            viterbi[s, t] = candidates[bp] + log_emit[s, t]
            backptr[s, t] = bp

    path = np.zeros(n_beats, dtype=int)
    path[-1] = int(np.argmax(viterbi[:, -1]))
    for t in range(n_beats - 2, -1, -1):
        path[t] = backptr[path[t + 1], t + 1]

    return [chord_list[ci] for ci in path]


def detect_chords(audio_path: str, hop_size: float = 0.5):
    """Run chord analysis in-process (avoids OOM from subprocess doubling memory)."""
    try:
        from analyze_chords import analyze as _analyze_chords
        data = _analyze_chords(audio_path)
        return data['chords'], data['bpm'], data['key'], data['beat_times'], data.get('downbeats')
    except Exception as e:
        print(f'[SeeChords] analyze_chords failed, falling back to librosa: {e}')
        import traceback; traceback.print_exc()

    chords, bpm, key, beat_times = _detect_chords_librosa(audio_path, hop_size)
    return chords, bpm, key, beat_times, None


def _detect_chords_librosa(audio_path: str, hop_size: float = 0.5):
    import librosa
    y, sr = librosa.load(audio_path, mono=True, sr=22050, duration=360)
    hop_length = 2048

    y_harm, y_perc = librosa.effects.hpss(y)

    chroma = librosa.feature.chroma_cens(
        y=y_harm, sr=sr, hop_length=hop_length, n_chroma=12,
    )

    tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr, hop_length=hop_length)
    tempo = np.asarray(tempo).item()
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()

    beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    n_beats = len(beat_times)
    beat_chroma = beat_chroma[:, :n_beats]

    key_idx = int(np.argmax(np.mean(beat_chroma, axis=1)))
    key = NOTES[key_idx]
    diatonic = _diatonic_set(key_idx)

    KEY_BOOST = 0.6
    simple_bias = np.zeros((len(SIMPLE_CHORDS), 1))
    for ci, name in enumerate(SIMPLE_CHORDS):
        if name in diatonic:
            simple_bias[ci, 0] = KEY_BOOST

    simple_path = _viterbi_decode(SIMPLE_MATRIX, SIMPLE_CHORDS, beat_chroma,
                                  self_prob=0.92, emission_bias=simple_bias)

    PROMOTE_THRESH = 0.12
    norms = np.linalg.norm(beat_chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    bc_normed = beat_chroma / norms
    full_sim = TEMPLATE_MATRIX @ bc_normed

    final_path = []
    for bi, simple_name in enumerate(simple_path):
        simple_score = full_sim[ALL_CHORDS.index(simple_name), bi]
        best_ext_name  = simple_name
        best_ext_score = simple_score
        for ext_name, parent in _EXTENDED_TO_SIMPLE.items():
            if parent == simple_name:
                ext_score = full_sim[ALL_CHORDS.index(ext_name), bi]
                if ext_score > best_ext_score + PROMOTE_THRESH:
                    best_ext_name  = ext_name
                    best_ext_score = ext_score
        final_path.append(best_ext_name)

    try:
        from analyze_chords import _uniformize_beat_times
        beat_times = _uniformize_beat_times(beat_times, float(tempo))
    except Exception:
        pass
    iv_tail = (beat_times[1] - beat_times[0]) if len(beat_times) >= 2 else 0.5

    merged = []
    for i, chord_name in enumerate(final_path):
        start = beat_times[i]
        end = beat_times[i + 1] if i + 1 < len(beat_times) else start + iv_tail
        if merged and merged[-1]['chord'] == chord_name:
            merged[-1]['end'] = round(end, 3)
        else:
            merged.append({'chord': chord_name, 'start': round(start, 3), 'end': round(end, 3)})

    return merged, tempo, key, beat_times


def _clean_title(name: str) -> str:
    title = re.sub(r'\.[A-Za-z0-9]{1,6}$', '', (name or '').strip())
    title = re.sub(r'[_\-]+', ' ', title).strip()
    return title[:120] if title else 'Uploaded Audio'


def _process_job_local(job_id: str, source_path: str, video_id: str, title: str):
    """Analyze uploaded audio locally (librosa fallback), store chords, delete audio file.
    Used for direct uploads to the API server when no worker is available."""
    temp_wav = os.path.join(UPLOAD_DIR, f'{job_id}.wav')
    try:
        _set_job(job_id, status='processing', progress=5, message='Preparing audio…')

        if not shutil.which('ffmpeg'):
            raise RuntimeError('FFmpeg is required to process uploaded files.')

        subprocess.run(
            ['ffmpeg', '-i', source_path, '-vn', '-ar', '44100', '-ac', '1',
             temp_wav, '-y'],
            capture_output=True, timeout=180,
        )
        if not os.path.exists(temp_wav):
            raise RuntimeError('Failed to convert uploaded file to wav.')

        _set_job(job_id, status='processing', progress=40, message='Analyzing chords…')
        chords_data, bpm_val, key_val, beat_times, downbeats = detect_chords(temp_wav)

        version_id = _cache_put(video_id, title, key_val, round(bpm_val, 1), chords_data, beat_times, downbeats)

        _set_job(
            job_id,
            status='done',
            progress=100,
            versionId=version_id,
            videoId=video_id,
            title=title,
            chords=chords_data,
            bpm=round(bpm_val, 1),
            key=key_val,
            beat_times=beat_times,
            source=CURRENT_MODEL_SOURCE,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set_job(job_id, status='error', message=f'Analysis failed: {exc}')
    finally:
        for f in (source_path, temp_wav):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/api/chords/<video_id>')
def get_chords(video_id):
    """Return chord data for a YouTube video ID (active version + all versions list)."""
    # Validate video_id format (11 chars, alphanumeric + _ -)
    if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return jsonify({'error': 'Invalid video ID.'}), 400

    active = _cache_get(video_id)
    if active:
        return jsonify(active)
    return jsonify({'error': 'No chords found for this video.'}), 404


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Upload audio + videoId to analyze chords."""
    video_id = request.form.get('videoId', '').strip()
    if not video_id or not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return jsonify({'error': 'Invalid or missing videoId.'}), 400

    if _has_verified_version(video_id):
        return jsonify({'error': 'This video has verified chords; re-analysis is disabled.'}), 400

    # Block re-analysis if already analyzed with the current model
    existing = _cache_get(video_id)
    if existing and existing.get('source') == CURRENT_MODEL_SOURCE:
        return jsonify({'error': f'Already analyzed with the current model ({CURRENT_MODEL_SOURCE}). No re-analysis needed.'}), 400

    file_obj = request.files.get('file')
    if not file_obj or not file_obj.filename:
        return jsonify({'error': 'Please upload an audio file.'}), 400

    filename = file_obj.filename.strip()
    ext = os.path.splitext(filename)[1].lower()
    allowed = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.webm', '.mp4'}
    if ext and ext not in allowed:
        return jsonify({'error': 'Unsupported file type.'}), 400

    job_id = str(uuid.uuid4())

    source_ext = ext if ext else '.bin'
    source_path = os.path.join(UPLOAD_DIR, f'{job_id}_source{source_ext}')
    file_obj.save(source_path)

    title = request.form.get('title', '').strip() or _clean_title(filename)

    _create_job(job_id, video_id, status='processing', progress=0, message='Upload received…')

    t = threading.Thread(target=_process_job_local, args=(job_id, source_path, video_id, title),
                         daemon=True)
    t.start()

    return jsonify({'job_id': job_id, 'cached': False})


FLY_API_TOKEN = os.environ.get('FLY_API_TOKEN', '')
FLY_WORKER_APP = os.environ.get('FLY_WORKER_APP', 'seechords-worker')
FLY_WORKER_IMAGE = os.environ.get('FLY_WORKER_IMAGE', f'registry.fly.io/{os.environ.get("FLY_WORKER_APP", "seechords-worker")}:latest')


def _spawn_worker(job_id: str, video_id: str, title: str = ''):
    """Spawn an ephemeral Fly Machine to run chord analysis."""
    import requests as req
    cookie_env = _worker_youtube_cookie_env()
    if cookie_env.get('YTDLP_COOKIES_B64'):
        n = len(cookie_env['YTDLP_COOKIES_B64'])
        print(f'[SeeChords] Worker spawn: forwarding YTDLP_COOKIES_B64 ({n} base64 chars)', flush=True)
    else:
        print(
            '[SeeChords] Worker spawn: no cookie payload from API env '
            '(set YTDLP_COOKIES_B64 or YTDLP_COOKIEFILE on app `seechords`; '
            'or set YTDLP_COOKIES_B64 on `seechords-worker` if Fly merges app secrets).',
            flush=True,
        )
    machine_env = {
        'JOB_ID': job_id,
        'VIDEO_ID': video_id,
        'TITLE': title or '',
        'TURSO_DATABASE_URL': TURSO_URL,
        'TURSO_AUTH_TOKEN': TURSO_TOKEN,
        'USE_BTC': '1',
        'USE_BEAT_THIS': '1',
        **cookie_env,
    }
    # Forward WAV cache secret so workers can upload/download cached audio
    wav_secret = os.environ.get('WAV_CACHE_SECRET', '')
    if wav_secret:
        machine_env['WAV_CACHE_SECRET'] = wav_secret
    resp = req.post(
        f'https://api.machines.dev/v1/apps/{FLY_WORKER_APP}/machines',
        headers={'Authorization': f'Bearer {FLY_API_TOKEN}'},
        json={
            'config': {
                'image': FLY_WORKER_IMAGE,
                'env': machine_env,
                'guest': {'cpu_kind': 'shared', 'cpus': 2, 'memory_mb': 4096},
                'auto_destroy': True,
            },
        },
        timeout=30,
    )
    if resp.status_code >= 400:
        print(f'[SeeChords] Machines API error {resp.status_code}: {resp.text[:500]}', flush=True)
        resp.raise_for_status()
    machine_id = resp.json().get('id', '')
    _set_job(job_id, worker_id=machine_id)
    return machine_id



@app.route('/api/internal/wav-cache-purge', methods=['POST'])
def purge_wav_cache():
    """Delete all cached WAV files from disk and DB. Auth: Bearer WAV_CACHE_SECRET."""
    if not _wav_cache_auth_ok():
        return jsonify({'error': 'Unauthorized.'}), 401
    deleted_files = 0
    deleted_db = 0
    # Delete files on disk
    if os.path.isdir(WAV_CACHE_DIR):
        for fname in os.listdir(WAV_CACHE_DIR):
            if fname.endswith('.wav'):
                try:
                    os.remove(os.path.join(WAV_CACHE_DIR, fname))
                    deleted_files += 1
                except OSError:
                    pass
    # Delete DB rows
    try:
        con = _get_db()
        cur = con.execute('DELETE FROM wav_cache_backups')
        deleted_db = cur.rowcount
        con.commit()
        con.close()
    except Exception:
        pass
    return jsonify({'deletedFiles': deleted_files, 'deletedDbRows': deleted_db})


@app.route('/api/analyze-youtube', methods=['POST'])
def analyze_youtube():
    """Download audio from YouTube via yt-dlp and analyze chords.
    Spawns an ephemeral Fly worker machine for the heavy ML work."""
    data = request.get_json(silent=True) or {}
    video_id = (data.get('videoId') or '').strip()
    if not video_id or not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return jsonify({'error': 'Invalid or missing videoId.'}), 400

    title = (data.get('title') or '').strip()

    if _has_verified_version(video_id):
        return jsonify({'error': 'This video has verified chords; re-analysis is disabled.'}), 400

    job_id = str(uuid.uuid4())
    _create_job(job_id, video_id, status='pending', progress=0, message='Queued for analysis…')

    try:
        machine_id = _spawn_worker(job_id, video_id, title)
        _set_job(job_id, status='processing', progress=5, message='Worker started…')
        print(f'[SeeChords] Spawned worker {machine_id} for job {job_id} / video {video_id}', flush=True)
    except Exception as e:
        print(f'[SeeChords] Worker spawn failed: {e}', flush=True)
        _set_job(job_id, status='error', message=f'Failed to start analysis worker: {e}')

    return jsonify({'job_id': job_id, 'cached': False})


def _search_youtube_fast(q, limit=8):
    """Search YouTube via the InnerTube API — fast, clean JSON, no page parsing."""
    import requests as req_lib, json
    resp = req_lib.post(
        'https://www.youtube.com/youtubei/v1/search',
        params={'prettyPrint': 'false'},
        headers={
            'Content-Type': 'application/json',
            'X-YouTube-Client-Name': '1',
            'X-YouTube-Client-Version': '2.20240101.00.00',
            'Accept-Language': 'en-US,en;q=0.9',
        },
        json={
            'query': q,
            'context': {
                'client': {
                    'clientName': 'WEB',
                    'clientVersion': '2.20240101.00.00',
                    'hl': 'en',
                    'gl': 'US',
                }
            }
        },
        timeout=8,
    )
    data = resp.json()
    items = (data.get('contents', {})
                 .get('twoColumnSearchResultsRenderer', {})
                 .get('primaryContents', {})
                 .get('sectionListRenderer', {})
                 .get('contents', [{}])[0]
                 .get('itemSectionRenderer', {})
                 .get('contents', []))
    results = []
    for item in items:
        vr = item.get('videoRenderer')
        if not vr:
            continue
        vid = vr.get('videoId', '')
        if not vid or len(vid) != 11:
            continue
        title = (vr.get('title', {}).get('runs') or [{}])[0].get('text', '')
        channel = ((vr.get('ownerText', {}).get('runs') or
                    vr.get('longBylineText', {}).get('runs') or [{}])[0].get('text', ''))
        dur = vr.get('lengthText', {}).get('simpleText', '')
        results.append({'videoId': vid, 'title': title, 'channel': channel,
                        'duration': dur, 'thumbnail': f'https://i.ytimg.com/vi/{vid}/mqdefault.jpg'})
        if len(results) >= limit:
            break
    return results


@app.route('/api/search-youtube')
def search_youtube():
    """Search YouTube and return top results. Tries fast page-parse first, falls back to yt-dlp."""
    q = (request.args.get('q') or '').strip()
    if not q:
        return jsonify({'error': 'Missing query'}), 400
    # Fast path: parse YouTube search page directly
    try:
        results = _search_youtube_fast(q)
        if results:
            return jsonify({'results': results})
    except Exception as e:
        print(f'[SeeChords] Fast YT search failed ({e}), falling back to yt-dlp', flush=True)
    # Fallback: yt-dlp
    try:
        import yt_dlp
        ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': True,
                    'default_search': 'ytsearch8', 'cookiefile': _ytdlp_cookiefile()}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(q, download=False)
        entries = info.get('entries') or []
        results = []
        for e in entries:
            vid = e.get('id') or e.get('url', '')
            if not vid or len(vid) != 11:
                continue
            dur = e.get('duration')
            dur_str = f'{int(dur)//60}:{int(dur)%60:02d}' if dur else ''
            results.append({'videoId': vid, 'title': e.get('title', ''),
                            'channel': e.get('uploader') or e.get('channel') or '',
                            'duration': dur_str,
                            'thumbnail': f'https://i.ytimg.com/vi/{vid}/mqdefault.jpg'})
        return jsonify({'results': results})
    except Exception as e:
        print(f'[SeeChords] YouTube search error: {e}', flush=True)
        return jsonify({'error': str(e)}), 500



@app.route('/api/status/<job_id>')
def job_status(job_id):
    return jsonify(_get_job(job_id))


@app.route('/api/analyze-chunk', methods=['POST'])
def analyze_chunk():
    """Analyze a short audio chunk (10-15s) from the extension's live capture.
    Tries BTC neural model first, falls back to fast librosa chromagram path."""
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'No audio file provided.'}), 400
    video_id = request.form.get('videoId', '').strip()
    chunk_index = int(request.form.get('chunkIndex', '0'))
    start_time = float(request.form.get('startTime', '0'))

    temp_in = os.path.join(UPLOAD_DIR, f'chunk_{video_id}_{chunk_index}.webm')
    temp_wav = os.path.join(UPLOAD_DIR, f'chunk_{video_id}_{chunk_index}.wav')
    try:
        f.save(temp_in)

        subprocess.run(
            ['ffmpeg', '-i', temp_in, '-vn', '-ar', '22050', '-ac', '1',
             temp_wav, '-y'],
            capture_output=True, timeout=30,
        )
        if not os.path.exists(temp_wav):
            return jsonify({'error': 'Failed to convert audio chunk.'}), 500

        chords_data, bpm_val, key_val, beat_times = detect_chords(temp_wav)

        return jsonify({
            'chords': chords_data,
            'bpm': round(bpm_val, 1),
            'key': key_val,
            'beat_times': beat_times,
            'chunkIndex': chunk_index,
            'startTime': start_time,
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Chunk analysis failed: {exc}'}), 500
    finally:
        for path in (temp_in, temp_wav):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


@app.route('/api/save-streamed', methods=['POST'])
def save_streamed():
    """Save accumulated streamed chord results to the database."""
    data = request.get_json(silent=True) or {}
    video_id = (data.get('videoId') or '').strip()
    if not video_id:
        return jsonify({'error': 'Missing videoId.'}), 400

    title = data.get('title', video_id)
    key_val = data.get('key', '?')
    bpm_val = data.get('bpm', 120)
    chords_data = data.get('chords', [])
    beat_times_data = data.get('beat_times', [])

    if not chords_data:
        return jsonify({'error': 'No chord data to save.'}), 400

    con = _get_db()
    con.execute('''
        INSERT INTO chord_versions
            (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, 'live-capture', ?, 1)
    ''', (video_id, title, key_val, round(float(bpm_val), 1),
          json.dumps(chords_data), json.dumps(beat_times_data),
          int(time.time())))
    con.execute('''
        UPDATE chord_versions SET is_active = 0
        WHERE video_id = ? AND source != 'live-capture'
        AND version_id != last_insert_rowid()
    ''', (video_id,))
    con.commit()
    version_id = con.execute('SELECT last_insert_rowid()').fetchone()[0]
    con.close()

    return jsonify({'versionId': version_id, 'saved': True})


@app.route('/api/health')
def health():
    essentia_ok = bool(shutil.which('python3'))
    ffmpeg_ok = shutil.which('ffmpeg') is not None
    return jsonify({
        'status': 'ok',
        'essentia_available': essentia_ok,
        'ffmpeg_available': ffmpeg_ok,
    })


# ─────────────────────────────────────────────
# Chord comparison / testing
# ─────────────────────────────────────────────

_CHORD_RE = re.compile(
    r'^[A-G][b#]?'
    r'(m7b5|m9|m11|m13|mmaj7|min7|maj7|dim7|min9|9sus4|7b13|7add13|add\d+|M7|m7|m6|m|min|maj|dim|aug|sus[24]?|7|9|11|13|6)?'
    r'(/[A-G][b#]?)?$'
)

# Normalize quality suffixes to canonical short forms (preserve extended types)
_QUALITY_MAP_SIMPLE = {
    '': '', 'maj': '', '7': '7', 'maj7': 'maj7', 'M7': 'maj7', '9': '9', '11': '11', '13': '13',
    '7b13': '7b13', '7add13': '7add13',
    'sus2': 'sus2', 'sus4': 'sus4', 'sus': 'sus4', 'add9': 'add9', '6': '6',
    'm': 'm', 'min': 'm', 'm7': 'm7', 'min7': 'm7', 'mmaj7': 'mmaj7', 'm6': 'm6',
    'm7b5': 'm7b5', 'm9': 'm9', 'm11': 'm11', 'm13': 'm13', 'min9': 'm9',
    'dim': 'dim', 'dim7': 'dim7', 'aug': 'aug', '9sus4': '9sus4',
}


# ── Display → Isophonics conversion (for saving .lab files) ──────
_DISPLAY_TO_ISO = [
    ('m7b5', ':hdim7'),
    ('mmaj7', ':minmaj7'), ('m7',  ':min7'), ('m6', ':min6'),
    ('m9',   ':min9'),     ('m11', ':min11'), ('m13', ':min13'),
    ('m',    ':min'),
    ('maj7', ':maj7'),     ('M7', ':maj7'),     ('maj9', ':maj9'),
    ('dim7', ':dim7'),     ('dim',  ':dim'),
    ('aug',  ':aug'),
    ('9sus4', ':9sus4'),   ('7b13', ':7b13'),   ('7add13', ':7add13'),
    ('sus2', ':sus2'),     ('sus4', ':sus4'),
    ('add9', ':add9'),
    ('(b5)', ':(b5)'),     ('(#5)', ':(#5)'),
    ('7',    ':7'),  ('9', ':9'), ('11', ':11'), ('13', ':13'),
    ('6',    ':6'),
]

def _display_to_iso(name):
    """Convert display chord name to Isophonics notation for .lab files.
    E.g. Amaj7 → A:maj7, C#m7 → C#:min7, G → G:maj, Bm7/E → B:min7/E"""
    if not name or name == 'N':
        return 'N'
    slash = ''
    if '/' in name:
        idx = name.index('/')
        slash = '/' + name[idx+1:]     # e.g. "/E"
        name = name[:idx]              # root+quality before slash
    m = re.match(r'^([A-G][#b]?)(.*)', name)
    if not m:
        return 'N'
    root, suffix = m.group(1), m.group(2)
    for disp, iso in _DISPLAY_TO_ISO:
        if suffix == disp:
            return root + iso + slash
    # No suffix matched — bare root means major, otherwise preserve as-is
    if not suffix:
        return root + ':maj' + slash
    return root + ':' + suffix + slash


def _normalize_chord_name(name):
    """Normalize chord — preserve extended quality, normalize root enharmonics."""
    if not name or name in ('N', 'X', '-'):
        return 'N'
    if '/' in name:
        name = name.split('/')[0]
    m = re.match(r'^([A-G][b#]?)(.*)', name)
    if not m:
        return 'N'
    root, quality = m.group(1), m.group(2)
    quality = _QUALITY_MAP_SIMPLE.get(quality, quality)
    return root + quality


def _parse_chord_text(text):
    """Parse a pasted chord sheet into a flat list of chord names."""
    chords = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or re.match(r'^\[.*\]$', line):
            continue
        if '|' in line:
            tokens = re.split(r'[\s|]+', line)
        else:
            tokens = line.split()
        for t in tokens:
            if _CHORD_RE.match(t):
                chords.append(_normalize_chord_name(t))
    return chords


_ROOT_TO_PC = {
    'C': 0, 'B#': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4,
    'F': 5, 'E#': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11,
}
_PC_TO_ROOT = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _parse_root_quality_display(chord):
    if not chord or chord == 'N':
        return None
    main = str(chord).split('/')[0].strip()
    m = re.match(r'^([A-G][b#]?)(.*)$', main)
    if not m:
        return None
    root = m.group(1)
    suffix = (m.group(2) or '').lower()
    quality = 'maj'
    if suffix.startswith('m') and not suffix.startswith('maj'):
        quality = 'min'
    if suffix.startswith('dim') or 'm7b5' in suffix:
        quality = 'dim'
    if suffix.startswith('aug'):
        quality = 'aug'
    if suffix.startswith('sus'):
        quality = 'sus'
    return root, quality


def _infer_key_from_segments(segments):
    """Infer key from edited display chords using simple weighted diatonic fit."""
    if not segments:
        return None

    maj_deg = {0, 2, 4, 5, 7, 9, 11}
    nat_min_deg = {0, 2, 3, 5, 7, 8, 10}
    best_score = -1.0
    best_key = None

    for tonic in range(12):
        maj_score = 0.0
        min_score = 0.0

        for seg in segments:
            ch = seg.get('chord', 'N')
            parsed = _parse_root_quality_display(ch)
            if not parsed:
                continue
            root, quality = parsed
            root_pc = _ROOT_TO_PC.get(root)
            if root_pc is None:
                continue

            deg = (root_pc - tonic) % 12
            try:
                dur = max(0.05, float(seg.get('end', 0)) - float(seg.get('start', 0)))
            except Exception:
                dur = 0.5

            if deg in maj_deg:
                w = 1.0
                if quality == 'maj':
                    w = 1.2
                elif quality in ('min', 'dim'):
                    w = 0.85
                maj_score += dur * w

            if deg in nat_min_deg:
                w = 1.0
                if quality == 'min':
                    w = 1.2
                elif quality in ('maj', 'dim'):
                    w = 0.85
                min_score += dur * w

            if deg == 0:
                if quality == 'maj':
                    maj_score += dur * 0.45
                if quality == 'min':
                    min_score += dur * 0.45
            if deg == 7:
                maj_score += dur * 0.2
                min_score += dur * 0.2

        tonic_name = _PC_TO_ROOT[tonic]
        if maj_score > best_score:
            best_score = maj_score
            best_key = tonic_name
        if min_score > best_score:
            best_score = min_score
            best_key = tonic_name + 'm'

    return best_key


@app.route('/api/compare', methods=['POST'])
def compare_chords_api():
    """Compare stored/analyzed chords against a pasted reference.

    Body JSON: { videoId, referenceText }
    Returns: { accuracy, segments: [{expected, predicted, match}], summary }
    """
    data = request.get_json(force=True)
    video_id = data.get('videoId', '').strip()
    ref_text = data.get('referenceText', '').strip()

    if not video_id or not ref_text:
        return jsonify({'error': 'videoId and referenceText required'}), 400

    # Get stored chords
    cached = _cache_get(video_id)
    if not cached:
        return jsonify({'error': 'No chords found for this video. Analyze first.'}), 404

    # Parse reference
    ref_chords = _parse_chord_text(ref_text)
    if not ref_chords:
        return jsonify({'error': 'No chords found in reference text.'}), 400

    # Collapse predicted into sequence (removing consecutive dupes)
    pred_seq = []
    for seg in cached['chords']:
        c = _normalize_chord_name(seg['chord'])
        if not pred_seq or pred_seq[-1] != c:
            pred_seq.append(c)

    # Collapse reference
    ref_seq = []
    for c in ref_chords:
        if not ref_seq or ref_seq[-1] != c:
            ref_seq.append(c)

    # LCS-based comparison
    m, n = len(ref_seq), len(pred_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_seq[i-1] == pred_seq[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    similarity = lcs_len / max(m, n) if max(m, n) > 0 else 0

    # Which chords in common / missing
    ref_set = set(ref_seq)
    pred_set = set(pred_seq)

    return jsonify({
        'similarity': round(similarity, 3),
        'referenceLength': len(ref_seq),
        'predictedLength': len(pred_seq),
        'lcsLength': lcs_len,
        'referenceSequence': ref_seq[:60],
        'predictedSequence': pred_seq[:60],
        'chordsFound': sorted(ref_set & pred_set),
        'chordsMissing': sorted(ref_set - pred_set),
        'chordsExtra': sorted(pred_set - ref_set),
        'referenceUniqueChords': sorted(ref_set),
    })


@app.route('/api/align', methods=['POST'])
def align_chords_api():
    """DP-align a pasted chord sheet to stored beat-level predictions.

    Body JSON: { videoId, referenceText }
    Returns: { segments: [{start, end, sheetChord, predChord, nBeats,
                           matches, match}],
               beatAccuracy, segmentAgreement }
    """
    data = request.get_json(force=True)
    video_id = data.get('videoId', '').strip()
    ref_text = data.get('referenceText', '').strip()

    if not video_id or not ref_text:
        return jsonify({'error': 'videoId and referenceText required'}), 400

    cached = _cache_get(video_id)
    if not cached:
        return jsonify({'error': 'No chords found for this video. Analyze first.'}), 404

    # Parse sheet chords using the same parser as training tools
    ref_chords = _parse_chord_text(ref_text)
    if not ref_chords:
        return jsonify({'error': 'No chords found in reference text.'}), 400

    # Reconstruct beat-level chords from cached analysis
    merged = cached['chords']       # [{chord, start, end}, ...]
    beat_times = cached.get('beat_times', [])
    if not beat_times:
        return jsonify({'error': 'No beat data available. Re-analyze.'}), 400

    beat_chords = []
    seg_idx = 0
    for bt in beat_times:
        while (seg_idx < len(merged) - 1
               and merged[seg_idx + 1]['start'] <= bt + 1e-4):
            seg_idx += 1
        beat_chords.append(_normalize_chord_name(merged[seg_idx]['chord']))

    # Import DP aligner
    import importlib.util
    align_path = os.path.join(
        os.path.dirname(__file__), 'tools', 'align_chords.py')
    spec = importlib.util.spec_from_file_location('align_chords', align_path)
    align_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_mod)

    segments = align_mod.dp_align(ref_chords, beat_chords, beat_times)
    if not segments:
        return jsonify({'error': 'Alignment failed — too few beats.'}), 400

    json_segments = align_mod.alignment_to_json(segments)

    total_beats = sum(s['nBeats'] for s in json_segments)
    total_matches = sum(s['matches'] for s in json_segments)
    agree = sum(1 for s in json_segments if s['match'])

    return jsonify({
        'segments': json_segments,
        'beatAccuracy': round(total_matches / total_beats, 3) if total_beats else 0,
        'segmentAgreement': round(agree / len(json_segments), 3) if json_segments else 0,
        'totalBeats': total_beats,
        'totalSegments': len(json_segments),
    })


# ─────────────────────────────────────────────
# Chord sheet ingest: PDF/text + MP3 → aligned .lab
# ─────────────────────────────────────────────
_ingest_jobs = {}
_ingest_lock = threading.Lock()


def _extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF, preserving spatial layout for chord sheets."""
    import pdfplumber
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if text:
                lines.append(text)
    return '\n'.join(lines)


def _update_ingest_job(job_id, new_data):
    """Update an ingest job dict, preserving the videoId field."""
    with _ingest_lock:
        old = _ingest_jobs.get(job_id) or {}
        vid = old.get('videoId')
        _ingest_jobs[job_id] = new_data
        if vid and 'videoId' not in new_data:
            _ingest_jobs[job_id]['videoId'] = vid


def _do_ingest(job_id: str, audio_path: str, chord_text: str, song_name: str):
    """Background worker: analyze audio, align chords, store result.

    Uses lyric-aware alignment (Whisper) when the chord sheet contains
    lyrics below chord lines, otherwise falls back to beat-based DP.
    """
    try:
        _update_ingest_job(job_id, {
            'status': 'processing', 'message': 'Parsing chord sheet…',
            'songName': song_name,
        })

        # Detect if the sheet has lyrics (spatial chord+lyric pairs)
        has_lyrics = _sheet_has_lyrics(chord_text)

        if has_lyrics:
            _do_ingest_lyric_align(job_id, audio_path, chord_text, song_name)
        else:
            _do_ingest_beat_align(job_id, audio_path, chord_text, song_name)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_ingest_job(job_id, {
            'status': 'error', 'message': f'Ingest failed: {exc}',
        })


def _sheet_has_lyrics(chord_text):
    """Check if a chord sheet has lyrics below chord lines (UG-style)."""
    lines = chord_text.strip().split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or re.match(r'^\[.*\]$', stripped):
            continue
        # If this is a chord line and the next non-empty line is NOT chords
        # and NOT a section header → it's probably lyrics
        tokens = stripped.split()
        clean = [t for t in tokens if t not in ('|', '/', '||')]
        clean = [t for t in clean if not re.match(r'^\[.*\]$', t)]
        if clean and sum(1 for t in clean if _CHORD_RE.match(t)) / len(clean) >= 0.5:
            # Found a chord line — check what follows
            for j in range(i + 1, len(lines)):
                next_stripped = lines[j].strip()
                if not next_stripped:
                    continue
                if re.match(r'^\[.*\]$', next_stripped):
                    break
                # Is the next non-empty line NOT a chord line?
                next_tokens = next_stripped.split()
                next_clean = [t for t in next_tokens if t not in ('|', '/')]
                if next_clean:
                    chord_ratio = sum(1 for t in next_clean if _CHORD_RE.match(t)) / len(next_clean)
                    if chord_ratio < 0.5:
                        return True  # Found lyrics below chords
                break
    return False


def _do_ingest_lyric_align(job_id, audio_path, chord_text, song_name):
    """Lyric-aware ingest: Whisper transcription + spatial chord parsing."""
    with _ingest_lock:
        _ingest_jobs[job_id]['message'] = 'Transcribing audio (Whisper)…'

    # Import the lyric aligner
    lyric_align_path = os.path.join(
        os.path.dirname(__file__), 'tools', 'lyric_align.py')
    import importlib.util
    spec = importlib.util.spec_from_file_location('lyric_align', lyric_align_path)
    lyric_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lyric_mod)

    # Run the full pipeline
    segments, transcript_words, spatial_chords, lyrics_words = \
        lyric_mod.align_chords_via_lyrics(chord_text, audio_path, model_size='base')

    if not segments:
        _update_ingest_job(job_id, {
            'status': 'error',
            'message': 'Lyric alignment failed — no chords or no transcript.',
        })
        return

    # Also run beat detection for BPM/key info
    with _ingest_lock:
        _ingest_jobs[job_id]['message'] = 'Detecting key & BPM…'
    chords_data, bpm_val, key_val, beat_times = detect_chords(audio_path)

    # Build JSON segments
    json_segments = []
    for seg in segments:
        json_segments.append({
            'start': round(seg['start'], 3) if seg['start'] is not None else 0,
            'end': round(seg['end'], 3) if seg['end'] is not None else 0,
            'sheetChord': seg['chord'],
            'predChord': seg['chord'],  # in lyric mode, sheet IS the truth
            'nBeats': 0,
            'matches': 0,
            'match': True,
            'word': seg.get('word', ''),
            'section': seg.get('section', ''),
            'confidence': seg.get('confidence', 0),
        })

    # Build transcript for UI display
    transcript_json = [
        {'word': w['word'], 'start': w['start'], 'end': w['end']}
        for w in transcript_words
    ]

    _update_ingest_job(job_id, {
        'status': 'done',
        'alignMode': 'lyric',
        'songName': song_name,
        'key': key_val,
        'bpm': round(bpm_val, 1),
        'beatTimes': beat_times,
        'segments': json_segments,
        'transcript': transcript_json,
        'sheetChords': [s['chord'] for s in segments],
        'lyricsWords': lyrics_words,
        'audioPath': audio_path,
        'chordText': chord_text,
    })


def _do_ingest_beat_align(job_id, audio_path, chord_text, song_name):
    """Beat-based ingest: DP alignment to beat-level predictions (original method)."""
    with _ingest_lock:
        _ingest_jobs[job_id]['message'] = 'Analyzing audio…'

    # 1. Parse chord tokens from text
    sheet_chords = _parse_chord_text(chord_text)
    if not sheet_chords:
        _update_ingest_job(job_id, {
            'status': 'error',
            'message': 'No chords found in the uploaded sheet.',
        })
        return

    # 2. Run Essentia / librosa analysis on the MP3
    chords_data, bpm_val, key_val, beat_times = detect_chords(audio_path)

    # 3. Reconstruct beat-level chords from merged segments
    beat_chords = []
    seg_idx = 0
    for bt in beat_times:
        while (seg_idx < len(chords_data) - 1
               and chords_data[seg_idx + 1]['start'] <= bt + 1e-4):
            seg_idx += 1
        beat_chords.append(_normalize_chord_name(chords_data[seg_idx]['chord']))

    # 4. DP alignment
    align_path = os.path.join(
        os.path.dirname(__file__), 'tools', 'align_chords.py')
    import importlib.util
    spec = importlib.util.spec_from_file_location('align_chords_tool', align_path)
    align_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_mod)

    segments = align_mod.dp_align(sheet_chords, beat_chords, beat_times)
    if not segments:
        _update_ingest_job(job_id, {
            'status': 'error',
            'message': 'Alignment failed — chord count vs song length mismatch.',
        })
        return

    json_segments = align_mod.alignment_to_json(segments)

    # 5. Store result for review
    _update_ingest_job(job_id, {
        'status': 'done',
        'alignMode': 'beat',
        'songName': song_name,
        'key': key_val,
        'bpm': round(bpm_val, 1),
        'beatTimes': beat_times,
        'segments': json_segments,
        'sheetChords': sheet_chords,
        'audioPath': audio_path,
        'chordText': chord_text,
    })


def _do_ingest_audio_only(job_id, audio_path, song_name):
    """Audio-only ingest: run beat detection, create blank segments for manual chord entry."""
    try:
        _analysis_start = time.time()
        _update_ingest_job(job_id, {
            'status': 'processing', 'message': 'Loading models and analyzing audio…',
            'songName': song_name,
        })
        app.logger.info(f'Ingest {job_id}: starting chord analysis for "{song_name}"')

        chords_data, bpm_val, key_val, beat_times = detect_chords(audio_path)
        app.logger.info(f'Ingest {job_id}: analysis done in {int(time.time() - _analysis_start)}s — '
                        f'key={key_val}, bpm={bpm_val:.1f}, beats={len(beat_times)}')

        # Group beats into segments of 4 beats each, with blank chord names
        beats_per_seg = 4
        json_segments = []
        for i in range(0, len(beat_times), beats_per_seg):
            start = beat_times[i]
            end_idx = min(i + beats_per_seg, len(beat_times))
            if end_idx < len(beat_times):
                end = beat_times[end_idx]
            elif beat_times:
                end = beat_times[-1] + (60.0 / max(bpm_val, 60))
            else:
                end = start + 2.0
            json_segments.append({
                'start': round(start, 3),
                'end': round(end, 3),
                'sheetChord': '',
                'predChord': '',
                'chord': '',
                'nBeats': min(beats_per_seg, end_idx - i),
                'matches': 0,
                'match': False,
            })

        _update_ingest_job(job_id, {
            'status': 'done',
            'alignMode': 'beat',
            'songName': song_name,
            'key': key_val,
            'bpm': round(bpm_val, 1),
            'beatTimes': beat_times,
            'segments': json_segments,
            'sheetChords': [],
            'audioPath': audio_path,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_ingest_job(job_id, {
            'status': 'error', 'message': f'Audio analysis failed: {exc}',
        })


# Public listing; override with SEECHORDS_CHROME_STORE_URL if the URL ever changes.
DEFAULT_CHROME_STORE_URL = (
    'https://chromewebstore.google.com/detail/SeeChords/bkmkkgblbnakckgdehgjaggnmglcljmj'
)
# Tip page; override with SEECHORDS_DONATION_URL if you switch platforms.
DEFAULT_DONATION_URL = 'https://buymeacoffee.com/devjinn'


@app.route('/')
def site_home():
    """Marketing landing page for the SeeChords browser extension."""
    donation_url = os.environ.get('SEECHORDS_DONATION_URL', DEFAULT_DONATION_URL).strip()
    chrome_store_url = os.environ.get('SEECHORDS_CHROME_STORE_URL', DEFAULT_CHROME_STORE_URL).strip()
    return render_template(
        'site_home.html',
        donation_url=donation_url,
        chrome_store_url=chrome_store_url,
    )


@app.route('/privacy')
def privacy_policy():
    """Serve the extension privacy policy."""
    pp_path = os.path.join(os.path.dirname(__file__), '..', 'extension', 'privacy-policy.html')
    if not os.path.isfile(pp_path):
        pp_path = os.path.join(os.path.dirname(__file__), 'privacy-policy.html')
    with open(pp_path) as f:
        return f.read()


@app.route('/report-bug')
def report_bug():
    """Bug report and feature request form for the SeeChords extension."""
    return render_template('report_bug.html')


@app.route('/play')
def play():
    """Web-based chord player: upload audio, get beat-synced chord display."""
    return render_template('play.html')


@app.route('/browse')
def browse():
    """Browse all analyzed songs in the database."""
    con = _get_db()
    rows = con.execute('''
        SELECT video_id, title, key, bpm, analyzed_at
        FROM chord_versions
        WHERE is_active = 1
        GROUP BY video_id
        ORDER BY analyzed_at DESC
    ''').fetchall()
    con.close()
    songs = []
    for r in rows:
        d = _row_to_dict(r, ['video_id', 'title', 'key', 'bpm', 'analyzed_at'])
        songs.append({
            'videoId': d['video_id'],
            'title': d['title'] or d['video_id'],
            'key': d['key'] or '',
            'bpm': round(d['bpm']) if d['bpm'] else '',
            'thumbnail': f"https://i.ytimg.com/vi/{d['video_id']}/mqdefault.jpg",
        })
    return render_template('browse.html', songs=songs)


@app.route('/chords/<video_id>')
def chord_viewer(video_id):
    """Standalone chord viewer: loads cached chords by video ID, no audio/analysis."""
    if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return 'Invalid video ID', 400
    return render_template('chords.html', video_id=video_id)


def _resolve_yt_stream(video_id):
    """Return (stream_url, content_type) for a YouTube video, using a 5-hour in-memory cache."""
    now = time.time()
    with _yt_stream_lock:
        cached = _yt_stream_cache.get(video_id)
        if cached and cached['expires'] > now:
            return cached['url'], cached['content_type']

    import yt_dlp
    import tempfile
    tmp_cookie = None
    yt_url = f'https://www.youtube.com/watch?v={video_id}'
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }
    cf = _ytdlp_cookiefile()
    if cf:
        ydl_opts['cookiefile'] = cf
    else:
        b64 = _normalize_ytdlp_cookies_b64(os.environ.get('YTDLP_COOKIES_B64') or '')
        if b64:
            cookie_bytes = base64.b64decode(b64)
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='wb') as tf:
                tf.write(cookie_bytes)
                tmp_cookie = tf.name
            ydl_opts['cookiefile'] = tmp_cookie

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(yt_url, download=False)

        stream_url = info.get('url')
        content_type = 'audio/mp4'
        if not stream_url and info.get('formats'):
            audio_fmts = [f for f in info['formats']
                          if f.get('vcodec') == 'none' and f.get('acodec') != 'none']
            best = audio_fmts[-1] if audio_fmts else info['formats'][-1]
            stream_url = best.get('url')
            if best.get('ext') == 'webm':
                content_type = 'audio/webm'

        if not stream_url:
            raise ValueError('No stream URL found in yt-dlp response')

        with _yt_stream_lock:
            _yt_stream_cache[video_id] = {
                'url': stream_url,
                'content_type': content_type,
                'expires': now + 5 * 3600,
            }
        return stream_url, content_type
    finally:
        if tmp_cookie:
            try:
                os.remove(tmp_cookie)
            except OSError:
                pass


@app.route('/api/youtube-stream/<video_id>')
def youtube_stream(video_id):
    """Proxy YouTube audio to the browser, forwarding Range headers so seeking works."""
    if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return jsonify({'error': 'Invalid video ID.'}), 400

    try:
        stream_url, content_type = _resolve_yt_stream(video_id)
    except Exception as e:
        print(f'[SeeChords] YouTube stream resolve failed for {video_id}: {e}', flush=True)
        return jsonify({'error': f'Stream unavailable: {e}'}), 500

    up_headers = {'User-Agent': 'Mozilla/5.0'}
    range_hdr = request.headers.get('Range')
    if range_hdr:
        up_headers['Range'] = range_hdr

    import requests as req_lib
    try:
        upstream = req_lib.get(stream_url, headers=up_headers, stream=True, timeout=30)
    except Exception as e:
        print(f'[SeeChords] YouTube proxy fetch failed for {video_id}: {e}', flush=True)
        return jsonify({'error': 'Upstream fetch failed'}), 502

    resp_headers = {
        'Content-Type': upstream.headers.get('Content-Type', content_type),
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-cache',
    }
    for h in ('Content-Length', 'Content-Range'):
        if h in upstream.headers:
            resp_headers[h] = upstream.headers[h]

    def generate():
        for chunk in upstream.iter_content(chunk_size=65536):
            if chunk:
                yield chunk

    return Response(stream_with_context(generate()),
                    status=upstream.status_code,
                    headers=resp_headers)


def _require_ingest_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('ingest_auth'):
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect('/ingest')
        return f(*args, **kwargs)
    return decorated


@app.route('/ingest/login', methods=['POST'])
def ingest_login():
    password = request.form.get('password', '')
    if not INGEST_PASSWORD:
        return redirect('/ingest')
    if password == INGEST_PASSWORD:
        session['ingest_auth'] = True
        return redirect('/ingest')
    return render_template('ingest.html', login_error='Incorrect password', show_login=True)


@app.route('/ingest/logout', methods=['POST'])
def ingest_logout():
    session.pop('ingest_auth', None)
    return redirect('/ingest')


@app.route('/ingest')
def ingest_page():
    """Serve the chord sheet ingest + review UI (auth-gated)."""
    if INGEST_PASSWORD and not session.get('ingest_auth'):
        return render_template('ingest.html', show_login=True)
    return render_template('ingest.html', show_login=False)


@app.route('/api/ingest/saved-labs')
@_require_ingest_auth
def list_saved_labs():
    """List all .lab files saved as training data (server + training dirs)."""
    labs = []
    seen = set()
    label_dirs = [
        ('server', os.path.join(SERVER_VERIFIED_DIR, 'labels')),
        ('training', os.path.join(TRAINING_VERIFIED_DIR, 'labels')),
    ]
    # Check which files have matching audio
    audio_dir = os.path.join(TRAINING_VERIFIED_DIR, 'audio')
    for source, label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        for f in sorted(os.listdir(label_dir)):
            if not f.endswith('.lab') or f in seen:
                continue
            seen.add(f)
            path = os.path.join(label_dir, f)
            stat = os.stat(path)
            with open(path) as fh:
                lines = [l for l in fh if l.strip()]
            stem = f[:-4]
            name = stem.replace('_', ' ')
            has_audio = os.path.isfile(os.path.join(audio_dir, f'{stem}.wav'))
            labs.append({
                'filename': f,
                'name': name,
                'segments': len(lines),
                'savedAt': int(stat.st_mtime),
                'size': stat.st_size,
                'source': source,
                'hasAudio': has_audio,
            })
    labs.sort(key=lambda x: x['savedAt'], reverse=True)
    return jsonify(labs)


@app.route('/api/ingest/saved-labs/<name>')
@_require_ingest_auth
def get_saved_lab(name):
    """Read a saved .lab file and return parsed segments + metadata from DB."""
    # Sanitise to prevent path traversal
    safe = os.path.basename(name)
    if not safe.endswith('.lab'):
        safe += '.lab'
    # Look in server dir first, then training dir
    path = os.path.join(SERVER_VERIFIED_DIR, 'labels', safe)
    if not os.path.isfile(path):
        path = os.path.join(TRAINING_VERIFIED_DIR, 'labels', safe)
    if not os.path.isfile(path):
        return jsonify({'error': 'Not found'}), 404
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                segments.append({
                    'start': float(parts[0]),
                    'end': float(parts[1]),
                    'chord': parts[2],
                })
    # Find matching audio file (audio lives in training dir only)
    stem = safe[:-4]
    audio_dir = os.path.join(TRAINING_VERIFIED_DIR, 'audio')
    audio_url = None
    if os.path.isdir(audio_dir):
        for ext in ('.wav', '.m4a', '.mp3', '.ogg', '.flac'):
            if os.path.isfile(os.path.join(audio_dir, stem + ext)):
                audio_url = f'/api/ingest/silver-audio/{stem}{ext}'
                break

    # Try to load metadata from video_map.json first (preserves original BPM/key)
    # Check video_map.json in both server and training dirs
    bpm = None
    key = None
    beat_times = None
    video_id = None
    for map_dir in (SERVER_VERIFIED_DIR, TRAINING_VERIFIED_DIR):
        mp = os.path.join(map_dir, 'video_map.json')
        if os.path.isfile(mp):
            with open(mp) as mf:
                vmap = json.load(mf)
            entry = vmap.get(stem, {})
            if entry:
                bpm = entry.get('bpm')
                key = entry.get('key')
                beat_times = entry.get('beatTimes')
                video_id = entry.get('videoId')
                break
    if not video_id:
        # Fall back to fuzzy DB lookup
        stem_lower = stem.replace('_', ' ').lower()
        stem_words = [w for w in stem_lower.split() if len(w) > 2]
        con = _get_db()
        best_match = None
        best_score = 0
        _fuzzy_cols = ['video_id', 'title', 'key', 'bpm', 'beat_times']
        if stem_words:
            rows = con.execute(
                'SELECT video_id, title, key, bpm, beat_times FROM chord_versions '
                'WHERE is_active = 1 ORDER BY version_id DESC'
            ).fetchall()
            for r in rows:
                rd = _row_to_dict(r, _fuzzy_cols)
                title_lower = (rd['title'] or '').lower()
                matches = sum(1 for w in stem_words if w in title_lower)
                score = matches / len(stem_words)
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_match = rd
            if best_match:
                bpm = best_match['bpm']
                key = best_match['key']
                beat_times = json.loads(best_match['beat_times']) if best_match['beat_times'] else None
                video_id = best_match['video_id']
        con.close()

    return jsonify({
        'filename': safe,
        'name': safe[:-4].replace('_', ' '),
        'segments': segments,
        'audioUrl': audio_url,
        'bpm': bpm,
        'key': key,
        'beatTimes': beat_times,
        'videoId': video_id,
    })


@app.route('/api/ingest/silver-audio/<path:filename>')
@_require_ingest_auth
def serve_silver_audio(filename):
    """Serve an audio file from training verified/audio/."""
    safe = os.path.basename(filename)
    audio_dir = os.path.join(TRAINING_VERIFIED_DIR, 'audio')
    path = os.path.join(audio_dir, safe)
    if not os.path.isfile(path):
        return jsonify({'error': 'Not found'}), 404
    return send_file(path)


@app.route('/api/ingest/saved-labs/<name>', methods=['PUT'])
@_require_ingest_auth
def update_saved_lab(name):
    """Update a saved .lab file with edited segments.
    Also creates a new chord_versions entry when a matching video exists."""
    safe = os.path.basename(name)
    if not safe.endswith('.lab'):
        safe += '.lab'
    path = os.path.join(SERVER_VERIFIED_DIR, 'labels', safe)
    if not os.path.isfile(path):
        return jsonify({'error': 'Not found'}), 404
    data = request.get_json(force=True)
    segments = data.get('segments', [])
    if not segments:
        return jsonify({'error': 'No segments provided.'}), 400
    # Write .lab file (Isophonics notation for training)
    with open(path, 'w') as f:
        for seg in segments:
            start = float(seg['start'])
            end = float(seg['end'])
            chord = seg.get('chord', 'N')
            f.write(f'{start:.6f} {end:.6f} {chord}\n')

    # Mirror to training directory
    training_labels_dir = os.path.join(TRAINING_VERIFIED_DIR, 'labels')
    if os.path.isdir(training_labels_dir):
        import shutil
        shutil.copy2(path, os.path.join(training_labels_dir, safe))

    # Update metadata in video_map.json if BPM/key provided, or preserve existing
    stem = safe[:-4]
    map_path = os.path.join(SERVER_VERIFIED_DIR, 'video_map.json')
    vmap = {}
    if os.path.isfile(map_path):
        with open(map_path) as mf:
            vmap = json.load(mf)
    entry = vmap.get(stem, {})
    if data.get('bpm') is not None:
        entry['bpm'] = data['bpm']
    if data.get('key') is not None:
        entry['key'] = data['key']
    if data.get('videoId') is not None:
        entry['videoId'] = data['videoId']
    if data.get('beatTimes') is not None:
        entry['beatTimes'] = data['beatTimes']
    vmap[stem] = entry
    _dump_video_map(vmap, map_path)

    # Also save as a chord_versions entry if we can match to a video
    video_id = data.get('videoId')
    version_id = None
    if not video_id:
        # Try to find a matching video by fuzzy title match on the filename stem
        stem = safe[:-4].replace('_', ' ').lower()
        con = _get_db()
        rows = con.execute('SELECT video_id, title FROM chord_versions GROUP BY video_id').fetchall()
        con.close()
        for r in rows:
            rd = _row_to_dict(r, ['video_id', 'title'])
            title_lower = (rd['title'] or '').lower()
            stem_words = [w for w in stem.split() if len(w) > 2]
            matches = sum(1 for w in stem_words if w in title_lower)
            if stem_words and matches >= len(stem_words) * 0.5:
                video_id = rd['video_id']
                break

    if video_id:
        # Convert Isophonics segments to simple chords for DB
        simple_chords = []
        for seg in segments:
            ch = seg.get('chord', 'N')
            if ch == 'N':
                continue
            # Isophonics -> simple: G:min -> Gm, C:maj -> C, C7:maj -> C7
            m = re.match(r'^([A-G][#b]?)(\d*):?(maj|min|dim|aug|sus2|sus4)?$', ch)
            if m:
                root, num, qual = m.group(1), m.group(2), m.group(3)
                simple = root + num
                if qual == 'min':
                    simple += 'm'
                elif qual and qual != 'maj':
                    simple += qual
                ch = simple
            simple_chords.append({
                'chord': ch,
                'start': round(float(seg['start']), 3),
                'end': round(float(seg['end']), 3),
            })

        con = _get_db()
        existing = con.execute(
            'SELECT beat_times, key, bpm, title FROM chord_versions WHERE video_id = ? ORDER BY version_id DESC LIMIT 1',
            (video_id,)
        ).fetchone()
        if existing:
            ed = _row_to_dict(existing, ['beat_times', 'key', 'bpm', 'title'])
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
            con.execute(
                'INSERT INTO chord_versions (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (video_id, ed['title'], ed['key'], ed['bpm'],
                 json.dumps(simple_chords), ed['beat_times'], 'user-edited', now, 1)
            )
            new_id = con.execute('SELECT last_insert_rowid()').fetchone()[0]
            con.execute('UPDATE chord_versions SET is_active = 0 WHERE video_id = ? AND version_id != ?', (video_id, new_id))
            con.commit()
            version_id = new_id
        con.close()

    return jsonify({'saved': True, 'filename': safe, 'segmentCount': len(segments), 'versionId': version_id})


@app.route('/api/ingest', methods=['POST'])
@_require_ingest_auth
def ingest_upload():
    """Upload audio for beat analysis. Chords are added manually in the UI.

    Form fields:
      - audio: MP3 file (required)
      - name: song name override (optional)
      - videoId: optional YouTube video ID
    """
    audio_file = request.files.get('audio')
    song_name = request.form.get('name', '').strip()

    if not audio_file or not audio_file.filename:
        return jsonify({'error': 'Audio file is required.'}), 400

    job_id = str(uuid.uuid4())

    # Save audio
    audio_ext = os.path.splitext(audio_file.filename)[1].lower() or '.mp3'
    allowed_audio = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}
    if audio_ext not in allowed_audio:
        return jsonify({'error': f'Unsupported audio format: {audio_ext}'}), 400
    audio_path = os.path.join(UPLOAD_DIR, f'{job_id}_audio{audio_ext}')
    audio_file.save(audio_path)

    if not song_name:
        song_name = _clean_title(audio_file.filename)

    # Optional videoId
    video_id_form = request.form.get('videoId', '').strip()

    # Store job with optional videoId
    with _ingest_lock:
        _ingest_jobs[job_id] = {
            'status': 'processing',
            'message': 'Uploading…',
        }
        if video_id_form:
            _ingest_jobs[job_id]['videoId'] = video_id_form

    # Run audio-only analysis in background
    t = threading.Thread(
        target=_do_ingest_audio_only,
        args=(job_id, audio_path, song_name),
        daemon=True,
    )
    t.start()

    return jsonify({'jobId': job_id})


@app.route('/api/ingest/youtube', methods=['POST'])
@_require_ingest_auth
def ingest_youtube():
    """Download audio from YouTube for beat analysis. Chords are added manually.

    Form fields:
      - videoId: YouTube video ID (required)
      - name: song name override (optional)
    """
    video_id = (request.form.get('videoId') or '').strip()
    if not video_id or not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return jsonify({'error': 'Invalid YouTube video ID.'}), 400

    song_name = request.form.get('name', '').strip()

    # Check if this video already has chord data — skip re-analysis
    # Return the full job data inline so the client can render immediately
    # (avoids multi-machine routing issues with in-memory job store)
    existing = _cache_get(video_id)
    if existing:
        chords = existing.get('chords', [])
        beat_times = existing.get('beat_times', [])
        segments = []
        for c in chords:
            segments.append({
                'start': round(float(c['start']), 3),
                'end': round(float(c['end']), 3),
                'chord': c.get('chord', 'N'),
                'sheetChord': c.get('chord', ''),
                'predChord': c.get('chord', ''),
                'nBeats': 4,
                'matches': 1,
                'match': True,
            })
        return jsonify({
            'cached': True,
            'status': 'done',
            'alignMode': 'beat',
            'songName': song_name or existing.get('title', video_id),
            'key': existing.get('key', '?'),
            'bpm': existing.get('bpm', 120),
            'beatTimes': beat_times,
            'segments': segments,
            'sheetChords': [],
            'videoId': video_id,
        })

    job_id = str(uuid.uuid4())

    with _ingest_lock:
        _ingest_jobs[job_id] = {
            'status': 'processing',
            'message': 'Downloading audio from YouTube…',
            'videoId': video_id,
        }

    def _yt_download_and_ingest():
        audio_path = os.path.join(UPLOAD_DIR, f'{job_id}_audio.m4a')
        _dl_start = time.time()
        try:
            import yt_dlp
            yt_url = f'https://www.youtube.com/watch?v={video_id}'

            def _dl_progress(d):
                if d.get('status') == 'downloading':
                    pct = d.get('_percent_str', '').strip()
                    speed = d.get('_speed_str', '').strip()
                    eta = d.get('_eta_str', '').strip()
                    elapsed = int(time.time() - _dl_start)
                    parts = [f'Downloading audio ({pct})' if pct else f'Downloading audio ({elapsed}s)']
                    if speed:
                        parts.append(speed)
                    if eta:
                        parts.append(f'ETA {eta}')
                    _update_ingest_job(job_id, {
                        'status': 'processing',
                        'message': ' — '.join(parts),
                    })
                elif d.get('status') == 'finished':
                    _update_ingest_job(job_id, {
                        'status': 'processing',
                        'message': 'Download complete, converting…',
                    })

            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio',
                'outtmpl': audio_path,
                'noplaylist': True,
                'quiet': True,
                'no_warnings': False,
                'progress_hooks': [_dl_progress],
                'socket_timeout': 30,
            }
            cf = _ytdlp_cookiefile()
            if cf:
                ydl_opts['cookiefile'] = cf
                app.logger.info(f'Ingest {job_id}: using cookie file for yt-dlp')
            else:
                app.logger.info(f'Ingest {job_id}: no cookies available')

            app.logger.info(f'Ingest {job_id}: starting download for {video_id}')
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(yt_url, download=True)
                yt_title = info.get('title', video_id)
            app.logger.info(f'Ingest {job_id}: download finished in {int(time.time() - _dl_start)}s')

            if not os.path.exists(audio_path):
                _update_ingest_job(job_id, {
                    'status': 'error',
                    'message': 'YouTube download failed — no audio file produced.',
                })
                return

            name = song_name or _clean_title(yt_title)
            _update_ingest_job(job_id, {
                'status': 'processing',
                'message': f'Download complete, analyzing beats and chords…',
            })

            _do_ingest_audio_only(job_id, audio_path, name)

            # Add audioUrl to the completed job so the browser can play it
            with _ingest_lock:
                job = _ingest_jobs.get(job_id)
                if job and job.get('status') == 'done':
                    job['audioUrl'] = f'/api/ingest/audio/{job_id}'

        except Exception as e:
            import traceback
            app.logger.error(f'Ingest {job_id}: download failed — {e}\n{traceback.format_exc()}')
            _update_ingest_job(job_id, {
                'status': 'error',
                'message': f'YouTube download failed: {e}',
            })

    t = threading.Thread(target=_yt_download_and_ingest, daemon=True)
    t.start()
    return jsonify({'jobId': job_id})


@app.route('/api/ingest/audio/<job_id>')
@_require_ingest_auth
def serve_ingest_audio(job_id):
    """Serve a downloaded/uploaded audio file for playback."""
    if not re.match(r'^[a-f0-9-]+$', job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    # Find the audio file for this job
    for ext in ('.m4a', '.mp3', '.wav', '.ogg', '.flac', '.aac'):
        path = os.path.join(UPLOAD_DIR, f'{job_id}_audio{ext}')
        if os.path.isfile(path):
            return send_file(path)
    return jsonify({'error': 'Audio not found'}), 404


def _find_ingest_audio_path(job_id):
    """Return on-disk uploaded audio path for an ingest job ID, if present."""
    if not job_id or not re.match(r'^[a-f0-9-]+$', job_id):
        return ''
    for ext in ('.m4a', '.mp3', '.wav', '.ogg', '.flac', '.aac'):
        path = os.path.join(UPLOAD_DIR, f'{job_id}_audio{ext}')
        if os.path.isfile(path):
            return path
    return ''


@app.route('/api/ingest/<job_id>')
@_require_ingest_auth
def ingest_status(job_id):
    """Check ingest job status / get alignment result for review."""
    with _ingest_lock:
        job = _ingest_jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(job)


@app.route('/api/ingest/<job_id>/save', methods=['POST'])
@_require_ingest_auth
def ingest_save(job_id):
    """Save reviewed alignment as a .lab file for training.

    Body JSON: { segments: [{start, end, chord}, ...], songName? }
    The user may have edited chords/boundaries in the review UI.
    """
    data = request.get_json(force=True)
    with _ingest_lock:
        job = _ingest_jobs.get(job_id)
    if not job or job.get('status') != 'done':
        job = {}

    segments = data.get('segments', [])
    song_name = data.get('songName', job.get('songName', 'unknown'))

    if not segments:
        return jsonify({'error': 'No segments to save.'}), 400

    # Sanitise song name for filesystem
    safe_name = re.sub(r'[^\w\s\-]', '', song_name).strip().replace(' ', '_')
    if not safe_name:
        safe_name = job_id[:8]

    # Build .lab content
    lab_content = ''
    for seg in segments:
        chord = seg.get('chord', 'N')
        start = float(seg['start'])
        end = float(seg['end'])
        iso = _display_to_iso(chord)
        lab_content += f'{start:.6f} {end:.6f} {iso}\n'

    # Create a chord_versions DB entry so the song appears in the extension
    video_id = job.get('videoId') or safe_name
    key_val = _infer_key_from_segments(segments) or job.get('key', '?')
    bpm_val = round(float(job.get('bpm', 120)), 1)
    beat_times = data.get('beatTimes', job.get('beatTimes', []))
    song_title = song_name or safe_name

    # Write labels to both dirs, meta only to server dir
    lab_path = os.path.join(SERVER_VERIFIED_DIR, 'labels', f'{safe_name}.lab')
    for base_dir in (SERVER_VERIFIED_DIR, TRAINING_VERIFIED_DIR):
        lbl_dir = os.path.join(base_dir, 'labels')
        os.makedirs(lbl_dir, exist_ok=True)
        with open(os.path.join(lbl_dir, f'{safe_name}.lab'), 'w') as f:
            f.write(lab_content)

    # Update video_map.json with metadata
    map_path = os.path.join(SERVER_VERIFIED_DIR, 'video_map.json')
    vmap = {}
    if os.path.isfile(map_path):
        with open(map_path) as f:
            vmap = json.load(f)
    vmap[safe_name] = {**vmap.get(safe_name, {}), 'videoId': video_id,
                       'title': song_title, 'bpm': bpm_val, 'key': key_val,
                       'beatTimes': beat_times}
    _dump_video_map(vmap, map_path)

    # Audio only to training dir
    aud_dir = os.path.join(TRAINING_VERIFIED_DIR, 'audio')
    os.makedirs(aud_dir, exist_ok=True)
    if audio_src and os.path.exists(audio_src):
        audio_dest = os.path.join(aud_dir, f'{safe_name}.wav')
        if not os.path.exists(audio_dest):
            subprocess.run(
                ['ffmpeg', '-i', audio_src, '-vn', '-ar', '44100', '-ac', '1',
                 audio_dest, '-y'],
                capture_output=True, timeout=180,
            )

    simple_chords = []
    for seg in segments:
        ch = seg.get('chord', 'N')
        if ch != 'N':
            simple_chords.append({
                'chord': ch,
                'start': round(float(seg['start']), 3),
                'end': round(float(seg['end']), 3),
            })

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    con = _get_db()
    con.execute(
        'UPDATE chord_versions SET is_active = 0 WHERE video_id = ?', (video_id,)
    )
    con.execute('''
        INSERT INTO chord_versions
            (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, 'ingest', ?, 1)
    ''', (video_id, song_title, key_val, bpm_val,
          json.dumps(simple_chords), json.dumps(beat_times), now))
    con.commit()
    con.close()

    # Keep job in memory so users can save again or promote after saving.
    with _ingest_lock:
        if job_id in _ingest_jobs:
            _ingest_jobs[job_id]['lastSavedAt'] = now
            _ingest_jobs[job_id]['lastSavedSong'] = safe_name

    return jsonify({
        'saved': True,
        'labPath': lab_path,
        'songName': safe_name,
        'segmentCount': len(segments),
        'videoId': video_id,
    })


# ── Save / Load chord versions (Play Along edits) ─────────────────

@app.route('/api/ingest/<job_id>/save-version', methods=['POST'])
@_require_ingest_auth
def ingest_save_version(job_id):
    """Save edited segments from Play Along as a new version in chord_versions DB."""
    data = request.get_json(force=True)
    with _ingest_lock:
        job = _ingest_jobs.get(job_id)
    if not job or job.get('status') != 'done':
        job = {}

    segments = data.get('segments', [])
    song_name = data.get('songName', job.get('songName', 'unknown'))
    key_val = _infer_key_from_segments(segments) or data.get('key', job.get('key', '?'))
    bpm_val = float(data.get('bpm', job.get('bpm', 120)))
    beat_times = data.get('beatTimes', job.get('beatTimes', []))

    if not segments:
        return jsonify({'error': 'No segments to save.'}), 400

    video_id = data.get('videoId') or job_id[:11]

    # Build chords_data in the format chord_versions expects
    chords_data = []
    for seg in segments:
        chords_data.append({
            'chord': seg.get('chord', 'N'),
            'start': float(seg['start']),
            'end': float(seg['end']),
        })

    # Upsert: delete any existing ingest-edit version, insert fresh
    con = _get_db()
    con.execute("DELETE FROM chord_versions WHERE video_id = ? AND source = 'ingest-edit'", (video_id,))
    con.execute('''
        INSERT INTO chord_versions
            (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, 'ingest-edit', ?, 1)
    ''', (video_id, song_name, key_val, round(bpm_val, 1),
          json.dumps(chords_data), json.dumps(beat_times),
          int(time.time())))
    con.commit()
    version_id = con.execute('SELECT last_insert_rowid()').fetchone()[0]
    con.close()

    return jsonify({
        'saved': True,
        'versionId': version_id,
        'videoId': video_id,
        'segmentCount': len(segments),
    })


@app.route('/api/ingest/<job_id>/versions')
@_require_ingest_auth
def ingest_list_versions(job_id):
    """List all saved versions for an ingest job."""
    video_id = job_id[:11]
    return _list_versions(video_id)


@app.route('/api/versions/<video_id>')
def list_versions(video_id):
    """List all versions for a video_id."""
    return _list_versions(video_id)


def _list_versions(video_id):
    con = _get_db()
    _ver_cols = ['version_id', 'key', 'bpm', 'analyzed_at', 'is_active', 'chords', 'source']
    rows = con.execute(
        '''SELECT version_id, key, bpm, analyzed_at, is_active, chords, source
           FROM chord_versions WHERE video_id = ? AND source != 'ingest-edit'
           ORDER BY
             (source = 'verified') DESC,
             (source = ?) DESC,
             version_id DESC''',
        (video_id, CURRENT_MODEL_SOURCE),
    ).fetchall()
    con.close()

    versions = []
    for r in rows:
        d = _row_to_dict(r, _ver_cols)
        chords = json.loads(d['chords']) if isinstance(d['chords'], str) else d['chords']
        versions.append({
            'versionId': d['version_id'],
            'key': d['key'],
            'bpm': d['bpm'],
            'analyzedAt': d['analyzed_at'],
            'isActive': bool(d['is_active']),
            'segmentCount': len(chords),
            'source': d['source'],
        })
    return jsonify({'versions': versions})


@app.route('/api/ingest/versions')
@_require_ingest_auth
def ingest_all_versions():
    """List all ingest-created and verified versions for the home page."""
    con = _get_db()
    rows = con.execute('''
        SELECT cv.version_id, cv.video_id, cv.title, cv.key, cv.bpm,
               cv.source, cv.analyzed_at, cv.is_active, cv.chords
        FROM chord_versions cv
        WHERE cv.source IN ('ingest-edit', 'verified')
        ORDER BY cv.analyzed_at DESC
    ''').fetchall()
    con.close()

    versions = []
    seen_videos = set()
    for r in rows:
        d = {
            'versionId': r[0], 'videoId': r[1], 'title': r[2],
            'key': r[3], 'bpm': r[4], 'source': r[5],
            'analyzedAt': r[6], 'isActive': bool(r[7]),
        }
        chords = json.loads(r[8]) if isinstance(r[8], str) else (r[8] or [])
        d['segmentCount'] = len(chords)
        vid = r[1]
        if vid in seen_videos:
            continue
        seen_videos.add(vid)
        versions.append(d)
    return jsonify({'versions': versions})


@app.route('/api/version/<int:version_id>')
def get_version(version_id):
    """Load a specific version by version_id."""
    con = _get_db()
    row = con.execute(
        'SELECT * FROM chord_versions WHERE version_id = ?',
        (version_id,),
    ).fetchone()
    con.close()

    if not row:
        return jsonify({'error': 'Version not found.'}), 404

    return jsonify(_version_row_to_dict(row))


# ── Batch re-analysis ────────────────────────────────────────────

@app.route('/api/ingest/reanalyze-batch', methods=['POST'])
@_require_ingest_auth
def reanalyze_batch():
    """Queue re-analysis for next batch of songs missing downbeats.
    Call repeatedly until remaining=0."""
    BATCH_SIZE = 5
    con = _get_db()
    rows = con.execute(
        "SELECT DISTINCT video_id, title FROM chord_versions WHERE downbeats IS NULL OR downbeats = '[]' LIMIT ?",
        (BATCH_SIZE,),
    ).fetchall()
    remaining = con.execute(
        "SELECT COUNT(DISTINCT video_id) FROM chord_versions WHERE downbeats IS NULL OR downbeats = '[]'"
    ).fetchone()[0]
    con.close()

    if not rows:
        return jsonify({'queued': 0, 'remaining': 0, 'message': 'All songs already have downbeats.'})

    queued = []
    for r in rows:
        video_id, title = r[0], r[1] or ''
        job_id = f'reanalyze-{video_id}-{int(time.time())}'
        try:
            _create_job(job_id, video_id, status='pending', message='Queued for re-analysis')
            _spawn_worker(job_id, video_id, title)
            queued.append(video_id)
        except Exception as e:
            print(f'[SeeChords] Failed to queue {video_id}: {e}', flush=True)

    return jsonify({'queued': len(queued), 'remaining': remaining - len(queued), 'videoIds': queued})


# ── Delete a version ──────────────────────────────────────────────

@app.route('/api/ingest/version/<int:version_id>', methods=['DELETE'])
@_require_ingest_auth
def delete_version(version_id):
    """Delete a chord version by ID."""
    con = _get_db()
    row = con.execute('SELECT version_id FROM chord_versions WHERE version_id = ?', (version_id,)).fetchone()
    if not row:
        con.close()
        return jsonify({'error': 'Version not found.'}), 404
    con.execute('DELETE FROM chord_versions WHERE version_id = ?', (version_id,))
    con.commit()
    con.close()
    return jsonify({'deleted': True, 'versionId': version_id})


# ── Promote a version to verified (human-checked) ─────────────────

@app.route('/api/ingest/<job_id>/promote-verified', methods=['POST'])
@_require_ingest_auth
def promote_to_verified(job_id):
    """Save & Upload: creates a chord_versions entry with source='verified'."""
    data = request.get_json(force=True)
    with _ingest_lock:
        job = _ingest_jobs.get(job_id)
    if not job or job.get('status') != 'done':
        job = {}

    segments = data.get('segments', [])
    song_name = data.get('songName', job.get('songName', 'unknown'))
    video_id = data.get('videoId') or job.get('videoId')

    if not segments:
        return jsonify({'error': 'No segments to save.'}), 400
    if not video_id or not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return jsonify({'error': 'Valid videoId required.'}), 400

    key_val = _infer_key_from_segments(segments) or data.get('key', job.get('key', '?'))
    bpm_val = round(float(data.get('bpm', job.get('bpm', 120))), 1)
    beat_times = data.get('beatTimes', job.get('beatTimes', []))

    display_chords = []
    for seg in segments:
        ch = seg.get('chord', 'N')
        if ch != 'N':
            display_chords.append({
                'chord': ch,
                'start': round(float(seg['start']), 3),
                'end': round(float(seg['end']), 3),
            })

    con = _get_db()
    # Delete previous verified and draft versions; keep model-generated ones as-is
    con.execute("DELETE FROM chord_versions WHERE video_id = ? AND source IN ('verified', 'ingest-edit')", (video_id,))
    con.execute('''
        INSERT INTO chord_versions
            (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, 'verified', ?, 1)
    ''', (video_id, song_name, key_val, bpm_val,
          json.dumps(display_chords), json.dumps(beat_times), int(time.time())))
    con.commit()
    version_id = con.execute('SELECT last_insert_rowid()').fetchone()[0]
    con.close()

    return jsonify({
        'saved': True,
        'promoted': True,
        'versionId': version_id,
        'videoId': video_id,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, port=port, use_reloader=False)
