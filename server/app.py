"""SeeChords backend – chord analysis API for the Chrome extension.

Endpoints:
  GET  /api/chords/<videoId>   → cached chord JSON or 404
  POST /api/analyze            → upload MP3 + videoId, returns job_id
  GET  /api/status/<job_id>    → poll analysis progress
  GET  /api/health             → diagnostic info
"""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # prevent OpenMP crash (essentia + torch)
import uuid
import threading
import shutil
import re
import json
import sqlite3
import time
import subprocess

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import librosa
import numpy as np

# Fix macOS Python SSL
try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
except ImportError:
    pass

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

DB_PATH              = os.environ.get('DB_PATH', os.path.join(os.path.dirname(__file__), 'seechords.db'))
UPLOAD_DIR           = os.environ.get('UPLOAD_DIR', os.path.join(os.path.dirname(__file__), 'uploads'))
TRAINING_VERIFIED_DIR = os.path.join(os.path.dirname(__file__), '..', 'training', 'data', 'verified')
SERVER_VERIFIED_DIR  = os.path.join(os.path.dirname(__file__), 'verified')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# SQLite persistent cache
# ─────────────────────────────────────────────

def _init_db():
    con = sqlite3.connect(DB_PATH)
    # New versioned table
    con.execute('''
        CREATE TABLE IF NOT EXISTS chord_versions (
            version_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id    TEXT NOT NULL,
            title       TEXT,
            key         TEXT,
            bpm         REAL,
            chords      TEXT,   -- JSON array
            beat_times  TEXT,   -- JSON array
            source      TEXT DEFAULT 'user-uploaded',
            analyzed_at INTEGER,
            is_active   INTEGER DEFAULT 0
        )
    ''')
    con.execute('CREATE INDEX IF NOT EXISTS idx_cv_video ON chord_versions(video_id)')
    # Ratings table
    con.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id  INTEGER NOT NULL,
            video_id    TEXT NOT NULL,
            stars       INTEGER NOT NULL CHECK(stars BETWEEN 1 AND 5),
            created_at  INTEGER
        )
    ''')
    con.execute('CREATE INDEX IF NOT EXISTS idx_ratings_version ON ratings(version_id)')
    # Migrate old chords table if it exists
    cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chords'")
    if cur.fetchone():
        rows = con.execute('SELECT * FROM chords').fetchall()
        for r in rows:
            con.execute('''
                INSERT INTO chord_versions
                    (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ''', (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
        con.execute('DROP TABLE chords')
    con.commit()
    con.close()

_init_db()


def _version_row_to_dict(row):
    return {
        'versionId':  row['version_id'],
        'videoId':    row['video_id'],
        'title':      row['title'],
        'key':        row['key'],
        'bpm':        row['bpm'],
        'chords':     json.loads(row['chords']),
        'beat_times': json.loads(row['beat_times']),
        'source':     row['source'],
        'analyzedAt': row['analyzed_at'],
        'isActive':   bool(row['is_active']),
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
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    for stem, meta in vmap.items():
        video_id = meta['videoId']
        title = meta.get('title', stem.replace('_', ' '))
        lab_path = os.path.join(label_dir, stem + '.lab')
        if not os.path.isfile(lab_path):
            continue

        # Skip if a verified version already exists for this video
        existing = con.execute(
            "SELECT version_id FROM chord_versions WHERE video_id = ? AND source = 'verified'",
            (video_id,),
        ).fetchone()
        if existing:
            continue

        # Parse .lab file
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

        # Pull key/bpm/beat_times from video_map entry, fall back to DB
        key_val = meta.get('key')
        bpm_val = meta.get('bpm')
        beat_times_list = meta.get('beatTimes', [])
        if not key_val or not bpm_val or not beat_times_list:
            ref = con.execute(
                'SELECT key, bpm, beat_times FROM chord_versions WHERE video_id = ? ORDER BY version_id DESC LIMIT 1',
                (video_id,),
            ).fetchone()
            key_val = key_val or (ref['key'] if ref else '?')
            bpm_val = bpm_val or (ref['bpm'] if ref else 120)
            if not beat_times_list and ref and ref['beat_times']:
                beat_times_list = json.loads(ref['beat_times'])
        if not beat_times_list:
            print(f"  WARNING: verified song '{stem}' has no beat_times (video_map empty, no DB fallback)")
        beat_times = json.dumps(beat_times_list)

        # Deactivate all other versions, insert verified as active
        con.execute('UPDATE chord_versions SET is_active = 0 WHERE video_id = ?', (video_id,))
        con.execute('''
            INSERT INTO chord_versions
                (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 'verified', ?, 1)
        ''', (video_id, title, key_val, bpm_val,
              json.dumps(segments), beat_times, int(time.time())))

    con.commit()
    con.close()

_import_verified()


def _cache_get(video_id: str):
    """Return the best version for a video: prefer verified, then active."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    # Prefer verified source, then is_active, then newest
    row = con.execute(
        '''SELECT * FROM chord_versions
           WHERE video_id = ?
           ORDER BY (source = 'verified') DESC, is_active DESC, version_id DESC
           LIMIT 1''',
        (video_id,),
    ).fetchone()
    con.close()
    if row is None:
        return None
    return _version_row_to_dict(row)


def _cache_put(video_id, title, key, bpm, chords_data, beat_times):
    con = sqlite3.connect(DB_PATH)
    # Delete any previous entries for this video
    con.execute('DELETE FROM chord_versions WHERE video_id = ?', (video_id,))
    con.execute('''
        INSERT INTO chord_versions
            (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, 'user-uploaded', ?, 1)
    ''', (video_id, title, key, bpm,
          json.dumps(chords_data), json.dumps(beat_times),
          int(time.time())))
    con.commit()
    version_id = con.execute('SELECT last_insert_rowid()').fetchone()[0]
    con.close()
    return version_id


# ─────────────────────────────────────────────
# In-memory job store
# ─────────────────────────────────────────────
_jobs = {}
_jobs_lock = threading.Lock()


def _set_job(job_id, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
        else:
            _jobs[job_id] = dict(kwargs)


def _get_job(job_id):
    with _jobs_lock:
        return dict(_jobs.get(job_id, {'status': 'not_found'}))


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


# Essentia subprocess detection
# Use ezchords' Python 3.12 venv which has essentia-tensorflow installed
_VENV312_PYTHON_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'ezchords', '.venv312', 'bin', 'python')
_VENV312_PYTHON = _VENV312_PYTHON_PATH if os.path.exists(_VENV312_PYTHON_PATH) else sys.executable
_ANALYZE_SCRIPT = os.path.join(os.path.dirname(__file__), 'analyze_chords.py')


def _detect_chords_essentia(audio_path: str):
    result = subprocess.run(
        [_VENV312_PYTHON, _ANALYZE_SCRIPT, audio_path],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f'Essentia analysis failed: {result.stderr[-500:]}')
    data = json.loads(result.stdout)
    if 'error' in data:
        raise RuntimeError(data['error'])
    return data['chords'], data['bpm'], data['key'], data['beat_times']


def detect_chords(audio_path: str, hop_size: float = 0.5):
    # Try Essentia first
    if os.path.exists(_VENV312_PYTHON) and os.path.exists(_ANALYZE_SCRIPT):
        try:
            return _detect_chords_essentia(audio_path)
        except Exception as e:
            print(f'[SeeChords] Essentia failed, falling back to librosa: {e}')

    return _detect_chords_librosa(audio_path, hop_size)


def _detect_chords_librosa(audio_path: str, hop_size: float = 0.5):
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

    merged = []
    for i, chord_name in enumerate(final_path):
        start = beat_times[i]
        end = beat_times[i + 1] if i + 1 < len(beat_times) else start + 0.5
        if merged and merged[-1]['chord'] == chord_name:
            merged[-1]['end'] = round(end, 3)
        else:
            merged.append({'chord': chord_name, 'start': round(start, 3), 'end': round(end, 3)})

    return merged, tempo, key, beat_times


def _clean_title(name: str) -> str:
    title = re.sub(r'\.[A-Za-z0-9]{1,6}$', '', (name or '').strip())
    title = re.sub(r'[_\-]+', ' ', title).strip()
    return title[:120] if title else 'Uploaded Audio'


def _process_job(job_id: str, source_path: str, video_id: str, title: str):
    """Analyze uploaded audio, store chords, delete audio file."""
    # Normalize to WAV (lossless) — mp3 compression degrades Essentia BPM accuracy
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
        chords_data, bpm_val, key_val, beat_times = detect_chords(temp_wav)

        version_id = _cache_put(video_id, title, key_val, round(bpm_val, 1), chords_data, beat_times)

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
            source='user-uploaded',
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
    """Upload audio + videoId to analyze chords. Always allows re-upload."""
    video_id = request.form.get('videoId', '').strip()
    if not video_id or not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return jsonify({'error': 'Invalid or missing videoId.'}), 400

    file_obj = request.files.get('file')
    if not file_obj or not file_obj.filename:
        return jsonify({'error': 'Please upload an audio file.'}), 400

    filename = file_obj.filename.strip()
    ext = os.path.splitext(filename)[1].lower()
    allowed = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.webm', '.mp4'}
    if ext and ext not in allowed:
        return jsonify({'error': 'Unsupported file type.'}), 400

    job_id = str(uuid.uuid4())
    _set_job(job_id, status='processing', progress=0, message='Upload received…')

    source_ext = ext if ext else '.bin'
    source_path = os.path.join(UPLOAD_DIR, f'{job_id}_source{source_ext}')
    file_obj.save(source_path)

    title = request.form.get('title', '').strip() or _clean_title(filename)

    t = threading.Thread(target=_process_job, args=(job_id, source_path, video_id, title),
                         daemon=True)
    t.start()

    return jsonify({'job_id': job_id, 'cached': False})


@app.route('/api/analyze-youtube', methods=['POST'])
def analyze_youtube():
    """Download audio from YouTube via yt-dlp and analyze chords."""
    data = request.get_json(silent=True) or {}
    video_id = (data.get('videoId') or '').strip()
    if not video_id or not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return jsonify({'error': 'Invalid or missing videoId.'}), 400

    title = (data.get('title') or '').strip()

    # Check if yt-dlp is available
    ytdlp = shutil.which('yt-dlp')
    if not ytdlp:
        return jsonify({'error': 'yt-dlp not installed on server.'}), 500

    job_id = str(uuid.uuid4())
    _set_job(job_id, status='processing', progress=0, message='Downloading audio from YouTube…')

    def _download_and_process():
        source_path = os.path.join(UPLOAD_DIR, f'{job_id}_source.m4a')
        try:
            _set_job(job_id, status='processing', progress=5, message='Downloading audio…')
            yt_url = f'https://www.youtube.com/watch?v={video_id}'
            result = subprocess.run(
                [ytdlp, '-f', 'bestaudio[ext=m4a]/bestaudio',
                 '--no-playlist', '--no-check-certificates',
                 '-o', source_path, yt_url],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0 or not os.path.exists(source_path):
                err_msg = result.stderr[:300] if result.stderr else 'yt-dlp failed'
                _set_job(job_id, status='error', message=f'Download failed: {err_msg}')
                return

            _set_job(job_id, status='processing', progress=15, message='Download complete, analyzing…')

            if not title:
                # Try to get title from yt-dlp
                try:
                    t_result = subprocess.run(
                        [ytdlp, '--get-title', '--no-playlist', yt_url],
                        capture_output=True, text=True, timeout=15
                    )
                    if t_result.returncode == 0 and t_result.stdout.strip():
                        title_val = t_result.stdout.strip()
                    else:
                        title_val = video_id
                except Exception:
                    title_val = video_id
            else:
                title_val = title

            _process_job(job_id, source_path, video_id, title_val)
        except subprocess.TimeoutExpired:
            _set_job(job_id, status='error', message='Download timed out')
            if os.path.exists(source_path):
                os.remove(source_path)
        except Exception as e:
            _set_job(job_id, status='error', message=str(e))

    t = threading.Thread(target=_download_and_process, daemon=True)
    t.start()
    return jsonify({'job_id': job_id, 'cached': False})


@app.route('/api/status/<job_id>')
def job_status(job_id):
    return jsonify(_get_job(job_id))


@app.route('/api/health')
def health():
    essentia_ok = os.path.exists(_VENV312_PYTHON)
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
        os.path.dirname(__file__), '..', 'training', 'tools', 'align_chords.py')
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
        os.path.dirname(__file__), '..', 'training', 'tools', 'lyric_align.py')
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
        os.path.dirname(__file__), '..', 'training', 'tools', 'align_chords.py')
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
        _update_ingest_job(job_id, {
            'status': 'processing', 'message': 'Analyzing audio (beat detection)…',
            'songName': song_name,
        })

        chords_data, bpm_val, key_val, beat_times = detect_chords(audio_path)

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


@app.route('/privacy')
def privacy_policy():
    """Serve the extension privacy policy."""
    pp_path = os.path.join(os.path.dirname(__file__), '..', 'extension', 'privacy-policy.html')
    if not os.path.isfile(pp_path):
        pp_path = os.path.join(os.path.dirname(__file__), 'privacy-policy.html')
    with open(pp_path) as f:
        return f.read()


@app.route('/ingest')
def ingest_page():
    """Serve the chord sheet ingest + review UI."""
    html_path = os.path.join(os.path.dirname(__file__), 'templates', 'ingest.html')
    with open(html_path) as f:
        return f.read()


@app.route('/api/ingest/saved-labs')
def list_saved_labs():
    """List all .lab files saved as training data."""
    label_dir = os.path.join(SERVER_VERIFIED_DIR, 'labels')
    if not os.path.isdir(label_dir):
        return jsonify([])
    labs = []
    for f in sorted(os.listdir(label_dir)):
        if not f.endswith('.lab'):
            continue
        path = os.path.join(label_dir, f)
        stat = os.stat(path)
        # Count segments (lines)
        with open(path) as fh:
            lines = [l for l in fh if l.strip()]
        name = f[:-4].replace('_', ' ')  # strip .lab, underscores → spaces
        labs.append({
            'filename': f,
            'name': name,
            'segments': len(lines),
            'savedAt': int(stat.st_mtime),
            'size': stat.st_size,
        })
    labs.sort(key=lambda x: x['savedAt'], reverse=True)
    return jsonify(labs)


@app.route('/api/ingest/saved-labs/<name>')
def get_saved_lab(name):
    """Read a saved .lab file and return parsed segments + metadata from DB."""
    # Sanitise to prevent path traversal
    safe = os.path.basename(name)
    if not safe.endswith('.lab'):
        safe += '.lab'
    path = os.path.join(SERVER_VERIFIED_DIR, 'labels', safe)
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
    map_path = os.path.join(SERVER_VERIFIED_DIR, 'video_map.json')
    bpm = None
    key = None
    beat_times = None
    video_id = None
    if os.path.isfile(map_path):
        with open(map_path) as mf:
            vmap = json.load(mf)
        entry = vmap.get(stem, {})
        bpm = entry.get('bpm')
        key = entry.get('key')
        beat_times = entry.get('beatTimes')
        video_id = entry.get('videoId')
    else:
        # Fall back to fuzzy DB lookup
        stem_lower = stem.replace('_', ' ').lower()
        stem_words = [w for w in stem_lower.split() if len(w) > 2]
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        best_match = None
        best_score = 0
        if stem_words:
            rows = con.execute(
                'SELECT video_id, title, key, bpm, beat_times FROM chord_versions '
                'WHERE is_active = 1 ORDER BY version_id DESC'
            ).fetchall()
            for r in rows:
                title_lower = (r['title'] or '').lower()
                matches = sum(1 for w in stem_words if w in title_lower)
                score = matches / len(stem_words)
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_match = r
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
def serve_silver_audio(filename):
    """Serve an audio file from training verified/audio/."""
    safe = os.path.basename(filename)
    audio_dir = os.path.join(TRAINING_VERIFIED_DIR, 'audio')
    path = os.path.join(audio_dir, safe)
    if not os.path.isfile(path):
        return jsonify({'error': 'Not found'}), 404
    return send_file(path)


@app.route('/api/ingest/saved-labs/<name>', methods=['PUT'])
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
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute('SELECT video_id, title FROM chord_versions GROUP BY video_id').fetchall()
        con.close()
        for r in rows:
            title_lower = (r['title'] or '').lower()
            # Check if enough words from the stem appear in the title
            stem_words = [w for w in stem.split() if len(w) > 2]
            matches = sum(1 for w in stem_words if w in title_lower)
            if stem_words and matches >= len(stem_words) * 0.5:
                video_id = r['video_id']
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

        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        existing = con.execute(
            'SELECT beat_times, key, bpm, title FROM chord_versions WHERE video_id = ? ORDER BY version_id DESC LIMIT 1',
            (video_id,)
        ).fetchone()
        if existing:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
            con.execute(
                'INSERT INTO chord_versions (video_id, title, key, bpm, chords, beat_times, source, analyzed_at, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (video_id, existing['title'], existing['key'], existing['bpm'],
                 json.dumps(simple_chords), existing['beat_times'], 'user-edited', now, 1)
            )
            # Mark older versions inactive
            new_id = con.execute('SELECT last_insert_rowid()').fetchone()[0]
            con.execute('UPDATE chord_versions SET is_active = 0 WHERE video_id = ? AND version_id != ?', (video_id, new_id))
            con.commit()
            version_id = new_id
        con.close()

    return jsonify({'saved': True, 'filename': safe, 'segmentCount': len(segments), 'versionId': version_id})


@app.route('/api/ingest', methods=['POST'])
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

    job_id = str(uuid.uuid4())

    with _ingest_lock:
        _ingest_jobs[job_id] = {
            'status': 'processing',
            'message': 'Downloading audio from YouTube…',
            'videoId': video_id,
        }

    def _yt_download_and_ingest():
        audio_path = os.path.join(UPLOAD_DIR, f'{job_id}_audio.m4a')
        try:
            import yt_dlp
            yt_url = f'https://www.youtube.com/watch?v={video_id}'
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio',
                'outtmpl': audio_path,
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(yt_url, download=True)
                yt_title = info.get('title', video_id)

            if not os.path.exists(audio_path):
                _update_ingest_job(job_id, {
                    'status': 'error',
                    'message': 'YouTube download failed — no audio file produced.',
                })
                return

            name = song_name or _clean_title(yt_title)
            with _ingest_lock:
                _ingest_jobs[job_id]['message'] = 'Download complete, analyzing…'

            _do_ingest_audio_only(job_id, audio_path, name)

            # Add audioUrl to the completed job so the browser can play it
            with _ingest_lock:
                job = _ingest_jobs.get(job_id)
                if job and job.get('status') == 'done':
                    job['audioUrl'] = f'/api/ingest/audio/{job_id}'

        except Exception as e:
            _update_ingest_job(job_id, {
                'status': 'error',
                'message': f'YouTube download failed: {e}',
            })

    t = threading.Thread(target=_yt_download_and_ingest, daemon=True)
    t.start()
    return jsonify({'jobId': job_id})


@app.route('/api/ingest/audio/<job_id>')
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
def ingest_status(job_id):
    """Check ingest job status / get alignment result for review."""
    with _ingest_lock:
        job = _ingest_jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(job)


@app.route('/api/ingest/<job_id>/save', methods=['POST'])
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

    audio_src = job.get('audioPath', '') or _find_ingest_audio_path(job_id)

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
    con = sqlite3.connect(DB_PATH)
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

    # Use job_id as video_id (stable identifier for this ingest session)
    video_id = job_id[:11]  # truncate to 11 chars for compat

    # Build chords_data in the format chord_versions expects
    chords_data = []
    for seg in segments:
        chords_data.append({
            'chord': seg.get('chord', 'N'),
            'start': float(seg['start']),
            'end': float(seg['end']),
        })

    # Insert as new version (don't delete old ones — keep history)
    con = sqlite3.connect(DB_PATH)
    # Deactivate previous versions
    con.execute('UPDATE chord_versions SET is_active = 0 WHERE video_id = ?', (video_id,))
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
def ingest_list_versions(job_id):
    """List all saved versions for an ingest job."""
    video_id = job_id[:11]
    return _list_versions(video_id)


@app.route('/api/versions/<video_id>')
def list_versions(video_id):
    """List all versions for a video_id."""
    return _list_versions(video_id)


def _list_versions(video_id):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        '''SELECT version_id, key, bpm, analyzed_at, is_active, chords, source
           FROM chord_versions WHERE video_id = ?
           ORDER BY (source = 'verified') DESC, is_active DESC, version_id DESC''',
        (video_id,),
    ).fetchall()
    con.close()

    versions = []
    for r in rows:
        chords = json.loads(r['chords'])
        versions.append({
            'versionId': r['version_id'],
            'key': r['key'],
            'bpm': r['bpm'],
            'analyzedAt': r['analyzed_at'],
            'isActive': bool(r['is_active']),
            'segmentCount': len(chords),
            'source': r['source'],
        })
    return jsonify({'versions': versions})


@app.route('/api/version/<int:version_id>')
def get_version(version_id):
    """Load a specific version by version_id."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute(
        'SELECT * FROM chord_versions WHERE version_id = ?',
        (version_id,),
    ).fetchone()
    con.close()

    if not row:
        return jsonify({'error': 'Version not found.'}), 404

    return jsonify(_version_row_to_dict(row))


# ── Rating endpoint ────────────────────────────────────────
@app.route('/api/rate', methods=['POST'])
def rate_version():
    """Submit a 1-5 star rating for a chord version."""
    data = request.get_json(force=True)
    version_id = data.get('versionId')
    video_id = data.get('videoId')
    stars = data.get('stars')

    if not version_id or not video_id or stars not in (1, 2, 3, 4, 5):
        return jsonify({'error': 'versionId, videoId, and stars (1-5) required.'}), 400

    con = sqlite3.connect(DB_PATH)
    # Upsert: one rating per version
    existing = con.execute(
        'SELECT id FROM ratings WHERE version_id = ?', (version_id,)
    ).fetchone()
    if existing:
        con.execute('UPDATE ratings SET stars = ?, created_at = ? WHERE version_id = ?',
                     (stars, int(time.time()), version_id))
    else:
        con.execute('INSERT INTO ratings (version_id, video_id, stars, created_at) VALUES (?, ?, ?, ?)',
                     (version_id, video_id, stars, int(time.time())))
    con.commit()
    con.close()
    return jsonify({'ok': True, 'stars': stars})


@app.route('/api/rating/<int:version_id>')
def get_rating(version_id):
    """Get the rating for a version."""
    con = sqlite3.connect(DB_PATH)
    row = con.execute('SELECT stars FROM ratings WHERE version_id = ?', (version_id,)).fetchone()
    con.close()
    return jsonify({'stars': row[0] if row else 0})


# ── Promote a version to verified (human-checked) ─────────────────

@app.route('/api/ingest/<job_id>/promote-verified', methods=['POST'])
def promote_to_verified(job_id):
    """Save & verify: writes .lab + audio to both server/verified/ and
    training/silver/, updates video_map.json, and creates a
    chord_versions entry with source='verified'."""
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

    # Sanitise name for filesystem
    safe_name = re.sub(r'[^\w\s\-]', '', song_name).strip().replace(' ', '_')
    if not safe_name:
        safe_name = job_id[:8]

    # Build Isophonics .lab content once
    lab_lines = []
    for seg in segments:
        chord = seg.get('chord', 'N')
        start = float(seg['start'])
        end = float(seg['end'])
        iso = _display_to_iso(chord)
        lab_lines.append(f'{start:.6f} {end:.6f} {iso}\n')
    lab_content = ''.join(lab_lines)

    audio_src = job.get('audioPath', '') or _find_ingest_audio_path(job_id)

    # Write labels to both dirs
    for base_dir in (SERVER_VERIFIED_DIR, TRAINING_VERIFIED_DIR):
        lbl_dir = os.path.join(base_dir, 'labels')
        os.makedirs(lbl_dir, exist_ok=True)
        with open(os.path.join(lbl_dir, f'{safe_name}.lab'), 'w') as f:
            f.write(lab_content)

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

    # Create verified chord_versions entry
    key_val = _infer_key_from_segments(segments) or job.get('key', '?')
    bpm_val = round(float(job.get('bpm', 120)), 1)
    beat_times = job.get('beatTimes', [])

    # Update verified/video_map.json (consolidated metadata)
    map_path = os.path.join(SERVER_VERIFIED_DIR, 'video_map.json')
    vmap = {}
    if os.path.isfile(map_path):
        with open(map_path) as f:
            vmap = json.load(f)
    vmap[safe_name] = {'videoId': video_id, 'title': song_name,
                       'bpm': bpm_val, 'key': key_val, 'beatTimes': beat_times}
    _dump_video_map(vmap, map_path)

    display_chords = []
    for seg in segments:
        ch = seg.get('chord', 'N')
        if ch != 'N':
            display_chords.append({
                'chord': ch,
                'start': round(float(seg['start']), 3),
                'end': round(float(seg['end']), 3),
            })

    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM chord_versions WHERE video_id = ? AND source = 'verified'", (video_id,))
    con.execute('UPDATE chord_versions SET is_active = 0 WHERE video_id = ?', (video_id,))
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
        'labPath': os.path.join(SERVER_VERIFIED_DIR, 'labels', f'{safe_name}.lab'),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, port=port, use_reloader=False)
