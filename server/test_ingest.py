#!/usr/bin/env python3
"""Quick smoke test for the ingest API."""
import urllib.request
import json
import time

def multipart_post(url, fields, files):
    boundary = 'BoundaryXYZ123'
    body = b''
    for key, val in fields.items():
        body += ('--' + boundary + '\r\n').encode()
        body += ('Content-Disposition: form-data; name="' + key + '"\r\n\r\n').encode()
        body += val.encode() + b'\r\n'
    for key, (filename, data, content_type) in files.items():
        body += ('--' + boundary + '\r\n').encode()
        body += ('Content-Disposition: form-data; name="' + key + '"; filename="' + filename + '"\r\n').encode()
        body += ('Content-Type: ' + content_type + '\r\n\r\n').encode()
        body += data + b'\r\n'
    body += ('--' + boundary + '--\r\n').encode()

    req = urllib.request.Request(url, data=body, method='POST')
    req.add_header('Content-Type', 'multipart/form-data; boundary=' + boundary)
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())

# Read yesterday chord sheet
with open('/Users/genej/projects/chords/seechords/training/test_sheets/yesterday.txt') as f:
    chord_text = f.read()

# Use Let It Be as audio for testing
audio_path = '/Users/genej/projects/chords/seechords/training/data/audio/12_-_Let_It_Be/06_-_Let_It_Be.mp3'
with open(audio_path, 'rb') as f:
    audio_data = f.read()

print("Uploading...")
result = multipart_post(
    'http://localhost:5002/api/ingest',
    {'chordText': chord_text, 'name': 'Test_Yesterday'},
    {'audio': ('test.mp3', audio_data, 'audio/mpeg')}
)
print('Upload:', result)

job_id = result['jobId']

for _ in range(30):
    time.sleep(2)
    resp = urllib.request.urlopen('http://localhost:5002/api/ingest/' + job_id)
    data = json.loads(resp.read())
    print('Status:', data['status'], data.get('message', ''))
    if data['status'] in ('done', 'error'):
        if data['status'] == 'done':
            print('Key:', data['key'], 'BPM:', data['bpm'], 'Segments:', len(data['segments']))
            for s in data['segments'][:10]:
                m = 'Y' if s['match'] else 'N'
                print(f"  {s['start']:5.1f}-{s['end']:5.1f}  sheet={s['sheetChord']:>5s}  pred={s['predChord']:>5s}  {m}")
            print("...")
        break
