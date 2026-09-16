#!/usr/bin/env python3
"""Attach-only browser shutter. Collection remains unassigned until an offline split."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cv_capture import checked_rgb

SERVICE = 'physicalai/camera/UVCCamera/4/frame'
STATIC = Path(__file__).resolve().parents[1] / 'secondlook' / 'capture_static'
LABELS = {'normal': 'Good', 'anomalous': 'Defective', 'unknown': 'Unsure'}


def append_record(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_journal(path):
    if not path.exists():
        return []
    # A partial tail is not silently discarded: repair is an explicit offline action.
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError(f'Incomplete journal tail: {path}; preserve and repair before resuming')
    return [json.loads(line) for line in raw.splitlines()]


class Collection:
    def __init__(self, root, *, synthetic=False, service=SERVICE):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        import fcntl
        self.lock_file = (self.root / '.capture.lock').open('a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock = threading.RLock()
        self.frames = OrderedDict()
        self.error = 'Waiting for fresh camera frame'
        self.service = service
        self.synthetic = synthetic
        descriptor = self.root / 'collection.json'
        if descriptor.exists():
            self.info = json.loads(descriptor.read_text())
            if self.info['synthetic'] != synthetic or self.info['camera_service'] != service:
                raise ValueError('Existing collection camera/evidence mode differs')
        else:
            self.info = {'schema': 'secondlook.operator_collection.v1', 'collection_id': uuid.uuid4().hex,
                         'synthetic': synthetic, 'camera_service': service, 'split_status': 'pending_offline_split'}
            with descriptor.open('x') as stream:
                json.dump(self.info, stream)
                stream.flush()
                os.fsync(stream.fileno())
        self.specimens = OrderedDict((row['specimen_id'], row) for row in read_journal(self.root / 'specimens.jsonl'))
        self.rows = read_journal(self.root / 'captures.jsonl')
        self.saved_frames = set()
        self.labels = {}
        for row in self.rows:
            if row['specimen_id'] not in self.specimens or not (self.root / row['path']).is_file():
                raise ValueError('Collection journal references missing specimen/image')
            if row['frame_id'] in self.saved_frames:
                raise ValueError('Duplicate frame in collection journal')
            prior = self.labels.setdefault(row['specimen_id'], row['label'])
            if prior != row['label']:
                raise ValueError('Conflicting specimen labels in collection journal')
            self.saved_frames.add(row['frame_id'])

    def close(self):
        self.lock_file.close()

    def state(self):
        with self.lock:
            try:
                mac_sync = json.loads((self.root / 'sync-status.json').read_text())
            except (OSError, ValueError):
                mac_sync = None
            return {**self.info, 'count': len(self.rows), 'target': 50, 'error': self.error,
                    'mac_sync': mac_sync,
                    'specimens': [{**row, 'label': self.labels.get(key)} for key, row in self.specimens.items()],
                    'recent': self.rows[-12:][::-1]}

    def new_specimen(self):
        with self.lock:
            row = {'specimen_id': 'block-' + uuid.uuid4().hex[:12], 'name': f'Block {len(self.specimens) + 1}'}
            append_record(self.root / 'specimens.jsonl', row)
            self.specimens[row['specimen_id']] = row
            return row

    def publish(self, rgb, metadata):
        from PIL import Image
        # Preserve lossless RGB; the JPEG is a preview of this same immutable array.
        rgb = rgb.copy()
        frame_id = f"{self.info['collection_id']}:{metadata['sequence']}:{metadata['capture_timestamp_ns']}"
        with self.lock:
            if frame_id in self.frames:
                return
        preview = Image.fromarray(rgb)
        preview.thumbnail((1100, 800))
        encoded = io.BytesIO()
        preview.save(encoded, format='JPEG', quality=90)
        with self.lock:
            self.frames[frame_id] = (rgb, dict(metadata), encoded.getvalue())
            while len(self.frames) > 24:
                self.frames.popitem(last=False)
            self.error = None

    def preview(self):
        with self.lock:
            if not self.frames:
                raise ValueError(self.error or 'Camera unavailable')
            key = next(reversed(self.frames))
            frame = self.frames[key]
            if not 0 <= time.monotonic() - frame[1]['capture_timestamp'] <= 3:
                raise ValueError('Camera frame is stale')
            return key, frame[2]

    def capture(self, payload):
        from PIL import Image
        if not isinstance(payload, dict) or set(payload) != {'frame_id', 'specimen_id', 'label'}:
            raise ValueError('Expected frame_id, specimen_id, and explicit label')
        if not all(isinstance(value, str) for value in payload.values()):
            raise ValueError('Capture fields must be strings')
        key, specimen, label = payload['frame_id'], payload['specimen_id'], payload['label']
        with self.lock:
            if label not in LABELS or specimen not in self.specimens:
                raise ValueError('Select a block and an explicit label')
            if self.synthetic and label != 'unknown':
                raise ValueError('SYNTHETIC fixtures allow Unsure only')
            if specimen in self.labels and self.labels[specimen] != label:
                raise ValueError('This block already has a different label; keep its original label')
            if key in self.saved_frames:
                raise ValueError('This exact frame is already saved; wait for a new frame')
            if key not in self.frames:
                raise ValueError('Displayed frame expired; wait for a fresh preview')
            rgb, metadata, _ = self.frames[key]
            age = time.monotonic() - metadata['capture_timestamp']
            if not 0 <= age <= 3:
                raise ValueError('Displayed frame is stale; wait for a fresh preview')
            filename = 'capture-' + uuid.uuid4().hex + '.png'
            path = self.root / filename
            temporary = self.root / ('.' + filename + '.tmp')
            with temporary.open('xb') as stream:
                Image.fromarray(rgb).save(stream, format='PNG')
                stream.flush()
                os.fsync(stream.fileno())
            # Unique final names and same-filesystem rename keep mirrors from seeing partial PNGs.
            os.rename(temporary, path)
            row = {'schema': 'secondlook.operator_collection.capture.v1',
                   'collection_id': self.info['collection_id'], 'collection_session_id': self.info['collection_id'],
                   'session_id': self.info['collection_id'],
                   'frame_id': key, 'specimen_id': specimen, 'label': label, 'role': 'unassigned',
                   'split_status': 'pending_offline_split', 'path': filename,
                   'ground_truth': {'source': f'Operator selected {LABELS[label]} in capture UI',
                                    'authoritative': label != 'unknown' and not self.synthetic},
                   'evidence_kind': 'synthetic' if self.synthetic else 'real',
                   'camera_service': self.service, 'capture_host': os.uname().nodename,
                   'capture_python': sys.executable, 'timestamp_source': 'fixture_monotonic' if self.synthetic else 'publisher_monotonic',
                   **metadata, 'shutter_frame_age_seconds': age,
                   'saved_at_utc': datetime.now(timezone.utc).isoformat(),
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            append_record(self.root / 'captures.jsonl', row)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self.rows.append(row)
            self.saved_frames.add(key)
            self.labels[specimen] = label
            return row


def camera_loop(collection, stop, fixture=None):
    camera = None
    try:
        if fixture:
            from PIL import Image
            import numpy as np
            rgb = np.asarray(Image.open(fixture).convert('RGB'))
        else:
            from physicalai.capture import SharedCamera
            camera = SharedCamera.from_publisher(collection.service, overwrite_settings=False, zero_copy=False)
            camera.connect(timeout=3)
        sequence = 0
        while not stop.is_set():
            try:
                if fixture:
                    sequence += 1
                    now = time.monotonic()
                    metadata = {'capture_timestamp': now, 'capture_timestamp_ns': int(now * 1e9),
                                'sequence': sequence, 'publisher_color': 'RGB', 'stored_color': 'RGB',
                                'width': rgb.shape[1], 'height': rgb.shape[0]}
                else:
                    frame = camera.read_latest()
                    rgb, metadata = checked_rgb(frame, camera._last_header, now=time.monotonic(), max_age=3)
                collection.publish(rgb, metadata)
            except Exception as exc:
                with collection.lock:
                    collection.error = f'{type(exc).__name__}: {exc}'
            stop.wait(.17)
    except Exception as exc:
        with collection.lock:
            collection.error = f'{type(exc).__name__}: {exc}'
    finally:
        if camera is not None:
            camera.disconnect()


def make_handler(collection, port):
    hosts = {f'localhost:{port}', f'127.0.0.1:{port}'}
    origins = {'http://' + host for host in hosts}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, body, content_type='application/json', headers=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def allowed(self, mutation=False):
            if self.headers.get('Host') not in hosts:
                return False
            origin = self.headers.get('Origin')
            return (origin in origins if mutation else origin is None or origin in origins) and self.headers.get('Sec-Fetch-Site') != 'cross-site'

        def do_GET(self):
            if not self.allowed():
                return self.reply(403, {'error': 'Loopback origin required'})
            try:
                if self.path == '/api/state':
                    return self.reply(200, collection.state())
                if self.path == '/api/preview':
                    key, jpeg = collection.preview()
                    return self.reply(200, jpeg, 'image/jpeg', {'X-Frame-ID': key})
                static = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}
                if self.path in static:
                    filename, mime = static[self.path]
                    return self.reply(200, (STATIC / filename).read_bytes(), mime)
                with collection.lock:
                    match = next((row for row in collection.rows if self.path == '/images/' + row['path']), None)
                if match:
                    return self.reply(200, (collection.root / match['path']).read_bytes(), 'image/png')
                return self.reply(404, {'error': 'Not found'})
            except ValueError as exc:
                return self.reply(409, {'error': str(exc)})
            except OSError:
                return self.reply(500, {'error': 'Storage unavailable'})

        def do_POST(self):
            if not self.allowed(mutation=True):
                return self.reply(403, {'error': 'Loopback origin required'})
            try:
                if self.headers.get('Content-Type') != 'application/json' or self.headers.get('Transfer-Encoding'):
                    raise ValueError('JSON body required')
                size = int(self.headers.get('Content-Length', '0'))
                if not 1 <= size <= 2048:
                    raise ValueError('Invalid body length')
                payload = json.loads(self.rfile.read(size))
                if self.path == '/api/specimens':
                    if payload != {}:
                        raise ValueError('New block request must be empty')
                    return self.reply(201, collection.new_specimen())
                if self.path == '/api/capture':
                    return self.reply(201, collection.capture(payload))
                return self.reply(404, {'error': 'Not found'})
            except (ValueError, UnicodeError) as exc:
                return self.reply(400, {'error': str(exc)})
            except OSError:
                return self.reply(500, {'error': 'Storage failed. Check collection disk before retrying.'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--service', default=SERVICE)
    parser.add_argument('--fixture-image', type=Path, help='SYNTHETIC preview only; use a separate output directory')
    args = parser.parse_args()
    collection = Collection(args.output, synthetic=bool(args.fixture_image), service=args.service)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(collection, args.port))
    server.daemon_threads = True
    stop = threading.Event()
    worker = threading.Thread(target=camera_loop, args=(collection, stop, args.fixture_image), daemon=True)
    worker.start()
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    print(f'Capture UI: http://127.0.0.1:{args.port} | {args.output} | {"SYNTHETIC" if args.fixture_image else "REAL"}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=4)
        collection.close()


if __name__ == '__main__':
    main()
