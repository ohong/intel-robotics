#!/usr/bin/env python3
"""Bounded, attach-only Studio capture with original timestamp and RGB PNG evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secondlook.cv_observation import quality_stats

STUDIO_PYTHON = '/home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python'


def checked_rgb(frame, header, *, now: float, max_age: float):
    """Validate the exact shared-memory frame/header association before conversion."""
    import numpy as np
    timestamp, sequence = frame.timestamp, frame.sequence
    if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0 or
            not 0 <= now - timestamp <= max_age or type(sequence) is not int or sequence < 0):
        raise ValueError('Stale or malformed publisher timestamp/sequence')
    if (header is None or type(header.timestamp_ns) is not int or header.timestamp_ns < 0 or
            type(header.sequence) is not int or header.sequence != sequence or header.timestamp_ns / 1e9 != timestamp):
        raise ValueError('Publisher header does not match captured frame')
    colors = {0: 'RGB', 1: 'BGR', 2: 'GRAY'}
    color = colors.get(header.color_mode)
    if color is None or header.dtype != 0:
        raise ValueError('Unsupported publisher color or dtype')
    width, height = header.width, header.height
    shape = (height, width) if color == 'GRAY' else (height, width, 3)
    data = np.asarray(frame.data)
    if (type(width) is not int or type(height) is not int or not 1 <= width <= 4096 or
            not 1 <= height <= 2160 or data.shape != shape or data.dtype != np.uint8):
        raise ValueError('Publisher shape/dtype does not match its header')
    rgb = data[..., ::-1] if color == 'BGR' else (np.repeat(data[..., None], 3, axis=2) if color == 'GRAY' else data)
    return np.ascontiguousarray(rgb).copy(), {'capture_timestamp': timestamp,
            'capture_timestamp_ns': header.timestamp_ns, 'sequence': sequence,
            'publisher_color': color, 'stored_color': 'RGB', 'width': width, 'height': height}


def roi_box(roi: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if (len(roi) != 4 or not all(math.isfinite(x) for x in roi) or
            not 0 <= roi[0] < roi[2] <= 1 or not 0 <= roi[1] < roi[3] <= 1):
        raise ValueError('ROI must be normalized left,top,right,bottom with positive area')
    box = (int(roi[0] * width), int(roi[1] * height), math.ceil(roi[2] * width), math.ceil(roi[3] * height))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('ROI has no pixels at this image resolution')
    return box


def save_capture(root: Path, rgb, metadata: dict, provenance: dict, *, roi: list[float], size: int) -> dict:
    """Save full RGB as the training source; crops are inspection previews only."""
    from PIL import Image
    stem = f"frame-{metadata['sequence']:012d}-{metadata['capture_timestamp_ns']}"
    image = Image.fromarray(rgb, mode='RGB')
    box = roi_box(roi, image.width, image.height)
    paths = {name: root / f'{stem}-{name}.png' for name in ('raw', 'crop', 'resized')}
    previews = {'raw': image, 'crop': image.crop(box)}
    previews['resized'] = previews['crop'].resize((size, size), Image.Resampling.BILINEAR)
    for name, path in paths.items():
        with path.open('xb') as stream:
            previews[name].save(stream, format='PNG')
    raw_path = paths['raw']
    return {**provenance, 'path': raw_path.name, 'sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            'frame_id': f"{provenance['session_id']}:{provenance['camera_service']}:{metadata['sequence']}:{metadata['capture_timestamp_ns']}",
            **metadata, 'timestamp_source': 'publisher_monotonic',
            'saved_at_utc': datetime.now(timezone.utc).isoformat(),
            'quality_stats': quality_stats(rgb), 'preview_roi_normalized': roi,
            'preview_roi_pixels': list(box), 'preview_resize': [size, size],
            'previews': {key: path.name for key, path in paths.items()},
            'object_presence': 'UNVERIFIED', 'occlusion': 'UNVERIFIED', 'defect_visibility': 'UNVERIFIED'}


def contact_sheet(root: Path, rows: list[dict]) -> None:
    from PIL import Image, ImageDraw, ImageOps
    width, height, columns = 320, 220, min(4, len(rows))
    sheet = Image.new('RGB', (columns * width, math.ceil(len(rows) / columns) * height), '#181818')
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x, y = index % columns * width, index // columns * height
        with Image.open(root / row['path']) as original:
            preview = ImageOps.contain(original, (width, height - 40))
            sheet.paste(preview, (x, y))
        draw.text((x + 5, y + height - 36), f"seq {row['sequence']} | {row['label']} | {row['evidence_kind']}", fill='white')
        draw.text((x + 5, y + height - 20), row['specimen_id'][:45], fill='white')
    with (root / 'contact-sheet.png').open('xb') as stream:
        sheet.save(stream, format='PNG')


def capture(args) -> int:
    # This is the only physicalai import: no discovery, camera creation or robot API.
    from physicalai.capture import SharedCamera
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    rows, camera, error = [], None, None
    provenance = {'session_id': args.session_id, 'specimen_id': args.specimen_id,
                  'role': args.role, 'label': args.label,
                  'ground_truth': {'source': args.ground_truth_source, 'authoritative': args.authoritative},
                  'camera_service': args.service, 'evidence_kind': 'real',
                  'capture_python': sys.executable, 'capture_host': os.uname().nodename}
    deadline = time.monotonic() + args.duration
    previous = (-1., -1)
    try:
        camera = SharedCamera.from_publisher(args.service, overwrite_settings=False, zero_copy=False)
        camera.connect(timeout=args.connect_timeout)
        while len(rows) < args.count and time.monotonic() < deadline:
            tick = time.monotonic()
            frame = camera.read_latest()
            rgb, metadata = checked_rgb(frame, camera._last_header, now=time.monotonic(), max_age=args.max_age)
            current = (metadata['capture_timestamp'], metadata['sequence'])
            if current[0] > previous[0] and current[1] > previous[1]:
                row = save_capture(root, rgb, metadata, provenance, roi=args.roi, size=args.preview_size)
                rows.append(row)
                # Append after each complete saved observation, preserving evidence if the worker is killed.
                with (root / 'captures.jsonl').open('a') as journal:
                    journal.write(json.dumps(row, allow_nan=False) + '\n')
                    journal.flush()
                    os.fsync(journal.fileno())
                previous = current
            elif current != previous:
                raise ValueError('Publisher timestamp/sequence regressed or lost association')
            time.sleep(min(max(0., args.interval - (time.monotonic() - tick)), max(0., deadline - time.monotonic())))
        if len(rows) < args.count:
            raise TimeoutError(f'Capture deadline reached: saved {len(rows)} of {args.count}')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if camera is not None:
            try:
                camera.disconnect()
            except Exception as exc:
                error = error or f'Disconnect failed: {exc}'
        manifest = {'schema_version': 1, 'capture_status': 'COMPLETE' if error is None else 'PARTIAL_OR_FAILED',
                    'capture_error': error, 'images': rows,
                    'note': 'Full RGB sources. Preview crops are not separate samples. Unknown labels support scouting only.'}
        with (root / 'manifest.json').open('x') as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write('\n')
        if rows:
            contact_sheet(root, rows)
    print(json.dumps({'output': str(root), 'saved': len(rows), 'status': manifest['capture_status'], 'error': error}))
    return 2 if error else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studio-python', default=STUDIO_PYTHON)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    for name in ('service', 'output', 'session-id', 'specimen-id', 'ground-truth-source'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--role', choices=('train', 'validation', 'test'), required=True)
    parser.add_argument('--label', choices=('normal', 'anomalous', 'unknown'), required=True)
    parser.add_argument('--authoritative', action='store_true')
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--interval', type=float, default=1.)
    parser.add_argument('--connect-timeout', type=float, default=3.)
    parser.add_argument('--max-age', type=float, default=1.)
    parser.add_argument('--roi', type=float, nargs=4, default=[0., 0., 1., 1.])
    parser.add_argument('--preview-size', type=int, default=256)
    args = parser.parse_args()
    if (not 1 <= args.count <= 1000 or not 1 <= args.preview_size <= 2048 or
            any(not math.isfinite(x) or not 0 < x <= 300 for x in
                (args.duration, args.interval, args.connect_timeout, args.max_age))):
        parser.error('Count, preview size and finite positive timing must be bounded')
    if any(not getattr(args, name).strip() for name in ('service', 'session_id', 'specimen_id', 'ground_truth_source')):
        parser.error('Provenance fields cannot be empty')
    if args.label == 'unknown' and args.authoritative:
        parser.error('Unknown labels cannot have authoritative ground truth')
    if args.role == 'train' and args.label == 'anomalous':
        parser.error('Known anomalous samples cannot enter the normal reference bank')
    try:
        roi_box(args.roi, 4096, 2160)
        if args.worker:
            if os.path.abspath(sys.executable) != os.path.abspath(args.studio_python):
                raise ValueError('Worker must run using the selected Studio Python')
            return capture(args)
        command = [args.studio_python, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
        return subprocess.run(command, timeout=args.duration + args.connect_timeout + 10, check=False).returncode
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f'Capture failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
