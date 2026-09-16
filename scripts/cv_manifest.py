#!/usr/bin/env python3
"""Merge complete capture manifests without changing labels or split assignments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secondlook.anomaly import validate_manifest


def merge(inputs: list[str], output: str, *, engineering_smoke: bool = False) -> dict:
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f'Output already exists: {target}')
    rows = []
    for filename in inputs:
        source = Path(filename).resolve()
        manifest = json.loads(source.read_text())
        if manifest.get('schema_version') != 1 or manifest.get('capture_status') != 'COMPLETE':
            raise ValueError(f'Capture must be COMPLETE with schema_version=1: {source}')
        if not isinstance(manifest.get('images'), list) or not manifest['images']:
            raise ValueError(f'Capture must contain images: {source}')
        for original in manifest['images']:
            row = dict(original)
            # Paths resolve from each source capture, then from the combined manifest.
            # Copy the complete dataset tree when moving a relative-path manifest.
            path = (source.parent / row['path']).resolve()
            row['path'] = os.path.relpath(path, target.parent)
            if isinstance(row.get('previews'), dict):
                row['previews'] = {key: os.path.relpath((source.parent / value).resolve(), target.parent)
                                   for key, value in row['previews'].items()}
            row['source_capture_manifest'] = os.path.relpath(source, target.parent)
            rows.append(row)
    combined = {'schema_version': 1, 'images': rows,
                'note': 'Merged capture provenance; labels and split assignments preserved.'}
    target.parent.mkdir(parents=True, exist_ok=True)
    # Validate the exact serialized output at its final relative-path base.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(combined, stream, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        validated = validate_manifest(temporary, engineering_smoke=engineering_smoke)
        with target.open('x') as stream:
            stream.write(temporary.read_text())
    finally:
        temporary.unlink(missing_ok=True)
    return {'output': str(target), 'count': len(rows), 'dataset_sha256': validated['dataset_sha256']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    combine = sub.add_parser('merge')
    combine.add_argument('--inputs', nargs='+', required=True)
    combine.add_argument('--output', required=True)
    combine.add_argument('--engineering-smoke', action='store_true')
    args = parser.parse_args()
    try:
        result = merge(args.inputs, args.output, engineering_smoke=args.engineering_smoke)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Manifest merge failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
