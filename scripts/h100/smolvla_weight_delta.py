#!/usr/bin/env python3
"""Lossless transfer of changed safetensors tensors against an explicitly pinned base.

Reconstruction reproduces the target file byte for byte, including its header.
This standard-library helper neither loads nor changes model tensor values.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import zipfile

BASE_SHA256 = '7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def layout(path):
    with Path(path).open('rb') as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, 'Truncated safetensors file')
        size = struct.unpack('<Q', prefix)[0]
        require(2 <= size <= 64 * 1024 * 1024, 'Invalid safetensors header length')
        header = stream.read(size)
    require(len(header) == size, 'Truncated safetensors header')
    entries = json.loads(header)
    entries.pop('__metadata__', None)
    ordered = sorted(entries, key=lambda key: entries[key]['data_offsets'])
    end = 0
    for key in ordered:
        start, stop = entries[key]['data_offsets']
        require(type(start) is int and type(stop) is int and start == end and stop >= start, 'Noncontiguous tensor byte ranges')
        end = stop
    require(end + 8 + size == Path(path).stat().st_size, 'Tensor lengths do not cover the file')
    return prefix + header, entries, ordered


def bytes_at(stream, header_size, entry):
    start, stop = entry['data_offsets']
    stream.seek(header_size + start)
    data = stream.read(stop - start)
    require(len(data) == stop - start, 'Truncated tensor')
    return data


def create(base, target, output, expected_base_sha=BASE_SHA256):
    require(not output.exists(), 'Preserve existing delta output')
    require(sha(base) == expected_base_sha, 'Pinned base checksum mismatch')
    base_header, base_entries, _ = layout(base)
    header, entries, ordered = layout(target)
    mapping = {key.replace('model.', '_model.', 1) if key.startswith('model.') else key: key for key in base_entries}
    require(len(mapping) == len(base_entries), 'Ambiguous canonical base tensor names')
    manifest = {'schema_version': 1, 'base_sha256': expected_base_sha, 'target_sha256': sha(target),
                'target_bytes': target.stat().st_size, 'header_base64': base64.b64encode(header).decode(),
                'tensors': [], 'controller_ready': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.weight-delta-', delete=False) as stream:
            temp = Path(stream.name)
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED) as archive, base.open('rb') as source, target.open('rb') as destination:
            for index, key in enumerate(ordered):
                data = bytes_at(destination, len(header), entries[key])
                old_key = mapping.get(key)
                reusable = old_key is not None and all(entries[key][field] == base_entries[old_key][field] for field in ('dtype', 'shape'))
                if reusable:
                    reusable = data == bytes_at(source, len(base_header), base_entries[old_key])
                record = {'key': key, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                if reusable:
                    record['base_key'] = old_key
                else:
                    record['entry'] = f'tensors/{index}.bin'
                    archive.writestr(record['entry'], data)
                manifest['tensors'].append(record)
            archive.writestr('delta.json', json.dumps(manifest, indent=2))
        temp.rename(output)
    finally:
        if temp and temp.exists():
            temp.unlink()
    changed = [entry for entry in manifest['tensors'] if 'entry' in entry]
    return {'target_sha256': manifest['target_sha256'], 'delta_sha256': sha(output),
            'target_bytes': manifest['target_bytes'], 'delta_bytes': output.stat().st_size,
            'changed_tensors': len(changed), 'reused_tensors': len(manifest['tensors']) - len(changed),
            'controller_ready': False}


def reconstruct(base, delta, output, expected_base_sha=BASE_SHA256):
    require(not output.exists(), 'Preserve existing reconstructed output')
    require(sha(base) == expected_base_sha, 'Pinned base checksum mismatch')
    base_header, base_entries, _ = layout(base)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with zipfile.ZipFile(delta) as archive, base.open('rb') as source:
            manifest = json.loads(archive.read('delta.json'))
            require(manifest['schema_version'] == 1 and manifest['base_sha256'] == expected_base_sha, 'Delta uses another base')
            header = base64.b64decode(manifest['header_base64'], validate=True)
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.weight-reconstruct-', delete=False) as stream:
                temp = Path(stream.name)
                stream.write(header)
                for record in manifest['tensors']:
                    require(('base_key' in record) != ('entry' in record), 'Ambiguous tensor source')
                    data = (bytes_at(source, len(base_header), base_entries[record['base_key']])
                            if 'base_key' in record else archive.read(record['entry']))
                    require(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], 'Corrupted tensor payload')
                    stream.write(data)
            require(temp.stat().st_size == manifest['target_bytes'] and sha(temp) == manifest['target_sha256'], 'Reconstructed target checksum mismatch')
            layout(temp)
            temp.rename(output)
            return {'status': 'BYTE_EXACT_RECONSTRUCTION_PASSED', 'target_sha256': manifest['target_sha256'],
                    'target_bytes': output.stat().st_size, 'controller_ready': False}
    finally:
        if temp and temp.exists():
            temp.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('create', 'reconstruct'):
        command = sub.add_parser(mode)
        command.add_argument('--base', type=Path, required=True)
        command.add_argument('--target' if mode == 'create' else '--delta', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = create(args.base, args.target, args.output) if args.mode == 'create' else reconstruct(args.base, args.delta, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
