#!/usr/bin/env python3
"""Offline only: validate a finalized Studio dataset and write training evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from contracts import require, snapshot, validate_groups, validate_rows


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(args):
    # These prevent library fallback downloads; no runtime, robot, or camera APIs are imported.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image, ImageDraw
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.set_num_threads(2)
    root, output = args.dataset.resolve(), args.output.resolve()
    require(args.finalized, 'Owner must confirm recording stopped and dataset finalized with --finalized')
    require(root.is_dir(), f'No dataset at {root}')
    require(not output.is_relative_to(root), 'Evidence directory must be outside the dataset')
    require(not output.exists(), 'Use a new evidence directory; never overwrite a prior validation')
    for required in ('meta/info.json', 'meta/stats.json', 'meta/tasks.parquet'):
        require((root / required).is_file(), f'Missing {required}')
    before = snapshot(root)
    info = json.loads((root / 'meta/info.json').read_text())
    require(0 < info['total_frames'] <= args.max_frames, f'Frame count exceeds bounded check ({args.max_frames}) or is empty')
    episode_files = sorted(root.glob('meta/episodes/**/*.parquet'))
    data_files = sorted(root.glob('data/**/*.parquet'))
    require(episode_files and data_files, 'Missing episode metadata or data Parquet files')
    episodes = [row for path in episode_files for row in pq.read_table(path).to_pylist()]
    rows = [row for path in data_files for row in pq.read_table(path).to_pylist()]
    val_ids = args.validation_episodes or [info['total_episodes'] - 1]
    report = validate_rows(info, episodes, rows, val_ids, args.gripper_min_range, args.final_eval_episodes)
    if args.evidence_kind == 'real':
        require(args.episode_labels is not None, 'Real episodes require --episode-labels with destination and observed outcome')
    labels = json.loads(args.episode_labels.read_text()) if args.episode_labels else {
        str(ep['episode_index']): {'destination': 'SYNTHETIC', 'outcome': 'synthetic_success'} for ep in episodes}
    validate_groups(report, labels, args.evidence_kind, args.split_scope)
    tasks = {int(row['task_index']) for row in rows}
    task_table = pq.read_table(root / 'meta/tasks.parquet').to_pandas()
    require(tasks <= set(int(value) for value in task_table['task_index']), 'Unknown task index')
    require(len(tasks) == 1, 'Keep one task/skill per standard ACT dataset; captions do not condition this policy')
    output.mkdir(parents=True)
    raw = LeRobotDataset(repo_id='snapshot', root=root, download_videos=False, video_backend='pyav')
    require(len(raw) == len(rows), 'Installed loader frame count differs')
    train_ids = set(report['train_episodes'])
    numeric_stats = {}
    for key in ('observation.state', 'action'):
        values = np.asarray([row[key] for row in rows if row['episode_index'] in train_ids], dtype=np.float64)
        numeric_stats[key] = {name: value.tolist() for name, value in {
            'mean': values.mean(axis=0), 'std': values.std(axis=0), 'min': values.min(axis=0),
            'max': values.max(axis=0), 'q01': np.quantile(values, 0.01, axis=0),
            'q99': np.quantile(values, 0.99, axis=0), 'count': np.array([len(values)])}.items()}
    image_moments = {key: {'count': 0, 'sum': np.zeros(3), 'squares': np.zeros(3),
                           'min': np.ones(3), 'max': np.zeros(3)} for key in report['cameras']}
    stream_evidence, sheet_files = [], []
    for ep in episodes:
        eid, length = int(ep['episode_index']), int(ep['length'])
        sampled = {round(i * (length - 1) / 7) for i in range(8)}
        tiles = []
        evidence = {key: {'decoded_frames': 0, 'longest_identical_run': 0, 'dark_frames': 0} for key in report['cameras']}
        previous, streak = {}, {}
        for frame in range(length):
            item = raw[int(ep['dataset_from_index']) + frame]
            for key in report['cameras']:
                array = item[key].detach().cpu().numpy()
                require(array.ndim == 3 and array.shape[0] == 3 and np.isfinite(array).all(), f'{key}: invalid decoded RGB tensor')
                require(array.min() >= 0 and array.max() <= 1, f'{key}: image range outside [0,1]')
                shape = info['features'][key]['shape']
                layout = info['features'][key]['names']
                expected = [shape[2], shape[0], shape[1]] if layout[2] in ('channel', 'channels') else shape
                require(list(array.shape) == expected, f'{key}: decoded shape differs from metadata')
                checksum = hashlib.sha256(array.tobytes()).hexdigest()
                streak[key] = streak.get(key, 0) + 1 if previous.get(key) == checksum else 1
                previous[key] = checksum
                ev = evidence[key]
                ev['decoded_frames'] += 1
                ev['longest_identical_run'] = max(ev['longest_identical_run'], streak[key])
                ev['dark_frames'] += int(array.max() < 0.02)
                if eid in train_ids:
                    pixels = array.reshape(3, -1).astype(np.float64)
                    moment = image_moments[key]
                    moment['count'] += pixels.shape[1]
                    moment['sum'] += pixels.sum(axis=1)
                    moment['squares'] += np.square(pixels).sum(axis=1)
                    moment['min'] = np.minimum(moment['min'], pixels.min(axis=1))
                    moment['max'] = np.maximum(moment['max'], pixels.max(axis=1))
                if frame in sampled:
                    img = Image.fromarray((array.transpose(1, 2, 0) * 255).round().astype('uint8'))
                    img.thumbnail((240, 180))
                    tile = Image.new('RGB', (240, 210), 'white')
                    tile.paste(img, (0, 25))
                    ImageDraw.Draw(tile).text((3, 3), f'ep {eid} f{frame} {key}', fill='black')
                    tiles.append(tile)
        sheet = Image.new('RGB', (240 * len(report['cameras']), 210 * len(sampled)), 'white')
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % len(report['cameras'])) * 240, (index // len(report['cameras'])) * 210))
        filename = f'episode-{eid:04d}-contact-sheet.jpg'
        sheet.save(output / filename)
        sheet_files.append(filename)
        for key, ev in evidence.items():
            ev.update({'episode_index': eid, 'camera': key})
            stream_evidence.append(ev)
            require(ev['dark_frames'] < length, f'Episode {eid} {key}: entirely dark video')
            require(ev['longest_identical_run'] <= max(2, info['fps'] * args.max_identical_seconds), f'Episode {eid} {key}: possible frozen video')
    for key, moment in image_moments.items():
        mean = moment['sum'] / moment['count']
        std = np.sqrt(np.maximum(0, moment['squares'] / moment['count'] - mean ** 2))
        numeric_stats[key] = {name: value.reshape(3, 1, 1).tolist() for name, value in {
            'mean': mean, 'std': std, 'min': moment['min'], 'max': moment['max']}.items()}
        numeric_stats[key]['count'] = [sum(row['episode_index'] in train_ids for row in rows)]
    require(snapshot(root) == before, 'Dataset changed during validation; discard this evidence and finalize again')
    report.update({'status': 'OFFLINE_DATA_VALIDATED', 'evidence_kind': args.evidence_kind, 'finalized': True, 'skill': args.skill,
                   'dataset': str(root), 'features': info['features'], 'files': before, 'decoded_streams': stream_evidence,
                   'contact_sheets': sheet_files, 'normalization': 'train episodes only; exact numeric quantiles and RGB pixel moments',
                   'versions': {name: importlib.metadata.version(name) for name in ('physicalai', 'lerobot', 'torch', 'lightning')},
                   'limits': ['Contact sheets require human/agent visual review; no grasp success inferred.',
                              'Recorder timestamps may be frame_index/fps; sensor wall-clock freshness is not established.',
                              'Joint units, calibration, action meaning, and physical success require capture contract evidence.',
                              'This pass does not execute ACT; run smoke_act.py separately.']})
    dump(output / 'train_stats.json', numeric_stats)
    report['train_stats_sha256'] = hashlib.sha256((output / 'train_stats.json').read_bytes()).hexdigest()
    dump(output / 'manifest.json', report)
    print(json.dumps({'manifest': str(output / 'manifest.json'), 'status': report['status'], 'evidence_kind': args.evidence_kind}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skill', required=True, help='One unambiguous manipulation skill')
    parser.add_argument('--evidence-kind', choices=('real', 'synthetic'), required=True)
    parser.add_argument('--finalized', action='store_true', help='Dataset owner confirms recording stopped and writers finalized')
    parser.add_argument('--validation-episodes', type=lambda value: [int(item) for item in value.split(',')])
    parser.add_argument('--final-eval-episodes', type=lambda value: [int(item) for item in value.split(',')], default=[])
    parser.add_argument('--episode-labels', type=Path, help='JSON keyed by episode ID: destination, outcome, outcome_evidence')
    parser.add_argument('--split-scope', choices=('loader_smoke', 'grouped_evaluation'), default='loader_smoke')
    parser.add_argument('--max-frames', type=int, default=20000)
    parser.add_argument('--gripper-min-range', type=float, default=1e-3)
    parser.add_argument('--max-identical-seconds', type=float, default=2)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(json.dumps({'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
