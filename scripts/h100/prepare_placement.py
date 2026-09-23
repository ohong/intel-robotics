#!/usr/bin/env python3
"""Prepare an operator-confirmed derivative recording for H100 fitting; never train."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from placement_contract import DECISIONS, INSTRUCTIONS, check_placement, frozen_capture
from training_act import dump, digest, read, require, snapshot


def split_routes(routes, seed):
    result = {'train_episodes': [], 'validation_episodes': [], 'final_eval_episodes': []}
    rng = random.Random(seed)
    for route in INSTRUCTIONS:
        ids = sorted(int(eid) for eid, destination in routes.items() if destination == route)
        require(len(ids) >= 5, f'Need at least five {route} placement demonstrations')
        rng.shuffle(ids)
        holdout = max(1, len(ids) // 5)
        result['validation_episodes'].extend(ids[:holdout])
        result['final_eval_episodes'].extend(ids[holdout:2 * holdout])
        result['train_episodes'].extend(ids[2 * holdout:])
    return {key: sorted(value) for key, value in result.items()}


def prepare(args):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, output = args.dataset_root.resolve(), args.output_root.resolve()
    require(root.is_dir() and args.derivative_copy_confirmed, 'Use a frozen derivative copy; preserve original Studio recordings')
    require(not output.exists(), 'Use a fresh metadata output directory')
    require(not output.is_relative_to(root), 'Metadata output must be outside the dataset snapshot')
    routes = {int(eid): value for eid, value in read(args.routes).items()}
    require(all(route in INSTRUCTIONS for route in routes.values()), 'Map every episode explicitly to BLUE or PINK')
    splits = split_routes(routes, args.seed)
    info, environment = read(root / 'meta/info.json'), read(args.environment)
    features = info['features']
    require(features['observation.state']['shape'] == features['action']['shape'] == [6], 'Need six-joint measured state/actions')
    require(features['observation.state']['names'] == features['action']['names'], 'State/action order differs')
    original_inventory = snapshot(root)
    rows = []
    data_paths = sorted((root / 'data').rglob('*.parquet'))
    require(data_paths, 'Missing recorded rows')
    original_tasks = pq.read_table(root / 'meta/tasks.parquet').to_pandas()
    original_task_text = {int(row['task_index']): str(index) for index, row in original_tasks.iterrows()}
    task_ids = {'BLUE': 0, 'PINK': 1}
    for path in data_paths:
        table = pq.read_table(path)
        values = table.select(['episode_index', 'frame_index', 'index', 'task_index', 'observation.state', 'action']).to_pylist()
        require(all(row['episode_index'] in routes for row in values), 'An unassigned source episode is present')
        for row in values:
            row['original_task'] = original_task_text[row['task_index']]
        rows.extend(values)
    by_id = {eid: [row for row in rows if row['episode_index'] == eid] for eid in routes}
    require(len({row['index'] for row in rows}) == len(rows), 'Duplicate global row index')
    for eid, episode_rows in by_id.items():
        require(episode_rows and sorted(row['frame_index'] for row in episode_rows) == list(range(len(episode_rows))),
                f'Episode {eid}: missing, duplicate, or truncated frame indices')
        for feature in ('observation.state', 'action'):
            values = np.asarray([row[feature] for row in episode_rows], dtype=np.float64)
            require(values.shape == (len(episode_rows), 6) and np.isfinite(values).all(), f'Episode {eid}: invalid {feature}')
    training = [row for row in rows if row['episode_index'] in splits['train_episodes']]
    stats = {}
    for feature in ('observation.state', 'action'):
        values = np.asarray([row[feature] for row in training], dtype=np.float64)
        stats[feature] = {'mean': values.mean(0).tolist(), 'std': values.std(0).tolist(), 'min': values.min(0).tolist(),
                          'max': values.max(0).tolist(), 'count': [len(training)]}
        for q in (1, 10, 50, 90, 99):
            stats[feature][f'q{q:02d}'] = np.quantile(values, q / 100, axis=0).tolist()
    require(args.visual_normalization_identity_verified, 'Verify pinned native SmolVLA VISUAL identity normalization before preparation')
    cameras = sorted(key for key in features if key.startswith('observation.images.'))
    camera_map, slots = {}, {}
    for key in cameras:
        # Runtime sanitization is the frozen Studio dataset convention; require one match.
        suffix = key.removeprefix('observation.images.')
        matches = [camera for camera in environment['cameras'] if camera['name'].replace('(', '_').replace(')', '_') == suffix]
        require(len(matches) == 1, f'Camera identity is ambiguous: {key}')
        camera = matches[0]
        serial = camera.get('fingerprint', {}).get('serial')
        require(serial, f'Missing camera serial: {key}')
        slot = 0 if 'low' in suffix else 1 if 'high' in suffix else 2 if 'wrist' in suffix else None
        require(slot is not None, f'Unknown camera slot: {key}')
        slots[suffix] = slot
        camera_map[key] = {'serial': serial, 'recorded_rgb_shape_hwc': features[key]['shape'],
                           'source_configured_width': camera['payload']['width'], 'source_configured_height': camera['payload']['height'],
                           'source_environment_camera_id': camera['id']}
        # These are fixed input-domain bounds, not statistics learned from any holdout.
        stats[key] = {'min': [[[0.0]]] * 3, 'max': [[[1.0]]] * 3, 'mean': [[[0.0]]] * 3,
                      'std': [[[1.0]]] * 3, 'count': [1]}
    require(len(set(slots.values())) == len(slots), 'Duplicate camera slot')
    follower = environment['robots'][0]['robot']
    require(follower['type'] == 'SO101_Follower', 'Unknown measured follower contract')
    calibration = follower['payload']['calibration']
    source_receipt = read(args.source_receipt)
    qc = read(args.visual_qc)
    require(qc, 'Need sampled visual QC evidence')
    operator_confirmation = args.operator_confirmation
    require(operator_confirmation.strip(), 'Need exact user-confirmed recording task scope')

    # No source writes occur until all portable/numeric checks above have passed.
    output.mkdir(parents=True)
    for path in data_paths:
        table = pq.read_table(path)
        task_column = pa.array([task_ids[routes[eid]] for eid in table['episode_index'].to_pylist()], type=table['task_index'].type)
        table = table.set_column(table.schema.get_field_index('task_index'), 'task_index', task_column)
        pq.write_table(table, path)
    task_frame = pd.DataFrame({'task_index': [0, 1]}, index=[INSTRUCTIONS['BLUE'], INSTRUCTIONS['PINK']])
    pq.write_table(pa.Table.from_pandas(task_frame), root / 'meta/tasks.parquet')
    for path in sorted((root / 'meta/episodes').rglob('*.parquet')):
        table = pq.read_table(path)
        if 'tasks' in table.column_names:
            texts = [[INSTRUCTIONS[routes[eid]]] for eid in table['episode_index'].to_pylist()]
            table = table.set_column(table.schema.get_field_index('tasks'), 'tasks', pa.array(texts, type=table['tasks'].type))
            pq.write_table(table, path)
    info['total_tasks'] = 2
    dump(root / 'meta/info.json', info)
    dump(root / 'meta/stats.json', stats)
    dump(output / 'train_stats.json', stats)
    mapping = {str(eid): {'destination': routes[eid], 'original_captions': sorted({row['original_task'] for row in episode_rows}),
                          'canonical_instruction': INSTRUCTIONS[routes[eid]]} for eid, episode_rows in by_id.items()}
    dump(output / 'caption-provenance.json', mapping)
    dump(output / 'source-inventory.json', original_inventory)
    dump(output / 'source-snapshot-receipt.json', source_receipt)
    dump(output / 'sampled-visual-qc.json', qc)
    dump(output / 'environment.json', environment)
    contract = {'start': 'operator_confirmed_inspection_area', 'actual_ACT_handoff_pose_verified': False,
                'end': 'released_block_and_cleared_bin',
                'qualification': 'operator_confirmed_demonstration_scope', 'operator_confirmation': operator_confirmation,
                'individual_success_verified': False, 'sampled_visual_qc': {'path': 'sampled-visual-qc.json', 'sha256': digest(output / 'sampled-visual-qc.json')},
                'handoff_reference': 'User confirms all 27 episodes start FROM INSPECTION; exact ACT joint-state parity remains a deployment check.',
                'robot_control_owner': 'existing Intel Studio robot-control owner; training does not acquire control'}
    manifest = {'schema_version': 1, 'status': 'OFFLINE_DATA_VALIDATED', 'skill': 'inspection_to_selected_bin',
                'evidence_kind': 'real', 'finalized': True, 'task_ready': False, 'controller_ready': False,
                'validation_scope': 'Frozen numeric/structural checks; user-confirmed placement scope with sampled visual QC; individual physical success unverified',
                'placement_contract': contract, 'joint_names': features['observation.state']['names'], 'cameras': cameras,
                'features': features, 'fps': info['fps'], 'total_episodes': len(by_id), 'total_frames': len(rows),
                'robot_contract': {'state_semantics': 'measured follower joint positions', 'joint_order': features['observation.state']['names'],
                                   'action_semantics': 'absolute joint position targets; not velocities or deltas',
                                   'body_units': 'normalized calibrated position [-100,100]', 'gripper_units': 'normalized calibrated position [0,100]',
                                   'calibration_sha256': hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()},
                'capture_provenance': {'environment_sha256': digest(output / 'environment.json'),
                                       'source_snapshot_sha256': digest(output / 'source-snapshot-receipt.json'),
                                       'camera_identity_mapping': camera_map,
                                       'timing_limits': 'Recorded frame cadence; hardware capture simultaneity and action acknowledgment timing are not certified'},
                'episodes': [{'episode_index': eid, 'destination': routes[eid], 'frames': len(episode_rows),
                              'task_caption': INSTRUCTIONS[routes[eid]], 'outcome': 'operator_confirmed_placement_demonstration',
                              'individual_success_verified': False, 'outcome_evidence': operator_confirmation}
                             for eid, episode_rows in sorted(by_id.items())],
                **splits, 'split_scope': 'episode_pilot', 'generalization_claim_permitted': False,
                'normalization': {'training_episodes_only': splits['train_episodes'], 'training_rows': len(training),
                                  'excluded_validation_episodes': splits['validation_episodes'], 'excluded_final_episodes': splits['final_eval_episodes'],
                                  'images': 'Fixed identity-domain bounds; native VISUAL normalization is IDENTITY; count=1 denotes one fixed specification, not a measured pixel; no validation/test pixels fitted'},
                'files': snapshot(root), 'train_stats_sha256': digest(output / 'train_stats.json')}
    config = read(args.prior_config)
    h100_root = Path(args.h100_root)
    config.update(training_scope='step_three_placement', dataset_root=str(h100_root / 'dataset'),
                  manifest=str(h100_root / 'prepared/placement-manifest.json'), output_dir=str(h100_root / 'runs/fit-v1'),
                  capture_manifest=None, capture_runtime=None, image_key_reorder_map=slots,
                  camera_mapping_evidence='Frozen Studio environment camera names/serials; low0 high1 wrist2; dimensions from derivative dataset',
                  conditioning={'mode': 'selected_instruction', 'claim': 'experimental_imitation', 'decision_mapping': DECISIONS, 'instructions': INSTRUCTIONS},
                  seed=args.seed, max_steps=1000, max_seconds=1800, batch_size=8, num_workers=0, precision='bf16-mixed',
                  validation_every=250, checkpoint_every=100, learning_rate=1e-4, warmup_steps=100, scheduler_decay_steps=2000)
    check_placement(manifest, config['conditioning'])
    frozen_capture(manifest)
    dump(output / 'placement-manifest.json', manifest)
    config['manifest_sha256'] = digest(output / 'placement-manifest.json')
    dump(output / 'placement-config.json', config)
    smoke = dict(config, output_dir=str(h100_root / 'runs/smoke-v1'), warmup_steps=0)
    dump(output / 'placement-smoke-config.json', smoke)
    print(json.dumps({'status': 'FROZEN_OPERATOR_CONFIRMED_PLACEMENT_PREPARED', 'episodes': len(by_id), 'frames': len(rows),
                      'splits': splits, 'manifest': str(output / 'placement-manifest.json'), 'training_executed': False,
                      'physical_success': 'NOT VERIFIED'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root', 'output-root', 'routes', 'environment', 'source-receipt', 'visual-qc', 'prior-config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--operator-confirmation', required=True)
    parser.add_argument('--h100-root', default='/workspace/second-look-h100/placement-v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--derivative-copy-confirmed', action='store_true')
    parser.add_argument('--visual-normalization-identity-verified', action='store_true')
    args = parser.parse_args()
    try:
        prepare(args)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(2, f'BLOCKED: {exc}\n')


if __name__ == '__main__':
    main()
