"""Dependency-free checks for finalized Studio LeRobot v3 episodes."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(root: Path):
    files = sorted(p for p in root.rglob('*') if p.is_file())
    require(files, 'Dataset is empty')
    result = {}
    for path in files:
        require(not path.is_symlink(), f'Symlink not accepted: {path}')
        require(path.suffix not in {'.tmp', '.partial'}, f'Unfinished file: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        result[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    return result


def validate_rows(info, episodes, rows, validation_episodes, gripper_min_range=1e-3, final_eval_episodes=()):
    require(info.get('codebase_version') == 'v3.0', 'Only installed Studio LeRobot v3.0 is supported')
    fps = info['fps']
    require(math.isfinite(fps) and fps > 0, 'FPS must be positive and finite')
    features = info['features']
    names = {}
    for key in ('observation.state', 'action'):
        feature = features.get(key, {})
        names[key] = feature.get('names')
        require(isinstance(names[key], list) and len(names[key]) == len(set(names[key])), f'{key}: unique joint names required')
        require(feature.get('shape') == [len(names[key])], f'{key}: shape does not match names')
        require(len(names[key]) > 0, f'{key}: no joints')
    require(names['action'] == names['observation.state'], 'State/action joint order differs')
    grippers = [i for i, name in enumerate(names['action']) if 'gripper' in name.lower()]
    require(len(grippers) == 1, 'Exactly one named gripper required')
    cameras = [key for key, feature in features.items() if feature.get('dtype') in {'video', 'image'}]
    require(cameras, 'ACT requires at least one image stream')
    require(len(episodes) == info['total_episodes'] >= 2, 'Need at least two complete episodes matching metadata')
    require(len(rows) == info['total_frames'], 'Metadata frame count differs from data rows')
    ids = [int(ep['episode_index']) for ep in episodes]
    require(ids == list(range(len(ids))), 'Episode metadata must be ordered, unique, and contiguous')
    validation = set(validation_episodes)
    final_eval = set(final_eval_episodes)
    require(not validation & final_eval, 'Validation and final evaluation must be disjoint')
    require(final_eval <= set(ids), 'Unknown final evaluation episode')
    require(len(final_eval) == len(final_eval_episodes), 'Repeated final evaluation episode')
    require(validation and validation | final_eval < set(ids), 'Train and validation must both contain whole episodes')
    require(len(validation) == len(validation_episodes), 'Repeated validation episode')
    report = []
    cursor = 0
    for ep in episodes:
        eid, size = int(ep['episode_index']), int(ep['length'])
        require(size >= 2, f'Episode {eid}: too short')
        require(ep['dataset_from_index'] == cursor and ep['dataset_to_index'] == cursor + size, f'Episode {eid}: inconsistent boundaries')
        selected = rows[cursor:cursor + size]
        require(len(selected) == size, f'Episode {eid}: truncated')
        for frame, row in enumerate(selected):
            require(row['episode_index'] == eid and row['frame_index'] == frame and row['index'] == cursor + frame, f'Episode {eid}: reordered/missing/duplicate frame {frame}')
            timestamp = float(row['timestamp'])
            require(math.isfinite(timestamp) and abs(timestamp - frame / fps) <= max(1e-4, 0.1 / fps), f'Episode {eid} frame {frame}: timestamp gap/order/FPS mismatch')
            for key in names:
                values = row[key]
                require(len(values) == len(names[key]), f'{key}: vector size mismatch')
                require(all(math.isfinite(float(value)) for value in values), f'{key}: nonfinite values')
        ranges = {}
        for key in names:
            values = [float(row[key][grippers[0]]) for row in selected]
            ranges[key] = max(values) - min(values)
            require(ranges[key] >= gripper_min_range, f'Episode {eid}: {key} gripper has no useful variation')
        value_ranges = {key: [[min(float(row[key][j]) for row in selected),
                              max(float(row[key][j]) for row in selected)]
                             for j in range(len(names[key]))] for key in names}
        report.append({'episode_index': eid, 'frames': size, 'seconds': size / fps,
                       'gripper_ranges': ranges, 'joint_ranges': value_ranges})
        cursor += size
    require(cursor == len(rows), 'Unindexed rows after last episode')
    return {'cameras': cameras, 'joint_names': names['action'], 'fps': fps, 'episodes': report,
            'train_episodes': [eid for eid in ids if eid not in validation | final_eval],
            'validation_episodes': sorted(validation), 'final_eval_episodes': sorted(final_eval)}


def validate_groups(report, labels, evidence_kind, split_scope):
    """Keep diagnostic same-specimen splits separate from evaluation evidence."""
    require(set(labels) == {str(ep['episode_index']) for ep in report['episodes']}, 'Labels must cover every episode exactly')
    destinations = set()
    for ep in report['episodes']:
        label = labels[str(ep['episode_index'])]
        required = ['destination']
        if evidence_kind == 'real':
            required += ['outcome_evidence', 'specimen_id', 'session_id']
        for field in required:
            require(isinstance(label.get(field), str) and label[field].strip(), f'Episode {ep["episode_index"]}: missing {field}')
        expected = 'complete_success' if evidence_kind == 'real' else 'synthetic_success'
        require(label.get('outcome') == expected, f'Episode {ep["episode_index"]}: requires {expected}; keep failures separately')
        destinations.add(label['destination'])
        ep.update({key: label[key] for key in ('destination', 'outcome', 'outcome_evidence', 'specimen_id', 'session_id') if key in label})
    require(len(destinations) == 1, 'One destination per standard ACT skill')
    if split_scope == 'grouped_evaluation':
        require(evidence_kind == 'real', 'Synthetic splits cannot establish grouped evaluation')
        for group in ('specimen_id', 'session_id'):
            sets = [{labels[str(eid)][group] for eid in report[key]} for key in
                    ('train_episodes', 'validation_episodes', 'final_eval_episodes')]
            require(not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]),
                    f'{group} overlaps across splits; use independent groups or loader_smoke')
    report['split_scope'] = split_scope
    report['split_limit'] = ('Software/loader diagnostic only; no specimen/session independence or quality claim'
                             if split_scope == 'loader_smoke' else
                             'Distinct labeled specimens and sessions; labels do not establish statistical power or physical success')
