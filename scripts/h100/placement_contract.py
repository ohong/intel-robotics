#!/usr/bin/env python3
"""Fail-closed step-three placement contract; importing this never connects hardware."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

INSTRUCTIONS = {
    'BLUE': 'Pick up the block from the inspection area and place it in the blue bin.',
    'PINK': 'Pick up the block from the inspection area and place it in the pink bin.',
}
DECISIONS = {'GOOD': 'BLUE', 'BAD': 'PINK', 'UNKNOWN': None}
SPLITS = {'train_episodes': 3, 'validation_episodes': 1, 'final_eval_episodes': 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_instruction(decision):
    """Anomalib owns the decision; UNKNOWN must never reach policy placement."""
    require(isinstance(decision, str) and decision in DECISIONS, 'Expected GOOD, BAD, or UNKNOWN')
    destination = DECISIONS[decision]
    return INSTRUCTIONS[destination] if destination else None


def placement_input(decision, cameras, measured_state):
    """Build policy inputs only. Freshness and action ownership remain with the controller."""
    instruction = selected_instruction(decision)
    if instruction is None:
        return None
    require(isinstance(cameras, dict) and cameras and all(
        isinstance(key, str) and key.startswith('observation.images.') and value is not None
        for key, value in cameras.items()), 'Need actual camera observations')
    require(isinstance(measured_state, (list, tuple)) and len(measured_state) == 6 and all(
        type(value) in (int, float) and math.isfinite(value) for value in measured_state),
        'Need six finite measured follower joint positions')
    return {'task': instruction, 'observation.state': list(measured_state), **cameras}


def check_placement(manifest, conditioning):
    """Validate semantic evidence before expensive dataset/model checks.

    Evidence references are recording-owner attestations, not proof from this validator.
    File inventories and frame contents are checked by the training runner separately.
    """
    require(manifest.get('skill') == 'inspection_to_selected_bin', 'Need step-three placement demonstrations; step-one ACT data is insufficient')
    require(manifest.get('evidence_kind') == 'real' and manifest.get('finalized') is True,
            'Need finalized real recordings')
    require(manifest.get('task_ready') is False and manifest.get('controller_ready') is False,
            'Recording qualification does not establish deployment readiness')
    require(conditioning.get('mode') == 'selected_instruction', 'Placement requires selected-instruction language, without anomaly scores')
    require(conditioning.get('claim') == 'experimental_imitation', 'Pilot training supports experimental imitation only')
    require(conditioning.get('decision_mapping') == DECISIONS, 'Wrong GOOD/BLUE, BAD/PINK, UNKNOWN/no-action mapping')
    require(conditioning.get('instructions') == INSTRUCTIONS, 'Use the exact placement instructions')
    contract = manifest.get('placement_contract', {})
    operator_scope = contract.get('qualification') == 'operator_confirmed_demonstration_scope'
    starts = ('actual_ACT_handoff_pose', 'operator_confirmed_inspection_area') if operator_scope else ('actual_ACT_handoff_pose',)
    require(contract.get('start') in starts and contract.get('end') == 'released_block_and_cleared_bin',
            'Record inspection handoff through release and bin clearance')
    require(contract.get('handoff_reference') and contract.get('robot_control_owner'), 'Need handoff reference and existing robot-control owner')
    if operator_scope:
        require(contract.get('operator_confirmation') and contract.get('sampled_visual_qc')
                and contract.get('individual_success_verified') is False,
                'Operator-confirmed scope needs sampled QC and explicit individual-success limits')
    robot = manifest.get('robot_contract', {})
    require(robot.get('state_semantics') == 'measured follower joint positions', 'Commanded targets cannot replace measured robot state')
    require(robot.get('joint_order') == manifest.get('joint_names') and len(manifest.get('joint_names', [])) == 6,
            'Need the actual six-joint order')
    cameras = manifest.get('cameras', [])
    features = manifest.get('features', {})
    require(cameras and set(cameras) == {key for key in features if key.startswith('observation.images.')},
            'Need every recorded camera with feature metadata')
    require(features.get('observation.state', {}).get('shape') == [6] and features.get('action', {}).get('shape') == [6],
            'Need measured state and action features')
    episodes = manifest.get('episodes', [])
    by_id = {}
    for episode in episodes:
        eid = episode.get('episode_index')
        require(type(eid) is int and eid >= 0 and eid not in by_id, 'Duplicate or invalid episode ID')
        by_id[eid] = episode
        require(episode.get('destination') in INSTRUCTIONS, f'Episode {eid}: destination must be BLUE or PINK')
        require(episode.get('task_caption') == INSTRUCTIONS[episode['destination']], f'Episode {eid}: wrong recorded task instruction')
        accepted = ('complete_success', 'operator_confirmed_placement_demonstration') if operator_scope else ('complete_success',)
        require(episode.get('outcome') in accepted and episode.get('outcome_evidence'), f'Episode {eid}: need placement demonstration evidence')
        frames = episode.get('frames')
        require(type(frames) is int and frames >= 3, f'Episode {eid}: need full placement recording')
        events = episode.get('placement_events', {})
        if operator_scope and episode.get('outcome') == 'operator_confirmed_placement_demonstration':
            require(episode.get('individual_success_verified') is False, f'Episode {eid}: do not invent individual success')
            continue
        require(events.get('starts_at_actual_ACT_handoff') is True and events.get('handoff_evidence'), f'Episode {eid}: ACT handoff is unverified')
        released, cleared = events.get('release_frame'), events.get('clear_frame')
        require(type(released) is int and type(cleared) is int and 0 < released < cleared < frames,
                f'Episode {eid}: require release then clearance inside recording')
        require(events.get('release_evidence') and events.get('clearance_evidence'), f'Episode {eid}: missing release/clear evidence')
    counts = {route: sum(ep['destination'] == route for ep in episodes) for route in INSTRUCTIONS}
    missing = {route: max(0, 5 - count) for route, count in counts.items()}
    require(not any(missing.values()), f'Missing qualified placement episodes: {missing}; require five pilots per route')
    assigned = []
    route_splits = {}
    for key, minimum in SPLITS.items():
        ids = manifest.get(key, [])
        require(isinstance(ids, list) and all(type(eid) is int and eid in by_id for eid in ids), f'Invalid {key}')
        assigned.extend(ids)
        route_splits[key] = {route: [eid for eid in ids if by_id[eid]['destination'] == route] for route in INSTRUCTIONS}
        require(all(len(ids) >= minimum for ids in route_splits[key].values()),
                f'{key}: need at least {minimum} whole episodes per destination')
    require(len(assigned) == len(set(assigned)) and set(assigned) == set(by_id), 'Whole episode splits must be disjoint and cover the dataset')
    normalization = manifest.get('normalization', {})
    require(isinstance(normalization, dict) and normalization.get('training_episodes_only') == manifest['train_episodes']
            and normalization.get('excluded_validation_episodes') == manifest['validation_episodes']
            and normalization.get('excluded_final_episodes') == manifest['final_eval_episodes']
            and normalization.get('training_rows') == sum(by_id[eid]['frames'] for eid in manifest['train_episodes']),
            'Fit normalization exclusively on training episodes; exclude both holdouts')
    require(manifest.get('split_scope') in ('episode_pilot', 'grouped_evaluation'), 'Use whole episode pilot or independent grouped evaluation')
    if manifest['split_scope'] == 'episode_pilot':
        require(manifest.get('generalization_claim_permitted') is False, 'Same-session pilots cannot establish independent generalization')
    return {'route_counts': counts, 'route_splits': route_splits, 'physical_success': 'NOT TESTED', 'controller_ready': False}


def frozen_capture(manifest):
    """Use frozen recording provenance without constructing a hardware runtime."""
    capture = manifest.get('capture_provenance', {})
    require(capture.get('environment_sha256') and capture.get('source_snapshot_sha256'), 'Missing frozen capture identities')
    require(capture.get('camera_identity_mapping') and set(capture['camera_identity_mapping']) == set(manifest['cameras']),
            'Missing frozen camera identities')
    for key, value in capture['camera_identity_mapping'].items():
        require(value.get('serial') and value.get('recorded_rgb_shape_hwc') == manifest['features'][key]['shape'],
                'Frozen camera identity or dimensions differ')
    robot = manifest['robot_contract']
    require(robot.get('action_semantics') == 'absolute joint position targets; not velocities or deltas'
            and robot.get('body_units') == 'normalized calibrated position [-100,100]'
            and robot.get('gripper_units') == 'normalized calibrated position [0,100]', 'Unknown recorded action units')
    require(robot.get('calibration_sha256') and capture.get('timing_limits'), 'Missing calibration or timing evidence limits')
    return capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check_placement(json.loads(args.manifest.read_text()), json.loads(args.config.read_text())['conditioning'])
        print(json.dumps({'status': 'PLACEMENT_SEMANTICS_CHECKED', 'training_executed': False, **result}, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'BLOCKED: {exc}\n')


if __name__ == '__main__':
    main()
