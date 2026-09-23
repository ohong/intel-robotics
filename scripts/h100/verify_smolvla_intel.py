#!/usr/bin/env python3
"""Offline native SmolVLA CPU references and Intel verification; never connects hardware."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

from export_smolvla import digest, dump, inventory, load_observation, read, require
from placement_contract import DECISIONS, INSTRUCTIONS, selected_instruction


def checked_inventory(root, entries):
    require(isinstance(entries, dict) and entries, 'Missing artifact inventory')
    for relative, expected in entries.items():
        path = root / relative
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'Inventory paths must remain inside artifact root')
        require(path.is_file() and path.stat().st_size == expected['bytes']
                and digest(path) == expected['sha256'], f'Artifact mismatch: {relative}')


def validate_observation(document, manifest, manifest_sha256):
    require(document.get('evidence_kind') == 'real' and document.get('manifest_sha256') == manifest_sha256,
            'Observation must come from the frozen real training dataset')
    require(manifest.get('evidence_kind') == 'real' and manifest.get('finalized') is True
            and manifest.get('status') == 'OFFLINE_DATA_VALIDATED', 'Dataset is not finalized real evidence')
    require(manifest.get('robot_contract', {}).get('state_semantics') == 'measured follower joint positions',
            'Measured follower state provenance is required; commanded targets are insufficient')
    require(document.get('episode_index') in manifest['validation_episodes'], 'Use validation only, not train or final evaluation')
    require(document.get('state_names') == manifest.get('joint_names')
            == manifest['robot_contract'].get('joint_order') and len(document['state_names']) == 6,
            'Measured state joint order differs')
    require(len(document.get('state', [])) == 6 and all(type(v) in (int, float) and math.isfinite(v)
            for v in document['state']), 'Need six finite measured state values')
    require(set(document.get('images', {})) == {k.removeprefix('observation.images.') for k in manifest['cameras']},
            'Use every frozen camera recording')


def compare_chunks(actual, expected, atol, rtol):
    require(len(actual) == len(expected) == 1 and len(actual[0]) == len(expected[0]) > 0,
            'Reference chunk shape differs')
    differences, failed = [], 0
    for row, reference in zip(actual[0], expected[0]):
        require(len(row) == len(reference) == 6, 'Expected six action joints')
        for value, target in zip(row, reference):
            require(type(value) in (float, int) and type(target) in (float, int)
                    and math.isfinite(value) and math.isfinite(target), 'Nonfinite or invalid action')
            error = abs(value - target)
            differences.append(error)
            failed += error > atol + rtol * abs(target)
    return {'passed': failed == 0, 'max_abs_error': max(differences),
            'mean_abs_error': statistics.mean(differences), 'elements_outside_tolerance': failed,
            'atol': atol, 'rtol': rtol}


def distribution(samples):
    require(samples and all(math.isfinite(v) and v >= 0 for v in samples), 'Invalid timing samples')
    ordered = sorted(samples)
    return {'samples_ms': samples, 'count': len(samples), 'min_ms': min(samples),
            'median_ms': statistics.median(samples), 'p95_ms': ordered[math.ceil(len(samples) * .95) - 1],
            'max_ms': max(samples)}


def infer(policy, observation, seed):
    """Observe native noise without changing it, then restore the model and CPU RNG."""
    import torch
    flow = policy.model._model
    original = flow._sample_noise
    prior_override = flow.__dict__.get('_sample_noise')
    had_override = '_sample_noise' in flow.__dict__
    noise_records = []

    def record_noise(shape, device):
        require(torch.device(device).type == 'cpu', 'Random noise must be generated on CPU')
        noise = original(shape, device)
        raw = noise.detach().contiguous().cpu().numpy().tobytes()
        noise_records.append({'shape': list(noise.shape), 'dtype': str(noise.dtype),
                              'sha256': hashlib.sha256(raw).hexdigest()})
        return noise

    policy.reset()
    # A fresh observation avoids mutable preprocessing state between routes or repeats.
    batch = copy.deepcopy(observation)
    flow._sample_noise = record_noise
    try:
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            started = time.perf_counter()
            action = policy.predict_action_chunk(batch).detach().float().cpu()
            elapsed = (time.perf_counter() - started) * 1000
        require(action.ndim == 3 and action.shape[0] == 1 and action.shape[2] == 6
                and action.shape[1] > 0 and bool(torch.isfinite(action).all()), 'Invalid action chunk')
        require(len(noise_records) == 1, 'Native noise sampling contract changed')
        return action.tolist(), elapsed, noise_records
    finally:
        if had_override:
            flow._sample_noise = prior_override
        else:
            delattr(flow, '_sample_noise')


def run(args):
    require(platform.system() == 'Linux', 'Inference is permitted only on Linux; use --self-test on Mac')
    require(not args.output.exists(), 'Choose a new output file; existing evidence is preserved')
    require(args.repeats >= 1 and args.threads >= 1 and 0 <= args.seed < 2**32, 'Invalid benchmark options')
    require(all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)), 'Invalid parity tolerance')
    require(all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')),
            'Set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 before launch')
    artifact = args.export_dir.resolve()
    metadata_path = artifact / 'export.json'
    require(digest(metadata_path) == args.export_sha256, 'Export receipt SHA mismatch')
    metadata = read(metadata_path)
    require(metadata.get('export_validation') == 'EXACT_NATIVE_HF_CONFIG_WEIGHTS_AND_STATS_PASSED'
            and metadata.get('controller_ready') is False and metadata.get('task_ready') is False,
            'Need a verified native export with readiness false')
    checked_inventory(artifact, metadata['files'])
    require({'config.json', 'model.safetensors', 'policy_preprocessor.json', 'training-normalization.safetensors'}
            <= set(metadata['files']), 'Incomplete exported checkpoint')
    backbone, library = args.backbone_dir.resolve(), args.studio_library.resolve()
    require(inventory(backbone) == metadata['backbone_files'], 'Local backbone inventory differs')
    require(metadata.get('studio_source_sha256'), 'Missing pinned Studio sources')
    for relative, expected in metadata['studio_source_sha256'].items():
        source = (library / relative).resolve()
        require(source.is_relative_to(library) and digest(source) == expected, 'Studio source differs from training')
    manifest_sha = digest(args.manifest)
    require(manifest_sha == metadata['manifest_sha256'], 'Frozen dataset manifest differs from export')
    document, manifest = read(args.observation), read(args.manifest)
    validate_observation(document, manifest, manifest_sha)
    identity = {'export_sha256': args.export_sha256, 'checkpoint_sha256': metadata['checkpoint_sha256'],
                'observation_sha256': digest(args.observation), 'manifest_sha256': manifest_sha,
                'seed': args.seed, 'device': 'cpu', 'precision': 'float32',
                'instructions': INSTRUCTIONS, 'decision_mapping': DECISIONS,
                'verifier_sha256': digest(__file__)}
    reference = None
    if args.mode == 'verify':
        require(args.reference is not None and args.reference_sha256, 'Provide pinned H100 reference and SHA')
        require(digest(args.reference) == args.reference_sha256, 'H100 reference SHA differs')
        reference = read(args.reference)
        require(reference.get('status') == 'H100_CPU_REFERENCE_RECORDED'
                and reference['identity'] == identity, 'Reference model, inputs, prompts, seed, or verifier differ')
    # Explicit pinned imports prevent an unrelated editable Studio install from taking precedence.
    sys.path.insert(0, str(library / 'src'))
    import torch
    from physicalai.policies.smolvla.policy import SmolVLA
    require(Path(inspect.getfile(SmolVLA)).resolve().is_relative_to(library), 'Wrong imported Studio library')
    torch.set_num_threads(args.threads)
    config = read(artifact / 'config.json')
    require(config.get('use_random_input_noise') is True and config.get('compile_model') is False
            and config.get('snapflow_enabled') is False, 'Unsupported deterministic-noise or compiled configuration')
    require(config['output_features']['action']['shape'] == [6], 'Expected six action joints')
    report = {'schema_version': 1, 'identity': identity, 'evidence_kind': 'real_recorded_observation_replay',
              'runtime': {'hostname': platform.node(), 'machine': platform.machine(), 'platform': platform.platform(),
                          'python': sys.version, 'executable': sys.executable, 'threads': args.threads,
                          'versions': {p: importlib.metadata.version(p) for p in
                                       ('torch', 'transformers', 'safetensors', 'lightning')}},
              'status': 'RUNNING', 'routes': {}, 'controller_ready': False, 'task_ready': False,
              'robot_connected': False, 'cameras_connected': False, 'training_executed': False,
              'physical_success': 'NOT TESTED', 'openvino': 'NOT TESTED',
              'reference_sha256': args.reference_sha256,
              'limits': ['One recorded observation; both instructions use identical cameras and measured state.',
                         'Recorded action ranges are descriptive, not physical safety limits.',
                         'CPU action-chunk latency includes native preprocessing and postprocessing plus noise hashing.',
                         'Cold chunk means first call per freshly loaded route; OS file cache may already be warm.',
                         'No autonomous routing or physical placement has been tested.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Relocate only the backbone path. Preserve exported bytes and share read-only large weights.
        with tempfile.TemporaryDirectory(prefix='smolvla-verify-', dir=args.output.parent) as temporary:
            local = Path(temporary)
            relocated = copy.deepcopy(config)
            relocated['vlm_model_name'] = str(backbone)
            for name in metadata['files']:
                require(len(Path(name).parts) == 1, 'Expected flat native export')
                if name != 'config.json':
                    (local / name).symlink_to(artifact / name)
            dump(local / 'config.json', relocated)
            for route, prompt in INSTRUCTIONS.items():
                print(f'{args.mode}: loading native CPU policy for {route}', flush=True)
                started = time.perf_counter()
                policy = SmolVLA(pretrained_name_or_path=local, **metadata['hf_constructor_kwargs']).cpu().float().eval()
                loaded_ms = (time.perf_counter() - started) * 1000
                require(all(p.device.type == 'cpu' and (not p.is_floating_point() or p.dtype == torch.float32)
                            for p in policy.parameters()), 'Model must use CPU FP32')
                expected_config = {k: v for k, v in relocated.items() if k not in ('input_features', 'output_features')}
                require(policy.config.to_dict() == expected_config, 'HF loader changed native configuration')
                observation = load_observation(args.observation, config)
                observation.task = [prompt]
                tokens = policy._preprocessor.tokenizer(prompt + '\n', truncation=False)['input_ids']
                require(len(tokens) <= config['tokenizer_max_length'], 'Exact route instruction would be truncated')
                action, cold_ms, noises = infer(policy, observation, args.seed)
                require(len(action[0]) == config['chunk_size'], 'Action chunk length differs from native config')
                warmed = []
                for _ in range(args.repeats):
                    repeated, elapsed, next_noises = infer(policy, observation, args.seed)
                    require(next_noises == noises, 'CPU seeded noise changed between repetitions')
                    require(compare_chunks(repeated, action, args.atol, args.rtol)['passed'], 'Repeated CPU output differs')
                    warmed.append(elapsed)
                from safetensors.torch import load_file
                stats = load_file(str(artifact / 'training-normalization.safetensors'))
                lower, upper = stats['action.min'].tolist(), stats['action.max'].tolist()
                outside = [sum(not lower[j] <= row[j] <= upper[j] for row in action[0]) for j in range(6)]
                item = {'decision': next(k for k, v in DECISIONS.items() if v == route), 'instruction': prompt,
                        'token_count': len(tokens), 'cold_load_ms': loaded_ms, 'cold_action_chunk_ms': cold_ms,
                        'warmed_action_chunk': distribution(warmed), 'noise': noises, 'actions': action,
                        'action_shape': [1, len(action[0]), 6], 'finite': True,
                        'min_per_joint': [min(row[j] for row in action[0]) for j in range(6)],
                        'max_per_joint': [max(row[j] for row in action[0]) for j in range(6)],
                        'training_min_per_joint': lower, 'training_max_per_joint': upper,
                        'outside_training_range_per_joint': outside,
                        'range_check': 'OUTSIDE_RECORDED_TRAINING_RANGE' if any(outside) else 'WITHIN_RECORDED_TRAINING_RANGE'}
                report['routes'][route] = item
                if reference:
                    require(reference['routes'][route]['noise'] == noises, 'CPU random noise differs; output parity is not comparable')
                    item['parity'] = compare_chunks(action, reference['routes'][route]['actions'], args.atol, args.rtol)
                    require(item['parity']['passed'], f'{route} CPU reference parity failed')
                del policy
            require(selected_instruction('UNKNOWN') is None, 'UNKNOWN must produce no placement instruction')
            report['unknown'] = {'instruction': None, 'policy_calls': 0, 'placement_action': None}
            left, right = (report['routes'][route]['actions'] for route in INSTRUCTIONS)
            difference = compare_chunks(left, right, 0, 0)['max_abs_error']
            report['instruction_sensitivity_max_abs'] = difference
            require(difference > 0, 'No measured instruction effect with identical recorded input and noise')
            report['status'] = 'INTEL_NATIVE_CPU_PARITY_PASSED' if reference else 'H100_CPU_REFERENCE_RECORDED'
    except BaseException as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        dump(args.output, report)
        raise
    dump(args.output, report)
    return {'status': report['status'], 'output': str(args.output), 'sha256': digest(args.output),
            'task_ready': False, 'controller_ready': False, 'physical_success': 'NOT TESTED'}


def self_test():
    import unittest

    class ContractTests(unittest.TestCase):
        def test_compare(self):
            chunk = [[[0., 1., 2., 3., 4., 5.]]]
            self.assertTrue(compare_chunks(chunk, chunk, 0, 0)['passed'])
            changed = copy.deepcopy(chunk)
            changed[0][0][0] = .1
            self.assertFalse(compare_chunks(changed, chunk, 1e-4, 1e-4)['passed'])
            changed[0][0][0] = float('nan')
            with self.assertRaises(ValueError):
                compare_chunks(changed, chunk, 1, 1)
            with self.assertRaises(ValueError):
                compare_chunks(chunk, [[[1.]]], 1, 1)

        def test_measured_provenance(self):
            manifest = {'evidence_kind': 'real', 'finalized': True, 'status': 'OFFLINE_DATA_VALIDATED',
                        'validation_episodes': [2], 'joint_names': list('abcdef'),
                        'robot_contract': {'state_semantics': 'measured follower joint positions', 'joint_order': list('abcdef')},
                        'cameras': ['observation.images.low', 'observation.images.high']}
            observation = {'evidence_kind': 'real', 'manifest_sha256': 'fixture', 'episode_index': 2,
                           'state_names': list('abcdef'), 'state': [1.] * 6, 'images': {'low': {}, 'high': {}}}
            validate_observation(observation, manifest, 'fixture')
            manifest['robot_contract']['state_semantics'] = 'commanded targets'
            with self.assertRaisesRegex(ValueError, 'Measured follower'):
                validate_observation(observation, manifest, 'fixture')
            manifest['robot_contract']['state_semantics'] = 'measured follower joint positions'
            observation['episode_index'] = 0
            with self.assertRaisesRegex(ValueError, 'validation only'):
                validate_observation(observation, manifest, 'fixture')

        def test_inventory_and_unknown(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'one').write_text('a')
                entries = inventory(root)
                checked_inventory(root, entries)
                (root / 'one').write_text('b')
                with self.assertRaises(ValueError):
                    checked_inventory(root, entries)
                with self.assertRaises(ValueError):
                    checked_inventory(root, {'../one': entries['one']})
            self.assertIsNone(selected_instruction('UNKNOWN'))
            self.assertEqual(selected_instruction('GOOD'), INSTRUCTIONS['BLUE'])
            self.assertEqual(selected_instruction('BAD'), INSTRUCTIONS['PINK'])
            with self.assertRaises(ValueError):
                selected_instruction('normal')

        def test_timings(self):
            self.assertEqual(distribution([3, 1, 2])['median_ms'], 2)
            with self.assertRaises(ValueError):
                distribution([float('inf')])

        def test_three_camera_export(self):
            from export_smolvla import export_plan, HF_OVERRIDES, STAT_FIELDS
            config = dict.fromkeys(HF_OVERRIDES, False)
            config.update(num_cameras=3, tokenizer_max_length=128, load_vlm_weights=False,
                          adapt_to_pi_aloha=False, n_action_steps=50,
                          image_key_reorder_map={'low': 0, 'high': 1, 'wrist': 2})
            manifest = {'evidence_kind': 'real', 'finalized': True, 'status': 'OFFLINE_DATA_VALIDATED',
                        'train_episodes': [0], 'validation_episodes': [1], 'final_eval_episodes': [],
                        'episodes': [{'episode_index': 0, 'frames': 10}, {'episode_index': 1, 'frames': 10}],
                        'normalization': 'train episodes only', 'joint_names': list('abcdef'),
                        'cameras': ['observation.images.' + key for key in ('low', 'high', 'wrist')]}
            stats = {key: {'count': [10], **{field: [0.] * 6 for field in STAT_FIELDS}}
                     for key in ('observation.state', 'action')}
            native = {key: {'name': key.removeprefix('observation.'), 'shape': [6], **value}
                      for key, value in stats.items()}
            native.update({'observation.' + key: {'name': key, 'shape': [3, 16, 16]}
                           for key in ('low', 'high', 'wrist')})
            _, _, _, slots = export_plan({'config': config, 'dataset_stats': native}, manifest, stats, Path('/fixture'))
            self.assertEqual(slots, ['low', 'high', 'wrist'])
            config['image_key_reorder_map']['wrist'] = 1
            with self.assertRaisesRegex(ValueError, 'distinct'):
                export_plan({'config': config, 'dataset_stats': native}, manifest, stats, Path('/fixture'))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    return 0 if result.wasSuccessful() else 1


def main():
    if sys.argv[1:] == ['--self-test']:
        raise SystemExit(self_test())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('reference', 'verify'))
    for name in ('export-dir', 'backbone-dir', 'studio-library', 'manifest', 'observation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--export-sha256', required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
