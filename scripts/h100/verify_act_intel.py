#!/usr/bin/env python3
"""Offline Studio native/OpenVINO replay parity. Never constructs a robot or camera."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import time

from training_act import digest, dump, read, require, snapshot


def check_action_shape(declared_shape, reference_shape, actual_shape, legacy_chunk_size):
    declared_shape = list(declared_shape)
    require(len(declared_shape) == 2 and all(type(size) is int and size > 0 for size in declared_shape),
            'Manifest action feature must declare [chunk, joints]')
    require(list(reference_shape) == declared_shape, 'Reference shape differs from manifest action feature')
    require(list(actual_shape) == declared_shape, 'Predicted shape differs from manifest action feature')
    return {'declared_action_shape': declared_shape, 'actual_action_shape': list(actual_shape),
            'legacy_chunk_size_property': legacy_chunk_size,
            'chunk_metadata_status': 'CONSISTENT' if legacy_chunk_size == declared_shape[0] else 'LEGACY_PROPERTY_MISMATCH'}


def run(args):
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    cpu_info = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').is_file() else ''
    require(platform.system() == 'Linux' and 'GenuineIntel' in cpu_info, 'Run Intel verification on the Intel Linux PC')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # Importing the adapter registers Studio's native checkpoint loader.
    import physicalai.inference.adapters.pytorch  # noqa: F401
    from physicalai.inference import InferenceModel

    root = args.run.resolve()
    evidence = read(root / 'result.json')
    require(evidence['status'] in ('EXPORTED_FOR_INTEL_VERIFICATION', 'UNSAFE_NOT_TASK_TRAINED'), 'Export is incomplete')
    for name, files in evidence['artifact_files'].items():
        require(snapshot(root / name) == files, f'Export changed during transfer: {name}')
    require(digest(root / 'export-replay.npz') == evidence['replay_sha256'], 'Replay fixture changed during transfer')
    with np.load(root / 'export-replay.npz', allow_pickle=False) as fixture:
        inputs = {name: fixture[name].copy() for name in fixture.files if name != 'expected_action'}
        reference = fixture['expected_action'].copy()
    # Studio InferenceModel deliberately returns [chunk, joints] for a single observation.
    if reference.ndim == 3 and reference.shape[0] == 1:
        reference = reference[0]
    require(np.isfinite(reference).all(), 'Nonfinite reference')
    result = {'status': 'STARTED', 'evidence_kind': evidence['evidence_kind'], 'host': platform.node(),
              'run_identity': evidence['run_identity'], 'native': {}, 'openvino': {},
              'cpu_models': sorted({line.split(':', 1)[1].strip() for line in cpu_info.splitlines() if line.startswith('model name')}),
              'cpu_thread_limit': 2, 'torch_interop_threads': 1,
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'openvino', 'physicalai')},
              'tolerance': {'atol': args.atol, 'rtol': args.rtol}, 'robot_execution': 'NOT TESTED',
              'limits': ['One saved observation only; this checks software compatibility, not task quality.',
                         'No camera, controller, robot connection, motion, or autonomous trial occurred.']}
    require(not args.output.exists(), 'Use a new parity evidence file')
    openvino_options = {'INFERENCE_PRECISION_HINT': 'f32'}
    if args.device == 'CPU':
        openvino_options.update(NUM_STREAMS='1', INFERENCE_NUM_THREADS='2')
    for name, directory, device, kwargs in (
        ('native', 'export-torch', 'cpu', {}),
        ('openvino', 'export-openvino', args.device, openvino_options),
    ):
        started = time.monotonic()
        try:
            model = InferenceModel(root / directory, device=device, **kwargs)
            load_seconds = time.monotonic() - started
            started = time.monotonic()
            with torch.inference_mode():
                actual = np.asarray(model.predict_action_chunk({key: value.copy() for key, value in inputs.items()}))
            seconds = time.monotonic() - started
            action_features = [feature for feature in model.output_features if feature.name == 'action']
            require(len(action_features) == 1, 'Export must declare exactly one action output feature')
            shape_evidence = check_action_shape(action_features[0].shape, reference.shape, actual.shape, model.chunk_size)
            require(actual.shape == reference.shape, f'{name} shape mismatch: {actual.shape} != {reference.shape}')
            require(np.isfinite(actual).all(), f'{name} nonfinite prediction')
            max_error = float(np.abs(actual - reference).max())
            np.testing.assert_allclose(actual, reference, atol=args.atol, rtol=args.rtol)
            latencies = []
            for _ in range(args.repeats):
                started = time.monotonic()
                with torch.inference_mode():
                    model.predict_action_chunk({key: value.copy() for key, value in inputs.items()})
                latencies.append((time.monotonic() - started) * 1000)
            result[name] = {'status': 'PASSED', 'device': device, 'load_seconds': load_seconds,
                            **shape_evidence,
                            'first_inference_seconds': seconds, 'max_abs_error': max_error,
                            'warm_median_ms': float(np.median(latencies)), 'repeats': args.repeats}
            if name == 'openvino':
                result[name]['execution_devices'] = list(model.adapter.compiled_model.get_property('EXECUTION_DEVICES'))
        except Exception as exc:
            result[name] = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
        dump(args.output, result)
    passed = all(result[name]['status'] == 'PASSED' for name in ('native', 'openvino'))
    result['status'] = 'PARITY_PASSED' if passed else 'PARITY_FAILED'
    if evidence['evidence_kind'] == 'synthetic':
        result['task_training_status'] = 'UNSAFE_NOT_TASK_TRAINED'
    dump(args.output, result)
    print(args.output)
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('CPU', 'GPU', 'NPU'), default='CPU')
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    require(args.repeats > 0 and args.atol > 0 and args.rtol >= 0, 'Invalid verification limits')
    raise SystemExit(run(args))
