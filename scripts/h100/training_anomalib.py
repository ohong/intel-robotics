#!/usr/bin/env python3
"""Fit the supplied Anomalib PatchCore on CUDA; export the existing CPU artifact contract.

PatchCore builds a normal-feature memory bank. It does not optimize weights.
No camera, controller, or robot API is imported. --check needs only Python.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/second-look-h100')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked(args):
    require(64 <= args.image_size <= 512 and args.image_size % 32 == 0, 'image_size must be a multiple of 32 in [64, 512]')
    require(math.isfinite(args.sampling_ratio) and 0 < args.sampling_ratio <= 1, 'Invalid sampling_ratio')
    require(1 <= args.max_images <= 256, 'max_images must be in [1, 256]')
    require(1 <= args.max_seconds <= 3600, 'max_seconds must be in [1, 3600]')
    require(1 <= args.num_neighbors <= 20, 'num_neighbors must be in [1, 20]')
    require(args.gpu >= 0, 'gpu must be a nonnegative physical device index')
    require(not (args.engineering_smoke and args.calibrate), 'Smoke cannot be calibrated')
    require(math.isfinite(args.uncertainty_fraction) and 0 <= args.uncertainty_fraction <= .5, 'Invalid uncertainty_fraction')
    source = args.source_root.resolve()
    module_path = source / 'secondlook/anomaly.py'
    require(module_path.is_file(), 'Missing frozen secondlook/anomaly.py')
    require(digest(module_path) == args.source_sha256, 'Frozen anomaly.py SHA-256 mismatch')
    require(not args.output.exists(), 'Output must not exist; never overwrite a previous run')
    sys.path.insert(0, str(source))
    anomaly = importlib.import_module('secondlook.anomaly')
    require(Path(anomaly.__file__).resolve() == module_path, 'Imported anomaly module differs from frozen source')
    dataset = anomaly.validate_manifest(args.manifest, engineering_smoke=args.engineering_smoke)
    train = [row for row in dataset['images'] if row['role'] == 'train']
    require(len(train) <= args.max_images, 'Manifest exceeds max_images')
    roi = anomaly._roi(args.roi)
    if args.calibrate:
        require({row['label'] for row in dataset['images'] if row['role'] == 'validation'} == {'normal', 'anomalous'}, 'Calibration requires normal and anomalous validation images')
    return anomaly, dataset, train, roi


@contextmanager
def gpu_lock(path):
    """Cooperating jobs retain one shared inode; do not delete the lock file."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('GPU ownership lock is held by another job') from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({'pid': os.getpid(), 'runner': str(Path(__file__).resolve())}) + '\n')
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_idle(gpu):
    command = ['nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits']
    active = subprocess.check_output(command, text=True, timeout=15).strip()
    require(not active, 'GPU has active compute processes; coordinate with their owner before starting')
    return subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader'], text=True, timeout=15).strip()


def train_cuda(args, checked_data):
    anomaly, dataset, train, roi = checked_data
    paths = [args.source_root, args.manifest, args.output, Path(__file__), *[Path(row['path']) for row in dataset['images']]]
    require(all(path.resolve().is_relative_to(ROOT) for path in paths), f'All source, data, script, and output paths must be under {ROOT}')
    # Set caches before any heavy import. Sponsor home directories stay untouched.
    for key, directory in {'XDG_CACHE_HOME': 'cache', 'HF_HOME': 'cache/huggingface', 'HF_HUB_CACHE': 'cache/huggingface/hub',
                           'HF_DATASETS_CACHE': 'cache/huggingface/datasets', 'TORCH_HOME': 'cache/torch',
                           'CUDA_CACHE_PATH': 'cache/cuda', 'TRITON_CACHE_DIR': 'cache/triton', 'TMPDIR': 'tmp'}.items():
        destination = ROOT / directory
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(destination)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    with gpu_lock(ROOT / '.gpu-job.lock'):
        inventory = gpu_idle(args.gpu)
        args.output.mkdir(parents=True, exist_ok=False)
        run = {'schema_version': 1, 'status': 'running', 'pid': os.getpid(), 'nvidia_smi': inventory,
               'runner_sha256': digest(__file__), 'source_sha256': args.source_sha256,
               'source_files': {str(p.relative_to(args.source_root)): digest(p) for p in sorted((args.source_root / 'secondlook').glob('*.py'))},
               'manifest_sha256': digest(args.manifest), 'dataset_sha256': dataset['dataset_sha256'],
               'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'seed': 17, 'training_kind': 'pretrained frozen feature extraction and k-center memory-bank fit; no optimizer',
               'test_inference_performed': False, 'split_limitations': dataset['split_limitations']}
        write(args.output / 'run.json', run)
        started = time.perf_counter()

        def event(stage, **fields):
            record = {'stage': stage, 'elapsed_seconds': time.perf_counter() - started, **fields}
            with (args.output / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)

        def timeout(signum, frame):
            raise TimeoutError(f'Run exceeded {args.max_seconds} seconds')

        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.max_seconds)
        try:
            import numpy as np
            import openvino as ov
            import torch
            require(torch.cuda.is_available(), 'CUDA is required; CPU fallback is forbidden')
            require(torch.cuda.device_count() == 1, 'Expected exactly one selected CUDA device')
            torch.set_num_threads(4)
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            properties = torch.cuda.get_device_properties(0)
            run.update(versions=anomaly._versions(), cuda_runtime=torch.version.cuda,
                       cuda_device={'index': 0, 'physical_index': args.gpu, 'name': properties.name,
                                    'total_memory_bytes': properties.total_memory, 'compute_capability': list(torch.cuda.get_device_capability(0))})
            run['installed_packages'] = json.loads(subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'], text=True, timeout=30))
            foreground = anomaly._foreground_config() if args.foreground_crop else None
            config = {'backbone': 'wide_resnet50_2', 'layers': ['layer2', 'layer3'], 'num_neighbors': args.num_neighbors,
                      'image_size': args.image_size, 'sampling_ratio': args.sampling_ratio, 'roi': roi,
                      'foreground_crop': foreground,
                      'preprocessing': 'RGB; fixed normalized ROI (floor left/top, ceil right/bottom); optional colored foreground bbox retaining original pixels; PIL bilinear square resize; float32 /255; ImageNet mean/std',
                      'seed': 17, 'torch_threads': 4}
            model = anomaly._model(config, pretrained=True).to('cuda:0')
            run['model_implementation'] = {'path': inspect.getfile(type(model)), 'sha256': digest(inspect.getfile(type(model)))}
            sampler = model.subsample_embedding.__func__.__globals__.get('KCenterGreedy')
            if sampler is not None:
                sampler_path = inspect.getfile(sampler)
                run['coreset_implementation'] = {'path': sampler_path, 'sha256': digest(sampler_path)}
            model.train()
            model.feature_extractor.eval()
            feature_batches = []

            def verify_features(module, inputs, outputs):
                values = list(outputs.values()) if isinstance(outputs, dict) else [outputs]
                require(values and all(torch.is_tensor(x) and x.is_cuda and torch.isfinite(x).all().item() for x in values), 'Feature extractor must return finite CUDA tensors')
                feature_batches.append([list(x.shape) for x in values])

            hook = model.feature_extractor.register_forward_hook(verify_features)
            torch.cuda.synchronize()
            fit_started = time.perf_counter()
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_start.record()
            with torch.inference_mode():
                for i, row in enumerate(train):
                    tensor = torch.from_numpy(anomaly._preprocess(row['path'], args.image_size, roi, foreground)).to('cuda:0')
                    model(tensor)
                    event('fit_image', index=i + 1, count=len(train), sha256=row['sha256'])
                embeddings = model.embedding_store
                require(len(embeddings) == len(train), 'Expected one stored embedding per training image')
                require(all(x.is_cuda and torch.isfinite(x).all().item() for x in embeddings), 'Stored embeddings must be finite CUDA tensors')
                run['cuda_embedding_shapes'] = [list(x.shape) for x in embeddings]
                model.subsample_embedding(sampling_ratio=args.sampling_ratio)
            gpu_end.record()
            torch.cuda.synchronize()
            hook.remove()
            require(len(model.memory_bank) >= args.num_neighbors and model.memory_bank.is_cuda and torch.isfinite(model.memory_bank).all().item(), 'Memory bank must be finite, CUDA-resident, and contain at least num_neighbors rows')
            run.update(cuda_fit_wall_seconds=time.perf_counter() - fit_started,
                       cuda_fit_event_ms=gpu_start.elapsed_time(gpu_end), cuda_feature_shapes=feature_batches,
                       cuda_memory_bank_shape=list(model.memory_bank.shape), cuda_memory_bank_device=str(model.memory_bank.device),
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            event('cuda_fit_complete', **{k: run[k] for k in ('cuda_fit_wall_seconds', 'cuda_fit_event_ms', 'cuda_memory_bank_shape', 'cuda_peak_allocated_bytes')})
            model.eval()
            example = torch.from_numpy(anomaly._preprocess(train[0]['path'], args.image_size, roi, foreground))
            with torch.inference_mode():
                gpu_prediction = model(example.to('cuda:0'))
                gpu_score = gpu_prediction.pred_score.detach().cpu()
                gpu_map = gpu_prediction.anomaly_map.detach().cpu()
            model.cpu()
            torch.save(model.state_dict(), args.output / 'model.pt')
            reloaded = anomaly._model(config, pretrained=False)
            reloaded.load_state_dict(torch.load(args.output / 'model.pt', map_location='cpu', weights_only=True))
            reloaded.eval()
            with torch.inference_mode():
                cpu_prediction = reloaded(example)
            for expected, actual in ((gpu_score, cpu_prediction.pred_score), (gpu_map, cpu_prediction.anomaly_map)):
                require(torch.isfinite(actual).all().item(), 'Reload returned nonfinite output')
                torch.testing.assert_close(actual, expected, atol=.1, rtol=.01)
            run['checkpoint_reload'] = {'passed': True, 'device': 'CPU', 'image_sha256': train[0]['sha256'],
                                        'score': float(cpu_prediction.pred_score.item()), 'cuda_score': float(gpu_score.item()),
                                        'atol': .1, 'rtol': .01, 'scope': 'one training-image engineering smoke; not held-out evaluation'}
            event('checkpoint_reload_complete')

            class Export(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, pixels):
                    prediction = self.inner(pixels)
                    return prediction.pred_score, prediction.anomaly_map

            converted = ov.convert_model(Export(reloaded).eval(), example_input=example,
                                         input=[(1, 3, args.image_size, args.image_size)])
            converted.output(0).get_tensor().set_names({'score'})
            converted.output(1).get_tensor().set_names({'anomaly_map'})
            ov.save_model(converted, args.output / 'model.xml', compress_to_fp16=False)
            metadata = {'schema_version': 1, 'model': 'Anomalib PatchCore', 'config': config,
                        'versions': anomaly._versions(), 'dataset': dataset,
                        'evidence_kind': 'engineering_smoke' if args.engineering_smoke else 'task_model_uncalibrated',
                        'threshold': None, 'calibration': None, 'evaluation': None, 'train_images': len(train),
                        'memory_bank_shape': list(reloaded.memory_bank.shape),
                        'fit_and_export_seconds': time.perf_counter() - fit_started,
                        'precision': 'FP32 export; device compilation precision reported by benchmark',
                        'files': {name: digest(args.output / name) for name in ('model.pt', 'model.xml', 'model.bin')}}
            metadata['model_id'] = hashlib.sha256(json.dumps({'config': config, 'files': metadata['files']}, sort_keys=True).encode()).hexdigest()
            metadata['artifact_id'] = anomaly._artifact_identity(metadata)
            write(args.output / 'metadata.json', metadata)
            detector = anomaly.AnomalyDetector(args.output, device='CPU', precision='f32')
            prediction = detector.infer(train[0]['path'])
            np.testing.assert_allclose(prediction['score'], cpu_prediction.pred_score.numpy().item(), atol=.1, rtol=.01)
            np.testing.assert_allclose(prediction['anomaly_map'], cpu_prediction.anomaly_map.numpy().squeeze(), atol=.1, rtol=.01)
            run['openvino_reload'] = {'passed': True, 'execution_devices': detector.execution_devices,
                                      'score': prediction['score'], 'atol': .1, 'rtol': .01,
                                      'scope': 'one training-image engineering smoke; not held-out evaluation'}
            event('openvino_export_verified')
            if args.calibrate:
                run['calibration'] = anomaly.calibrate(args.output, args.manifest, uncertainty_fraction=args.uncertainty_fraction)
                event('validation_calibration_complete')
            final = anomaly._metadata(args.output)
            require(final['evaluation'] is None, 'This runner must never perform final evaluation')
            require(digest(args.source_root / 'secondlook/anomaly.py') == args.source_sha256, 'Frozen source changed during execution')
            require(anomaly.validate_manifest(args.manifest, engineering_smoke=args.engineering_smoke)['dataset_sha256'] == dataset['dataset_sha256'], 'Dataset changed during execution')
            run.update(status='complete', artifact_id=final['artifact_id'], total_seconds=time.perf_counter() - started)
        except BaseException as exc:
            run.update(status='failed', error_type=type(exc).__name__, error=str(exc), total_seconds=time.perf_counter() - started)
            event('failed', error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            write(args.output / 'run.json', run)
    print(json.dumps({'status': run['status'], 'output': str(args.output), 'artifact_id': run.get('artifact_id')}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--sampling-ratio', type=float, default=.02)
    p.add_argument('--max-images', type=int, default=64)
    p.add_argument('--max-seconds', type=int, default=600)
    p.add_argument('--num-neighbors', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--roi', type=float, nargs=4)
    p.add_argument('--foreground-crop', action='store_true')
    p.add_argument('--engineering-smoke', action='store_true')
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--uncertainty-fraction', type=float, default=.05)
    p.add_argument('--check', action='store_true')
    return p


def main():
    args = parser().parse_args()
    data = checked(args)
    if args.check:
        print(json.dumps({'status': 'checked_only_no_cuda_execution', 'dataset_sha256': data[1]['dataset_sha256'], 'train_images': len(data[2]), 'split_limitations': data[1]['split_limitations']}))
    else:
        train_cuda(args, data)


if __name__ == '__main__':
    main()
