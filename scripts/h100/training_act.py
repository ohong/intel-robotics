#!/usr/bin/env python3
"""Bounded Studio ACT training from a frozen, real, single-skill recording snapshot."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

STUDIO_REVISION = 'c4ff730fb49f84e5102d01088d52cfff1ba62854'
OWN_ROOT = Path('/workspace/second-look-h100')
BACKBONE_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def snapshot(root):
    require(root.is_dir() and not root.is_symlink(), 'Dataset must be a real directory')
    result = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), f'Symlink forbidden in snapshot: {path}')
        if path.is_file():
            require(path.suffix not in {'.tmp', '.partial'}, f'Unfinished file: {path}')
            result[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    require(result, 'Empty snapshot')
    return result


def checked_path(value, name, directory=False):
    require(isinstance(value, str) and value, f'Provide {name}')
    path = Path(value)
    require(path.is_absolute(), f'{name} must be absolute')
    require(path.is_dir() if directory else path.is_file(), f'Missing {name}: {path}')
    return path.resolve()


def check_splits(manifest, mode):
    keys = ('train_episodes', 'validation_episodes', 'final_eval_episodes')
    splits = [manifest.get(key, []) for key in keys]
    for key, split in zip(keys, splits):
        require(isinstance(split, list) and all(type(x) is int and x >= 0 for x in split), f'Invalid {key}')
        require(len(set(split)) == len(split), f'Duplicate episode in {key}')
    require(splits[0] and splits[1], 'Need nonempty whole-episode train and validation splits')
    if mode not in ('smoke', 'software-qualification'):
        require(splits[2], 'Training requires a separate final_eval_episodes holdout')
    all_ids = [x for split in splits for x in split]
    require(len(all_ids) == len(set(all_ids)), 'Episode leakage across splits')
    episodes = [item['episode_index'] for item in manifest['episodes']]
    require(len(episodes) == len(set(episodes)) and set(episodes) == set(all_ids), 'Splits must cover every episode exactly')
    if mode in ('train', 'check'):
        require(manifest.get('split_scope') == 'grouped_evaluation', 'Training requires grouped_evaluation split scope')
        by_id = {episode['episode_index']: episode for episode in manifest['episodes']}
        for field in ('specimen_id', 'session_id'):
            groups = []
            for split in splits:
                values = {by_id[eid].get(field) for eid in split}
                require(all(isinstance(value, str) and value.strip() for value in values), f'Missing {field}')
                require(not any(values & prior for prior in groups), f'{field} leaks across episode splits')
                groups.append(values)


def check_capture(config, manifest):
    """Read capture provenance as data only; never instantiate runtime YAML classes."""
    capture_path = checked_path(config.get('capture_manifest'), 'capture_manifest')
    runtime_path = checked_path(config.get('capture_runtime'), 'capture_runtime')
    capture = read(capture_path)
    require(capture.get('real_episodes') == len(manifest['episodes']), 'Refresh capture evidence after real collection')
    require(digest(runtime_path) == capture['sources_sha256']['artifacts/recording/runtime.yaml'], 'Capture runtime changed after capture manifest')
    cameras = capture['environment']['cameras']
    expected = {'observation.images.' + camera['runtime_camera_key']: camera for camera in cameras}
    require(set(manifest['cameras']) == set(expected), 'Dataset camera keys differ from capture contract')
    for key, camera in expected.items():
        feature = manifest['features'][key]
        require(feature['shape'] == [camera['configured_height'], camera['configured_width'], 3], 'Dataset camera resolution changed')
        require(camera['color_mode'] == 'RGB', 'Capture contract must identify RGB frames')
    follower = [robot for robot in capture['environment']['robots'] if robot['role'] == 'follower']
    require(len(follower) == 1 and follower[0]['configured_unit'] == 'normalized', 'Need normalized follower capture contract')
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    require([name.removesuffix('.pos') for name in manifest['joint_names']] == joint_names, 'Unknown capture joint order')
    return {'capture_manifest_sha256': digest(capture_path), 'capture_runtime_sha256': digest(runtime_path),
            'follower_calibration_sha256': follower[0]['supplied_calibration_file_sha256'],
            'action_semantics': 'absolute normalized positions', 'joint_names': manifest['joint_names'],
            'body_range': [-100, 100], 'gripper_range': [0, 100],
            'timing_limit': capture['timing']['limits'], 'capture_manifest': capture}


def check(config, mode):
    require(config.get('studio_revision') == STUDIO_REVISION, 'Wrong Studio revision')
    manifest_path = checked_path(config.get('manifest'), 'manifest')
    manifest = read(manifest_path)
    require(manifest.get('status') == 'OFFLINE_DATA_VALIDATED', 'Recording validation must pass first')
    synthetic = mode == 'software-qualification'
    require(manifest.get('evidence_kind') == ('synthetic' if synthetic else 'real'),
            'Qualification requires synthetic data; real smoke/training requires real recordings')
    require(manifest.get('finalized') is True, 'Dataset owner must finalize recordings')
    require(isinstance(manifest.get('skill'), str) and manifest['skill'].strip(), 'Missing single-skill identity')
    require(config.get('skill') == manifest['skill'], 'Config skill must exactly match the frozen manifest')
    check_splits(manifest, mode)
    destinations = set()
    for episode in manifest['episodes']:
        require(episode.get('outcome') == ('synthetic_success' if synthetic else 'complete_success'), 'Wrong episode outcome label')
        require(synthetic or episode.get('outcome_evidence'), 'Every real episode needs observed success evidence')
        require(isinstance(episode.get('destination'), str) and episode['destination'].strip(), 'Missing physical destination label')
        destinations.add(episode['destination'])
    require(len(destinations) == 1, 'Use separate ACT runs for each physical destination')
    capture = None if synthetic else check_capture(config, manifest)
    root = checked_path(config.get('dataset_root'), 'dataset_root', directory=True)
    require(snapshot(root) == manifest['files'], 'Dataset files differ from the validated frozen snapshot')
    # The source machine path in the manifest remains provenance; relocated content is hash-checked.
    stats_path = manifest_path.parent / 'train_stats.json'
    require(digest(stats_path) == manifest['train_stats_sha256'], 'Training statistics changed after validation')
    require(manifest.get('normalization', '').startswith('train episodes only'), 'Train-only normalization evidence required')
    stats = read(stats_path)
    for key in ('observation.state', 'action'):
        values = stats[key]
        for stat in ('mean', 'std', 'min', 'max', 'q01', 'q99'):
            require(isinstance(values.get(stat), list) and all(type(v) in (int, float) and math.isfinite(v) for v in values[stat]), f'Invalid {key}/{stat}')
        require(len(values['mean']) == len(manifest['joint_names']), f'{key} dimensions differ')
        require(all(v >= 0 for v in values['std']), 'Negative standard deviation')
    train_frames = sum(ep['frames'] for ep in manifest['episodes'] if ep['episode_index'] in manifest['train_episodes'])
    require(stats['action']['count'] == [train_frames], 'Statistics frame count does not match training split')
    for key in ('batch_size', 'max_steps', 'max_seconds', 'validation_every', 'checkpoint_every'):
        require(type(config.get(key)) is int and config[key] > 0, f'Invalid {key}')
    require(config['batch_size'] <= train_frames, 'Batch size drops every training frame')
    require(config['validation_every'] <= config['max_steps'], 'Validation must run within the step budget')
    require(type(config.get('seed')) is int and 0 <= config['seed'] < 2**32, 'Invalid seed')
    require(type(config.get('num_workers')) is int and 0 <= config['num_workers'] <= 8, 'Invalid worker count')
    require(config.get('precision') in ('32-true', 'bf16-mixed'), 'Unsupported precision')
    options = config.get('policy', {})
    require(options.get('pretrained_backbone_weights') is None, 'Offline route requires explicit local backbone support before pretrained weights')
    require(options.get('compile_model') is False, 'Compilation is disabled for bounded initial runs')
    require(options.get('n_action_steps') == options.get('chunk_size') and type(options.get('chunk_size')) is int and options['chunk_size'] > 0,
            'Keep action steps equal to positive chunk size for this export contract')
    require(not any(k in options for k in ('dataset_stats', 'pretrained_name_or_path')), 'Dataset statistics and task checkpoint cannot override manifest')
    output = Path(config.get('output_dir') or '')
    require(output.is_absolute() and output.resolve().is_relative_to(OWN_ROOT), 'Output must be under owned /workspace/second-look-h100')
    require(not output.resolve().is_relative_to(root), 'Output cannot be inside the snapshot')
    return {'manifest': manifest, 'manifest_path': manifest_path, 'root': root, 'stats_path': stats_path,
            'stats': stats, 'output': output.resolve(), 'capture': capture}


def identity(config, checked):
    # Budgets may grow on resume, while data, architecture, and optimizer settings stay fixed.
    fixed = {key: value for key, value in config.items() if key not in (
        'max_steps', 'max_seconds', 'output_dir', 'manifest', 'dataset_root', 'smoke_result', 'intel_parity_result')}
    fixed['manifest_sha256'] = digest(checked['manifest_path'])
    fixed['launcher_sha256'] = digest(__file__)
    fixed['capture'] = checked.get('capture')
    return hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()


def check_smoke_gate(config, checked):
    smoke_path = checked_path(config.get('smoke_result'), 'smoke_result')
    parity_path = checked_path(config.get('intel_parity_result'), 'intel_parity_result')
    smoke, parity = read(smoke_path), read(parity_path)
    run_id = identity(config, checked)
    require(smoke.get('mode') == 'smoke' and smoke.get('evidence_kind') == 'real', 'Need a real one-step smoke before budgeted training')
    require(smoke.get('status') == 'EXPORTED_FOR_INTEL_VERIFICATION', 'Real smoke must reload and export successfully')
    require(smoke.get('run_identity') == run_id == parity.get('run_identity'), 'Smoke/parity must match this frozen data and model configuration')
    require(parity.get('evidence_kind') == 'real' and parity.get('status') == 'PARITY_PASSED', 'Early real checkpoint must pass native/OpenVINO Intel parity')
    return {'smoke_result_sha256': digest(smoke_path), 'intel_parity_result_sha256': digest(parity_path)}


def check_resume_packages(previous_versions, current_versions, previous_freeze, current_freeze):
    require(previous_versions == current_versions, 'Package versions changed since the resumable run')
    require(sorted(previous_freeze.splitlines()) == sorted(current_freeze.splitlines()),
            'Installed package freeze changed since the resumable run')


def final_eval_status(manifest):
    if manifest['evidence_kind'] == 'synthetic':
        return 'NOT_APPLICABLE_SYNTHETIC'
    return 'UNTOUCHED' if manifest.get('final_eval_episodes') else 'NOT_ALLOCATED'


def run(config, checked, mode, resume):
    import fcntl
    one_step = mode in ('smoke', 'software-qualification')
    smoke_gate = check_smoke_gate(config, checked) if mode == 'train' else None
    require(platform.system() == 'Linux' and OWN_ROOT.is_dir(), 'Run on the prepared Linux GPU workspace')
    for name in ('dataset_root', 'manifest', 'studio_library', 'capture_manifest', 'capture_runtime',
                 'smoke_result', 'intel_parity_result'):
        if config.get(name):
            require(Path(config[name]).resolve().is_relative_to(OWN_ROOT), f'{name} must be staged under the owned workspace')
    lock = (OWN_ROOT / '.gpu-job.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError('Another owned GPU job holds the lock') from exc
    occupancy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
                               capture_output=True, text=True, check=True)
    require(not occupancy.stdout.strip(), 'GPU already has a compute process; coordinate resource ownership first')
    for variable, directory in {'HF_HOME': 'cache/huggingface', 'HF_HUB_CACHE': 'cache/huggingface/hub',
                                'HF_DATASETS_CACHE': 'cache/huggingface/datasets', 'TORCH_HOME': 'cache/torch',
                                'CUDA_CACHE_PATH': 'cache/cuda', 'TRITON_CACHE_DIR': 'cache/triton',
                                'XDG_CACHE_HOME': 'cache', 'TMPDIR': 'tmp'}.items():
        location = OWN_ROOT / directory
        location.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(location)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.dont_write_bytecode = True
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    import_start = time.monotonic()
    import numpy as np
    import torch
    from lightning.pytorch import Callback, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from physicalai.data.lerobot import LeRobotDataModule
    from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter
    from physicalai.policies import ACT
    from physicalai.train import Trainer

    require(platform.system() == 'Linux' and torch.cuda.is_available(), 'Actual Linux CUDA runtime is required')
    import_seconds = time.monotonic() - import_start
    library = checked_path(config.get('studio_library'), 'studio_library', directory=True)
    expected_sources = read(Path(__file__).with_name('act-studio-sources.json'))
    for relative, expected in expected_sources.items():
        require(digest(library / relative) == expected, f'Pinned Studio source changed: {relative}')
    require(Path(inspect.getfile(ACT)).resolve().is_relative_to(library), 'Imported ACT does not come from the pinned Studio library')
    backbone = OWN_ROOT / 'cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
    require(digest(backbone) == BACKBONE_SHA256, 'Missing or altered official ImageNet ResNet18 backbone')
    require(set(config['policy']) <= set(inspect.signature(ACT).parameters), 'Unknown ACT constructor option')
    output = checked['output']
    run_id = identity(config, checked)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'physicalai', 'lerobot', 'lightning', 'openvino')}
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    if resume:
        require(mode == 'train', 'Only training can resume')
        previous_run = read(output / 'run.json')
        require(previous_run['run_identity'] == run_id, 'Resume config/data/source identity mismatch')
        check_resume_packages(previous_run['versions'], versions, (output / 'packages.txt').read_text(), freeze.stdout)
        resume = checked_path(resume, 'resume checkpoint')
        require(resume.is_relative_to(output), 'Resume only an owned checkpoint from this run')
    else:
        require(not output.exists(), 'Use a fresh output directory, or resume explicitly')
        output.mkdir(parents=True)
    seed_everything(config['seed'], workers=True)
    torch.set_num_threads(4)
    stats = {key: {name: np.asarray(value) for name, value in values.items()} for key, values in checked['stats'].items()}
    manifest = checked['manifest']

    def load(ids):
        raw = LeRobotDataset(repo_id='snapshot', root=checked['root'], episodes=ids,
                             download_videos=False, video_backend='pyav')
        raw.meta.stats = stats
        return raw

    dm = LeRobotDataModule(dataset=load(manifest['train_episodes']), train_batch_size=config['batch_size'],
                           val_batch_size=config['batch_size'], num_workers=config['num_workers'])
    dm.val_eval_dataset = _LeRobotDatasetAdapter.from_lerobot(load(manifest['validation_episodes']))
    start = time.monotonic()
    record = {'run_identity': run_id, 'mode': mode, 'config': config, 'manifest': manifest,
              'manifest_sha256': digest(checked['manifest_path']), 'train_stats_sha256': digest(checked['stats_path']),
              'host': platform.node(), 'python': sys.version, 'cuda': torch.version.cuda,
              'device': torch.cuda.get_device_name(0), 'seed': config['seed'],
              'initialization': 'New ACT task policy with official ImageNet ResNet18 backbone; no task checkpoint fine-tuning claim',
              'backbone_sha256': BACKBONE_SHA256, 'cold_import_seconds': import_seconds,
              'capture': checked.get('capture'),
              'real_smoke_gate': smoke_gate,
              'versions': versions,
              'studio_source_sha256': expected_sources, 'launcher_sha256': digest(__file__),
              'loaded_sources': {obj.__name__: {'path': inspect.getfile(obj), 'sha256': digest(inspect.getfile(obj))}
                                 for obj in (LeRobotDataset, LeRobotDataModule, ACT, Trainer)},
              'resume_checkpoint': str(resume) if resume else None,
              'resume_sha256': digest(resume) if resume else None,
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (output / 'packages.txt').write_text(freeze.stdout)
    record['packages_sha256'] = digest(output / 'packages.txt')
    dump(output / 'run.json', record)

    class Evidence(Callback):
        gradients = None
        last_step = 0

        def on_fit_start(self, trainer, model):
            if not resume:
                # Initialize only the backbone. Keep constructor weights=None in checkpoints,
                # so Intel reloads never download ImageNet weights over the saved learned state.
                weights = torch.load(backbone, map_location='cpu', weights_only=True)
                weights = {name: value for name, value in weights.items() if not name.startswith('fc.')}
                model.model._model.backbone.load_state_dict(weights, strict=True)

        def on_after_backward(self, trainer, model):
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            require(gradients and all(torch.isfinite(g).all().item() for g in gradients), 'Missing/nonfinite CUDA gradients')
            if self.gradients is None:
                norm = sum(g.detach().float().square().sum().item() for g in gradients) ** 0.5
                require(norm > 0, 'Zero CUDA gradient norm')
                self.gradients = {'count': len(gradients), 'l2_norm': norm}

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            require(torch.isfinite(outputs['loss']).all().item(), 'Nonfinite training loss')
            self.last_step = trainer.global_step
            if trainer.global_step % 10 == 0 or one_step:
                dump(output / 'progress.json', {'step': trainer.global_step, 'elapsed_seconds': time.monotonic() - start,
                                               'loss': float(outputs['loss'].detach().cpu())})

        def on_save_checkpoint(self, trainer, model, checkpoint):
            checkpoint['second_look_run_identity'] = run_id

        def on_load_checkpoint(self, trainer, model, checkpoint):
            require(checkpoint.get('second_look_run_identity') == run_id, 'Checkpoint identity mismatch')

    evidence = Evidence()
    best = ModelCheckpoint(dirpath=output / 'checkpoints', filename='selected-{step:06d}', monitor='val/loss',
                           mode='min', save_top_k=1, save_on_train_epoch_end=False)
    recovery = ModelCheckpoint(dirpath=output / 'recovery', filename='step-{step:06d}', every_n_train_steps=config['checkpoint_every'],
                               save_top_k=1, save_last=True, save_on_train_epoch_end=False)
    policy = ACT(**config['policy'])
    options = dict(accelerator='gpu', devices=1, precision=config['precision'], max_epochs=-1,
                   max_steps=1 if one_step else config['max_steps'], max_time={'seconds': config['max_seconds']},
                   enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                   deterministic=True, cudnn_benchmark=False, log_every_n_steps=1 if one_step else 10,
                   logger=CSVLogger(str(output), name='metrics'), default_root_dir=str(output),
                   callbacks=[evidence, best, recovery], check_val_every_n_epoch=None,
                   val_check_interval=1 if one_step else config['validation_every'])
    if one_step:
        options.update(limit_train_batches=1, limit_val_batches=1)
    trainer = Trainer(**options)
    torch.cuda.reset_peak_memory_stats()
    fit_start = time.monotonic()
    trainer.fit(policy, datamodule=dm, ckpt_path=str(resume) if resume else None)
    fit_seconds = time.monotonic() - fit_start
    require(evidence.gradients and trainer.global_step > 0, 'No completed CUDA update')
    last = output / 'last.ckpt'
    trainer.save_checkpoint(last)
    require(best.best_model_path and math.isfinite(float(best.best_model_score)), 'No validated checkpoint selected; last.ckpt is recovery only')
    selected_path = Path(best.best_model_path)
    selected = ACT.load_from_checkpoint(selected_path, map_location='cpu').eval()
    # Verify exact save/reload output parity for the just-trained state, independently of selection.
    restored_last = ACT.load_from_checkpoint(last, map_location='cpu').eval()
    batch = next(iter(dm.val_dataloader()))
    policy.cpu().eval()
    with torch.no_grad():
        expected = policy.predict_action_chunk(batch)
        actual = restored_last.predict_action_chunk(batch)
    require(torch.isfinite(actual).all().item(), 'Nonfinite reloaded prediction')
    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
    result = {'status': 'CUDA_SMOKE_RELOAD_PASSED' if mode == 'smoke' else 'VALIDATION_SELECTED', 'mode': mode,
              'evidence_kind': manifest['evidence_kind'], 'run_identity': run_id, 'global_step': trainer.global_step,
              'fit_seconds': fit_seconds, 'gradient_check': evidence.gradients,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'selection_metric': 'val/loss', 'selection_score': float(best.best_model_score),
              'selected_checkpoint': str(selected_path), 'selected_sha256': digest(selected_path),
              'reload_max_abs_error': float((expected - actual).abs().max()),
              'final_eval_status': final_eval_status(manifest), 'robot_execution': 'NOT TESTED',
              'limits': ['Offline action loss is not physical task success.', 'ACT does not use task captions to choose the destination.',
                         'Each skill needs a separate frozen dataset and run.', 'Wall budget bounds fit cooperatively; export and verification are separate.']}
    dump(output / 'result.json', result)
    if mode in ('train', 'smoke', 'software-qualification'):
        export_start = time.monotonic()
        selected.export(output / 'export-torch', backend='torch')
        export_reloaded = ACT.load_from_checkpoint(output / 'export-torch' / 'act.pt', map_location='cpu').eval()
        with torch.no_grad():
            reference = selected.predict_action_chunk(batch)
            torch.testing.assert_close(reference, export_reloaded.predict_action_chunk(batch), atol=1e-5, rtol=1e-5)
        require(torch.isfinite(reference).all().item(), 'Selected checkpoint produced nonfinite actions')
        result['validation_prediction_min_per_joint'] = reference.amin(dim=(0, 1)).tolist()
        result['validation_prediction_max_per_joint'] = reference.amax(dim=(0, 1)).tolist()
        # Real validation frames only. Export handles core normalization, with letterbox resizing
        # recorded in the Studio manifest. This fixture compares its complete Intel pipeline.
        flat = batch.to_dict()
        if 'images' not in flat:
            camera_keys = [key for key in flat if key.startswith('images.') and 'is_pad' not in key]
            if len(camera_keys) == 1:
                flat['images'] = flat[camera_keys[0]]
        sample = {feature.name: flat[feature.name][:1].cpu() for feature in selected.inputs_schema}
        processed_sample = selected._preprocessor(sample)
        export_args = selected._get_export_extra_args
        selected._get_export_extra_args = lambda backend: replace(export_args(backend), compress_to_fp16=False) if str(backend) == 'openvino' else export_args(backend)
        selected.export(output / 'export-openvino', backend='openvino', input_sample=processed_sample)
        with torch.no_grad():
            native = selected.model.predict_action_chunk(processed_sample)
        np.savez_compressed(output / 'export-replay.npz', **{name: value.numpy() for name, value in sample.items()}, expected_action=native.numpy())
        result.update(status='EXPORTED_FOR_INTEL_VERIFICATION', export_torch=str(output / 'export-torch'),
                      export_openvino=str(output / 'export-openvino'), export_seconds=time.monotonic() - export_start,
                      openvino_parity='NOT TESTED', openvino_weight_precision='FP32', final_eval_status=final_eval_status(manifest))
        if mode == 'software-qualification':
            result['status'] = 'UNSAFE_NOT_TASK_TRAINED'
        result['artifact_files'] = {name: snapshot(output / name) for name in ('export-torch', 'export-openvino')}
        result['replay_sha256'] = digest(output / 'export-replay.npz')
        dump(output / 'result.json', result)
    if snapshot(checked['root']) != manifest['files']:
        result['status'] = 'INVALID_DATA_CHANGED'
        dump(output / 'result.json', result)
        raise ValueError('Dataset changed during training; artifacts are invalid')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--mode', choices=('check', 'smoke', 'train', 'software-qualification'), default='check')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = read(args.config)
    checked = check(config, args.mode)
    if args.mode == 'check':
        print(json.dumps({'status': 'FROZEN_REAL_DATA_CHECKED', 'skill': config['skill'], 'robot_execution': 'NOT TESTED'}))
    else:
        run(config, checked, args.mode, args.resume)


if __name__ == '__main__':
    main()
