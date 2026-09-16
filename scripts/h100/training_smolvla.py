#!/usr/bin/env python3
"""Bounded native Studio SmolVLA fine tuning; every output remains experimental."""
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
import subprocess
import sys
import time

# Reuse the already exercised frozen-recording and GPU-run primitives.
from training_act import (OWN_ROOT, STUDIO_REVISION, check_capture, checked_path,
                          check_resume_packages, check_splits, digest, dump, read,
                          require, snapshot)

BASE_REVISION = 'c83c3163b8ca9b7e67c509fffd9121e66cb96205'
BASE_SHA256 = '7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb'
INACTIVE_KEYS = {f'_model.target_time_mlp_{part}.{kind}' for part in ('in', 'out') for kind in ('weight', 'bias')}
SOURCE_FILES = [f'src/physicalai/policies/smolvla/{name}.py' for name in
                ('policy', 'model', 'preprocessor', 'pretrained_utils', 'config')]
SOURCE_FILES += ['src/physicalai/data/lerobot/dataset.py', 'src/physicalai/data/lerobot/datamodule.py',
                 'src/physicalai/train/utils.py', 'src/physicalai/policies/base/policy.py',
                 'src/physicalai/policies/mixins/snapflow.py']


def verify_weight_keys(weight_keys, model_keys, snapflow_enabled):
    require(not snapflow_enabled, 'Legacy base requires SnapFlow disabled')
    missing, unexpected = model_keys - weight_keys, weight_keys - model_keys
    require(not unexpected and missing <= INACTIVE_KEYS,
            f'Active base weights mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    return sorted(missing)


def require_prediction_devices(model_device, observation_device):
    # Equal numeric seeds do not align CPU and CUDA normal-sampling streams.
    require(str(model_device) == str(observation_device) == 'cuda:0',
            'Reload comparisons require policy and observation on CUDA device zero')


def artifact_inventory(root):
    """Permit model aliases only when their resolved files stay in the owned workspace."""
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            require(not path.is_symlink() or path.resolve().is_relative_to(OWN_ROOT), 'Model alias escapes owned workspace')
            result[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    require(result, 'Empty model inventory')
    return result


def annotation_text(instruction, row, threshold):
    require(isinstance(instruction, str) and instruction.strip(), 'Need generic task instruction')
    require(row.get('eligible') is True, 'Ineligible detection cannot condition an action start')
    score = row.get('score')
    require(type(score) in (float, int) and math.isfinite(score), 'Eligible detection needs finite score')
    decision = str(row.get('decision', '')).lower()
    require(decision in ('normal', 'anomalous'), 'Invalid detector decision')
    require(decision == ('anomalous' if score >= threshold else 'normal'), 'Decision disagrees with frozen threshold')
    return (f"{instruction}\nInspection: {decision}; score={score:.6g}; "
            f"threshold={threshold:.6g}.\n")


def check_annotations(config, manifest, *, for_fit=False):
    conditioning = config['conditioning']
    path = checked_path(conditioning.get('annotation_manifest'), 'annotation_manifest')
    require(digest(path) == conditioning.get('annotation_sha256'), 'Annotation manifest checksum differs')
    document = read(path)
    require(document.get('schema_version') == 1, 'Unknown annotation schema')
    require(document.get('rgb_hash_format') == 'uint8_HWC_RGB_contiguous', 'Unknown annotated image hash format')
    require(document.get('camera_key') in camera_keys(manifest), 'Annotation camera is absent from dataset')
    require(isinstance(document.get('artifact_id'), str) and document['artifact_id'], 'Missing frozen detector artifact identity')
    require(document['artifact_id'] == conditioning.get('artifact_id'), 'Wrong detector artifact')
    threshold = document.get('threshold')
    require(type(threshold) in (int, float) and math.isfinite(threshold), 'Invalid annotation threshold')
    require(threshold == conditioning.get('threshold'), 'Frozen detector threshold changed')
    margin = document.get('uncertainty_margin', 0.0)
    require(type(margin) in (int, float) and math.isfinite(margin) and margin >= 0, 'Invalid detector uncertainty margin')
    band = document.get('uncertainty_band')
    if band is not None:
        require(isinstance(band, list) and len(band) == 2 and all(type(v) in (float, int) and math.isfinite(v) for v in band)
                and band[0] <= threshold <= band[1], 'Invalid detector uncertainty band')
    snapshot_hash = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(document.get('dataset_sha256') == snapshot_hash or
            (conditioning.get('snapshot_receipt_sha256') and document.get('snapshot_receipt_sha256') == conditioning['snapshot_receipt_sha256']),
            'Detector annotations belong to a different recording snapshot')
    by_episode = {ep['episode_index']: ep for ep in manifest['episodes']}
    indexes, frames = set(), set()
    for row in document['rows']:
        require(type(row.get('index')) is int and row['index'] >= 0 and row['index'] not in indexes, 'Duplicate/invalid annotation index')
        eid, frame = row.get('episode_index'), row.get('frame_index')
        require(type(eid) is int and eid in by_episode and type(frame) is int and 0 <= frame < by_episode[eid]['frames'], 'Annotation episode/frame out of bounds')
        require((eid, frame) not in frames, 'Duplicate annotation frame')
        indexes.add(row['index'])
        frames.add((eid, frame))
        require(type(row.get('eligible')) is bool, 'Annotation must explicitly mark eligibility')
        require(isinstance(row.get('image_sha256'), str) and len(row['image_sha256']) == 64 and all(c in '0123456789abcdef' for c in row['image_sha256']), 'Missing image byte identity')
        if row['eligible']:
            score = row.get('score')
            require(type(score) in (int, float) and math.isfinite(score), 'Eligible detection needs finite score')
            require(not (margin > 0 and threshold - margin <= score <= threshold + margin), 'Eligible score lies in detector uncertainty band')
            require(band is None or not band[0] <= score <= band[1], 'Eligible score lies in detector uncertainty band')
            text = annotation_text(conditioning.get('instruction'), row, threshold)
            require('text' not in row or row['text'] == text, 'Annotation text differs from deployment conditioning format')
        else:
            require(row.get('reason'), 'Ineligible start needs a reason')
    require(document['rows'], 'Empty annotations')
    if for_fit:
        require(document.get('conditioning_verified') is True and document.get('target_association') == 'VERIFIED',
                'Detector pixels are not verified as the manipulated object; fitting is blocked')
        used = set(manifest['train_episodes'] + manifest['validation_episodes'])
        for row in document['rows']:
            if row['episode_index'] in used and row['eligible']:
                require(row.get('conditioning_verified', document['conditioning_verified']) is True and
                        row.get('target_association', document['target_association']) == 'VERIFIED',
                        'An eligible start lacks verified detector-to-manipulated-object association')
    return document


def check_conditioning(config, manifest, *, for_fit=True):
    conditioning = config.get('conditioning', {})
    require(conditioning.get('mode') in ('recorded_task', 'episode_task', 'anomalib_rows'), 'Choose an explicit supported conditioning mode')
    require(conditioning.get('claim') == 'experimental_imitation', 'This runner supports experimental imitation claims only')
    if conditioning['mode'] == 'episode_task':
        prompts = conditioning.get('episodes', {})
        require(set(prompts) == {str(ep['episode_index']) for ep in manifest['episodes']}, 'Condition every episode exactly once')
        text_destinations = {}
        for episode in manifest['episodes']:
            entry = prompts[str(episode['episode_index'])]
            require(isinstance(entry.get('text'), str) and entry['text'].strip(), 'Missing task text')
            require(isinstance(entry.get('evidence'), str) and entry['evidence'].strip(), 'Task conditioning needs source evidence')
            previous = text_destinations.setdefault(entry['text'], episode['destination'])
            require(previous == episode['destination'], 'Identical task text labels conflicting destinations')
    if conditioning['mode'] == 'anomalib_rows':
        check_annotations(config, manifest, for_fit=for_fit)
    return conditioning


def camera_keys(manifest):
    keys = manifest.get('cameras')
    if keys is None:
        keys = list(manifest.get('camera_identity_mapping', {}))
    return keys


def observed_destination(episode):
    return episode.get('destination', episode.get('observed_destination'))


def check_pipeline_probe(config, manifest):
    require(config.get('training_scope') == 'real_data_pipeline_probe', 'Missing explicit probe scope')
    require(type(config.get('max_steps')) is int and 1 <= config['max_steps'] <= 20, 'Pipeline probe is limited to twenty total optimizer updates')
    require(config.get('conditioning', {}).get('mode') == 'recorded_task', 'Pipeline probe preserves original recorded captions only')
    require(manifest.get('task_ready') is False and manifest.get('controller_ready') is False, 'Probe manifest must remain nondeployable')
    require(manifest.get('split_scope') == 'loader_smoke' and manifest.get('generalization_claim_permitted') is False,
            'Probe cannot claim grouped independence or generalization')
    for episode in manifest['episodes']:
        require(episode.get('outcome') == 'visual_transfer_observed' and episode.get('observed_transfer') is True,
                'Probe needs observed transfer evidence without invented task success')
        require(isinstance(episode.get('visual_evidence'), str) and episode['visual_evidence'], 'Missing visual transfer evidence')
        require(isinstance(episode.get('observed_destination'), str) and episode['observed_destination'], 'Missing observed destination')
        require(isinstance(episode.get('task_caption'), str) and episode['task_caption'], 'Missing original recorded caption')
    robot = manifest.get('robot_contract', {})
    require(robot.get('joint_order') == manifest['joint_names'], 'Frozen robot joint order mismatch')
    require(robot.get('state_semantics') == 'measured follower joint positions', 'Unknown recorded state semantics')
    require(robot.get('action_semantics') == 'absolute joint position targets; not velocities or deltas', 'Unknown recorded action semantics')
    require(robot.get('body_units') == 'normalized calibrated position [-100,100]' and
            robot.get('gripper_units') == 'normalized calibrated position [0,100]', 'Unknown recorded joint units')
    require(robot.get('calibration_id_provenance') and robot.get('action_ack_limit'), 'Missing calibration or acknowledgment evidence limits')
    require(manifest.get('camera_identity_provenance') and manifest.get('timing_limits'), 'Missing camera/timing provenance')
    cameras = manifest['camera_identity_mapping']
    require(cameras and set(cameras) == {key for key in manifest['features'] if key.startswith('observation.images.')}, 'Frozen camera map mismatch')
    for key, camera in cameras.items():
        require(camera.get('serial') and camera['recorded_rgb_shape_hwc'] == manifest['features'][key]['shape'], 'Camera identity or dimensions mismatch')
    return {'source': 'frozen exploratory manifest; no live capture access', 'robot_contract': robot,
            'camera_identity_mapping': cameras, 'camera_identity_provenance': manifest['camera_identity_provenance'],
            'timing_limits': manifest['timing_limits'], 'physical_task_success': 'NOT VALIDATED'}


def check_normalization(manifest, training_frames):
    normalization = manifest.get('normalization', '')
    if isinstance(normalization, dict):
        require(normalization.get('training_episodes_only') == manifest['train_episodes'], 'Training normalization episode list differs')
        require(normalization.get('training_rows') == training_frames, 'Training normalization row count differs')
        require(normalization.get('excluded_validation_episodes') == manifest['validation_episodes'], 'Validation exclusion differs')
        require(normalization.get('excluded_final_episodes') == manifest.get('final_eval_episodes', []), 'Final holdout exclusion differs')
    else:
        require(isinstance(normalization, str) and normalization.startswith('train episodes only'), 'Need training-only normalization')


def check(config, *, purpose='fit'):
    require(config.get('studio_revision') == STUDIO_REVISION, 'Wrong Studio revision')
    require(config.get('base_revision') == BASE_REVISION, 'Wrong base revision')
    manifest_path = checked_path(config.get('manifest'), 'manifest')
    manifest = read(manifest_path)
    require(manifest.get('status') == 'OFFLINE_DATA_VALIDATED' and manifest.get('finalized') is True,
            'Require a closed, validated recording snapshot')
    require(manifest.get('evidence_kind') == 'real', 'This fine-tuning runner requires real recordings')
    # Pilot splits can share sessions, but never rows. Such results carry no generalization claim.
    check_splits(manifest, 'train' if manifest.get('split_scope') == 'grouped_evaluation' else 'smoke')
    require(manifest.get('split_scope') in ('grouped_evaluation', 'loader_smoke'), 'Unknown split scope')
    probe = config.get('training_scope') == 'real_data_pipeline_probe'
    frozen_capture = check_pipeline_probe(config, manifest) if probe else None
    if probe:
        require(digest(manifest_path) == config.get('manifest_sha256'), 'Probe manifest checksum differs')
    if purpose == 'fit' and not probe:
        for ep in manifest['episodes']:
            accepted_outcomes = {'complete_success'}
            if config.get('training_scope') == 'developmental_observed_transfer':
                accepted_outcomes.add('observed_transfer')
            require(ep.get('outcome') in accepted_outcomes and ep.get('outcome_evidence'), 'Need observed demonstration outcome evidence')
            require(isinstance(ep.get('destination'), str) and ep['destination'].strip(), 'Missing destination evidence')
    root = checked_path(config.get('dataset_root'), 'dataset_root', directory=True)
    require(snapshot(root) == manifest['files'], 'Frozen dataset content changed')
    stats_path = manifest_path.parent / 'train_stats.json'
    require(digest(stats_path) == manifest['train_stats_sha256'], 'Training statistics changed')
    stats = read(stats_path)
    frames = sum(ep['frames'] for ep in manifest['episodes'] if ep['episode_index'] in manifest['train_episodes'])
    check_normalization(manifest, frames)
    for feature in ('observation.state', 'action'):
        for name in ('mean', 'std', 'min', 'max', 'q01', 'q99'):
            vector = stats[feature][name]
            require(len(vector) == len(manifest['joint_names']) and all(type(x) in (float, int) and math.isfinite(x) for x in vector),
                    f'Invalid training statistic {feature}/{name}')
        require(all(x >= 0 for x in stats[feature]['std']), 'Negative standard deviation')
        require(stats[feature]['count'] == [frames], 'Statistics include the wrong frame count')
    cameras = {key.removeprefix('observation.images.') for key in camera_keys(manifest)}
    mapping = config.get('image_key_reorder_map')
    require(isinstance(mapping, dict) and cameras and set(mapping) == cameras, 'Map every actual camera suffix')
    require(all(type(slot) is int and 0 <= slot < 3 for slot in mapping.values()) and len(set(mapping.values())) == len(mapping),
            'Camera slots must be distinct integers 0..2')
    require(isinstance(config.get('camera_mapping_evidence'), str) and config['camera_mapping_evidence'].strip(), 'Record camera mapping provenance')
    base = checked_path(config.get('base_dir'), 'base_dir', directory=True)
    backbone = checked_path(config.get('backbone_dir'), 'backbone_dir', directory=True)
    require(digest(base / 'model.safetensors') == BASE_SHA256, 'Base weights differ from pinned upstream digest')
    require(artifact_inventory(base) == config.get('base_files'), 'Require exact base file size/SHA-256 inventory')
    require(artifact_inventory(backbone) == config.get('backbone_files'), 'Require exact backbone/tokenizer file inventory')
    base_config = read(base / 'config.json')
    require(Path(base_config.get('vlm_model_name', '')).resolve() == backbone, 'Base config must select local pinned backbone')
    require(base_config.get('load_vlm_weights') is False, 'Base checkpoint already includes VLM weights; disable duplicate download')
    for key in ('max_steps', 'max_seconds', 'batch_size', 'checkpoint_every', 'validation_every', 'tokenizer_max_length', 'scheduler_decay_steps'):
        require(type(config.get(key)) is int and config[key] > 0, f'{key} must be a positive integer')
    require(config['batch_size'] <= frames, 'Batch size drops every training frame')
    require(config['validation_every'] <= config['max_steps'], 'Validation must fit within the step budget')
    require(type(config.get('warmup_steps')) is int and 0 <= config['warmup_steps'] < config['scheduler_decay_steps'], 'Invalid warmup_steps')
    require(type(config.get('num_workers')) is int and 0 <= config['num_workers'] <= 8, 'Invalid worker count')
    require(type(config.get('seed')) is int and 0 <= config['seed'] < 2**32, 'Invalid seed')
    require(type(config.get('learning_rate')) in (int, float) and math.isfinite(config['learning_rate']) and config['learning_rate'] > 0, 'Invalid learning rate')
    require(config.get('precision') in ('32-true', 'bf16-mixed'), 'Unsupported precision')
    output = Path(config.get('output_dir', ''))
    require(output.is_absolute() and output.resolve().is_relative_to(OWN_ROOT), 'Output must remain in owned workspace')
    require(not output.resolve().is_relative_to(root), 'Output cannot alter the recording snapshot')
    return {'root': root, 'manifest': manifest, 'manifest_path': manifest_path, 'stats': stats,
            'stats_path': stats_path, 'base': base, 'backbone': backbone, 'output': output.resolve(),
            'capture': frozen_capture if probe else (check_capture(config, manifest) if purpose == 'fit' else None),
            'conditioning': check_conditioning(config, manifest, for_fit=purpose == 'fit')}


def identity(config, checked):
    fixed = {key: value for key, value in config.items() if key not in ('max_steps', 'max_seconds', 'output_dir')}
    fixed.update(manifest_sha256=digest(checked['manifest_path']), launcher_sha256=digest(__file__),
                 shared_helpers_sha256=digest(Path(__file__).with_name('training_act.py')))
    return hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()


def prepare_owned_environment():
    require(platform.system() == 'Linux' and OWN_ROOT.is_dir(), 'Use the prepared owned Linux workspace')
    for variable, directory in {'HF_HOME': 'cache/huggingface', 'HF_HUB_CACHE': 'cache/huggingface/hub',
                                'HF_DATASETS_CACHE': 'cache/huggingface/datasets', 'TORCH_HOME': 'cache/torch',
                                'CUDA_CACHE_PATH': 'cache/cuda', 'TRITON_CACHE_DIR': 'cache/triton',
                                'XDG_CACHE_HOME': 'cache', 'TMPDIR': 'tmp'}.items():
        path = OWN_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.dont_write_bytecode = True


def decode_check(config, checked, output_path):
    """Compare native loader RGB with annotation decoder RGB, without a model or GPU."""
    prepare_owned_environment()
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(2)
    annotations = check_annotations(config, checked['manifest'])
    by_episode = {}
    for row in annotations['rows']:
        by_episode.setdefault(row['episode_index'], []).append(row)
    requested = []
    manifest = checked['manifest']
    ids = manifest['train_episodes'] + manifest['validation_episodes']
    for eid in ids:
        ordered = sorted(by_episode.get(eid, []), key=lambda row: row['frame_index'])
        require(ordered, f'No annotated frames in episode {eid}')
        requested.extend(ordered[index] for index in sorted({0, len(ordered) // 2, len(ordered) - 1}))
    raw = LeRobotDataset(repo_id='snapshot', root=checked['root'], episodes=ids, download_videos=False, video_backend='pyav')
    local_indexes = {int(row['index']): index for index, row in enumerate(raw.hf_dataset.select_columns(['index']))}
    samples = []
    for row in requested:
        item = raw[local_indexes[row['index']]]
        require(int(item['episode_index']) == row['episode_index'] and int(item['frame_index']) == row['frame_index'], 'Native loader returned another episode/frame')
        rgb = item[annotations['camera_key']]
        array = (rgb.permute(1, 2, 0).numpy() * 255).round().astype('uint8')
        actual = hashlib.sha256(array.tobytes(order='C')).hexdigest()
        require(actual == row['image_sha256'], f"RGB decoder mismatch at episode {row['episode_index']} frame {row['frame_index']}")
        samples.append({'index': row['index'], 'episode_index': row['episode_index'], 'frame_index': row['frame_index'],
                        'shape': list(array.shape), 'sha256': actual})
    require(snapshot(checked['root']) == manifest['files'], 'Dataset changed during decoder check')
    result = {'status': 'NATIVE_RGB_DECODER_PARITY_PASSED', 'annotation_sha256': config['conditioning']['annotation_sha256'],
              'manifest_sha256': digest(checked['manifest_path']), 'samples': samples,
              'sampling': 'first/middle/last annotated frame of each train and validation episode',
              'model_loaded': False, 'gpu_work': False, 'controller_ready': False}
    require(not output_path.exists(), 'Preserve prior decoder evidence; choose a new result path')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(output_path, result)
    return result


def run(config, checked, mode, resume):
    import fcntl
    require(platform.system() == 'Linux' and OWN_ROOT.is_dir(), 'Use the allocated Linux H100 workspace')
    for key in ('dataset_root', 'manifest', 'capture_manifest', 'capture_runtime', 'base_dir', 'backbone_dir', 'studio_library'):
        if config.get('training_scope') == 'real_data_pipeline_probe' and key in ('capture_manifest', 'capture_runtime'):
            continue
        require(Path(config[key]).resolve().is_relative_to(OWN_ROOT), f'{key} must be staged in owned workspace')
    lock = (OWN_ROOT / '.gpu-job.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError('Another owned GPU job holds the lock') from exc
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip(),
            'Another GPU compute process is active')
    prepare_owned_environment()
    import numpy as np
    import torch
    from lightning.pytorch import Callback, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from physicalai.data.lerobot import LeRobotDataModule
    from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter
    from physicalai.policies import SmolVLA
    from physicalai.policies.smolvla.pretrained_utils import fix_state_dict_keys
    from physicalai.train import Trainer
    from safetensors.torch import load_file

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name(0), 'Expose the single allocated H100')
    library = checked_path(config['studio_library'], 'studio_library', directory=True)
    require(Path(inspect.getfile(SmolVLA)).resolve().is_relative_to(library), 'Imported policy is outside pinned Studio checkout')
    revision = subprocess.check_output(['git', '-C', str(library), 'rev-parse', 'HEAD'], text=True).strip()
    require(revision == STUDIO_REVISION, 'Installed Studio revision differs')
    subprocess.run(['git', '-C', str(library), 'diff', '--exit-code', '--quiet', 'HEAD', '--', *SOURCE_FILES], check=True)
    sources = {name: digest(library / name) for name in SOURCE_FILES}
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'physicalai', 'lerobot', 'lightning', 'transformers')}
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    output, manifest = checked['output'], checked['manifest']
    run_id = identity(config, checked)
    if resume:
        require(mode == 'train', 'Resume requires train mode')
        previous = read(output / 'run.json')
        require(previous['run_identity'] == run_id, 'Resume data/config/source identity differs')
        check_resume_packages(previous['versions'], versions, (output / 'packages.txt').read_text(), freeze)
        require(previous['studio_source_sha256'] == sources, 'Resume Studio sources differ')
        resume = checked_path(str(resume), 'resume checkpoint')
        require(resume.is_relative_to(output), 'Resume only this run\'s checkpoints')
    else:
        require(not output.exists(), 'Use a fresh output directory or explicit resume')
        output.mkdir(parents=True)
    seed_everything(config['seed'], workers=True)
    torch.set_num_threads(4)
    stats = {key: {name: np.asarray(value) for name, value in values.items()} for key, values in checked['stats'].items()}
    conditioning = checked['conditioning']
    annotations = check_annotations(config, manifest, for_fit=True) if conditioning['mode'] == 'anomalib_rows' else None
    annotated_rows = {row['index']: row for row in annotations['rows']} if annotations else {}

    class ConditionedDataset(LeRobotDataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            if conditioning['mode'] == 'episode_task':
                eid = str(int(item['episode_index']))
                item['task'] = conditioning['episodes'][eid]['text']
            elif annotations:
                row = annotated_rows[int(item['index'])]
                require(row['eligible'], 'Ineligible observation escaped the start-window filter')
                require(int(item['episode_index']) == row['episode_index'] and int(item['frame_index']) == row['frame_index'], 'Annotation row identity mismatch')
                rgb = item[annotations['camera_key']]
                if rgb.ndim == 4:
                    rgb = rgb[-1]
                array = (rgb.permute(1, 2, 0).numpy() * 255).round().astype('uint8')
                require(hashlib.sha256(array.tobytes(order='C')).hexdigest() == row['image_sha256'], 'Actual decoded image differs from detector annotation')
                item['task'] = annotation_text(conditioning['instruction'], row, annotations['threshold'])
            return item

    def load(ids):
        raw = ConditionedDataset(repo_id='snapshot', root=checked['root'], episodes=ids, download_videos=False, video_backend='pyav')
        raw.meta.stats = stats
        return raw

    train_data, validation_data = load(manifest['train_episodes']), load(manifest['validation_episodes'])
    dm = LeRobotDataModule(dataset=train_data, train_batch_size=config['batch_size'], val_batch_size=config['batch_size'], num_workers=config['num_workers'])
    dm.val_eval_dataset = _LeRobotDatasetAdapter.from_lerobot(validation_data)
    class EligibleAdapter(_LeRobotDatasetAdapter):
        def __init__(self, raw):
            self._lerobot_dataset = raw
            self.start_indices = []
            for local_index, item in enumerate(raw.hf_dataset.select_columns(['index', 'episode_index', 'frame_index'])):
                row = annotated_rows.get(int(item['index']))
                if row is None:
                    continue
                require(int(item['episode_index']) == row['episode_index'] and int(item['frame_index']) == row['frame_index'], 'Annotation index points to a different episode/frame')
                if row['eligible']:
                    self.start_indices.append(local_index)
            require(self.start_indices, 'Split has no eligible detector-conditioned action starts')

        def __len__(self):
            return len(self.start_indices)

        def __getitem__(self, index):
            return super().__getitem__(self.start_indices[index])

    if annotations:
        # Only choose valid observation starts. Future action deltas still use the complete raw episode.
        dm.train_dataset = EligibleAdapter(train_data)
        dm.val_eval_dataset = EligibleAdapter(validation_data)
        require(len(dm.train_dataset) >= config['batch_size'], 'Too few eligible training starts for one batch')
    # No final-evaluation observations are loaded or forwarded through the model.
    import pyarrow.parquet as pq
    task_frame = pq.read_table(checked['root'] / 'meta/tasks.parquet').to_pandas()
    task_text = {int(row['task_index']): str(index) for index, row in task_frame.iterrows()}
    prompts_by_episode = {}
    used_ids = set(manifest['train_episodes'] + manifest['validation_episodes'])
    for path in sorted((checked['root'] / 'data').rglob('*.parquet')):
        for row in pq.read_table(path, columns=['index', 'episode_index', 'task_index']).to_pylist():
            eid = int(row['episode_index'])
            if eid not in used_ids:
                continue
            if annotations:
                annotation = annotated_rows.get(int(row['index']))
                if annotation is None or not annotation['eligible']:
                    continue
                text = annotation_text(conditioning['instruction'], annotation, annotations['threshold'])
            else:
                text = conditioning['episodes'][str(eid)]['text'] if conditioning['mode'] == 'episode_task' else task_text[int(row['task_index'])]
            require(text.strip(), 'Empty task text')
            prompts_by_episode.setdefault(eid, set()).add(text)
    require(set(prompts_by_episode) == used_ids, 'Every training/validation episode needs eligible language-conditioned starts')
    if not annotations:
        text_destinations = {}
        for ep in manifest['episodes']:
            if ep['episode_index'] in used_ids:
                for text in prompts_by_episode[ep['episode_index']]:
                    destination = text_destinations.setdefault(text, observed_destination(ep))
                    require(destination == observed_destination(ep), 'Same caption labels conflicting destinations; provide evidenced conditioning')
    if config.get('training_scope') == 'real_data_pipeline_probe':
        for episode in manifest['episodes']:
            if episode['episode_index'] in used_ids:
                require(prompts_by_episode[episode['episode_index']] == {episode['task_caption']}, 'Loaded caption differs from frozen recorded caption')
    all_prompts = sorted({text for prompts in prompts_by_episode.values() for text in prompts})

    start = time.monotonic()
    record = {'status': 'RUNNING', 'run_identity': run_id, 'config': config, 'mode': mode,
              'versions': versions, 'studio_source_sha256': sources, 'host': platform.node(),
              'manifest_sha256': digest(checked['manifest_path']), 'train_stats_sha256': digest(checked['stats_path']),
              'base_sha256': BASE_SHA256, 'base_revision': BASE_REVISION, 'capture': checked['capture'],
              'resume_checkpoint': str(resume) if resume else None, 'resume_sha256': digest(resume) if resume else None,
              'controller_ready': False, 'task_ready': False, 'physical_success': 'NOT TESTED',
              'language': {str(key): sorted(value) for key, value in prompts_by_episode.items()},
              'eligible_train_starts': len(dm.train_dataset), 'eligible_validation_starts': len(dm.val_eval_dataset),
              'annotation_sha256': conditioning.get('annotation_sha256'),
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (output / 'packages.txt').write_text(freeze)
    dump(output / 'run.json', record)
    try:
        base_options = read(checked['base'] / 'config.json')
        require(not base_options.get('adapt_to_pi_aloha', False), 'SO-101 route cannot enable Aloha conversion')
        accepted = set(inspect.signature(SmolVLA).parameters)
        options = {key: value for key, value in base_options.items() if key in accepted}
        options.update(pretrained_name_or_path=None, dataset_stats=dm.train_dataset.stats,
                       image_key_reorder_map=config['image_key_reorder_map'], num_cameras=3,
                       tokenizer_max_length=config['tokenizer_max_length'], compile_model=False,
                       snapflow_enabled=False, load_vlm_weights=False, use_random_input_noise=True,
                       optimizer_lr=config['learning_rate'], scheduler_warmup_steps=config['warmup_steps'],
                       scheduler_decay_steps=config['scheduler_decay_steps'])
        policy = SmolVLA(**options)
        weights = fix_state_dict_keys(load_file(str(checked['base'] / 'model.safetensors')))
        weight_keys = set(weights)
        record['inactive_missing_keys'] = verify_weight_keys(weight_keys, set(policy.model.state_dict()), policy.model._model._snapflow_enabled)
        missing, unexpected = policy.model.load_state_dict(weights, strict=False)
        require(set(missing) <= INACTIVE_KEYS and not unexpected, 'Active pretrained weights failed to load')
        del weights
        policy.model._model.set_requires_grad()
        policy.model._model.vlm_with_expert.set_requires_grad()
        record['pretrained_active_keys'] = len(weight_keys)
        # Replace the base normalizers before any task-data forward pass.
        policy._update_preprocessor_stats(dm.train_dataset.stats)
        token_counts = {text: len(policy._preprocessor.tokenizer(text if text.endswith('\n') else text + '\n', truncation=False)['input_ids'])
                        for text in all_prompts}
        require(all(count <= config['tokenizer_max_length'] for count in token_counts.values()), 'Conditioning would be truncated')
        record['token_counts'] = token_counts
        dump(output / 'run.json', record)

        class Evidence(Callback):
            gradient = None
            rng_context = None

            def on_fit_start(self, trainer, model):
                for feature in ('observation.state', 'action'):
                    for name in ('mean', 'std'):
                        np.testing.assert_allclose(model.hparams['dataset_stats'][feature][name], checked['stats'][feature][name], rtol=1e-6, atol=1e-7)
                record['train_normalizers_verified'] = True

            def on_after_backward(self, trainer, model):
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                require(gradients and all(torch.isfinite(g).all().item() for g in gradients), 'Missing/nonfinite gradients')
                if self.gradient is None:
                    norm = sum(g.detach().float().square().sum().item() for g in gradients) ** .5
                    require(norm > 0, 'Zero gradient norm')
                    self.gradient = {'tensor_count': len(gradients), 'l2_norm': norm}

            def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                require(torch.isfinite(loss).all().item(), 'Nonfinite training loss')
                dump(output / 'progress.json', {'step': trainer.global_step, 'loss': float(loss.detach().cpu()), 'elapsed_seconds': time.monotonic() - start})

            def on_validation_epoch_start(self, trainer, model):
                # Fixed flow-matching noise/time for comparable checkpoint selection; restore train RNG afterward.
                self.rng_context = torch.random.fork_rng(devices=[0])
                self.rng_context.__enter__()
                torch.manual_seed(config['seed'] + 1)
                torch.cuda.manual_seed_all(config['seed'] + 1)

            def on_validation_epoch_end(self, trainer, model):
                self.rng_context.__exit__(None, None, None)
                self.rng_context = None

            def on_save_checkpoint(self, trainer, model, checkpoint):
                checkpoint['second_look_run_identity'] = run_id

            def on_load_checkpoint(self, trainer, model, checkpoint):
                require(checkpoint.get('second_look_run_identity') == run_id, 'Resume checkpoint identity differs')

        evidence = Evidence()
        best = ModelCheckpoint(dirpath=output / 'checkpoints', filename='selected-{step:06d}', monitor='val/loss', mode='min', save_top_k=1, save_on_train_epoch_end=False)
        recovery = ModelCheckpoint(dirpath=output / 'recovery', filename='step-{step:06d}', every_n_train_steps=config['checkpoint_every'], save_top_k=1, save_last=True, save_on_train_epoch_end=False)
        trainer = Trainer(accelerator='gpu', devices=1, precision=config['precision'], max_epochs=-1,
                          max_steps=1 if mode == 'smoke' else config['max_steps'], max_time={'seconds': config['max_seconds']},
                          callbacks=[evidence, best, recovery], check_val_every_n_epoch=None,
                          val_check_interval=1 if mode == 'smoke' else config['validation_every'],
                          limit_val_batches=1 if mode == 'smoke' else 1.0,
                          num_sanity_val_steps=0, enable_progress_bar=False, enable_model_summary=False,
                          deterministic='warn', cudnn_benchmark=False, logger=CSVLogger(str(output), name='metrics'),
                          default_root_dir=str(output), log_every_n_steps=1)
        torch.cuda.reset_peak_memory_stats()
        trainer.fit(policy, datamodule=dm, ckpt_path=str(resume) if resume else None)
        last = output / 'last.ckpt'
        trainer.save_checkpoint(last)
        record.update(global_step=trainer.global_step, gradient_check=evidence.gradient,
                      post_fit_model_device=str(policy.device), last_checkpoint=str(last),
                      last_sha256=digest(last), selected_checkpoint=best.best_model_path)
        dump(output / 'run.json', record)
        require(evidence.gradient and trainer.global_step > 0, 'No completed CUDA update')
        require(best.best_model_path and math.isfinite(float(best.best_model_score)), 'No held-out checkpoint selected; last.ckpt is recovery only')
        peak = torch.cuda.max_memory_allocated()
        # Native checkpoint round trip, on the same real validation minibatch and device.
        # Lightning teardown can move the fitted module to CPU. Seed equality does
        # not make CPU and CUDA normal samplers produce the same flow noise.
        policy.to('cuda').eval()

        def predict_fixed(model, observation):
            observation = copy.deepcopy(observation).to('cuda:0')
            require_prediction_devices(model.device, observation.state.device)
            with torch.random.fork_rng(devices=[0]), torch.inference_mode():
                torch.manual_seed(config['seed'] + 2)
                torch.cuda.manual_seed_all(config['seed'] + 2)
                return model.predict_action_chunk(observation).float().cpu()

        batch = next(iter(dm.val_dataloader())).to('cuda')
        reference = predict_fixed(policy, batch)
        policy.cpu()
        restored = SmolVLA.load_from_checkpoint(last, map_location='cpu').to('cuda').eval()
        reloaded = predict_fixed(restored, batch)
        torch.testing.assert_close(reference, reloaded, atol=1e-5, rtol=1e-5)
        require(torch.isfinite(reloaded).all().item(), 'Reload produced nonfinite actions')
        restored.cpu()
        selected_path = Path(best.best_model_path)
        selected = SmolVLA.load_from_checkpoint(selected_path, map_location='cpu').to('cuda').eval()
        selected_action = predict_fixed(selected, batch)
        require(torch.isfinite(selected_action).all().item() and selected_action.shape[-1] == len(manifest['joint_names']), 'Selected checkpoint invalid action dimensions/values')
        # This verifies functional language use, never correct destination behavior.
        prompts = sorted(token_counts)
        alternate = prompts[1] if len(prompts) > 1 and batch.task[0] == prompts[0] else prompts[0]
        if alternate == batch.task[0]:
            alternate = 'Language sensitivity diagnostic only. Do not move the robot.'
        from dataclasses import replace
        alternate_batch = replace(batch, task=[alternate] * selected_action.shape[0])
        alternate_action = predict_fixed(selected, alternate_batch)
        sensitivity = float((selected_action - alternate_action).abs().max())
        require(math.isfinite(sensitivity) and sensitivity > 0, 'No measured task-language effect for fixed real inputs')
        result = {'status': 'EXPERIMENTAL_SMOLVLA_SMOKE_PASSED' if mode == 'smoke' else 'EXPERIMENTAL_SMOLVLA_VALIDATION_SELECTED',
                  'run_identity': run_id, 'mode': mode, 'evidence_kind': 'real', 'global_step': trainer.global_step,
                  'elapsed_seconds': time.monotonic() - start, 'gradient_check': evidence.gradient,
                  'selected_checkpoint': str(selected_path), 'selected_sha256': digest(selected_path),
                  'last_checkpoint': str(last), 'last_sha256': digest(last),
                  'selection_metric': 'val/loss', 'selection_score': float(best.best_model_score),
                  'validation_rng': {'seed': config['seed'] + 1, 'method': 'fork CPU/CUDA RNG for each whole validation epoch; restore training RNG afterward'},
                  'prediction_rng': {'seed': config['seed'] + 2, 'method': 'same forked CPU/CUDA RNG seed for every compared native prediction', 'device': 'cuda:0'},
                  'validation_scope': 'one minibatch loader diagnostic' if mode == 'smoke' else 'all eligible starts in every held-out validation episode',
                  'validation_episode_ids': manifest['validation_episodes'], 'split_scope': manifest['split_scope'],
                  'peak_cuda_allocated_bytes': peak, 'reload_max_abs_error': float((reference - reloaded).abs().max()),
                  'action_shape': list(selected_action.shape), 'fixed_input_caption_max_abs_difference': sensitivity,
                  'alternate_diagnostic_caption': alternate, 'base_sha256': BASE_SHA256,
                  'controller_ready': False, 'task_ready': False, 'physical_success': 'NOT TESTED',
                  'anomalib_conditioned': bool(annotations), 'final_eval_status': 'UNTOUCHED' if manifest.get('final_eval_episodes') else 'NOT_ALLOCATED',
                  'limits': ['Real demonstration imitation; no autonomous sorting or physical success measured.',
                             'Caption sensitivity proves functional conditioning, not correct destination selection.',
                             'No deployment adapter or Intel parity is established by this native checkpoint.',
                             'Fit wall-clock budget is cooperative; checkpoint verification adds bounded work.',
                             'Shared specimens/sessions cannot support generalization claims.']}
        if config.get('training_scope') == 'real_data_pipeline_probe':
            result.update(status='REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE',
                          training_scope='real_data_pipeline_probe', conditioning_source='original recorded captions',
                          ground_truth_caption_warning='Recorded good/defective captions are not independently verified physical defect labels.',
                          detector_conditioned=False, max_authorized_optimizer_updates=20)
        require(snapshot(checked['root']) == manifest['files'], 'Dataset changed during fitting; artifacts are invalid')
        dump(output / 'result.json', result)
        record.update(status=result['status'], global_step=trainer.global_step)
        print(json.dumps(result, indent=2))
    except BaseException as exc:
        record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        dump(output / 'run.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('check', 'decode-check', 'smoke', 'train'), default='check')
    parser.add_argument('--check', action='store_true', help='Compatibility alias for --mode check')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--decode-output', type=Path, help='New JSON path for no-fit native RGB parity evidence')
    args = parser.parse_args()
    try:
        config = read(args.config)
        checked = check(config, purpose='decode' if args.mode == 'decode-check' and not args.check else 'fit')
        if args.check or args.mode == 'check':
            print(json.dumps({'status': 'FROZEN_REAL_INPUTS_CHECKED', 'controller_ready': False, 'training_executed': False}))
        elif args.mode == 'decode-check':
            require(args.decode_output is not None, 'Provide --decode-output for decoder parity evidence')
            print(json.dumps(decode_check(config, checked, args.decode_output.resolve()), indent=2))
        else:
            run(config, checked, args.mode, args.resume)
    except (ValueError, KeyError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'BLOCKED: {exc}\n')


if __name__ == '__main__':
    main()
