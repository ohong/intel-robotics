#!/usr/bin/env python3
"""Export a selected native Studio checkpoint with exact weights and training statistics.

Runtime dependencies are imported only after provenance checks. No network, package
installation, environment mutation, training, or robot access is performed here.
"""
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
import shutil
import tempfile

STAT_FIELDS = ('mean', 'std', 'min', 'max', 'q01', 'q99')
INACTIVE_KEYS = {f'_model.target_time_mlp_{part}.{kind}'
                 for part in ('in', 'out') for kind in ('weight', 'bias')}
# Studio's HF loader overwrites these fields even when config.json contains them.
HF_OVERRIDES = ('tokenizer_max_length', 'pad_language_to', 'use_random_input_noise',
                'image_key_reorder_map', 'num_cameras', 'compile_model',
                'snapflow_enabled', 'snapflow_alpha', 'snapflow_lambda',
                'snapflow_num_inference_steps', 'num_steps', 'use_cache',
                'freeze_vision_encoder', 'train_expert_only', 'train_state_proj',
                'optimizer_lr', 'optimizer_betas', 'optimizer_eps',
                'optimizer_weight_decay', 'optimizer_grad_clip_norm',
                'scheduler_warmup_steps', 'scheduler_decay_steps', 'scheduler_decay_lr')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inventory(root):
    return {str(path.relative_to(root)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
            for path in sorted(root.rglob('*')) if path.is_file()}


def validate_training_stats(manifest, stats, native_stats):
    """Verify saved statistics against the closed real training split; never synthesize them."""
    require(manifest.get('evidence_kind') == 'real' and manifest.get('finalized') is True
            and manifest.get('status') == 'OFFLINE_DATA_VALIDATED', 'Require a finalized real dataset')
    splits = [manifest.get(key, []) for key in ('train_episodes', 'validation_episodes', 'final_eval_episodes')]
    require(splits[0] and splits[1], 'Require train and validation episodes')
    ids = [eid for split in splits for eid in split]
    episode_ids = [episode['episode_index'] for episode in manifest['episodes']]
    require(len(ids) == len(set(ids)) and len(episode_ids) == len(set(episode_ids))
            and set(ids) == set(episode_ids), 'Episode split leakage or incomplete coverage')
    frames = sum(ep['frames'] for ep in manifest['episodes'] if ep['episode_index'] in splits[0])
    require(frames > 0, 'Empty training split')
    normalization = manifest.get('normalization', '')
    if isinstance(normalization, dict):
        require(normalization.get('training_episodes_only') == splits[0]
                and normalization.get('training_rows') == frames
                and normalization.get('excluded_validation_episodes') == splits[1]
                and normalization.get('excluded_final_episodes') == splits[2], 'Training-only normalization split differs')
    else:
        require(isinstance(normalization, str) and normalization.startswith('train episodes only'),
                'Require training-only statistics')
    width = len(manifest['joint_names'])
    require(width > 0, 'Missing joint names')
    result = {}
    for key in ('observation.state', 'action'):
        source, native = stats[key], native_stats[key]
        require(source.get('count') == [frames], f'{key}: wrong training frame count')
        require(list(native['shape']) == [width], f'{key}: native feature dimension differs')
        require(native['name'] == key.removeprefix('observation.'), f'{key}: unsupported native feature alias')
        for field in STAT_FIELDS:
            values = source[field]
            require(isinstance(values, list) and len(values) == width
                    and all(type(x) in (int, float) and math.isfinite(x) for x in values),
                    f'{key}/{field}: invalid statistics')
            require(native.get(field) == values, f'{key}/{field}: checkpoint differs from training statistics')
        require(all(v >= 0 for v in source['std']), f'{key}: negative standard deviation')
        require(all(low <= high for low, high in zip(source['min'], source['max'])), f'{key}: inverted range')
        result[key] = {field: source[field] for field in STAT_FIELDS}
    return result


def export_plan(hparams, manifest, stats, backbone):
    config = copy.deepcopy(hparams['config'])
    native_stats = hparams['dataset_stats']
    normalizers = validate_training_stats(manifest, stats, native_stats)
    require(config.get('num_cameras') == 3, 'Expected native three-slot camera padding')
    require(config.get('tokenizer_max_length') == 128, 'Expected tokenizer length 128')
    require(config.get('load_vlm_weights') is False and config.get('compile_model') is False,
            'Require local uncompiled model initialization without backbone weight loading')
    require(config.get('adapt_to_pi_aloha') is False and config.get('snapflow_enabled') is False,
            'Unsupported Aloha conversion or SnapFlow mode')
    mapping = config['image_key_reorder_map']
    cameras = [key.removeprefix('observation.images.') for key in manifest.get('cameras', list(manifest.get('camera_identity_mapping', {})))]
    require(len(cameras) == 2 and set(mapping) == set(cameras), 'Require the two actual camera keys')
    require(all(type(v) is int and 0 <= v < 3 for v in mapping.values())
            and len(set(mapping.values())) == 2, 'Camera slots must be distinct integers 0..2')
    features = {'observation.state': {'type': 'STATE', 'shape': list(native_stats['observation.state']['shape'])}}
    for camera in cameras:
        candidate = native_stats.get(f'observation.{camera}')
        require(candidate is not None and candidate.get('name') == camera, f'Missing native camera feature {camera}')
        shape = list(candidate['shape'])
        require(len(shape) == 3 and shape[0] == 3 and all(type(v) is int and v > 0 for v in shape),
                f'Camera {camera}: expected native CHW shape')
        features[f'observation.images.{camera}'] = {'type': 'VISUAL', 'shape': shape}
    config.update(vlm_model_name=str(backbone), input_features=features,
                  output_features={'action': {'type': 'ACTION', 'shape': list(native_stats['action']['shape'])}})
    overrides = {key: config[key] for key in HF_OVERRIDES}
    # The policy superclass receives this before the HF config is loaded.
    overrides['n_action_steps'] = config['n_action_steps']
    layout = [next((key for key, slot in mapping.items() if slot == index), None) for index in range(3)]
    return config, normalizers, overrides, layout


def model_weights(checkpoint_state, expected_keys):
    """Strip only the Lightning wrapper prefix, preserving inactive parameters too."""
    weights = {key[len('model.'):]: tensor for key, tensor in checkpoint_state.items() if key.startswith('model.')}
    require(set(weights) == set(expected_keys), 'Native checkpoint model keys do not exactly match current Studio')
    require(INACTIVE_KEYS <= set(weights), 'Native checkpoint must contain all inactive target-time parameters')
    return weights


def check_provenance(checkpoint, run_path, result_path, manifest_path, stats_path, studio_library, backbone):
    run, result, manifest, stats = map(read, (run_path, result_path, manifest_path, stats_path))
    require(run['run_identity'] == result['run_identity'], 'Run/result identity mismatch')
    require(result.get('evidence_kind') == 'real' and result.get('global_step', 0) > 0,
            'Require a real completed training result')
    require(result.get('status') in ('EXPERIMENTAL_SMOLVLA_SMOKE_PASSED', 'EXPERIMENTAL_SMOLVLA_VALIDATION_SELECTED', 'REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE'),
            'Require successfully verified native training')
    if run['config'].get('training_scope') == 'real_data_pipeline_probe':
        require(result['status'] == 'REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE'
                and result.get('controller_ready') is False and result.get('task_ready') is False
                and result.get('anomalib_conditioned') is False, 'Probe result must remain nondeployable')
    require(digest(checkpoint) == result['selected_sha256'], 'Only the validation-selected checkpoint may be exported')
    require(digest(manifest_path) == run['manifest_sha256'], 'Dataset manifest hash mismatch')
    require(digest(stats_path) == run['train_stats_sha256'] == manifest['train_stats_sha256'], 'Training statistics hash mismatch')
    require(run.get('studio_source_sha256'), 'Missing Studio source provenance')
    for relative, expected in run['studio_source_sha256'].items():
        path = (studio_library / relative).resolve()
        require(path.is_relative_to(studio_library.resolve()) and digest(path) == expected, 'Studio source differs from training')
    require(backbone.is_dir() and inventory(backbone) == run['config']['backbone_files'], 'Local backbone inventory differs from training')
    return run, result, manifest, stats


def assert_exact_weights(torch, left, right):
    require(set(left) == set(right), 'Model state keys changed during export/load')
    for key in left:
        require(left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
                and torch.equal(left[key].cpu(), right[key].cpu()), f'Model tensor changed: {key}')


def compare_predictions(native, exported, observation, *, seed, atol=1e-5, rtol=1e-5):
    """Compare full action chunks on CPU, restoring caller RNG state after each pass."""
    import torch
    predictions = []
    for policy in (native, exported):
        require(next(policy.parameters()).device.type == 'cpu', 'Parity uses CPU only')
        policy.eval()
        policy.reset()
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            action = policy.predict_action_chunk(copy.deepcopy(observation)).detach().float().cpu()
        require(bool(torch.isfinite(action).all()), 'Nonfinite parity prediction')
        predictions.append(action)
    torch.testing.assert_close(*predictions, atol=atol, rtol=rtol)
    return {'status': 'NATIVE_HF_CPU_PARITY_PASSED', 'seed': seed, 'atol': atol, 'rtol': rtol,
            'action_shape': list(predictions[0].shape),
            'max_abs_error': float((predictions[0] - predictions[1]).abs().max()),
            'physical_success': 'NOT TESTED', 'controller_ready': False}


def load_observation(path, config):
    import numpy as np
    import torch
    from PIL import Image
    from physicalai.data import Observation
    document = read(path)
    require(document.get('evidence_kind') == 'real' and document.get('observation_id'), 'Parity requires identified real observation')
    require(set(document['images']) == set(config['image_key_reorder_map']), 'Parity camera keys differ')
    state = document['state']
    require(len(state) == config['input_features']['observation.state']['shape'][0]
            and all(type(v) in (int, float) and math.isfinite(v) for v in state), 'Invalid parity state')
    require(isinstance(document['task'], str) and document['task'].strip(), 'Missing parity task text')
    require(type(document.get('timestamp')) in (int, float) and math.isfinite(document['timestamp'])
            and document['timestamp'] >= 0, 'Missing recorded nominal timestamp')
    require(all(type(document.get(key)) is int and document[key] >= 0 for key in ('episode_index', 'frame_index')),
            'Missing recorded episode/frame identity')
    images = {}
    for key, entry in document['images'].items():
        image_path = Path(entry['path'])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        require(digest(image_path) == entry['sha256'], 'Parity camera checksum differs')
        with Image.open(image_path) as image:
            rgb = np.array(image.convert('RGB'), dtype=np.uint8)
        require(hashlib.sha256(rgb.tobytes(order='C')).hexdigest() == entry['raw_rgb_sha256'],
                'Decoded RGB differs from recorded source frame')
        value = rgb.astype(np.float32) / 255
        images[key] = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0)
    return Observation(state=torch.tensor([state], dtype=torch.float32), images=images, task=[document['task']],
                       timestamp=torch.tensor([document['timestamp']], dtype=torch.float32),
                       episode_index=torch.tensor([document['episode_index']]), frame_index=torch.tensor([document['frame_index']]))


def export(args):
    require(not args.output.exists(), 'Choose a new output directory; existing artifacts are preserved')
    run, result, manifest, stats = check_provenance(args.checkpoint, args.run, args.result, args.manifest,
                                                  args.stats, args.studio_library, args.backbone_dir)
    require(all(os.environ.get(key) == '1' for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')),
            'Launch with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; exporter will not change environment')
    import torch
    from safetensors.torch import save_file
    from physicalai.policies.smolvla.policy import SmolVLA
    from physicalai.policies.pi05.pretrained_utils import parse_preprocessor_stats
    from physicalai.policies.smolvla.pretrained_utils import extract_dataset_stats
    source = Path(inspect.getfile(SmolVLA)).resolve()
    require(source.is_relative_to(args.studio_library.resolve()), 'Imported Studio is outside the verified library')
    # This is a trusted, locally trained Lightning artifact, authenticated by the selected SHA above.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    require(checkpoint.get('second_look_run_identity') == run['run_identity'], 'Checkpoint run identity differs')
    hparams = checkpoint['hyper_parameters']
    config, normalizers, overrides, layout = export_plan(hparams, manifest, stats, args.backbone_dir.resolve())
    native = SmolVLA.load_from_checkpoint(args.checkpoint, map_location='cpu', strict=True,
                                         vlm_model_name=str(args.backbone_dir.resolve()), compile_model=False).cpu().eval()
    weights = model_weights(checkpoint['state_dict'], native.model.state_dict())
    assert_exact_weights(torch, weights, native.model.state_dict())
    require(native.config.to_dict() == {key: value for key, value in config.items()
                                       if key not in ('input_features', 'output_features')}, 'Native resolved configuration differs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.smolvla-export-', dir=args.output.parent))
    try:
        dump(temporary / 'config.json', config)
        # Clones preserve every named tensor even when native tensors share storage.
        save_file({key: value.detach().cpu().contiguous().clone() for key, value in weights.items()}, str(temporary / 'model.safetensors'))
        # Float64 JSON round trips preserve the original training statistics exactly.
        save_file({f'{key}.{field}': torch.tensor(values, dtype=torch.float64)
                   for key, fields in normalizers.items() for field, values in fields.items()}, str(temporary / 'training-normalization.safetensors'))
        processor = {'steps': [{'registry_name': 'normalizer_processor', 'state_file': 'training-normalization.safetensors',
                               'config': {'norm_map': {'STATE': 'MEAN_STD', 'ACTION': 'MEAN_STD'}}}]}
        dump(temporary / 'policy_preprocessor.json', processor)
        parsed = parse_preprocessor_stats(processor, config, temporary)
        runtime_stats = extract_dataset_stats(config, temporary / 'policy_preprocessor.json', temporary)
        for key, fields in normalizers.items():
            for field, values in fields.items():
                require(parsed[key][field] == values, f'Parser changed {key}/{field}')
            for field in ('mean', 'std'):
                require(runtime_stats[key][field] == fields[field], 'Studio substituted normalization statistics')
        exported = SmolVLA(pretrained_name_or_path=temporary, **overrides).cpu().eval()
        require(exported.config.to_dict() == native.config.to_dict(), 'HF loader changed native behavior configuration')
        assert_exact_weights(torch, native.model.state_dict(), exported.model.state_dict())
        parity = {'status': 'NOT TESTED', 'reason': 'No --observation supplied'}
        if args.observation:
            observation = load_observation(args.observation, config)
            for policy in (native, exported):
                text = observation.task[0]
                tokens = policy._preprocessor.tokenizer(text if text.endswith('\n') else text + '\n', truncation=False)['input_ids']
                require(len(tokens) <= 128, 'Parity prompt would be truncated')
            parity = compare_predictions(native, exported, observation, seed=args.seed)
            original = read(args.observation)
            parity.update(observation_sha256=digest(args.observation), observation_id=original['observation_id'],
                          episode_index=original['episode_index'], frame_index=original['frame_index'],
                          recorded_nominal_timestamp=original['timestamp'], task=original['task'],
                          images={key: {'encoded_file_sha256': entry['sha256'], 'source_raw_rgb_sha256': entry['raw_rgb_sha256'],
                                        'raw_format': 'uint8_HWC_RGB_contiguous'} for key, entry in original['images'].items()})
        deployment_backbone = args.deployment_backbone_dir or args.backbone_dir.resolve()
        require(deployment_backbone.is_absolute(), 'Deployment backbone path must be absolute')
        config['vlm_model_name'] = str(deployment_backbone)
        dump(temporary / 'config.json', config)
        metadata = {'schema_version': 1, 'status': ('REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE'
                    if run['config'].get('training_scope') == 'real_data_pipeline_probe' else 'EXPERIMENTAL_NATIVE_HF_EXPORT_VERIFIED'),
                    'export_validation': 'EXACT_NATIVE_HF_CONFIG_WEIGHTS_AND_STATS_PASSED',
                    'training_scope': run['config'].get('training_scope'),
                    'anomalib_conditioned': result.get('anomalib_conditioned', False),
                    'detector_conditioned': result.get('detector_conditioned', False),
                    'controller_ready': False, 'task_ready': False, 'physical_success': 'NOT TESTED',
                    'checkpoint_sha256': digest(args.checkpoint), 'run_identity': run['run_identity'],
                    'run_sha256': digest(args.run), 'result_sha256': digest(args.result),
                    'manifest_sha256': digest(args.manifest), 'train_stats_sha256': digest(args.stats),
                    'exporter_sha256': digest(__file__), 'studio_revision': run['config']['studio_revision'],
                    'studio_source_sha256': run['studio_source_sha256'], 'training_versions': run['versions'],
                    'export_versions': {name: importlib.metadata.version(name) for name in ('torch', 'safetensors', 'transformers', 'lightning')},
                    'base_revision': run['base_revision'], 'base_sha256': run['base_sha256'],
                    'native_config': hparams['config'], 'hf_constructor_kwargs': overrides,
                    'actual_camera_keys': list(config['image_key_reorder_map']), 'camera_slots': layout,
                    'backbone_files': run['config']['backbone_files'], 'backbone_dir': str(args.backbone_dir.resolve()),
                    'deployment_backbone_dir': str(deployment_backbone),
                    'deployment_backbone_verified': deployment_backbone == args.backbone_dir.resolve(),
                    'backbone_relocation': 'Destination must verify backbone_files inventory before using deployment_backbone_dir.',
                    'all_model_keys_preserved': True, 'model_tensor_count': len(weights),
                    'inactive_keys_preserved': sorted(INACTIVE_KEYS), 'normalization_source': 'train episodes only',
                    'train_episodes': manifest['train_episodes'], 'parity': parity,
                    'files': inventory(temporary)}
        dump(temporary / 'export.json', metadata)
        temporary.rename(args.output)
        return metadata
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'run', 'result', 'manifest', 'stats', 'studio-library', 'backbone-dir', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--deployment-backbone-dir', type=Path, help='Optional absolute destination backbone path; destination must verify its pinned inventory')
    parser.add_argument('--observation', type=Path, help='Optional real observation JSON for CPU action-chunk parity')
    parser.add_argument('--seed', type=int, default=20260915)
    args = parser.parse_args()
    require(0 <= args.seed < 2**32, 'Invalid parity seed')
    print(json.dumps(export(args), indent=2))


if __name__ == '__main__':
    main()
