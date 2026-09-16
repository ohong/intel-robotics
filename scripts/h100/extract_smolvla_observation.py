#!/usr/bin/env python3
"""Extract one identified held-out observation with the native offline LeRobot decoder."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

from training_smolvla import OWN_ROOT, camera_keys, digest, dump, prepare_owned_environment, read, require, snapshot


def extract(args):
    root, manifest_path, output = args.dataset_root.resolve(), args.manifest.resolve(), args.output.resolve()
    require(all(path.is_relative_to(OWN_ROOT) for path in (root, manifest_path, output)), 'All paths must stay inside the owned workspace')
    require(not output.exists() and not output.is_relative_to(root), 'Choose a new output directory outside the frozen dataset')
    require(digest(manifest_path) == args.manifest_sha256, 'Frozen manifest checksum mismatch')
    manifest = read(manifest_path)
    require(manifest.get('evidence_kind') == 'real' and manifest.get('finalized') is True
            and manifest.get('status') == 'OFFLINE_DATA_VALIDATED', 'Require a finalized real recording')
    require(args.episode in manifest['validation_episodes'], 'Extract a held-out validation observation')
    require(len(manifest['joint_names']) == 6, 'Expected six recorded joints')
    require(snapshot(root) == manifest['files'], 'Frozen recording file inventory changed')
    prepare_owned_environment()
    import numpy as np
    import torch
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(2)
    require(importlib.metadata.version('lerobot') == '0.6.0', 'Use the prepared LeRobot 0.6.0 environment')
    dataset = LeRobotDataset(repo_id='snapshot', root=root, episodes=[args.episode], download_videos=False, video_backend='pyav')
    matches = [(index, row) for index, row in enumerate(dataset.hf_dataset.select_columns(['index', 'episode_index', 'frame_index']))
               if int(row['episode_index']) == args.episode and int(row['frame_index']) == args.frame]
    require(len(matches) == 1 and int(matches[0][1]['index']) == args.expected_index, 'Episode/frame/global-index identity mismatch')
    item = dataset[matches[0][0]]
    require(int(item['episode_index']) == args.episode and int(item['frame_index']) == args.frame
            and int(item['index']) == args.expected_index, 'Native decoder returned another observation')
    state, timestamp, task = item['observation.state'].reshape(-1), float(item['timestamp']), item['task']
    require(state.shape == (6,) and bool(torch.isfinite(state).all()), 'Invalid recorded state')
    require(np.isfinite(timestamp) and timestamp >= 0 and isinstance(task, str) and task.strip(), 'Invalid recorded timestamp or caption')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.observation-', dir=output.parent))
    try:
        images = {}
        keys = camera_keys(manifest)
        require(len(keys) == 2, 'Require both actual camera recordings')
        for index, key in enumerate(keys):
            pixels = item[key]
            require(pixels.ndim == 3 and pixels.shape[0] == 3 and bool(torch.isfinite(pixels).all())
                    and float(pixels.min()) >= 0 and float(pixels.max()) <= 1, 'Invalid native RGB tensor')
            rgb = (pixels.permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            require(list(rgb.shape) == manifest['camera_identity_mapping'][key]['recorded_rgb_shape_hwc'], 'Recorded camera shape differs')
            filename = f'camera-{index}.png'
            Image.fromarray(rgb).save(temporary / filename, format='PNG')
            images[key.removeprefix('observation.images.')] = {
                'path': filename, 'sha256': digest(temporary / filename),
                'raw_rgb_sha256': hashlib.sha256(rgb.tobytes(order='C')).hexdigest(),
                'shape': list(rgb.shape), 'raw_format': 'uint8_HWC_RGB_contiguous',
                'camera_identity': manifest['camera_identity_mapping'][key]}
        observation = {
            'evidence_kind': 'real', 'observation_id': f'{args.manifest_sha256}:episode-{args.episode}:frame-{args.frame}',
            'episode_index': args.episode, 'frame_index': args.frame, 'index': args.expected_index,
            'timestamp': timestamp, 'time_basis': 'recorded nominal dataset timestamp in seconds; not Unix wall time',
            'state': state.tolist(), 'state_names': manifest['joint_names'], 'task': task, 'images': images,
            'manifest_sha256': args.manifest_sha256, 'snapshot_receipt_sha256': manifest.get('snapshot_receipt_sha256'),
            'extractor_sha256': digest(__file__), 'decoder': {'lerobot': importlib.metadata.version('lerobot'), 'video_backend': 'pyav', 'av': importlib.metadata.version('av')},
            'controller_ready': False, 'task_ready': False, 'model_loaded': False, 'gpu_work': False,
            'target_association': 'NOT ASSESSED', 'physical_success': 'NOT TESTED'}
        require(snapshot(root) == manifest['files'] and digest(manifest_path) == args.manifest_sha256, 'Frozen recording changed during decode')
        dump(temporary / 'observation.json', observation)
        temporary.rename(output)
        return {'status': 'REAL_HELD_OUT_OBSERVATION_EXTRACTED', 'observation': str(output / 'observation.json'),
                'sha256': digest(output / 'observation.json'), 'controller_ready': False, 'model_loaded': False, 'gpu_work': False}
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset-root', 'manifest', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--episode', type=int, default=2)
    parser.add_argument('--frame', type=int, default=458)
    parser.add_argument('--expected-index', type=int, default=1730)
    print(json.dumps(extract(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
