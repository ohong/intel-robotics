#!/usr/bin/env python3
"""One CPU training step through Studio's native ACT loader, then checkpoint reload."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path

from contracts import require, snapshot
from validate_dataset import dump


def run(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from lightning.pytorch import Callback, seed_everything
    from lightning.pytorch.loggers import CSVLogger
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from physicalai.data.lerobot import LeRobotDataModule
    from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter
    from physicalai.policies import ACT
    from physicalai.train.trainer import Trainer

    torch.set_num_threads(2)
    seed_everything(42, workers=True)
    manifest = json.loads(args.manifest.read_text())
    require(manifest['status'] == 'OFFLINE_DATA_VALIDATED', 'Validation must pass first')
    require(manifest.get('finalized') is True, 'Dataset owner finalization declaration required')
    root = (args.dataset or Path(manifest['dataset'])).resolve()
    output = args.output.resolve()
    require(not output.exists() and not output.is_relative_to(root), 'Use a new output outside the dataset')
    require(snapshot(root) == manifest['files'], 'Dataset differs from validated snapshot')
    stats_path = args.manifest.parent / 'train_stats.json'
    require(hashlib.sha256(stats_path.read_bytes()).hexdigest() == manifest['train_stats_sha256'], 'Training statistics changed after validation')
    stats = {key: {name: np.array(value) for name, value in values.items()}
             for key, values in json.loads(stats_path.read_text()).items()}

    # Construct the same installed reader, replacing only its in-memory statistics.
    # Passing a prebuilt dataset avoids Studio's all-episode quantile calculation.
    def load(episodes):
        raw = LeRobotDataset(repo_id='snapshot', root=root, episodes=episodes,
                             download_videos=False, video_backend='pyav')
        raw.meta.stats = stats
        return raw

    dm = LeRobotDataModule(dataset=load(manifest['train_episodes']), train_batch_size=2, num_workers=0)
    dm.val_eval_dataset = _LeRobotDatasetAdapter.from_lerobot(load(manifest['validation_episodes']))
    options = {'pretrained_backbone_weights': None, 'compile_model': False}
    if args.profile == 'small':
        options.update(chunk_size=4, n_action_steps=4, image_size=(64, 64), dim_model=32,
                       n_heads=4, dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
                       n_vae_encoder_layers=1, latent_dim=8)
    policy = ACT(**options)

    class GradientCheck(Callback):
        evidence = {}

        def on_after_backward(self, trainer, model):
            gradients = [param.grad for param in model.parameters() if param.grad is not None]
            require(gradients and all(torch.isfinite(grad).all().item() for grad in gradients), 'Missing/nonfinite gradients')
            norm = sum(grad.detach().float().square().sum().item() for grad in gradients) ** 0.5
            require(norm > 0, 'All gradients are zero')
            self.evidence = {'gradient_tensors': len(gradients), 'gradient_l2_norm': norm}

    check = GradientCheck()
    trainer = Trainer(accelerator='cpu', devices=1, precision='32-true', max_steps=1, max_epochs=1,
                      limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                      callbacks=[check], logger=CSVLogger(str(output), name='log'), log_every_n_steps=1,
                      default_root_dir=str(output))
    trainer.fit(policy, datamodule=dm)
    require(trainer.global_step == 1 and check.evidence, 'Expected exactly one backward/optimizer step')
    checkpoint = output / 'smoke-model.ckpt'
    trainer.save_checkpoint(checkpoint)
    reloaded = ACT.load_from_checkpoint(checkpoint, map_location='cpu')
    policy.eval()
    reloaded.eval()
    batch = next(iter(dm.val_dataloader()))
    with torch.no_grad():
        first = policy.predict_action_chunk(batch)
        second = reloaded.predict_action_chunk(batch)
        changed_caption = replace(batch, task=['A different destination caption'] * second.shape[0])
        caption_result = reloaded.predict_action_chunk(changed_caption)
    require(torch.isfinite(first).all().item() and torch.isfinite(second).all().item(), 'Nonfinite reloaded output')
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(second, caption_result, atol=0, rtol=0)
    require(snapshot(root) == manifest['files'], 'Dataset changed during smoke')
    sources = {}
    for obj in (LeRobotDataset, LeRobotDataModule, ACT, Trainer):
        source = Path(inspect.getfile(obj))
        sources[obj.__name__] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    report = {'status': 'ONE_STEP_CHECKPOINT_RELOAD_PASSED', 'evidence_kind': manifest['evidence_kind'],
              'manifest': str(args.manifest.resolve()), 'profile': args.profile, 'policy_options': options,
              'dataset': str(root), 'final_eval_episodes': manifest['final_eval_episodes'],
              'split_scope': manifest['split_scope'],
              'sources': sources, 'gradient_check': check.evidence, 'global_step': trainer.global_step,
              'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'train_stats_sha256': hashlib.sha256(stats_path.read_bytes()).hexdigest(),
              'predicted_shape': list(second.shape), 'reload_max_abs_error': (first - second).abs().max().item(),
              'task_caption_max_abs_error': (second - caption_result).abs().max().item(),
              'limits': ['Software smoke only. Random policy; do not deploy or command the robot.',
                         'Small profile changes architecture sizes; Studio profile preserves default sizes.',
                         'No pretrained weights, training-quality, OpenVINO-export, or physical-success claim.']}
    dump(output / 'smoke-result.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, help='Relocated frozen dataset; every file hash must still match')
    parser.add_argument('--profile', choices=('small', 'studio'), default='small')
    run(parser.parse_args())
