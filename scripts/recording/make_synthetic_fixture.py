#!/usr/bin/env python3
"""Create TWO SYNTHETIC software-only LeRobot episodes. No hardware APIs."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must not exist')
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    features = {key: {'dtype': 'float32', 'shape': (6,), 'names': names} for key in ('observation.state', 'action')}
    features['observation.images.synthetic'] = {'dtype': 'video', 'shape': (64, 64, 3), 'names': ['height', 'width', 'channels']}
    dataset = LeRobotDataset.create(repo_id='synthetic/act-loader-check', root=args.output, fps=10,
                                   features=features, robot_type='SYNTHETIC_NOT_A_ROBOT', use_videos=True)
    for episode in range(2):
        for frame in range(8):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, :, 0] = 20 + frame * 20
            image[:, :, 1] = np.arange(64)[None, :] * 3
            image[:, :, 2] = np.arange(64)[:, None] * 3
            state = np.array([frame, frame * 2, episode, frame / 2, -frame, frame * 10], dtype=np.float32)
            dataset.add_frame({'observation.state': state, 'action': state + 0.25,
                               'observation.images.synthetic': image, 'task': 'SYNTHETIC loader test only'})
        dataset.save_episode()
    dataset.finalize()
    print(f'SYNTHETIC fixture only: {args.output}')


if __name__ == '__main__':
    main()
