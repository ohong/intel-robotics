#!/usr/bin/env python3
"""Read the known follower without configuring motors or changing torque."""
import argparse
import dataclasses
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    args = parser.parse_args()
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupied = subprocess.run(['fuser', args.port], capture_output=True, text=True)
        if occupied.returncode != 1:
            raise RuntimeError('Serial port is in use or ownership could not be checked')
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        raw = args.calibration.read_bytes()
        calibration = {k: MotorCalibration(**v) for k, v in json.loads(raw).items()}
        expected = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        if list(calibration) != expected:
            raise ValueError('Unexpected calibration joint order')
        motors = {
            name: Motor(calibration[name].id, 'sts3215',
                        MotorNormMode.RANGE_0_100 if name == 'gripper' else MotorNormMode.RANGE_M100_100)
            for name in expected
        }
        bus = FeetechMotorsBus(args.port, motors, calibration)
        try:
            # Reviewed LeRobot 0.6.1 handshake sends only ping/read requests.
            # Do not use SOFollower.connect(), configure(), or calibration helpers.
            bus.connect()
            start = time.monotonic()
            positions = bus.sync_read('Present_Position')
            raw_positions = bus.sync_read('Present_Position', normalize=False)
            torque = {name: bus.read('Torque_Enable', name) for name in expected}
            observed_calibration = bus.read_calibration()
            record = {
                'evidence_kind': 'real_read_only_robot_state',
                'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'captured_at_monotonic': start,
                'port': args.port,
                'joints': expected,
                'units': ['range_m100_100'] * 5 + ['range_0_100'],
                'values': [positions[k] for k in expected],
                'raw_encoder_positions': raw_positions,
                'torque_enabled': torque,
                'calibration_sha256': hashlib.sha256(raw).hexdigest(),
                'calibration_matches': all(dataclasses.asdict(observed_calibration[k]) == dataclasses.asdict(calibration[k]) for k in expected),
                'observed_calibration': {k: dataclasses.asdict(v) for k, v in observed_calibration.items()},
                'read_duration_ms': (time.monotonic() - start) * 1000,
                'motion_commanded': False,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps(record))
        finally:
            # Default disconnect disables torque; bypass that side effect deliberately.
            if bus.is_connected:
                bus.disconnect(disable_torque=False)


if __name__ == '__main__':
    main()
