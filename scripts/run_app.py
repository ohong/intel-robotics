#!/usr/bin/env python3
"""Launch the camera-only application. This script never connects to the robot."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from secondlook.runtime import CameraSource, InspectionRuntime
from secondlook.live import LiveClient
from secondlook.voice import FalClient, load_commands
from secondlook.web import make_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    camera_input = parser.add_mutually_exclusive_group(required=True)
    camera_input.add_argument("--camera", help="Explicit direct V4L2 camera mode")
    camera_input.add_argument("--studio-camera-service", help="Attach only to an existing Studio shared camera publisher")
    parser.add_argument("--studio-python", default="/home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--frame-max-age", type=float, default=1.0)
    parser.add_argument("--detection-max-age", type=float, default=3.0)
    parser.add_argument("--inference-interval", type=float, default=.5)
    parser.add_argument("--anomaly-artifact", type=Path)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--backend", default="openvino")
    parser.add_argument("--precision", default="f32")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--instruction-file", type=Path)
    parser.add_argument("--policy-blocker", default=(
        "No task-ready VLA/Physical AI Studio policy connected; base checkpoint lacks "
        "compatible state normalization and verified robot action conventions"))
    parser.add_argument("--voice-commands", type=Path, default=Path("config/voice-commands.json"))
    parser.add_argument("--studio-robot-session",
                        help="Observe joint telemetry of Studio runtime session rt-<follower id>, read-only")
    parser.add_argument("--replay-dataset", type=Path,
                        help="Recorded LeRobot v3 dataset shown as REPLAY in mission control")
    parser.add_argument("--revision", default="unknown")
    parser.add_argument("--evidence", type=Path, default=Path("artifacts/runtime/observations.jsonl"))
    parser.add_argument("--lock-file", type=Path, default=Path("/tmp/secondlook-runtime.lock"))
    args = parser.parse_args()

    lock = args.lock_file.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(2, "Another Second Look runtime already owns the camera session.\n")
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()

    instruction = args.instruction_file.read_text().strip() if args.instruction_file else args.instruction

    def load_detector():
        from secondlook.anomaly import AnomalyDetector
        return AnomalyDetector(args.anomaly_artifact, device=args.device, backend=args.backend,
                               precision=args.precision)

    if args.studio_camera_service:
        from secondlook.studio_camera import StudioCameraSource
        camera = StudioCameraSource(args.studio_camera_service, args.studio_python,
                                    max_age=args.frame_max_age)
    else:
        camera = CameraSource(args.camera, args.width, args.height, args.frame_max_age)
    runtime = InspectionRuntime(
        camera,
        detector_loader=load_detector if args.anomaly_artifact else None,
        task_instruction=instruction, evidence_path=args.evidence,
        revision=args.revision, detection_max_age=args.detection_max_age,
        policy_blocker=args.policy_blocker,
        inference_interval=args.inference_interval,
        voice=FalClient.from_env(), voice_commands=load_commands(args.voice_commands),
    )
    replay = None
    if args.replay_dataset:
        from secondlook.replay import ReplayDataset
        replay = ReplayDataset(args.replay_dataset)
    robot = None
    if args.studio_robot_session:
        from secondlook.studio_robot import StudioRobotSource
        robot = StudioRobotSource(args.studio_robot_session, args.studio_python)
    server = make_server(runtime, port=args.port, replay=replay, robot=robot,
                         live=LiveClient.from_env())

    def shutdown(*_args):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    runtime.start()
    if robot:
        robot.start()
    print(f"Second Look: http://127.0.0.1:{args.port} — software DISARMED; robot connection unavailable", flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        runtime.close()
        if robot:
            robot.close()
        server.server_close()
        lock.close()


if __name__ == "__main__":
    main()
