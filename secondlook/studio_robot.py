"""Read-only joint telemetry from a running Studio teleoperation session.

A helper in Studio's Python environment observes the session's zenoh ``tick``
events and prints JSON lines. This process never imports Studio, opens the
serial port, or sends a command. When no fresh packet exists, the pose is
reported as unavailable instead of holding the last one as if it were live.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MAX_LINE_BYTES = 8192


class RobotProtocolError(ValueError):
    pass


class StudioRobotSource:
    def __init__(self, session: str, studio_python: str, *, max_age: float = .5, read_timeout: float = 3.,
                 retry_interval: float = 5., worker_path: Path | None = None):
        if not session.startswith("rt-") or any(c in session for c in "\x00\r\n/ "):
            raise ValueError("A Studio runtime session name rt-<follower id> is required")
        self.session, self.studio_python = session, studio_python
        self.max_age, self.read_timeout, self.retry_interval = max_age, read_timeout, retry_interval
        self.worker_path = worker_path or Path(__file__).resolve().parents[1] / "scripts/studio_robot_worker.py"
        self.changed = threading.Condition()
        self._packet: dict | None = None
        self._received = float("-inf")
        self.error: str | None = "Waiting for Studio joint telemetry"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None

    def parse(self, line: bytes, now: float) -> dict:
        if not line or len(line) > MAX_LINE_BYTES:
            raise RobotProtocolError("Empty or oversized robot telemetry line")
        try:
            packet = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise RobotProtocolError("Malformed robot telemetry JSON") from error
        if not isinstance(packet, dict) or packet.get("protocol") != 1:
            raise RobotProtocolError("Unsupported robot telemetry packet")
        if packet.get("type") == "error":
            raise RobotProtocolError(f"Studio robot observer: {str(packet.get('error', 'unavailable'))[:500]}")
        if packet.get("type") != "joints" or packet.get("session") != self.session:
            raise RobotProtocolError("Robot telemetry type or session does not match")
        if packet.get("joint_names") != list(JOINTS):
            raise RobotProtocolError("Robot telemetry joint names do not match SO101")
        timestamp, sequence = packet.get("timestamp"), packet.get("sequence")
        if type(timestamp) not in (int, float) or not 0 <= now - timestamp <= self.max_age:
            raise RobotProtocolError("Robot telemetry is stale or from a future clock")
        if type(sequence) is not int or (self._packet and sequence <= self._packet["sequence"]):
            raise RobotProtocolError("Duplicate or reversed robot telemetry")
        for key in ("state", "action"):
            vector = packet.get(key)
            if vector is None and key == "action":
                continue
            if (not isinstance(vector, list) or len(vector) != len(JOINTS) or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
                raise RobotProtocolError(f"Robot telemetry {key} is not six finite joint values")
        return packet

    def publish(self, packet: dict, now: float) -> None:
        with self.changed:
            self._packet, self._received, self.error = packet, now, None
            self.changed.notify_all()

    def fail(self, message: str) -> None:
        with self.changed:
            self.error = message
            self.changed.notify_all()

    def latest(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        with self.changed:
            packet, received, error = self._packet, self._received, self.error
        base = {"source": "studio_runtime_tick", "session": self.session, "joint_names": list(JOINTS),
                "units": "degrees; gripper 0-100"}
        age = now - received
        if packet is None or error is not None or age > self.max_age:
            return {**base, "status": "UNAVAILABLE",
                    "error": error or f"No Studio joint telemetry for {age:.1f} s"}
        return {**base, "status": "LIVE", "sequence": packet["sequence"], "age_ms": round(age * 1000, 1),
                "state": packet["state"], "action": packet["action"]}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="studio-robot", daemon=True)
        self._thread.start()

    def _consume(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        buffer = bytearray()
        deadline = time.monotonic() + self.read_timeout + 10  # zenoh import and connect
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while not self._stop.is_set():
                if time.monotonic() >= deadline:
                    raise RobotProtocolError("Studio robot observer produced no telemetry in time")
                if not selector.select(.1):
                    if process.poll() is not None:
                        raise RobotProtocolError(f"Studio robot observer exited ({process.returncode})")
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RobotProtocolError("Studio robot observer closed its stream")
                buffer.extend(chunk)
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    now = time.monotonic()
                    self.publish(self.parse(bytes(line), now), now)
                    deadline = now + self.read_timeout
                if len(buffer) > MAX_LINE_BYTES:
                    raise RobotProtocolError("Studio robot observer exceeded line size limit")

    def _run(self) -> None:
        # A teleoperation session can start after this app; keep observing with a fixed retry pause.
        while not self._stop.is_set():
            process = None
            try:
                process = subprocess.Popen(
                    [self.studio_python, "-u", str(self.worker_path), "--session", self.session,
                     "--read-timeout", str(self.read_timeout)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=None, bufsize=0)
                self._process = process
                with self.changed:
                    self._packet = None  # sequence numbers restart with each observer process
                self._consume(process)
            except Exception as error:
                self.fail(str(error))
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                    process.stdout.close()
                self._process = None
            self._stop.wait(self.retry_interval)

    def close(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._thread:
            self._thread.join(timeout=4)
        self.fail("Studio robot observer stopped")
