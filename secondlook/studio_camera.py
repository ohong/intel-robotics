"""Bounded subscriber bridge to Studio's existing shared camera publisher.

The helper runs in Studio's Python environment. It only attaches to a named
publisher; the Intel runtime never imports Studio or opens that camera device.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time
from typing import Callable

from .runtime import CameraSource, Frame


MAX_JPEG_BYTES = 2 * 1024 * 1024
MAX_LINE_BYTES = (MAX_JPEG_BYTES * 4 // 3) + 8192
MAX_PIXELS = 4096 * 2160


class CameraProtocolError(ValueError):
    pass


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Read JPEG dimensions before an image decoder can allocate its output."""
    if not data.startswith(b"\xff\xd8"):
        raise CameraProtocolError("Invalid JPEG signature")
    offset = 2
    starts_of_frame = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                       0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            raise CameraProtocolError("Malformed JPEG marker")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xD8, 0xD9, 0xDA):
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        size = int.from_bytes(data[offset:offset + 2], "big")
        if size < 2 or offset + size > len(data):
            raise CameraProtocolError("Truncated JPEG segment")
        if marker in starts_of_frame:
            if size < 8:
                raise CameraProtocolError("Invalid JPEG dimensions segment")
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            return width, height
        offset += size
    raise CameraProtocolError("JPEG has no bounded image dimensions")


def decode_jpeg(data: bytes):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or not image.size:
        raise CameraProtocolError("JPEG decode failed")
    image.flags.writeable = False
    return image


class StudioCameraSource(CameraSource):
    def __init__(self, service_name: str, studio_python: str, max_age: float = 1.0,
                 *, read_timeout: float = 2.0, startup_timeout: float = 15.0,
                 connect_timeout: float = 3.0, attempts: int = 3,
                 reconnect_interval: float = 1.0, fps: float = 10.0,
                 worker_path: Path | None = None, image_decoder: Callable = decode_jpeg):
        super().__init__(device=f"studio:{service_name}", max_age=max_age)
        if not service_name.strip() or len(service_name) > 512 or any(c in service_name for c in "\x00\r\n"):
            raise ValueError("An explicit Studio publisher service name is required")
        bounds = (read_timeout, startup_timeout, connect_timeout, reconnect_interval, fps)
        if (any(not math.isfinite(x) or x <= 0 for x in bounds) or not 1 <= fps <= 15 or
                type(attempts) is not int or not 1 <= attempts <= 5):
            raise ValueError("Invalid shared-camera timing or retry bounds")
        self.service_name, self.studio_python = service_name, studio_python
        self.read_timeout, self.startup_timeout = read_timeout, startup_timeout
        self.connect_timeout, self.attempts = connect_timeout, attempts
        self.reconnect_interval, self.fps = reconnect_interval, fps
        self.worker_path = worker_path or Path(__file__).resolve().parents[1] / "scripts/studio_camera_worker.py"
        self.image_decoder = image_decoder
        self.timestamp_source = "Studio publisher capture time.monotonic; UTC derived from same-host clock offset"
        self.transport = "studio_shared_memory_attach_only"
        self._process: subprocess.Popen | None = None
        self._process_lock = threading.Lock()
        self._last_timestamp = float("-inf")
        self._last_sequence = -1
        self._attempt = 0

    def parse_frame(self, line: bytes, now: float | None = None) -> Frame:
        if not line or len(line) > MAX_LINE_BYTES:
            raise CameraProtocolError("Empty or oversized shared-camera packet")
        try:
            packet = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise CameraProtocolError("Malformed shared-camera JSON") from error
        if not isinstance(packet, dict):
            raise CameraProtocolError("Shared-camera packet must be an object")
        if packet.get("type") == "error":
            raise CameraProtocolError(f"Studio camera helper: {str(packet.get('error', 'unavailable'))[:500]}")
        if packet.get("type") != "frame" or packet.get("protocol") != 1:
            raise CameraProtocolError("Unsupported shared-camera packet")
        if packet.get("source") != self.service_name or packet.get("color") != "BGR":
            raise CameraProtocolError("Shared-camera source or output color does not match")
        timestamp, sequence = packet.get("timestamp"), packet.get("sequence")
        if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0 or
                type(sequence) is not int or sequence < 0):
            raise CameraProtocolError("Invalid capture timestamp or sequence")
        clock = time.monotonic() if now is None else now
        age = clock - timestamp
        if not math.isfinite(age) or age < 0 or age > self.max_age:
            raise CameraProtocolError("Shared-camera frame is stale or from a future clock")
        if timestamp <= self._last_timestamp or sequence <= self._last_sequence:
            raise CameraProtocolError("Duplicate or reversed shared-camera frame")
        width, height = packet.get("width"), packet.get("height")
        if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0 or
                width > 4096 or height > 2160 or width * height > MAX_PIXELS):
            raise CameraProtocolError("Shared-camera frame dimensions exceed limits")
        encoded = packet.get("jpeg_base64")
        if not isinstance(encoded, str) or len(encoded) > MAX_JPEG_BYTES * 4 // 3 + 4:
            raise CameraProtocolError("Missing or oversized JPEG payload")
        try:
            jpeg = base64.b64decode(encoded, validate=True)
        except (ValueError, UnicodeError) as error:
            raise CameraProtocolError("Malformed JPEG base64") from error
        if len(jpeg) > MAX_JPEG_BYTES or jpeg_dimensions(jpeg) != (width, height):
            raise CameraProtocolError("JPEG dimensions or payload size do not match")
        image = self.image_decoder(jpeg)
        if getattr(image, "shape", None) != (height, width, 3):
            raise CameraProtocolError("Decoded JPEG shape does not match")
        self._last_timestamp, self._last_sequence = timestamp, sequence
        # Preserve capture time. UTC is an explicitly approximate clock conversion.
        captured_utc = datetime.fromtimestamp(time.time() - age, timezone.utc).isoformat()
        return Frame(f"{self.service_name}:{sequence}:{timestamp:.9f}", timestamp,
                     captured_utc, image, source_sequence=sequence)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if process.stdout:
            process.stdout.close()

    def _consume(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        buffer = bytearray()
        deadline = time.monotonic() + self.startup_timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraProtocolError("Shared-camera helper timed out without a fresh frame")
                if not selector.select(min(.1, remaining)):
                    if process.poll() is not None:
                        raise CameraProtocolError(f"Shared-camera helper exited ({process.returncode})")
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise CameraProtocolError("Shared-camera helper exited or closed its stream")
                buffer.extend(chunk)
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    frame = self.parse_frame(bytes(line))
                    if not self._stop.is_set():
                        self.publish(frame)
                    deadline = time.monotonic() + self.read_timeout
                if len(buffer) > MAX_LINE_BYTES:
                    raise CameraProtocolError("Shared-camera helper exceeded packet size limit")

    def _read(self) -> None:
        for attempt in range(1, self.attempts + 1):
            if self._stop.is_set():
                return
            self._attempt = attempt
            self._last_sequence = -1  # A restarted publisher can reset its sequence.
            process = None
            try:
                command = [self.studio_python, "-u", str(self.worker_path),
                           "--service", self.service_name, "--fps", str(self.fps),
                           "--connect-timeout", str(self.connect_timeout),
                           "--frame-max-age", str(self.max_age),
                           "--read-timeout", str(self.read_timeout)]
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=None, bufsize=0)
                with self._process_lock:
                    self._process = process
                self._consume(process)
            except Exception as error:
                self.fail(f"{error} (attach attempt {attempt}/{self.attempts})")
            finally:
                if process is not None:
                    self._terminate(process)
                with self._process_lock:
                    self._process = None
            if self._stop.wait(self.reconnect_interval):
                return
        self.fail(f"{self.error}; reconnect budget exhausted")

    def status(self) -> dict:
        result = super().status()
        result.update(service_name=self.service_name, attach_attempt=self._attempt,
                      attach_attempt_limit=self.attempts, publisher_reconfigured=False)
        return result

    def close(self) -> None:
        self._stop.set()
        with self._process_lock:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
        if self._thread:
            self._thread.join(timeout=4)
        self.fail("Studio camera subscriber stopped")
