#!/usr/bin/env python3
"""Attach to one existing Studio camera publisher; never create or configure it."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import signal
import sys
import threading
import time

MAX_JPEG_BYTES = 2 * 1024 * 1024


def frame_packet(frame, header, service: str, now: float, max_age: float, encode) -> dict:
    """Header color describes actual publisher bytes, not subscriber preference."""
    timestamp, sequence = frame.timestamp, frame.sequence
    if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0 or
            not 0 <= now - timestamp <= max_age or type(sequence) is not int or sequence < 0):
        raise ValueError("Publisher returned a stale or malformed capture timestamp/sequence")
    if header is None or header.sequence != sequence or abs(header.timestamp_ns / 1e9 - timestamp) > 1e-6:
        raise ValueError("Publisher color header does not match the captured frame")
    colors = {0: "RGB", 1: "BGR", 2: "GRAY"}
    color = colors.get(header.color_mode)
    if color is None or header.dtype != 0:
        raise ValueError("Publisher must provide a supported uint8 color image")
    width, height = int(header.width), int(header.height)
    expected = (height, width) if color == "GRAY" else (height, width, 3)
    if (width <= 0 or height <= 0 or width > 4096 or height > 2160 or
            tuple(frame.data.shape) != expected or str(frame.data.dtype) != "uint8"):
        raise ValueError("Publisher image dimensions or dtype do not match its header")
    jpeg = encode(frame.data, color)
    if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
        raise ValueError("Encoded publisher image exceeds payload limit")
    return {"protocol": 1, "type": "frame", "source": service,
            "timestamp": timestamp, "sequence": sequence, "color": "BGR",
            "publisher_color": color, "width": width, "height": height,
            "jpeg_base64": base64.b64encode(jpeg).decode("ascii")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--connect-timeout", type=float, default=3.)
    parser.add_argument("--frame-max-age", type=float, default=1.)
    parser.add_argument("--read-timeout", type=float, default=2.)
    args = parser.parse_args()
    if (not args.service.strip() or not 1 <= args.fps <= 15 or
            any(not math.isfinite(x) or x <= 0 for x in
                (args.connect_timeout, args.frame_max_age, args.read_timeout))):
        parser.error("Service name and bounded positive camera timing are required")
    # Reserve a private protocol FD before importing native libraries. Both native
    # and Python stdout logging goes to stderr, so it cannot corrupt JSON framing.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    camera = None
    try:
        import cv2
        from physicalai.capture import SharedCamera

        def encode(data, color):
            bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR) if color == "RGB" else (
                cv2.cvtColor(data, cv2.COLOR_GRAY2BGR) if color == "GRAY" else data)
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("Could not encode shared camera frame")
            return encoded.tobytes()

        camera = SharedCamera.from_publisher(args.service, overwrite_settings=False, zero_copy=False)
        camera.connect(timeout=args.connect_timeout)
        previous_timestamp, previous_sequence = float("-inf"), -1
        deadline = time.monotonic() + args.read_timeout
        while not stop.is_set():
            tick = time.monotonic()
            frame = camera.read_latest()
            if frame.timestamp > previous_timestamp and frame.sequence > previous_sequence:
                packet = frame_packet(frame, camera._last_header, args.service, time.monotonic(),
                                      args.frame_max_age, encode)
                protocol.write(json.dumps(packet, separators=(",", ":"), allow_nan=False) + "\n")
                previous_timestamp, previous_sequence = frame.timestamp, frame.sequence
                deadline = time.monotonic() + args.read_timeout
            elif tick >= deadline:
                raise TimeoutError("Publisher stopped delivering new capture timestamps")
            stop.wait(max(0., 1 / args.fps - (time.monotonic() - tick)))
        return 0
    except Exception as error:
        try:
            protocol.write(json.dumps({"protocol": 1, "type": "error", "error": str(error)[:500]}) + "\n")
        except (BrokenPipeError, OSError):
            pass
        print(f"Studio camera subscriber stopped: {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        if camera is not None:
            camera.disconnect()
        protocol.close()


if __name__ == "__main__":
    raise SystemExit(main())
