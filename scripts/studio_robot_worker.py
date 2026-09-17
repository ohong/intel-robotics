#!/usr/bin/env python3
"""Observe one Studio runtime session's joint telemetry; never command or attach as a client.

Studio's runtime session publishes msgpack events on ``studio/rt/<session>/*``.
This worker subscribes only to ``tick`` (joint observations and action targets)
and ``lifecycle``. It never subscribes to ``state``: Studio counts ``state``
subscribers as attached clients, and that count drives its switch to hold and its
idle shutdown when the operator's UI leaves. It declares no publisher or query.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import sys
import threading
import time

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SESSION = re.compile(r"^rt-[A-Za-z0-9_-]+$")
KEY_PREFIX = "studio/rt"
MAX_PAYLOAD_BYTES = 1024 * 1024


def session_port(session: str) -> int:
    """Studio's deterministic loopback port (runtime/transport/ids.py derive_endpoint_port)."""
    digest = hashlib.sha256(f"{KEY_PREFIX}/{session}".encode()).digest()
    return 10000 + int.from_bytes(digest[:4], "big") % 10000


def joint_vector(values, label: str) -> list[float]:
    if not isinstance(values, dict):
        raise ValueError(f"Studio tick {label} must be a mapping")
    vector = [values.get(f"{name}.pos") for name in JOINTS]
    if any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1000 for value in vector):
        raise ValueError(f"Studio tick {label} lacks finite SO101 joint positions")
    return [float(value) for value in vector]


def tick_packet(payload, session: str, now: float, sequence: int) -> dict:
    """Validate one decoded runtime envelope; unexpected shapes fail rather than guess a pose."""
    event = payload.get("event") if isinstance(payload, dict) else None
    if not isinstance(event, dict) or event.get("event") != "observation":
        raise ValueError("Studio tick is not a runtime observation event")
    actions = event.get("actions")
    return {"protocol": 1, "type": "joints", "session": session, "timestamp": now, "sequence": sequence,
            "joint_names": list(JOINTS), "state": joint_vector(event.get("data"), "data"),
            "action": None if actions is None else joint_vector(actions, "actions")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Studio runtime session name, rt-<follower id>")
    parser.add_argument("--hz", type=float, default=30.)
    parser.add_argument("--read-timeout", type=float, default=3.)
    args = parser.parse_args()
    if not SESSION.fullmatch(args.session) or not 1 <= args.hz <= 60 or not 0 < args.read_timeout <= 30:
        parser.error("A Studio session name rt-<id> and bounded timing are required")
    # Private protocol FD first, so native zenoh logging on stdout cannot corrupt JSON lines.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    lock = threading.Lock()
    latest: dict = {"payload": None, "count": 0, "error": None}
    session = None
    try:
        import msgpack
        import zenoh

        def decode(sample):
            data = sample.payload.to_bytes()
            if len(data) > MAX_PAYLOAD_BYTES:
                raise ValueError("Studio telemetry payload exceeds 1 MiB")
            return msgpack.unpackb(data, raw=False)

        def on_tick(sample):
            try:
                payload = decode(sample)
                with lock:
                    latest["payload"], latest["count"] = payload, latest["count"] + 1
            except Exception as error:
                latest["error"] = f"Undecodable Studio tick: {error}"
                stop.set()

        def on_lifecycle(sample):
            try:
                event = decode(sample).get("event", {})
                if event.get("data", {}).get("event") == "shutdown":
                    latest["error"] = f"Studio runtime session shut down ({event['data'].get('reason')})"
                    stop.set()
            except Exception as error:
                latest["error"] = f"Undecodable Studio lifecycle event: {error}"
                stop.set()

        config = zenoh.Config()
        config.insert_json5("mode", '"peer"')
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("scouting/gossip/enabled", "false")
        config.insert_json5("connect/endpoints", f'["tcp/127.0.0.1:{session_port(args.session)}"]')
        session = zenoh.open(config)
        prefix = f"{KEY_PREFIX}/{args.session}"
        # Hold references for the session lifetime; closing the session undeclares them.
        subscribers = [session.declare_subscriber(f"{prefix}/tick", on_tick),
                       session.declare_subscriber(f"{prefix}/lifecycle", on_lifecycle)]
        sent, deadline = 0, time.monotonic() + args.read_timeout
        while not stop.is_set():
            started = time.monotonic()
            with lock:
                payload, count = latest["payload"], latest["count"]
            if count != sent:
                packet = tick_packet(payload, args.session, time.monotonic(), count)
                protocol.write(json.dumps(packet, separators=(",", ":"), allow_nan=False) + "\n")
                sent, deadline = count, time.monotonic() + args.read_timeout
            elif started >= deadline:
                raise TimeoutError("No Studio joint telemetry; is a teleoperation session running?")
            stop.wait(max(0., 1 / args.hz - (time.monotonic() - started)))
        if latest["error"]:
            raise RuntimeError(latest["error"])
        return 0
    except Exception as error:
        try:
            protocol.write(json.dumps({"protocol": 1, "type": "error", "error": str(error)[:500]}) + "\n")
        except (BrokenPipeError, OSError):
            pass
        print(f"Studio robot observer stopped: {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        if session is not None:
            session.close()
        protocol.close()


if __name__ == "__main__":
    raise SystemExit(main())
