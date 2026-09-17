import importlib.util
import json
import re
from pathlib import Path
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from urllib.request import urlopen

from secondlook.runtime import CameraSource, InspectionRuntime
from secondlook.studio_robot import JOINTS, RobotProtocolError, StudioRobotSource
from secondlook.web import make_server

SESSION = "rt-3f2c9a"
spec = importlib.util.spec_from_file_location("studio_robot_worker", "scripts/studio_robot_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

# Stands in for the zenoh observer: prints valid joint packets, then an error packet, like the real worker.
STAND_IN = textwrap.dedent(f"""
    import json, sys, time
    for sequence in range(1, {{count}} + 1):
        print(json.dumps({{{{"protocol": 1, "type": "joints", "session": "{SESSION}", "timestamp": time.monotonic(),
                          "sequence": sequence, "joint_names": {list(JOINTS)!r},
                          "state": [float(sequence), -10.0, 20.0, 0.0, 5.0, 40.0], "action": None}}}}), flush=True)
        time.sleep(1 / 30)
    time.sleep({{hold}})
    print(json.dumps({{{{"protocol": 1, "type": "error", "error": "Studio runtime session shut down (idle)"}}}}), flush=True)
    sys.exit(2)
""")


def tick(data, actions=None):
    return {"instance_id": "x", "event": {"event": "observation", "data": data, "actions": actions}, "_fatal": False}


class WorkerContractTests(unittest.TestCase):
    pose = {f"{name}.pos": float(index) for index, name in enumerate(JOINTS)}

    def test_session_port_matches_studio_derivation(self):
        port = worker.session_port(SESSION)
        self.assertTrue(10000 <= port < 20000)
        self.assertEqual(port, worker.session_port(SESSION))
        self.assertNotEqual(port, worker.session_port("rt-other"))

    def test_tick_packet_orders_joints_and_keeps_missing_actions_empty(self):
        packet = tick_packet = worker.tick_packet(tick(dict(reversed(self.pose.items()))), SESSION, 4.0, 7)
        self.assertEqual(packet["state"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertIsNone(tick_packet["action"])
        self.assertEqual(worker.tick_packet(tick(self.pose, self.pose), SESSION, 4.0, 8)["action"], packet["state"])

    def test_unexpected_tick_shapes_fail_instead_of_guessing_a_pose(self):
        missing = {key: value for key, value in self.pose.items() if not key.startswith("gripper")}
        for payload in (tick(missing), tick({**self.pose, "elbow_flex.pos": float("nan")}),
                        tick({**self.pose, "wrist_roll.pos": "3"}), {"event": {"event": "state"}}, []):
            with self.assertRaises(ValueError):
                worker.tick_packet(payload, SESSION, 1.0, 1)

    def test_worker_only_subscribes_to_tick_and_lifecycle(self):
        # Studio counts subscribers on the state key as attached clients, so observing it would
        # keep a session from switching to hold when the operator leaves.
        source = Path("scripts/studio_robot_worker.py").read_text()
        self.assertEqual(set(re.findall(r"\bsession\.(\w+)\(", source)), {"declare_subscriber", "close"})
        self.assertEqual(re.findall(r'declare_subscriber\(f"\{prefix\}/(\w+)"', source), ["tick", "lifecycle"])


class StudioRobotSourceTests(unittest.TestCase):
    def source(self, tmp, count, hold, **options):
        script = Path(tmp) / "stand_in.py"
        script.write_text(STAND_IN.format(count=count, hold=hold))
        return StudioRobotSource(SESSION, sys.executable, worker_path=script, **options)

    def wait_for(self, source, status, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = source.latest()
            if state["status"] == status:
                return state
            time.sleep(.02)
        self.fail(f"source never reported {status}: {source.latest()}")

    def test_live_packets_then_worker_error_becomes_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.source(tmp, count=10, hold=.3, retry_interval=30)
            source.start()
            try:
                live = self.wait_for(source, "LIVE")
                self.assertEqual(live["joint_names"], list(JOINTS))
                self.assertEqual(len(live["state"]), 6)
                gone = self.wait_for(source, "UNAVAILABLE")
                self.assertIn("shut down", gone["error"])
                self.assertNotIn("state", gone)
            finally:
                source.close()

    def test_silent_worker_goes_stale_without_holding_the_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.source(tmp, count=3, hold=5, max_age=.2, retry_interval=30)
            source.start()
            try:
                self.wait_for(source, "LIVE")
                stale = self.wait_for(source, "UNAVAILABLE", timeout=2)
                self.assertIn("No Studio joint telemetry", stale["error"])
            finally:
                source.close()

    def test_parse_rejects_wrong_session_stale_and_reversed_packets(self):
        source = StudioRobotSource(SESSION, sys.executable)
        good = {"protocol": 1, "type": "joints", "session": SESSION, "timestamp": 10.0, "sequence": 2,
                "joint_names": list(JOINTS), "state": [0.0] * 6, "action": None}
        source.publish(source.parse(json.dumps(good).encode(), 10.1), 10.1)
        for change, now in (({"session": "rt-other"}, 10.1), ({"sequence": 3}, 11.0), ({"sequence": 2}, 10.1),
                            ({"sequence": 3, "state": [0.0] * 5}, 10.1),
                            ({"sequence": 3, "action": [float("inf")] * 6}, 10.1)):
            with self.assertRaises(RobotProtocolError):
                source.parse(json.dumps({**good, **change}).encode(), now)
        with self.assertRaises(ValueError):
            StudioRobotSource("follower/../x", sys.executable)


class RobotRouteTests(unittest.TestCase):
    def test_latest_and_stream_serve_live_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = InspectionRuntime(CameraSource(max_age=1),
                                        evidence_path=Path(tmp) / "e.jsonl", evidence_kind="mock")
            source = StudioRobotSourceTests.source(None, tmp, count=60, hold=.2, retry_interval=30)
            server = make_server(runtime, port=0, robot=source)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_port}"
            source.start()
            try:
                with urlopen(base + "/api/robot/stream", timeout=5) as stream:
                    self.assertEqual(stream.headers["Content-Type"], "text/event-stream")
                    events, deadline = [], time.monotonic() + 8
                    while time.monotonic() < deadline:
                        line = stream.readline().decode()
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))
                            if events[-1]["status"] == "LIVE" and len(events) > 3:
                                break
                        else:
                            self.assertEqual(line, "\n")
                live = [event for event in events if event["status"] == "LIVE"]
                self.assertTrue(live, events)
                sequences = [event["sequence"] for event in live]
                self.assertEqual(sequences, sorted(set(sequences)))
                latest = json.load(urlopen(base + "/api/robot/latest", timeout=5))
                self.assertEqual(latest["session"], SESSION)
            finally:
                source.close()
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
