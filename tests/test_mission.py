import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from secondlook.runtime import CameraSource, InspectionRuntime
from secondlook.web import byte_range, make_server

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW = True
except ImportError:
    PYARROW = False

SNAPSHOT = Path("artifacts/recording/pilot-five-20260916T004831Z/dataset")
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
CAMERA = "observation.images.cam"


def write_dataset(root: Path, lengths=(3, 2)) -> None:
    """Write a minimal LeRobot v3 dataset: real parquet files, fake video bytes."""
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    video = root / f"videos/{CAMERA}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(bytes(range(256)) * 4)
    info = {"codebase_version": "v3.0", "fps": 30, "total_episodes": len(lengths),
            "total_frames": sum(lengths), "robot_type": "SO101 Follower",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {"action": {"dtype": "float32", "names": JOINTS},
                         "observation.state": {"dtype": "float32", "names": JOINTS},
                         CAMERA: {"dtype": "video"}}}
    (root / "meta/info.json").write_text(json.dumps(info))
    episodes, frames, offset = [], [], 0.0
    for index, length in enumerate(lengths):
        episodes.append({"episode_index": index, "tasks": [f"task {index}"], "length": length,
                         f"videos/{CAMERA}/chunk_index": 0, f"videos/{CAMERA}/file_index": 0,
                         f"videos/{CAMERA}/from_timestamp": offset,
                         f"videos/{CAMERA}/to_timestamp": offset + length / 30})
        offset += length / 30
        for frame in reversed(range(length)):
            frames.append({"episode_index": index, "frame_index": frame,
                           "observation.state": [float(frame)] * 6, "action": [frame + .5] * 6})
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(frames), root / "data/chunk-000/file-000.parquet")


class ByteRangeTests(unittest.TestCase):
    def test_parses_open_closed_and_suffix_ranges(self):
        self.assertIsNone(byte_range(None, 100))
        self.assertEqual(byte_range("bytes=0-", 100), (0, 99))
        self.assertEqual(byte_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(byte_range("bytes=90-500", 100), (90, 99))
        self.assertEqual(byte_range("bytes=-5", 100), (95, 99))

    def test_rejects_unsatisfiable_range(self):
        with self.assertRaises(ValueError):
            byte_range("bytes=100-", 100)


@unittest.skipUnless(PYARROW, "pyarrow not installed (cv group)")
class ReplayTests(unittest.TestCase):
    def setUp(self):
        from secondlook.replay import ReplayDataset

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "dataset"
        write_dataset(root)
        self.replay = ReplayDataset(root)
        runtime = InspectionRuntime(CameraSource(max_age=1),
                                    evidence_path=Path(self.tmp.name) / "evidence.jsonl", evidence_kind="mock")
        self.server = make_server(runtime, port=0, replay=self.replay)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def get(self, path, headers=None):
        return urlopen(Request(self.base + path, headers=headers or {}), timeout=5)

    def test_episode_frames_are_ordered_and_labelled_replay(self):
        summary = json.load(self.get("/api/replay/episodes"))
        self.assertEqual(summary["source"], "REPLAY")
        self.assertEqual([e["length"] for e in summary["episodes"]], [3, 2])
        episode = json.load(self.get("/api/replay/episodes/0"))
        self.assertEqual(episode["joint_names"][0], "shoulder_pan")
        self.assertEqual([row[0] for row in episode["state"]], [0, 1, 2])
        self.assertEqual(episode["action"][2], [2.5] * 6)
        self.assertEqual(json.load(self.get("/api/replay/episodes/1"))["videos"][0]["from_timestamp"], .1)

    def test_video_supports_range_requests(self):
        url = json.load(self.get("/api/replay/episodes/1"))["videos"][0]["url"]
        whole = self.get(url)
        self.assertEqual((whole.status, len(whole.read())), (200, 1024))
        part = self.get(url, {"Range": "bytes=10-13"})
        self.assertEqual(part.status, 206)
        self.assertEqual(part.headers["Content-Range"], "bytes 10-13/1024")
        self.assertEqual(part.read(), bytes([10, 11, 12, 13]))

    def test_unknown_episode_video_and_bad_range_are_rejected(self):
        for path, headers, code in (("/api/replay/episodes/9", {}, 404),
                                    ("/api/replay/video/0/0/1", {}, 404),
                                    ("/api/replay/video/1/0/0", {}, 404),
                                    ("/api/replay/video/0/0/0", {"Range": "bytes=5000-"}, 416)):
            with self.assertRaises(HTTPError) as caught:
                self.get(path, headers)
            self.assertEqual(caught.exception.code, code, path)

    def test_inconsistent_metadata_fails_loudly(self):
        from secondlook.replay import ReplayDataset, ReplayError

        root = Path(self.tmp.name) / "broken"
        write_dataset(root)
        info = json.loads((root / "meta/info.json").read_text())
        info["total_frames"] = 99
        (root / "meta/info.json").write_text(json.dumps(info))
        with self.assertRaises(ReplayError):
            ReplayDataset(root)

    @unittest.skipUnless(SNAPSHOT.is_dir(), "pilot recording snapshot not present")
    def test_pilot_snapshot_matches_recorded_counts(self):
        from secondlook.replay import ReplayDataset

        real = ReplayDataset(SNAPSHOT)
        summary = real.summary()
        self.assertEqual(sum(e["length"] for e in summary["episodes"]), 3249)
        self.assertEqual(len(summary["cameras"]), 2)
        self.assertTrue(all(len(row) == 6 for row in real.episode(4)["state"]))


class EvidenceTailTests(unittest.TestCase):
    def test_recent_evidence_returns_newest_records_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = InspectionRuntime(CameraSource(max_age=1),
                                        evidence_path=Path(tmp) / "e.jsonl", evidence_kind="mock")
            self.assertEqual(runtime.evidence.tail(5), [])
            for index in range(30):
                runtime.evidence.append({"event": "fixture", "evidence_kind": "mock",
                                         "observation_id": f"obs-{index}", "provenance": {"test": True},
                                         "outcome": "NOT_TESTED"})
            self.assertEqual([r["observation_id"] for r in runtime.evidence.tail(3, max_bytes=1000)],
                             ["obs-27", "obs-28", "obs-29"])
            server = make_server(runtime, port=0)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_port}/api/evidence/recent"
            try:
                records = json.load(urlopen(base + "?n=2", timeout=5))["records"]
                self.assertEqual([r["observation_id"] for r in records], ["obs-28", "obs-29"])
                self.assertEqual(records[0]["evidence_kind"], "mock")
                with self.assertRaises(HTTPError) as caught:
                    urlopen(base + "?n=x", timeout=5)
                self.assertEqual(caught.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()


class ReplayDisabledTests(unittest.TestCase):
    def test_replay_routes_404_without_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = InspectionRuntime(CameraSource(max_age=1),
                                        evidence_path=Path(tmp) / "e.jsonl", evidence_kind="mock")
            server = make_server(runtime, port=0)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                with self.assertRaises(HTTPError) as caught:
                    urlopen(f"http://127.0.0.1:{server.server_port}/api/replay/episodes", timeout=5)
                self.assertEqual(caught.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()


class MissionStaticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        runtime = InspectionRuntime(CameraSource(max_age=1),
                                    evidence_path=Path(self.tmp.name) / "e.jsonl", evidence_kind="mock")
        self.server = make_server(runtime, port=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_mission_page_and_vendored_modules_are_served(self):
        page = urlopen(self.base + "/mission", timeout=5).read().decode()
        self.assertIn('type="importmap"', page)
        module = urlopen(self.base + "/static/vendor/three/three.module.min.js", timeout=5)
        self.assertEqual(module.headers["Content-Type"], "text/javascript")
        self.assertIn(b"three.core.min.js", module.read())

    def test_every_element_id_used_by_mission_modules_exists_in_page(self):
        static = Path(__file__).resolve().parents[1] / "secondlook" / "static"
        page_ids = set(re.findall(r'\bid="([^"]+)"', (static / "mission.html").read_text()))
        for module in sorted((static / "mission").glob("*.js")):
            used = set(re.findall(r"(?:\bel|getElementById)\('([^']+)'\)", module.read_text()))
            self.assertFalse(used - page_ids, f"{module.name} references missing ids")

    def test_static_route_rejects_traversal_and_unknown_types(self):
        for path in ("/static/../web.py", "/static/%2e%2e/web.py", "/static/vendor/README.md",
                     "/static/vendor/three/LICENSE", "/static/missing.js"):
            with self.assertRaises(HTTPError) as caught:
                urlopen(self.base + path, timeout=5)
            self.assertEqual(caught.exception.code, 404, path)
