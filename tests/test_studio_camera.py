import base64
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

from secondlook.studio_camera import (CameraProtocolError, MAX_LINE_BYTES,
                                     StudioCameraSource, jpeg_dimensions)


# Minimal JPEG header for parser tests. Image decoding is explicitly mocked.
JPEG_HEADER = bytes.fromhex("ffd8 ffc0 0011 08 0001 0001 03 01 11 00 02 11 00 03 11 00 ffd9")
SERVICE = "physicalai/camera/fixture/123/frame"


class FakeImage:
    shape = (1, 1, 3)
    size = 3
    dtype = "uint8"


class StudioCameraTests(unittest.TestCase):
    def source(self, **kwargs):
        return StudioCameraSource(SERVICE, sys.executable,
                                  image_decoder=lambda _: FakeImage(), **kwargs)

    def packet(self, **changes):
        packet = {"protocol": 1, "type": "frame", "source": SERVICE,
                  "timestamp": 99.8, "sequence": 5, "color": "BGR",
                  "width": 1, "height": 1,
                  "jpeg_base64": base64.b64encode(JPEG_HEADER).decode()}
        packet.update(changes)
        return json.dumps(packet).encode()

    def test_preserves_capture_time_sequence_and_source(self):
        source = self.source()
        frame = source.parse_frame(self.packet(), now=100)
        self.assertEqual(frame.captured_at, 99.8)
        self.assertEqual(frame.source_sequence, 5)
        self.assertIn(SERVICE, frame.observation_id)
        source.publish(frame)
        self.assertIs(source.latest(now=100), frame)
        self.assertIsNone(source.latest(now=101))
        self.assertEqual(source.transport, "studio_shared_memory_attach_only")

    def test_rejects_stale_future_duplicate_reversed_and_bool_metadata(self):
        for fields in ({"timestamp": 90}, {"timestamp": 101}, {"timestamp": -1},
                       {"timestamp": True}, {"timestamp": float("nan")},
                       {"sequence": True}, {"sequence": -1}):
            with self.subTest(fields=fields), self.assertRaises(CameraProtocolError):
                self.source().parse_frame(self.packet(**fields), now=100)
        source = self.source()
        source.parse_frame(self.packet(), now=100)
        for fields in ({}, {"timestamp": 99.9}, {"timestamp": 99.7, "sequence": 6}):
            with self.subTest(fields=fields), self.assertRaises(CameraProtocolError):
                source.parse_frame(self.packet(**fields), now=100)
        source._last_sequence = -1  # New helper epoch permits sequence reset, not stale capture time.
        with self.assertRaises(CameraProtocolError):
            source.parse_frame(self.packet(sequence=0), now=100)
        frame = source.parse_frame(self.packet(timestamp=99.9, sequence=0), now=100)
        self.assertEqual(frame.source_sequence, 0)

    def test_rejects_malformed_oversized_and_wrong_source_images(self):
        malformed = [b"", b"not json", b"[]", b"x" * (MAX_LINE_BYTES + 1),
                     self.packet(source="another-publisher"), self.packet(color="RGB"),
                     self.packet(width=99999), self.packet(width=2),
                     self.packet(jpeg_base64="not-base64!"),
                     self.packet(jpeg_base64=base64.b64encode(b"notjpeg").decode()),
                     self.packet(type="unexpected")]
        for line in malformed:
            with self.subTest(length=len(line)), self.assertRaises(CameraProtocolError):
                self.source().parse_frame(line, now=100)
        self.assertEqual(jpeg_dimensions(JPEG_HEADER), (1, 1))

    def run_fixture(self, code, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        worker = Path(temporary.name) / "fake_worker.py"
        worker.write_text(code)
        source = self.source(worker_path=worker, attempts=1, reconnect_interval=.01, **kwargs)
        source.start()
        self.addCleanup(source.close)
        return source

    def test_missing_publisher_and_helper_exit_are_unavailable(self):
        fixtures = ["import json; print(json.dumps({'type':'error','error':'No publisher responded'}),flush=True)",
                    "raise SystemExit(7)"]
        for fixture in fixtures:
            source = self.run_fixture(fixture, startup_timeout=.5)
            source._thread.join(timeout=2)
            self.assertFalse(source._thread.is_alive())
            self.assertIsNone(source.latest())
            self.assertEqual(source.status()["status"], "UNAVAILABLE")
            self.assertIn("reconnect budget exhausted", source.error)

    def test_hung_helper_is_bounded_and_terminated(self):
        source = self.run_fixture("import time; time.sleep(60)", startup_timeout=.1)
        source._thread.join(timeout=2)
        self.assertFalse(source._thread.is_alive())
        self.assertIn("timed out", source.error)
        self.assertIsNone(source._process)

    def test_read_hang_after_valid_frame_clears_live_and_reaps_helper(self):
        packet = json.loads(self.packet())
        code = ("import json,time\n"
                f"packet={packet!r}\n"
                "packet['timestamp']=time.monotonic()\n"
                "print(json.dumps(packet),flush=True)\n"
                "time.sleep(60)\n")
        source = self.run_fixture(code, startup_timeout=.5, read_timeout=.15)
        deadline = time.monotonic() + .4
        while source.latest() is None and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertIsNotNone(source.latest())
        source._thread.join(timeout=2)
        self.assertFalse(source._thread.is_alive())
        self.assertIsNone(source.latest())
        self.assertIn("timed out", source.error)
        self.assertIsNone(source._process)

    def test_close_terminates_only_owned_helper(self):
        source = self.run_fixture("import time; time.sleep(60)")
        deadline = time.monotonic() + 1
        while source._process is None and time.monotonic() < deadline:
            time.sleep(.01)
        process = source._process
        self.assertIsNotNone(process)
        source.close()
        self.assertIsNotNone(process.poll())
        self.assertFalse(source._thread.is_alive())
        self.assertEqual(source.status()["status"], "UNAVAILABLE")

    def test_malformed_stream_never_publishes(self):
        source = self.run_fixture("print('not JSON',flush=True)", startup_timeout=.5)
        source._thread.join(timeout=2)
        self.assertIsNone(source.latest())
        self.assertIn("Malformed", source.error)

    def test_worker_uses_header_color_and_original_timestamp(self):
        path = Path(__file__).resolve().parents[1] / "scripts/studio_camera_worker.py"
        spec = importlib.util.spec_from_file_location("studio_worker_fixture", path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        frame = SimpleNamespace(timestamp=99.8, sequence=5, data=FakeImage())
        header = SimpleNamespace(timestamp_ns=99800000000, sequence=5,
                                 color_mode=0, dtype=0, width=1, height=1)
        seen = []
        def encode(data, color):
            seen.append(color)
            return JPEG_HEADER
        packet = worker.frame_packet(frame, header, SERVICE, 100, 1, encode)
        self.assertEqual(seen, ["RGB"])
        self.assertEqual(packet["timestamp"], 99.8)
        self.assertEqual(packet["publisher_color"], "RGB")
        self.assertEqual(packet["color"], "BGR")
        header.color_mode = 99
        with self.assertRaises(ValueError):
            worker.frame_packet(frame, header, SERVICE, 100, 1, encode)
        header.color_mode, header.width = 0, 2
        with self.assertRaises(ValueError):
            worker.frame_packet(frame, header, SERVICE, 100, 1, encode)


if __name__ == "__main__":
    unittest.main()
