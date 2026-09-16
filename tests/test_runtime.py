import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

from secondlook.runtime import (CameraSource, EvidenceLog, Frame, InspectionRuntime,
                                localization_roi, render_anomaly_overlay)
from secondlook.web import make_server
from secondlook.anomaly import ObservationInvalidError

try:
    import numpy as np
    NUMPY_DEPS = True
except ImportError:
    NUMPY_DEPS = False

try:
    import cv2
    IMAGE_DEPS = NUMPY_DEPS
except ImportError:
    IMAGE_DEPS = False


def mock_observe(detector, rgb, *_args, **_kwargs):
    return detector.infer(rgb)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.camera = CameraSource(max_age=1)
        self.runtime = InspectionRuntime(self.camera,
            evidence_path=Path(self.tmp.name) / "evidence.jsonl", evidence_kind="mock")

    def tearDown(self):
        self.tmp.cleanup()

    def test_unavailable_camera_and_policy_stay_blocked(self):
        status = self.runtime.status()
        self.assertEqual(status["camera"]["status"], "UNAVAILABLE")
        self.assertEqual(status["detector"]["status"], "BLOCKED")
        self.assertEqual(status["controller"]["status"], "DISARMED")
        self.assertEqual(status["controller"]["backend"], "UNAVAILABLE")
        self.assertIsNone(status["policy"]["result"])
        self.assertEqual(status["evidence"]["physical_trials"], 0)

    def test_evidence_requires_truthful_kind_and_provenance(self):
        with self.assertRaises(ValueError):
            self.runtime.evidence.append({"event": "trial"})
        with self.assertRaises(ValueError):
            self.runtime.evidence.append({"event": "trial", "evidence_kind": "mock"})
        record = {"event": "fixture", "evidence_kind": "mock", "observation_id": "fixture-1",
                  "provenance": {"test": True}, "outcome": "NOT_TESTED"}
        self.runtime.evidence.append(record)
        self.assertEqual(json.loads(self.runtime.evidence.path.read_text())["evidence_kind"], "mock")

    def test_loopback_only(self):
        with self.assertRaises(ValueError):
            make_server(self.runtime, host="0.0.0.0", port=0)

    def test_http_truthful_health_no_motion_routes(self):
        server = make_server(self.runtime, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/health") as response:
                status = json.load(response)
                self.assertEqual(status["service"], "ok")
                self.assertFalse(status["task_ready"])
                self.assertEqual(status["detector"]["status"], "BLOCKED")
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + "/api/arm")
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()
            with urlopen(base + "/") as response:
                self.assertIn(b"Motor torque may remain enabled", response.read())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    @unittest.skipUnless(IMAGE_DEPS, "cv2/numpy required for image checks")
    def test_empty_stale_and_lost_frames_cannot_be_live(self):
        self.camera.publish(Frame("empty", time.monotonic(), "fixture", np.zeros((0, 0, 3))))
        self.assertIsNone(self.camera.latest())
        frame = Frame("old", time.monotonic() - 2, "fixture", np.zeros((8, 8, 3), np.uint8))
        self.camera.publish(frame)
        self.assertIsNone(self.camera.latest())
        self.assertIsNone(self.runtime.frame_jpeg())
        fresh = Frame("fresh", time.monotonic(), "fixture", frame.image)
        self.camera.publish(fresh)
        self.assertIsNotNone(self.runtime.frame_jpeg())
        self.camera.fail("mock unplug")
        self.assertIsNone(self.runtime.frame_jpeg())

    @unittest.skipUnless(IMAGE_DEPS, "cv2/numpy required for image checks")
    def test_detection_keeps_original_observation_and_blocks_stale_image(self):
        class Detector:
            def infer(self, rgb):
                return {"score": .2, "disposition": "UNKNOWN", "threshold": None,
                        "anomaly_map": np.zeros(rgb.shape[:2]), "latency_ms": 1.,
                        "backend": "fixture", "device": "fixture", "artifact_id": "fixture",
                        "evidence_kind": "mock"}
        self.runtime.detector = Detector()
        frame = Frame("original", time.monotonic(), "fixture", np.arange(192, dtype=np.uint8).reshape(8, 8, 3))
        self.camera.publish(frame)
        self.runtime.infer_frame(frame)
        result = self.runtime.status()["detector"]
        self.assertEqual(result["observation_id"], "original")
        self.assertEqual(result["status"], "FRESH")
        self.assertIsNotNone(self.runtime.detector_jpeg())
        self.assertIsNone(self.runtime.detector_jpeg("different-observation"))
        with patch("secondlook.runtime.time.monotonic", return_value=time.monotonic() + 10):
            self.assertEqual(self.runtime.status()["detector"]["status"], "STALE")
            self.assertIsNone(self.runtime.detector_jpeg())
        log = json.loads(self.runtime.evidence.path.read_text())
        self.assertEqual(log["evidence_kind"], "mock")
        self.assertEqual(log["outcome"], "NOT_TESTED")

    @unittest.skipUnless(IMAGE_DEPS, "cv2/numpy required for image checks")
    def test_failed_inference_clears_prior_result(self):
        class Detector:
            def infer(self, _rgb):
                raise TimeoutError("fixture inference timeout")
        self.runtime.detector = Detector()
        self.runtime._result = {"score": .2}
        frame = Frame("timeout", time.monotonic(), "fixture", np.arange(192, dtype=np.uint8).reshape(8, 8, 3))
        self.runtime.infer_frame(frame)
        status = self.runtime.status()
        self.assertEqual(status["detector"]["status"], "BLOCKED")
        self.assertIn("timeout", status["detector"]["reason"])
        self.assertEqual(self.runtime.evidence.count, 0)

    def test_overlay_fetch_survives_newer_result_but_expires_and_clears(self):
        # Mock encoding makes this transport-race regression portable without cv2.
        class Image:
            def __init__(self, name):
                self.name = name
            def copy(self):
                return self
        cv2 = SimpleNamespace(COLOR_BGR2RGB=1, cvtColor=lambda image, _: image,
            imencode=lambda _, image: (True, SimpleNamespace(tobytes=lambda: image.name.encode())))
        self.runtime.detector = SimpleNamespace(infer=lambda _: {"score": .2, "disposition": "UNKNOWN"})
        with patch.dict("sys.modules", {"cv2": cv2, "numpy": SimpleNamespace()}), \
                patch("secondlook.runtime.observe_frame", side_effect=mock_observe), \
                patch("secondlook.runtime.time.monotonic", return_value=100.5) as clock:
            for name, captured_at in (("A", 100.), ("B", 100.1)):
                self.runtime.infer_frame(Frame(name, captured_at, "fixture", Image(name)))
            self.assertEqual(self.runtime.status()["detector"]["observation_id"], "B")
            self.assertEqual(self.runtime.detector_jpeg("A"), (b"A", "A"))
            self.assertEqual(self.runtime.detector_jpeg("B"), (b"B", "B"))
            self.assertIsNone(self.runtime.detector_jpeg("unknown"))
            clock.return_value = 103.05
            self.assertIsNone(self.runtime.detector_jpeg("A"))
            self.assertEqual(self.runtime.detector_jpeg("B"), (b"B", "B"))
            self.assertEqual(len(self.runtime._detector_images), 1)
            def failed(_):
                raise TimeoutError("fixture timeout")
            self.runtime.detector.infer = failed
            self.runtime.infer_frame(Frame("failed", 103., "fixture", Image("failed")))
            self.assertIsNone(self.runtime.detector_jpeg("B"))
            self.assertEqual(len(self.runtime._detector_images), 0)

    def test_overlay_cache_is_bounded_and_close_cannot_be_repopulated(self):
        class Image:
            def copy(self):
                return self
        cv2 = SimpleNamespace(COLOR_BGR2RGB=1, cvtColor=lambda image, _: image,
            imencode=lambda *_: (True, SimpleNamespace(tobytes=lambda: b"jpeg")))
        self.runtime.detector = SimpleNamespace(infer=lambda _: {"score": .2, "disposition": "UNKNOWN"})
        with patch.dict("sys.modules", {"cv2": cv2, "numpy": SimpleNamespace()}), \
                patch("secondlook.runtime.observe_frame", side_effect=mock_observe), \
                patch("secondlook.runtime.time.monotonic", return_value=100.5):
            self.assertEqual(self.runtime._detector_cache_limit, 8)
            for index in range(9):
                self.runtime.infer_frame(Frame(str(index), 100., "fixture", Image()))
            self.assertEqual(len(self.runtime._detector_images), 8)
            self.assertIsNone(self.runtime.detector_jpeg("0"))
            self.assertEqual(self.runtime.detector_jpeg("1"), (b"jpeg", "1"))
            self.runtime.close()
            self.assertEqual(len(self.runtime._detector_images), 0)
            self.runtime.infer_frame(Frame("late", 100., "fixture", Image()))
            self.assertIsNone(self.runtime.detector_jpeg("late"))
            self.assertEqual(len(self.runtime._detector_images), 0)

    def test_delayed_exact_overlay_survives_four_new_results_but_not_capture_ttl(self):
        class Image:
            def copy(self):
                return self
        cv2 = SimpleNamespace(COLOR_BGR2RGB=1, cvtColor=lambda image, _: image,
            imencode=lambda *_: (True, SimpleNamespace(tobytes=lambda: b"jpeg")))
        self.runtime.detector = SimpleNamespace(infer=lambda _: {"score": .2, "disposition": "UNKNOWN"})
        with patch.dict("sys.modules", {"cv2": cv2, "numpy": SimpleNamespace()}), \
                patch("secondlook.runtime.observe_frame", side_effect=mock_observe), \
                patch("secondlook.runtime.time.monotonic", return_value=100.) as clock:
            for index in range(5):
                clock.return_value = 100. + index * .5
                self.runtime.infer_frame(Frame(str(index), clock.return_value, "fixture", Image()))
            self.assertEqual(self.runtime.status()["detector"]["observation_id"], "4")
            clock.return_value = 102.1
            self.assertEqual(self.runtime.detector_jpeg("0"), (b"jpeg", "0"))
            clock.return_value = 103.001
            self.assertIsNone(self.runtime.detector_jpeg("0"))
            self.assertEqual(self.runtime.detector_jpeg("4"), (b"jpeg", "4"))
        bounded = InspectionRuntime(self.camera, detection_max_age=60., inference_interval=.1)
        self.assertEqual(bounded._detector_cache_limit, 32)

    def test_localization_rejects_inconsistent_nonfinite_and_outside_geometry(self):
        geometry = {"image_size": [10, 6], "map_size": [3, 2],
                    "roi_pixels": [2, 1, 8, 3], "roi_normalized": [.2, 1 / 6, .8, .5]}
        self.assertEqual(localization_roi(geometry, (6, 10, 3), (2, 3)), (2, 1, 8, 3))
        for changes in ({"image_size": [6, 10]}, {"map_size": [2, 3]},
                        {"roi_pixels": [-1, 1, 8, 3]}, {"roi_pixels": [2, 1, 11, 3]},
                        {"roi_pixels": [2, 1, 8, 1]}, {"roi_pixels": [2, 1, True, 3]},
                        {"roi_normalized": [.2, float("nan"), .8, .5]},
                        {"roi_normalized": [.2, 0., .8, .5]},
                        {"roi_normalized": [.2, 1 / 6, .8, 1.1]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                localization_roi({**geometry, **changes}, (6, 10, 3), (2, 3))

    def test_foreground_box_is_validated_inside_declared_parent_crop(self):
        geometry = {"image_size": [10, 6], "map_size": [3, 2],
                    "roi_pixels": [3, 2, 7, 4], "parent_roi_pixels": [1, 1, 9, 5],
                    "roi_normalized": [.1, 1 / 6, .9, 5 / 6]}
        self.assertEqual(localization_roi(geometry, (6, 10, 3), (2, 3)), (3, 2, 7, 4))
        for changes in ({"parent_roi_pixels": [0, 1, 9, 5]},
                        {"parent_roi_pixels": [True, 1, 9, 5]},
                        {"parent_roi_pixels": None}, {"parent_roi_pixels": [1, 1, 11, 5]},
                        {"roi_pixels": [0, 2, 7, 4]}, {"roi_pixels": [3, 0, 7, 4]},
                        {"roi_pixels": [3, 2, 10, 4]}, {"roi_pixels": [3, 2, 7, 6]},
                        {"roi_pixels": [3, 2, 3, 4]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                localization_roi({**geometry, **changes}, (6, 10, 3), (2, 3))

    @unittest.skipUnless(NUMPY_DEPS, "numpy required for pixel mapping")
    def test_nonsquare_roi_overlay_preserves_every_exterior_pixel(self):
        image = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
        values = np.array([[0., 1., 2.], [3., 4., 5.]], dtype=np.float32)
        geometry = {"image_size": [10, 6], "map_size": [3, 2],
                    "roi_pixels": [2, 1, 8, 3], "roi_normalized": [.2, 1 / 6, .8, .5]}
        sizes = []
        def resize(array, size):
            sizes.append(size)
            x = np.linspace(0, array.shape[1] - 1, size[0]).astype(int)
            y = np.linspace(0, array.shape[0] - 1, size[1]).astype(int)
            return array[y[:, None], x]
        fake_cv2 = SimpleNamespace(COLORMAP_TURBO=1, resize=resize,
            applyColorMap=lambda a, _: np.stack((a, a, a), axis=2),
            addWeighted=lambda a, aw, b, bw, _: (a * aw + b * bw).astype(np.uint8))
        with patch.dict("sys.modules", {"cv2": fake_cv2}):
            overlay = render_anomaly_overlay(image, values, geometry)
            self.assertEqual(sizes, [(6, 2)])
            exterior = np.ones(image.shape[:2], dtype=bool)
            exterior[1:3, 2:8] = False
            np.testing.assert_array_equal(overlay[exterior], image[exterior])
            self.assertTrue(np.any(overlay[1:3, 2:8] != image[1:3, 2:8]))
            np.testing.assert_array_equal(image, np.arange(180, dtype=np.uint8).reshape(6, 10, 3))
            render_anomaly_overlay(image, values)
            self.assertEqual(sizes[-1], (10, 6))  # Legacy artifacts retain full-frame mapping.
            foreground = {"image_size": [10, 6], "map_size": [3, 2],
                          "roi_pixels": [3, 2, 7, 4], "parent_roi_pixels": [1, 1, 9, 5],
                          "roi_normalized": [.1, 1 / 6, .9, 5 / 6]}
            cropped_overlay = render_anomaly_overlay(image, values, foreground)
            self.assertEqual(sizes[-1], (4, 2))
            exterior[:] = True
            exterior[2:4, 3:7] = False
            np.testing.assert_array_equal(cropped_overlay[exterior], image[exterior])
            self.assertTrue(np.any(cropped_overlay[2:4, 3:7] != image[2:4, 3:7]))

    @unittest.skipUnless(NUMPY_DEPS, "numpy required for observation integration")
    def test_roi_geometry_is_preserved_in_status_and_evidence(self):
        geometry = {"image_size": [10, 6], "map_size": [3, 2],
                    "roi_pixels": [2, 1, 8, 3], "roi_normalized": [.2, 1 / 6, .8, .5]}
        image = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
        self.runtime.detector = SimpleNamespace(infer=lambda _: {
            "score": .2, "disposition": "UNKNOWN", "localization_geometry": geometry,
            "anomaly_map": np.arange(6, dtype=np.float32).reshape(2, 3)})
        fake_cv2 = SimpleNamespace(COLOR_BGR2RGB=1, cvtColor=lambda image, _: image,
            imencode=lambda *_: (True, SimpleNamespace(tobytes=lambda: b"jpeg")))
        with patch.dict("sys.modules", {"cv2": fake_cv2}), \
                patch("secondlook.runtime.render_anomaly_overlay", return_value=image) as render, \
                patch("secondlook.runtime.time.monotonic", return_value=100.5):
            self.runtime.infer_frame(Frame("roi", 100., "fixture", image))
            self.assertEqual(render.call_args.args[2], geometry)
            self.assertEqual(self.runtime.status()["detector"]["localization_geometry"], geometry)
            record = json.loads(self.runtime.evidence.path.read_text())
            self.assertEqual(record["detector"]["localization_geometry"], geometry)
            self.assertEqual(record["detector"]["observation_validity"], "UNVERIFIED")

    @unittest.skipUnless(NUMPY_DEPS, "numpy required for observation integration")
    def test_invalid_observation_clears_overlay_and_records_invalid_without_inference(self):
        calls = []
        def infer(_image):
            calls.append(True)
            return {"score": .2, "disposition": "UNKNOWN"}
        self.runtime.detector = SimpleNamespace(infer=infer)
        fake_cv2 = SimpleNamespace(COLOR_BGR2RGB=1, cvtColor=lambda image, _: image,
            imencode=lambda *_: (True, SimpleNamespace(tobytes=lambda: b"jpeg")))
        with patch.dict("sys.modules", {"cv2": fake_cv2}), \
                patch("secondlook.runtime.time.monotonic", return_value=100.5):
            valid = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
            self.runtime.infer_frame(Frame("valid", 100., "fixture", valid))
            self.assertEqual(self.runtime.detector_jpeg("valid"), (b"jpeg", "valid"))
            self.assertEqual(self.runtime.status()["detector"]["observation_validity"], "UNVERIFIED")
            self.runtime.infer_frame(Frame("dark", 100.1, "fixture", np.zeros_like(valid)))
            status = self.runtime.status()["detector"]
            self.assertEqual(status["status"], "INVALID")
            self.assertEqual(status["decision"], "INVALID")
            self.assertIsNone(status["score"])
            self.assertFalse(status["inference_executed"])
            self.assertEqual(status["object_presence"]["status"], "UNVERIFIED")
            self.assertIsNone(self.runtime.detector_jpeg("valid"))
            self.assertEqual(len(calls), 1)
            records = [json.loads(line) for line in self.runtime.evidence.path.read_text().splitlines()]
            self.assertEqual(records[-1]["detector"]["frame_validity"], "INVALID")
            self.assertEqual(records[-1]["detector"]["score"], None)
            self.assertEqual(records[-1]["event"], "observation_invalid")
            self.runtime.infer_frame(Frame("valid-again", 100.2, "fixture", valid))
            self.assertIsNotNone(self.runtime.detector_jpeg("valid-again"))
            self.runtime.detector.metadata = {"artifact_id": "fixture-artifact"}
            def reject_foreground(_image):
                raise ObservationInvalidError("colored foreground absent or too small")
            self.runtime.detector.infer = reject_foreground
            self.runtime.infer_frame(Frame("foreground-invalid", 100.3, "fixture", valid))
            status = self.runtime.status()["detector"]
            self.assertEqual(status["status"], "INVALID")
            self.assertEqual(status["frame_validity"], "VALID")
            self.assertEqual(status["artifact_id"], "fixture-artifact")
            self.assertEqual(status["decision_reason"], "colored foreground absent or too small")
            self.assertIsNone(status["score"])
            self.assertFalse(status["inference_executed"])
            self.assertIsNone(self.runtime.detector_jpeg("valid-again"))
            self.assertEqual(len(self.runtime._detector_images), 0)


if __name__ == "__main__":
    unittest.main()
