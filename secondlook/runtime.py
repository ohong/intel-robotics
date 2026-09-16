"""One camera reader and one detector worker; no robot connection or commands."""
from __future__ import annotations

import copy
from collections import deque
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .cv_observation import observe_frame


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_validation_summary(metadata: Any) -> dict:
    """Expose recorded validation counts without turning missing data into zeros."""
    metadata = metadata if isinstance(metadata, dict) else {}
    calibration = metadata.get("calibration")
    if not isinstance(calibration, dict) or not calibration:
        return {"status": "UNAVAILABLE", "artifact_id": metadata.get("artifact_id"),
                "reason": "No validation calibration recorded for this artifact"}
    def count(value):
        return value if type(value) is int and value >= 0 else None
    distributions = calibration.get("score_distributions")
    anomalous = distributions.get("anomalous") if isinstance(distributions, dict) else None
    limitations = calibration.get("split_limitations")
    return {
        "status": "EXPERIMENTAL PILOT", "scope": "VALIDATION ONLY",
        "artifact_id": metadata.get("artifact_id"),
        "validation_images": count(calibration.get("count")),
        "defect_images": count(anomalous.get("count")) if isinstance(anomalous, dict) else None,
        **{key: count(calibration.get(key)) for key in
           ("missed_defects", "rejected_good_parts", "uncertain_defects", "uncertain_good_parts")},
        "split_limitations": [value for value in limitations if isinstance(value, str) and value.strip()]
            if isinstance(limitations, list) else [],
        "unseen_block_test": "PENDING" if not metadata.get("evaluation") else "EVALUATION RECORDED; COVERAGE NOT VERIFIED HERE",
        "unit": "images; repeated views of a specimen are not independent trials",
    }


def localization_roi(geometry: dict, image_shape: tuple, map_shape: tuple) -> tuple[int, int, int, int]:
    """Validate the detector's exact crop transform before drawing model pixels."""
    if not isinstance(geometry, dict):
        raise ValueError("Localization geometry must be an object")
    for name, count in (("image_size", 2), ("map_size", 2), ("roi_pixels", 4)):
        value = geometry.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != count or any(type(x) is not int for x in value):
            raise ValueError(f"Invalid localization {name}")
    height, width = image_shape[:2]
    if list(geometry["image_size"]) != [width, height] or list(geometry["map_size"]) != list(map_shape[::-1]):
        raise ValueError("Localization image or map dimensions do not match pixels")
    left, top, right, bottom = geometry["roi_pixels"]
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise ValueError("Localization ROI is empty or outside the image")
    normalized = geometry.get("roi_normalized")
    if (not isinstance(normalized, (list, tuple)) or len(normalized) != 4 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in normalized)):
        raise ValueError("Localization normalized ROI must contain finite coordinates")
    nl, nt, nr, nb = normalized
    if not 0 <= nl < nr <= 1 or not 0 <= nt < nb <= 1:
        raise ValueError("Localization normalized ROI is outside [0,1]")
    expected = [math.floor(nl * width), math.floor(nt * height),
                math.ceil(nr * width), math.ceil(nb * height)]
    # Foreground crops keep their effective box inside the declared fixed ROI.
    # Legacy artifacts use one box for both roles.
    parent = geometry.get("parent_roi_pixels", geometry["roi_pixels"])
    if (not isinstance(parent, (list, tuple)) or len(parent) != 4 or
            any(type(x) is not int for x in parent) or list(parent) != expected):
        raise ValueError("Localization parent ROI disagrees with normalized crop")
    if not parent[0] <= left < right <= parent[2] or not parent[1] <= top < bottom <= parent[3]:
        raise ValueError("Localization effective ROI extends outside its parent crop")
    return left, top, right, bottom


def render_anomaly_overlay(image: Any, anomaly_map: Any, geometry: dict | None = None):
    import cv2
    import numpy as np

    overlay = image.copy()
    if anomaly_map is None:
        if geometry is not None:
            raise ValueError("Localization geometry requires an anomaly map")
        return overlay
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or not values.size:
        raise ValueError("Detector returned an invalid anomaly map")
    height, width = overlay.shape[:2]
    left, top, right, bottom = ((0, 0, width, height) if geometry is None else
                               localization_roi(geometry, overlay.shape, values.shape))
    values = cv2.resize(values, (right - left, bottom - top))
    low, high = float(values.min()), float(values.max())
    scaled = (values - low) / (high - low) if high > low else np.zeros_like(values)
    heat = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay[top:bottom, left:right] = cv2.addWeighted(overlay[top:bottom, left:right], .6, heat, .4, 0)
    return overlay


@dataclass(frozen=True)
class Frame:
    observation_id: str
    captured_at: float
    captured_at_utc: str
    image: Any
    source_sequence: int | None = None


class CameraSource:
    """Drains capture continuously, keeping only the latest successful frame.

    The timestamp is host receipt time, not a hardware exposure timestamp.
    Driver buffer-size configuration is best effort and reported separately.
    """

    def __init__(self, device: str = "/dev/video10", width: int = 640,
                 height: int = 480, max_age: float = 1.0,
                 capture_factory: Callable | None = None):
        if width <= 0 or height <= 0 or not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("Camera dimensions and freshness bound must be positive")
        self.device, self.width, self.height, self.max_age = device, width, height, max_age
        self.capture_factory = capture_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: Frame | None = None
        self.error: str | None = "Camera has not started"
        self.buffer_size_requested = 1
        self.buffer_size_accepted = False
        self.timestamp_source = "host receipt after capture.read; exposure time unavailable"
        self.transport = "direct_v4l2"

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Camera reader already started")
        self._thread = threading.Thread(target=self._read, name="camera-reader", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        import cv2

        capture = None
        try:
            if self.capture_factory:
                capture = self.capture_factory(self.device)
            else:
                capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not capture.isOpened():
                raise RuntimeError(f"Camera unavailable: {self.device}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.buffer_size_accepted = bool(capture.set(cv2.CAP_PROP_BUFFERSIZE, 1))
            while not self._stop.is_set():
                ok, bgr = capture.read()
                received = time.monotonic()
                if self._stop.is_set():
                    break
                if not ok or bgr is None or not getattr(bgr, "size", 0):
                    raise RuntimeError("Camera returned an empty frame")
                image = bgr.copy()
                image.flags.writeable = False
                self.publish(Frame(str(uuid.uuid4()), received, utc_now(), image))
        except Exception as error:
            self.fail(str(error))
        finally:
            if capture is not None:
                capture.release()

    def publish(self, frame: Frame) -> None:
        if frame.image is None or not getattr(frame.image, "size", 0):
            self.fail("Camera returned an empty frame")
            return
        with self._lock:
            self._latest = frame
            self.error = None

    def fail(self, error: str) -> None:
        with self._lock:
            self.error = error
            self._latest = None

    def latest(self, now: float | None = None) -> Frame | None:
        with self._lock:
            frame = self._latest
        if frame is None:
            return None
        age = (time.monotonic() if now is None else now) - frame.captured_at
        return frame if math.isfinite(age) and 0 <= age <= self.max_age else None

    def status(self) -> dict:
        frame = self.latest()
        return {
            "status": "LIVE" if frame else "UNAVAILABLE",
            "device": self.device,
            "error": self.error if self.error else (None if frame else "No fresh camera frame"),
            "observation_id": frame.observation_id if frame else None,
            "captured_at_utc": frame.captured_at_utc if frame else None,
            "frame_age_ms": (time.monotonic() - frame.captured_at) * 1000 if frame else None,
            "timestamp_source": self.timestamp_source,
            "transport": self.transport,
            "source_sequence": frame.source_sequence if frame else None,
            "buffer_size_requested": self.buffer_size_requested,
            "buffer_size_accepted": self.buffer_size_accepted,
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.fail("Camera stopped")


class EvidenceLog:
    """Append-only evidence. Callers must provide provenance and outcome type."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.count = 0

    def append(self, record: dict) -> None:
        if record.get("evidence_kind") not in ("real", "replay", "synthetic", "mock"):
            raise ValueError("An explicit evidence kind is required")
        for required in ("event", "observation_id", "provenance", "outcome"):
            if not record.get(required):
                raise ValueError(f"Evidence requires {required}")
        line = json.dumps({"recorded_at_utc": utc_now(), **record}, allow_nan=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.count += 1


class InspectionRuntime:
    def __init__(self, camera: CameraSource, detector: Any = None,
                 detector_loader: Callable | None = None, task_instruction: str = "",
                 evidence_path: Path = Path("artifacts/runtime/observations.jsonl"),
                 revision: str = "unknown", detection_max_age: float = 3.0,
                 detector_error: str | None = None,
                 policy_blocker: str = "No task-ready VLA/Physical AI Studio policy connected",
                 evidence_kind: str = "real", inference_interval: float = 0.5):
        if not math.isfinite(detection_max_age) or detection_max_age <= 0:
            raise ValueError("Detection freshness bound must be positive and finite")
        self.camera, self.detector = camera, detector
        if evidence_kind not in ("real", "replay", "synthetic", "mock"):
            raise ValueError("An explicit valid evidence kind is required")
        self.evidence_kind = evidence_kind
        self.policy_blocker = policy_blocker
        if not math.isfinite(inference_interval) or inference_interval < 0.1:
            raise ValueError("Inference interval must be finite and at least 0.1 seconds")
        self.inference_interval = inference_interval
        self.detector_loader = detector_loader
        self.task_instruction, self.revision = task_instruction, revision
        self.detection_max_age = detection_max_age
        self.evidence = EvidenceLog(evidence_path)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: dict | None = None
        # Keep the normal freshness window across separate status/JPEG requests.
        # Two spare slots cover scheduling boundaries; cap retained image memory.
        self._detector_cache_limit = min(32, math.ceil(detection_max_age / inference_interval) + 2)
        self._detector_images: deque[tuple[str, float, bytes]] = deque(maxlen=self._detector_cache_limit)
        self._detector_error = detector_error or (
            "Loading artifact" if detector_loader else
            (None if detector else "No trained anomaly artifact configured"))
        self._evidence_error: str | None = None
        self.started_at = utc_now()

    def start(self) -> None:
        self.camera.start()
        self._thread = threading.Thread(target=self._infer_loop, name="anomaly-inference", daemon=True)
        self._thread.start()

    def _infer_loop(self) -> None:
        if self.detector_loader:
            try:
                self.detector = self.detector_loader()
                self._detector_error = None
            except Exception as error:
                self._detector_error = f"Artifact load failed: {error}"
                return
        previous = None
        while not self._stop.is_set():
            frame = self.camera.latest()
            if self.detector is None or frame is None or frame.observation_id == previous:
                self._stop.wait(.05)
                continue
            previous = frame.observation_id
            self.infer_frame(frame)
            self._stop.wait(self.inference_interval)

    def infer_frame(self, frame: Frame) -> None:
        """Inference accepts pixels only. Evaluation labels never enter this path."""
        import cv2

        started = time.monotonic()
        try:
            raw = observe_frame(self.detector, cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB),
                                frame.observation_id, frame.captured_at, max_age=self.camera.max_age)
            invalid = (raw.get("frame_validity") == "INVALID" or
                       raw.get("observation_validity") == "INVALID" or raw.get("decision") == "INVALID")
            score = None if invalid else float(raw["score"])
            if not invalid and not math.isfinite(score):
                raise ValueError("Detector returned a nonfinite score")
            if raw.get("disposition") not in ("UNKNOWN", "NORMAL", "ANOMALOUS", "UNCERTAIN"):
                raise ValueError("Detector returned an invalid disposition")
            result = {key: raw.get(key) for key in (
                "disposition", "threshold", "latency_ms", "model_latency_ms", "backend", "device",
                "artifact_id", "evidence_kind", "execution_devices", "inference_precision",
                "localization_geometry", "uncertainty_band", "decision_reason",
                "frame_validity", "observation_validity", "quality_flags", "quality_stats", "object_presence",
                "occlusion", "defect_visibility", "inference_executed", "anomaly_status",
                "anomaly_score", "decision", "model_version", "inference_ms")}
            result["execution_devices"] = raw.get("execution_devices", getattr(self.detector, "execution_devices", None))
            result["inference_precision"] = raw.get("inference_precision", getattr(self.detector, "inference_precision", None))
            result.update(score=score, observation_id=frame.observation_id,
                          captured_at=frame.captured_at,
                          source_sequence=frame.source_sequence,
                          timestamp_source=self.camera.timestamp_source,
                          captured_at_utc=frame.captured_at_utc,
                          completed_at_utc=utc_now(),
                          processing_ms=(time.monotonic() - started) * 1000,
                          observation_to_result_ms=(time.monotonic() - frame.captured_at) * 1000)
            # Fail clearly before placing a non-JSON result into shared state.
            json.dumps(result, allow_nan=False)
            result["map_display"] = "per-frame min/max color scale; not a probability"
            if invalid:
                result["reason"] = "Observation invalid: " + ", ".join(raw.get("quality_flags", []))
                encoded = None
            else:
                overlay = render_anomaly_overlay(frame.image, raw.get("anomaly_map"), raw.get("localization_geometry"))
                ok, encoded = cv2.imencode(".jpg", overlay)
                if not ok:
                    raise RuntimeError("Could not encode detector observation")
            with self._lock:
                if self._stop.is_set():
                    return
                self._result = result
                if invalid:
                    self._detector_images.clear()
                else:
                    self._detector_images.append((frame.observation_id, frame.captured_at,
                                                  encoded.tobytes()))
                self._detector_error = None
            record = {
                "event": "observation_invalid" if invalid else "observation_inference", "evidence_kind": self.evidence_kind,
                "observation_id": frame.observation_id,
                "provenance": {"revision": self.revision, "camera": self.camera.device,
                               "camera_transport": self.camera.transport,
                               "timestamp_source": self.camera.timestamp_source,
                               "model": result.get("artifact_id"),
                               "model_evidence_kind": result.get("evidence_kind")},
                "task_instruction": self.task_instruction or None,
                "detector": {k: v for k, v in result.items() if k != "captured_at"},
                "policy": {"status": "BLOCKED", "reason": "No task-ready VLA connected"},
                "robot_state": None, "controller_result": "DISARMED",
                "outcome": "NOT_TESTED", "intervention": None,
                "note": "Camera inference only; no physical trial or verified outcome",
            }
            try:
                self.evidence.append(record)
                self._evidence_error = None
            except Exception as error:
                self._evidence_error = str(error)
        except Exception as error:
            with self._lock:
                self._result = None
                self._detector_images.clear()
                self._detector_error = str(error)

    def status(self) -> dict:
        with self._lock:
            result = copy.deepcopy(self._result)
            error = self._detector_error
        camera = self.camera.status()
        if result:
            age = time.monotonic() - result.pop("captured_at")
            result["age_ms"] = age * 1000
            result["status"] = ("INVALID" if result.get("decision") == "INVALID" or result.get("frame_validity") == "INVALID" else
                                "FRESH" if 0 <= age <= self.detection_max_age else "STALE")
            result["observation_available"] = camera["status"] == "LIVE"
        else:
            result = {"status": "BLOCKED", "reason": error}
        return {
            "application": "Second Look", "mode": "LIVE" if self.evidence_kind == "real" else self.evidence_kind.upper(), "started_at_utc": self.started_at,
            "revision": self.revision, "camera": camera, "detector": result,
            "model_validation": model_validation_summary(getattr(self.detector, "metadata", None)),
            "task_instruction": self.task_instruction or None,
            "task_status": "CONFIGURED" if self.task_instruction else "BLOCKED — exact task not supplied",
            "policy": {"status": "BLOCKED", "provenance": None, "result": None,
                       "reason": self.policy_blocker},
            "controller": {"status": "DISARMED", "backend": "UNAVAILABLE",
                           "reason": "Software disarm only; hardware torque state is independent. Physical limits, stop procedure, supervision and run authorization required"},
            "evidence": {"status": "ERROR" if self._evidence_error else "OBSERVATIONS_ONLY",
                         "path": str(self.evidence.path), "records_this_session": self.evidence.count,
                         "error": self._evidence_error, "physical_trials": 0,
                         "physical_outcome": "NOT TESTED"},
        }

    def frame_jpeg(self) -> tuple[bytes, str] | None:
        import cv2

        frame = self.camera.latest()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame.image)
        return (encoded.tobytes(), frame.observation_id) if ok else None

    def detector_jpeg(self, observation_id: str | None = None) -> tuple[bytes, str] | None:
        with self._lock:
            if self._result is None:
                return None
            expected = observation_id if observation_id is not None else self._result["observation_id"]
            now = time.monotonic()
            self._detector_images = deque(
                (entry for entry in self._detector_images
                 if 0 <= now - entry[1] <= self.detection_max_age), maxlen=self._detector_cache_limit)
            for image_id, _captured_at, jpeg in reversed(self._detector_images):
                if image_id == expected:
                    return jpeg, image_id
            return None

    def close(self) -> None:
        self._stop.set()
        self.camera.close()
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            self._result = None
            self._detector_images.clear()
            self._detector_error = "Runtime stopped"
