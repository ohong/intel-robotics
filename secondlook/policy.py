"""Fail-closed JSON boundary to the installed Physical AI Studio interpreter.

This module imports no model or robot SDK. Policy candidates are evidence only;
a controller must separately validate freshness, limits, supervision, and stops.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

STUDIO_REVISION = "c4ff730fb49f84e5102d01088d52cfff1ba62854"
SCHEMA_VERSION = 1
MODES = {"replay", "live"}


class PolicyContractError(ValueError):
    """No usable policy candidate can be returned for this request."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyContractError(message)


def _number(value: Any) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _vector(value: Any, size: int, label: str) -> None:
    _require(isinstance(value, list) and len(value) == size and all(_number(v) for v in value),
             f"{label}: expected {size} finite numbers")


def validate_manifest(manifest: dict) -> None:
    _require(isinstance(manifest, dict) and manifest.get("schema_version") == SCHEMA_VERSION,
             "Unsupported policy manifest schema")
    for field in ("artifact_id", "checkpoint_revision", "calibration_id", "task_id",
                  "checkpoint_dir", "backbone_dir", "studio_root"):
        _require(_text(manifest.get(field)), f"Manifest requires {field}")
    _require(manifest.get("studio_revision") == STUDIO_REVISION, "Unsupported Studio revision")
    _require(manifest.get("adaptation") in {"base", "task_adapted"}, "Unknown adaptation status")
    _require(manifest.get("action_semantics") == "absolute_joint_position",
             "Only explicitly mapped absolute joint positions are supported")
    _require(isinstance(manifest.get("normalization"), dict), "Normalization contract is required")
    for prefix in ("state", "action"):
        names, units = manifest.get(prefix + "_names"), manifest.get(prefix + "_units")
        _require(isinstance(names, list) and bool(names) and all(_text(n) for n in names)
                 and len(set(names)) == len(names), f"Invalid {prefix} names/order")
        _require(isinstance(units, list) and len(units) == len(names) and all(_text(u) for u in units),
                 f"Explicit {prefix} units are required")
        stats = manifest.get("normalization", {}).get(prefix, {})
        _require(isinstance(stats, dict), "Normalization statistics must be objects")
        _vector(stats.get("mean"), len(names), prefix + " mean")
        _vector(stats.get("std"), len(names), prefix + " std")
        _require(all(v > 0 for v in stats["std"]), "Normalization std must be positive")
    _require(manifest["state_names"] == manifest["action_names"] and
             manifest["state_units"] == manifest["action_units"],
             "State and action joint mappings must match")
    cameras = manifest.get("cameras")
    _require(isinstance(cameras, dict) and bool(cameras), "Camera contract is required")
    slots = []
    for key, camera in cameras.items():
        _require(_text(key) and "." not in key and isinstance(camera, dict), "Invalid camera key")
        shape = camera.get("shape")
        _require(isinstance(shape, list) and len(shape) == 3 and shape[2] == 3 and
                 all(type(n) is int and n > 0 for n in shape), "Camera shape must be [height,width,3]")
        _require(type(camera.get("slot")) is int, "Camera slot must be an integer")
        _require(_text(camera.get("source_id")), "Camera source identity is required")
        slots.append(camera["slot"])
    _require(sorted(slots) == list(range(len(cameras))), "Camera slots must be complete and unique")
    anomaly_cameras = manifest.get("anomaly_camera_keys")
    _require(isinstance(anomaly_cameras, list) and bool(anomaly_cameras) and
             all(_text(key) for key in anomaly_cameras) and len(set(anomaly_cameras)) == len(anomaly_cameras) and
             set(anomaly_cameras) <= set(cameras),
             "Explicit unique anomaly camera keys must refer to configured cameras")
    for name in ("max_age_seconds", "max_sensor_skew_seconds"):
        _require(_number(manifest.get(name)) and 0 < manifest[name] <= 60, f"Invalid {name}")
    _require(type(manifest.get("tokenizer_max_length")) is int and
             1 <= manifest["tokenizer_max_length"] <= 512, "Explicit tokenizer limit is required")
    _require(manifest.get("device") in {"cpu", "xpu"}, "Only local CPU or XPU inference is supported")
    _require(isinstance(manifest.get("runtime_versions"), dict) and
             all(_text(manifest["runtime_versions"].get(k)) for k in ("physicalai", "physicalai-train", "torch")),
             "Pinned runtime versions are required")
    files = manifest.get("files")
    _require(isinstance(files, dict) and bool(files) and
             all(_text(k) and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                 for k, v in files.items()), "SHA-256 artifact inventory is required")


def artifact_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    _require(not Path(relative).is_absolute() and path.is_relative_to(root.resolve()),
             "Artifact paths must stay under the manifest directory")
    return path


def verify_artifacts(manifest: dict, root: Path) -> tuple[Path, Path]:
    """Hash all local checkpoint and backbone files before importing model code."""
    checkpoint = artifact_path(root, manifest["checkpoint_dir"])
    backbone = artifact_path(root, manifest["backbone_dir"])
    for directory in (checkpoint, backbone):
        _require(directory.is_dir(), f"Missing artifact directory: {directory}")
        for path in directory.rglob("*"):
            if path.is_file():
                _require(path.relative_to(root).as_posix() in manifest["files"],
                         f"Unpinned artifact: {path.name}")
    for name, expected in manifest["files"].items():
        path = artifact_path(root, name)
        _require(path.is_file(), f"Missing artifact: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        _require(actual == expected, f"Artifact checksum mismatch: {name}")
    _require((checkpoint / "model.safetensors").is_file(), "Missing model.safetensors")
    config = json.loads((checkpoint / "config.json").read_text())
    _require(Path(config.get("vlm_model_name", "")).resolve() == backbone,
             "Checkpoint must reference the pinned local backbone directory")
    return checkpoint, backbone


def validate_observation(manifest: dict, observation: dict, mode: str,
                         now: float | None = None) -> str:
    validate_manifest(manifest)
    _require(mode in MODES, "Task policy accepts replay/live only; use the dedicated engineering CLI")
    _require(isinstance(observation, dict) and observation.get("schema_version") == SCHEMA_VERSION,
             "Unsupported observation schema")
    _require(_text(observation.get("observation_id")), "Observation identity is required")
    _require(observation.get("task_id") == manifest["task_id"], "Task identity mismatch")
    _require(_text(observation.get("task")), "Task instruction is required")
    _require(observation.get("calibration_id") == manifest["calibration_id"], "Calibration mismatch")
    provenance = observation.get("provenance")
    _require(provenance in {"real", "recorded_real"}, "Task policy requires real measured or recorded state")
    if mode == "live":
        _require(provenance == "real", "Live input must contain real measured state and images")
        _require(manifest["adaptation"] == "task_adapted" and
                 manifest.get("task_validation_passed") is True and
                 manifest.get("task_validation_evidence") in manifest["files"],
                 "Live inference requires a task-adapted policy and pinned validation evidence")
    state = observation.get("state", {})
    _require(isinstance(state, dict), "Measured state object is required")
    _require(state.get("names") == manifest["state_names"] and
             state.get("units") == manifest["state_units"], "State joint names/order/units mismatch")
    _vector(state.get("values"), len(manifest["state_names"]), "state")
    images = observation.get("images")
    _require(isinstance(images, dict) and set(images) == set(manifest["cameras"]), "Camera keys mismatch")
    timestamps = [observation.get("captured_at"), state.get("captured_at")]
    for key, expected in manifest["cameras"].items():
        actual = images[key]
        _require(isinstance(actual, dict) and actual.get("shape") == expected["shape"] and
                 actual.get("source_id") == expected["source_id"] and actual.get("color") == "RGB",
                 f"Camera {key}: shape, source, or color mismatch")
        _require(_text(actual.get("path")) and Path(actual["path"]).is_absolute(), "Image path must be absolute")
        _require(isinstance(actual.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", actual["sha256"]) is not None,
                 "Each image requires its content hash")
        timestamps.append(actual.get("captured_at"))
    anomaly = observation.get("anomaly", {})
    _require(isinstance(anomaly, dict) and anomaly.get("observation_id") == observation["observation_id"],
             "Anomaly must refer to this observation")
    _require(_number(anomaly.get("score")) and _number(anomaly.get("threshold")) and
             _text(anomaly.get("model_id")) and anomaly.get("decision") in {"normal", "anomalous", "ambiguous"},
             "Complete anomaly score, threshold, model, and decision are required")
    _require(anomaly.get("evidence") == "real_inference", "Task policy requires real anomaly inference")
    sources = anomaly.get("source_images")
    _require(isinstance(sources, dict) and set(sources) == set(manifest["anomaly_camera_keys"]),
             "Anomaly source camera keys must match the configured detector cameras exactly")
    for key, source in sources.items():
        _require(isinstance(source, dict) and set(source) == {"source_id", "sha256"} and
                 source["source_id"] == images[key]["source_id"] and source["sha256"] == images[key]["sha256"],
                 f"Anomaly source image/camera mismatch: {key}")
    timestamps.append(anomaly.get("captured_at"))
    _require(all(_number(t) for t in timestamps), "All sensor timestamps must be finite Unix seconds")
    _require(max(timestamps) - min(timestamps) <= manifest["max_sensor_skew_seconds"], "Sensor timestamps disagree")
    if mode == "live":
        now = time.time() if now is None else now
        _require(_number(now) and all(0 <= now - t <= manifest["max_age_seconds"] for t in timestamps),
                 "Stale or future observation/state/anomaly")
    # This is conditioning text, not a claim that the pretrained model understands defects.
    return (f"{observation['task']}\n"
            f"Inspection: {anomaly['decision']}; score={anomaly['score']:.6g}; "
            f"threshold={anomaly['threshold']:.6g}.\n")


def engineering_conditioning(image_sha256: str, report: dict | None = None) -> tuple[str, str]:
    """Keep uncalibrated detector evidence usable only in an explicit probe."""
    task = "Sort LEGO into defective and non-defective piles.\n"
    if report is None:
        return task + "Inspection: anomalous; score=0.8; threshold=0.5.\n", "synthetic fixture; not detector measurement"
    _require(isinstance(report, dict) and report.get("schema_version") == 1 and
             report.get("stage") == "anomalib" and report.get("model_inference_executed") is True,
             "Engineering detector report requires actual Anomalib inference")
    image = report.get("image", {})
    _require(isinstance(image, dict) and image.get("sha256") == image_sha256 and
             image.get("provenance") == "recorded_real", "Detector and VLA image content must match")
    result = report.get("result", {})
    _require(isinstance(result, dict) and _number(result.get("score")) and
             "threshold" in result and result["threshold"] is None and result.get("disposition") == "UNKNOWN" and
             result.get("evidence_kind") == "engineering_smoke" and _text(result.get("artifact_id")),
             "Same-image smoke probe requires a real finite score, UNKNOWN decision, and null threshold")
    score = json.dumps(result["score"], allow_nan=False)
    return task + f"Inspection: UNKNOWN; anomaly score={score}; threshold=null (uncalibrated).\n", \
        "real Anomalib inference on the same recorded image; uncalibrated engineering smoke"


def validate_engineering_weight_keys(loaded: set[str], expected: set[str],
                                     snapflow_enabled: bool) -> list[str]:
    """Permit only Studio's inactive new target-time branch on the old base model."""
    inactive = {f"_model.target_time_mlp_{part}.{parameter}"
                for part in ("in", "out") for parameter in ("weight", "bias")}
    missing = expected - loaded
    _require(not snapflow_enabled, "Legacy engineering probe requires SnapFlow disabled")
    _require(not (loaded - expected) and missing <= inactive,
             f"Incomplete active pretrained weights: missing={sorted(missing - inactive)[:8]}; "
             f"unexpected={sorted(loaded - expected)[:8]}")
    return sorted(missing)


class StudioPolicyBridge:
    """One bounded inference subprocess; model-load latency is included in wall time."""

    def __init__(self, python: str, manifest_path: str | Path | None = None,
                 worker_path: str | Path | None = None, timeout_seconds: float = 60):
        self.python = python
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.worker_path = Path(worker_path or Path(__file__).resolve().parents[1] / "scripts/policy_worker.py")
        _require(_number(timeout_seconds) and timeout_seconds > 0, "Invalid policy timeout")
        self.timeout_seconds = timeout_seconds

    def status(self) -> dict:
        if self.manifest_path is None or not self.manifest_path.is_file():
            return {"status": "BLOCKED", "reason": "No policy artifact manifest; no task-ready checkpoint found",
                    "controller_ready": False}
        try:
            manifest = json.loads(self.manifest_path.read_text())
            validate_manifest(manifest)
        except (ValueError, TypeError, OSError, AttributeError) as error:
            return {"status": "BLOCKED", "reason": str(error), "controller_ready": False}
        return {"status": "NOT TESTED", "artifact_id": manifest["artifact_id"],
                "adaptation": manifest["adaptation"], "controller_ready": False,
                "reason": "Manifest schema accepted; model execution and task validation are separate gates"}

    def infer(self, observation: dict, mode: str = "replay") -> dict:
        _require(mode in MODES, "Task policy accepts replay/live only; use the dedicated engineering CLI")
        _require(self.manifest_path is not None and self.manifest_path.is_file(), "No policy artifact manifest")
        manifest = json.loads(self.manifest_path.read_text())
        validate_observation(manifest, observation, mode)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        started = time.monotonic()
        try:
            result = subprocess.run([self.python, str(self.worker_path), "--manifest", str(self.manifest_path)],
                                    input=json.dumps({"mode": mode, "observation": observation}, allow_nan=False),
                                    capture_output=True, text=True, timeout=self.timeout_seconds, env=env, check=False)
        except subprocess.TimeoutExpired as error:
            raise PolicyContractError("Policy worker timed out; no action returned") from error
        except OSError as error:
            raise PolicyContractError(f"Policy interpreter unavailable: {error}") from error
        try:
            response = json.loads(result.stdout)
        except ValueError as error:
            raise PolicyContractError("Policy worker returned invalid JSON") from error
        _require(isinstance(response, dict), "Policy worker response must be an object")
        _require(result.returncode == 0 and response.get("status") == "CANDIDATE", response.get("error", "Policy worker failed"))
        _require(response.get("model_inference_executed") is True, "No genuine model inference in response")
        _require(response.get("observation_id") == observation["observation_id"] and
                 response.get("artifact_id") == manifest["artifact_id"] and response.get("mode") == mode,
                 "Policy response identity mismatch")
        _require(response.get("action_names") == manifest["action_names"] and
                 response.get("action_units") == manifest["action_units"], "Policy action mapping mismatch")
        _vector(response.get("action"), len(manifest["action_names"]), "action")
        # Recheck after inference; a valid input can expire during model execution.
        validate_observation(manifest, observation, mode)
        response["controller_ready"] = False
        response["wall_seconds"] = time.monotonic() - started
        return response
