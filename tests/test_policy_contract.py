"""Portable contract tests; all fixtures are synthetic and no model is executed."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from secondlook.policy import (PolicyContractError, STUDIO_REVISION, StudioPolicyBridge,
                               validate_manifest, validate_observation, verify_artifacts, validate_engineering_weight_keys, engineering_conditioning)


def manifest_fixture():
    return {"schema_version": 1, "artifact_id": "test-only", "checkpoint_revision": "test-revision",
            "calibration_id": "test-calibration", "task_id": "test-task", "checkpoint_dir": "checkpoint",
            "backbone_dir": "backbone", "studio_root": "/test/studio", "studio_revision": STUDIO_REVISION,
            "adaptation": "task_adapted", "task_validation_passed": True,
            "task_validation_evidence": "validation.json", "action_semantics": "absolute_joint_position",
            "state_names": ["joint", "gripper"], "action_names": ["joint", "gripper"],
            "state_units": ["degree", "percent"], "action_units": ["degree", "percent"],
            "normalization": {key: {"mean": [0, 0], "std": [1, 1]} for key in ("state", "action")},
            "cameras": {"top": {"shape": [480, 640, 3], "source_id": "test-camera", "slot": 0}},
            "anomaly_camera_keys": ["top"],
            "max_age_seconds": 0.5, "max_sensor_skew_seconds": 0.1, "tokenizer_max_length": 48,
            "device": "cpu", "runtime_versions": {k: "test" for k in ("physicalai", "physicalai-train", "torch")},
            "files": {"validation.json": "a" * 64}}


def observation_fixture():
    return {"schema_version": 1, "observation_id": "test-observation", "task_id": "test-task",
            "task": "Hold the object.", "calibration_id": "test-calibration", "provenance": "real",
            "captured_at": 100., "state": {"names": ["joint", "gripper"], "units": ["degree", "percent"],
                                           "values": [3, 4], "captured_at": 100.},
            "images": {"top": {"shape": [480, 640, 3], "source_id": "test-camera", "color": "RGB",
                               "path": "/synthetic/fixture.png", "sha256": "b" * 64, "captured_at": 100.}},
            "anomaly": {"observation_id": "test-observation", "score": .8, "threshold": .5,
                        "decision": "anomalous", "model_id": "test-model", "evidence": "real_inference",
                        "source_images": {"top": {"source_id": "test-camera", "sha256": "b" * 64}},
                        "captured_at": 100.}}


class PolicyContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.obs = manifest_fixture(), observation_fixture()

    def test_task_and_anomaly_reach_conditioning(self):
        text = validate_observation(self.manifest, self.obs, "live", now=100.1)
        self.assertIn("Hold the object.", text)
        self.assertIn("anomalous; score=0.8; threshold=0.5", text)

    def test_missing_or_mismatched_state_is_rejected(self):
        for field, value in (("values", [0]), ("values", [float("nan"), 0]),
                             ("values", [True, 0]), ("names", ["gripper", "joint"]),
                             ("units", ["radian", "percent"])):
            obs = copy.deepcopy(self.obs)
            obs["state"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(PolicyContractError):
                validate_observation(self.manifest, obs, "replay")
        del self.obs["state"]
        with self.assertRaises(PolicyContractError):
            validate_observation(self.manifest, self.obs, "replay")

    def test_camera_contract_is_exact(self):
        for field, value in (("source_id", "other"), ("color", "BGR"), ("shape", [640, 480, 3]),
                             ("path", "relative.png"), ("sha256", "wrong")):
            obs = copy.deepcopy(self.obs)
            obs["images"]["top"][field] = value
            with self.subTest(field=field), self.assertRaises(PolicyContractError):
                validate_observation(self.manifest, obs, "replay")
        self.obs["images"]["unexpected"] = self.obs["images"]["top"]
        with self.assertRaises(PolicyContractError):
            validate_observation(self.manifest, self.obs, "replay")

    def test_stale_future_and_unsynchronized_input_rejected(self):
        for now in (99.9, 101.):
            with self.assertRaises(PolicyContractError):
                validate_observation(self.manifest, self.obs, "live", now=now)
        self.obs["state"]["captured_at"] = 90.
        with self.assertRaises(PolicyContractError):
            validate_observation(self.manifest, self.obs, "replay")

    def test_base_checkpoint_never_live(self):
        self.manifest["adaptation"] = "base"
        with self.assertRaisesRegex(PolicyContractError, "task-adapted"):
            validate_observation(self.manifest, self.obs, "live", now=100.1)
        validate_observation(self.manifest, self.obs, "replay")

    def test_task_validator_rejects_synthetic_state_and_engineering_mode(self):
        self.obs["provenance"] = "real_image_synthetic_state"
        for mode in ("live", "replay", "engineering_probe"):
            with self.assertRaises(PolicyContractError):
                validate_observation(self.manifest, self.obs, mode, now=100.1)

    def test_observation_and_anomaly_association_required(self):
        self.obs["anomaly"]["observation_id"] = "other-object"
        with self.assertRaises(PolicyContractError):
            validate_observation(self.manifest, self.obs, "replay")

    def test_same_identity_and_time_cannot_hide_swapped_detector_image(self):
        # The ID and every timestamp stay unchanged; only VLA image content changes.
        self.obs["images"]["top"]["sha256"] = "c" * 64
        for mode in ("replay", "live"):
            with self.subTest(mode=mode), self.assertRaisesRegex(PolicyContractError, "source image/camera mismatch"):
                validate_observation(self.manifest, self.obs, mode, now=100.1)

    def test_bridge_rejects_content_swap_before_starting_worker(self):
        self.obs["images"]["top"]["sha256"] = "c" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(self.manifest))
            with patch("secondlook.policy.subprocess.run") as worker:
                for mode in ("replay", "live"):
                    with self.subTest(mode=mode), self.assertRaisesRegex(PolicyContractError, "source image/camera mismatch"):
                        StudioPolicyBridge(sys.executable, path).infer(self.obs, mode=mode)
                worker.assert_not_called()

    def test_direct_worker_rejects_content_swap_before_model_import(self):
        self.obs["images"]["top"]["sha256"] = "c" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(self.manifest))
            worker = Path(__file__).resolve().parents[1] / "scripts/policy_worker.py"
            for mode in ("replay", "live"):
                result = subprocess.run([sys.executable, str(worker), "--manifest", str(path)],
                                        input=json.dumps({"mode": mode, "observation": self.obs}),
                                        capture_output=True, text=True, timeout=5)
                response = json.loads(result.stdout)
                with self.subTest(mode=mode):
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("source image/camera mismatch", response["error"])
                    self.assertFalse(response["model_inference_executed"])
                    self.assertNotIn("action", response)

    def test_anomaly_source_mapping_is_exact(self):
        cases = [None, {}, {"other": {"source_id": "test-camera", "sha256": "b" * 64}},
                 {"top": {"source_id": "other-camera", "sha256": "b" * 64}},
                 {"top": {"source_id": "test-camera"}},
                 {"top": {"source_id": "test-camera", "sha256": "b" * 64, "extra": 1}}]
        for source_images in cases:
            obs = copy.deepcopy(self.obs)
            obs["anomaly"]["source_images"] = source_images
            with self.subTest(source_images=source_images), self.assertRaises(PolicyContractError):
                validate_observation(self.manifest, obs, "replay")

    def test_detector_camera_subset_is_explicit_not_assumed(self):
        self.manifest["cameras"]["wrist"] = {"shape": [480, 640, 3], "source_id": "wrist-camera", "slot": 1}
        self.obs["images"]["wrist"] = {**self.obs["images"]["top"], "source_id": "wrist-camera", "sha256": "d" * 64}
        validate_observation(self.manifest, self.obs, "replay")
        self.obs["anomaly"]["source_images"]["wrist"] = {"source_id": "wrist-camera", "sha256": "d" * 64}
        with self.assertRaisesRegex(PolicyContractError, "source camera keys"):
            validate_observation(self.manifest, self.obs, "replay")
        self.manifest["anomaly_camera_keys"] = ["top", "wrist"]
        validate_observation(self.manifest, self.obs, "replay")

    def test_anomaly_camera_manifest_rejects_missing_duplicate_unknown_keys(self):
        for keys in (None, [], ["top", "top"], ["missing"]):
            manifest = copy.deepcopy(self.manifest)
            manifest["anomaly_camera_keys"] = keys
            with self.subTest(keys=keys), self.assertRaises(PolicyContractError):
                validate_manifest(manifest)

    def test_task_bridge_never_starts_worker_for_engineering_mode(self):
        with patch("secondlook.policy.subprocess.run") as worker:
            with self.assertRaisesRegex(PolicyContractError, "dedicated engineering CLI"):
                StudioPolicyBridge(sys.executable).infer(self.obs, mode="engineering_probe")
            worker.assert_not_called()

    def test_direct_task_worker_rejects_engineering_mode_before_model_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(self.manifest))
            worker = Path(__file__).resolve().parents[1] / "scripts/policy_worker.py"
            result = subprocess.run([sys.executable, str(worker), "--manifest", str(path)],
                                    input=json.dumps({"mode": "engineering_probe", "observation": self.obs}),
                                    capture_output=True, text=True, timeout=5)
            response = json.loads(result.stdout)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(response["status"], "BLOCKED")
            self.assertIn("dedicated engineering CLI", response["error"])
            self.assertFalse(response["model_inference_executed"])
            self.assertFalse(response["task_ready"])
            self.assertNotIn("action", response)

    def test_normalization_cannot_be_missing(self):
        del self.manifest["normalization"]["state"]
        with self.assertRaises(PolicyContractError):
            validate_manifest(self.manifest)

    def test_unknown_semantics_and_schema_rejected(self):
        for field, value in (("action_semantics", None), ("schema_version", 2),
                             ("calibration_id", None), ("state_units", [])):
            manifest = copy.deepcopy(self.manifest)
            manifest[field] = value
            with self.subTest(field=field), self.assertRaises(PolicyContractError):
                validate_manifest(manifest)

    def test_missing_manifest_reports_blocked(self):
        bridge = StudioPolicyBridge(sys.executable)
        self.assertEqual(bridge.status()["status"], "BLOCKED")
        with self.assertRaises(PolicyContractError):
            bridge.infer(self.obs)

    def test_artifact_escape_rejected(self):
        self.manifest["checkpoint_dir"] = "../outside"
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(PolicyContractError):
            verify_artifacts(self.manifest, Path(tmp))

    def test_worker_failure_replaces_old_success_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            output.write_text(json.dumps({"status": "ENGINEERING_PROBE_ONLY"}))
            worker = Path(__file__).resolve().parents[1] / "scripts/policy_worker.py"
            result = subprocess.run([sys.executable, str(worker), "--manifest", str(Path(tmp) / "missing.json"),
                                     "--output", str(output)], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(output.read_text())["status"], "BLOCKED")
            self.assertFalse(json.loads(output.read_text())["model_inference_executed"])

    def test_worker_timeout_cannot_return_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(self.manifest))
            with patch("secondlook.policy.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", .1)):
                with self.assertRaisesRegex(PolicyContractError, "timed out"):
                    StudioPolicyBridge(sys.executable, path).infer(self.obs)

    def test_input_expiring_during_inference_is_rejected(self):
        response = {"status": "CANDIDATE", "model_inference_executed": True,
                    "artifact_id": "test-only", "observation_id": "test-observation",
                    "mode": "live", "action_names": ["joint", "gripper"],
                    "action_units": ["degree", "percent"], "action": [2, 3]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(self.manifest))
            result = subprocess.CompletedProcess([], 0, stdout=json.dumps(response), stderr="")
            with patch("secondlook.policy.subprocess.run", return_value=result), \
                 patch("secondlook.policy.time.time", side_effect=[100.1, 101.]):
                with self.assertRaisesRegex(PolicyContractError, "Stale"):
                    StudioPolicyBridge(sys.executable, path).infer(self.obs, mode="live")

    def test_same_image_uncalibrated_anomaly_conditioning(self):
        report = {"schema_version": 1, "stage": "anomalib", "model_inference_executed": True,
                  "image": {"sha256": "a" * 64, "provenance": "recorded_real"},
                  "result": {"score": 3.125, "threshold": None, "disposition": "UNKNOWN",
                             "evidence_kind": "engineering_smoke", "artifact_id": "test-artifact"}}
        prompt, provenance = engineering_conditioning("a" * 64, report)
        self.assertIn("UNKNOWN; anomaly score=3.125; threshold=null (uncalibrated)", prompt)
        self.assertIn("real Anomalib inference", provenance)
        with self.assertRaises(PolicyContractError):
            engineering_conditioning("b" * 64, report)
        for field, value in (("score", float("nan")), ("threshold", .5), ("disposition", "ANOMALOUS")):
            bad = copy.deepcopy(report)
            bad["result"][field] = value
            with self.subTest(field=field), self.assertRaises(PolicyContractError):
                engineering_conditioning("a" * 64, bad)

    def test_legacy_engineering_keys_only_allow_inactive_snapflow_branch(self):
        loaded = {"_model.action_in_proj.weight"}
        expected = loaded | {"_model.target_time_mlp_in.weight"}
        self.assertEqual(validate_engineering_weight_keys(loaded, expected, False),
                         ["_model.target_time_mlp_in.weight"])
        for actual, needed, enabled in ((loaded, expected, True),
                                        (set(), expected, False),
                                        (loaded | {"unexpected"}, expected, False)):
            with self.assertRaises(PolicyContractError):
                validate_engineering_weight_keys(actual, needed, enabled)

    def test_engineering_probe_output_cannot_enter_task_bridge(self):
        response = {"status": "ENGINEERING_PROBE_ONLY", "model_inference_executed": True,
                    "output_vector": [1, 2], "controller_ready": False}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(self.manifest))
            result = subprocess.CompletedProcess([], 0, stdout=json.dumps(response), stderr="")
            with patch("secondlook.policy.subprocess.run", return_value=result), self.assertRaises(PolicyContractError):
                StudioPolicyBridge(sys.executable, path).infer(self.obs)

    def test_malformed_worker_action_rejected(self):
        response = {"status": "CANDIDATE", "model_inference_executed": True, "artifact_id": "test-only", "observation_id": "test-observation",
                    "mode": "replay", "action_names": ["joint", "gripper"], "action_units": ["degree", "percent"],
                    "action": [2]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(self.manifest))
            result = subprocess.CompletedProcess([], 0, stdout=json.dumps(response), stderr="")
            with patch("secondlook.policy.subprocess.run", return_value=result), self.assertRaises(PolicyContractError):
                StudioPolicyBridge(sys.executable, path).infer(self.obs)


if __name__ == "__main__":
    unittest.main()
