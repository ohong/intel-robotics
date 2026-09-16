#!/usr/bin/env python3
"""Studio-only inference worker. No camera, serial, gym, or robot is opened."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secondlook.policy import (PolicyContractError, STUDIO_REVISION, _require,
                               artifact_path, validate_manifest, validate_observation,
                               verify_artifacts, validate_engineering_weight_keys, engineering_conditioning)


MODEL_INFERENCE_EXECUTED = False


def inspect_contract(checkpoint: Path | None, state_file: Path | None) -> dict:
    import torch
    from physicalai.data import Observation
    from physicalai.policies.smolvla.policy import SmolVLA
    from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor
    from physicalai.policies.pi05.pretrained_utils import parse_preprocessor_stats

    result = {"status": "BLOCKED", "model_inference_executed": False, "controller_ready": False,
              "versions": {name: version(name) for name in ("physicalai", "physicalai-train", "torch")},
              "studio_import_verified": SmolVLA.__name__ == "SmolVLA", "blockers": []}
    if checkpoint:
        config = json.loads((checkpoint / "config.json").read_text())
        preprocessor = json.loads((checkpoint / "policy_preprocessor.json").read_text())
        # Read actual saved tensors. Unlike extract_dataset_stats, this has no identity fallback.
        actual = parse_preprocessor_stats(preprocessor, config, checkpoint)
        result["checkpoint_input_features"] = config["input_features"]
        result["checkpoint_output_features"] = config["output_features"]
        result["actual_normalization"] = actual
        result["files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in checkpoint.iterdir() if p.is_file()}
        state_keys = [key for key in actual if "state" in key.lower()]
        action_keys = [key for key in actual if "action" in key.lower()]
        if not state_keys:
            result["blockers"].append("Checkpoint has no measured state normalization statistics")
        if len(action_keys) != 1:
            result["blockers"].append("Checkpoint has multiple embodiment action statistics; mapping is unknown")
        camera_names = [k.removeprefix("observation.images.") for k in config["input_features"]
                        if k.startswith("observation.images.")]
        pre = SmolVLAPreprocessor(image_key_reorder_map={k: i for i, k in enumerate(camera_names)},
                                 num_cameras=len(camera_names))
        result["camera_slots_verified"] = pre._camera_slot_layout([f"images.{k}" for k in camera_names])
        try:
            pre._camera_slot_layout(["images.wrong_camera"])
        except ValueError:
            result["camera_mismatch_rejected"] = True
    if state_file:
        state = json.loads(state_file.read_text())
        obs = Observation(state=torch.tensor([state["values"]], dtype=torch.float32),
                          task=["Recorded contract inspection only; no motion."])
        result["recorded_state_shape"] = list(obs.state.shape)
        result["recorded_state_names"] = state["joints"]
        result["recorded_state_units"] = state["units"]
        result["recorded_state_captured_at"] = state["captured_at_utc"]
        result["state_source_sha256"] = hashlib.sha256(state_file.read_bytes()).hexdigest()
        result["state_evidence"] = "recorded_real; no camera synchronization established"
    result["blockers"].append("No task-adapted checkpoint, task dataset, or verified action semantics")
    return result


def run(manifest_path: Path, request: dict) -> dict:
    global MODEL_INFERENCE_EXECUTED
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    mode, observation = request["mode"], request["observation"]
    prompt = validate_observation(manifest, observation, mode)
    root = manifest_path.parent.resolve()
    checkpoint, backbone = verify_artifacts(manifest, root)
    revision = subprocess.run(["git", "-C", manifest["studio_root"], "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    _require(revision == STUDIO_REVISION, "Installed Studio revision changed")
    for package, expected in manifest["runtime_versions"].items():
        _require(version(package) == expected, f"Runtime version mismatch: {package}")

    import numpy as np
    import torch
    from PIL import Image
    from safetensors import safe_open
    from physicalai.data import Observation
    from physicalai.policies.smolvla.policy import SmolVLA
    from physicalai.policies.smolvla.pretrained_utils import extract_dataset_stats, fix_state_dict_keys
    from physicalai.policies.pi05.pretrained_utils import parse_preprocessor_stats

    config = json.loads((checkpoint / "config.json").read_text())
    preprocessor = json.loads((checkpoint / "policy_preprocessor.json").read_text())
    for step in preprocessor.get("steps", []):
        if step.get("state_file"):
            path = artifact_path(root, str(Path(manifest["checkpoint_dir"]) / step["state_file"]))
            _require(path.relative_to(root).as_posix() in manifest["files"], "Unpinned normalizer file")
    saved_stats = parse_preprocessor_stats(preprocessor, config, checkpoint)
    runtime_stats = extract_dataset_stats(config, checkpoint / "policy_preprocessor.json", checkpoint)
    for kind, key in (("state", "observation.state"), ("action", "action")):
        candidates = [value for name, value in saved_stats.items() if kind in name.lower()]
        _require(len(candidates) == 1, f"Missing or ambiguous saved {kind} normalization; identity fallback refused")
        _require(runtime_stats.get(key, {}).get("shape") == (len(manifest[kind + "_names"]),),
                 f"Checkpoint {kind} dimension mismatch")
        for stat in ("mean", "std"):
            expected = manifest["normalization"][kind][stat]
            _require(candidates[0].get(stat) == expected and runtime_stats[key].get(stat) == expected,
                     f"Checkpoint {kind} {stat} mismatch")
    expected_cameras = {f"observation.images.{name}" for name in manifest["cameras"]}
    actual_cameras = {name for name, feature in config["input_features"].items() if feature["type"] == "VISUAL"}
    _require(expected_cameras == actual_cameras, "Checkpoint camera feature keys mismatch")
    images = {}
    for key, image in observation["images"].items():
        path = Path(image["path"])
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == image["sha256"], f"Camera {key} image checksum changed")
        import io
        with Image.open(io.BytesIO(raw)) as decoded:
            rgb = decoded.convert("RGB")
            _require([rgb.height, rgb.width, 3] == image["shape"], "Decoded image shape mismatch")
            images[key] = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    load_start = time.monotonic()
    policy = SmolVLA(pretrained_name_or_path=checkpoint,
                     image_key_reorder_map={key: item["slot"] for key, item in manifest["cameras"].items()},
                     num_cameras=len(images), tokenizer_max_length=manifest["tokenizer_max_length"])
    # Studio loads strict=False. Refuse missing/unexpected weights instead of silently using random tensors.
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as weights:
        keys = set(fix_state_dict_keys({key: None for key in weights.keys()}))
    _require(keys == set(policy.model.state_dict()), "Checkpoint weights do not exactly match the Studio model")
    policy = policy.to(manifest["device"]).eval()
    encoded = policy._preprocessor.tokenizer(prompt, truncation=False)["input_ids"]
    _require(len(encoded) <= manifest["tokenizer_max_length"], "Task/anomaly text would be truncated")
    obs = Observation(state=torch.tensor([observation["state"]["values"]], dtype=torch.float32),
                      images=images, task=[prompt])
    policy.reset()  # Each request must use its own observation, never an older action chunk.
    load_seconds = time.monotonic() - load_start
    started = time.monotonic()
    with torch.inference_mode():
        action = policy.select_action(obs).detach().cpu()
        MODEL_INFERENCE_EXECUTED = True
    _require(list(action.shape) == [1, len(manifest["action_names"])], "Policy returned an unexpected action shape")
    _require(bool(torch.isfinite(action).all()), "Policy returned nonfinite actions")
    validate_observation(manifest, observation, mode)
    return {"status": "CANDIDATE", "model_inference_executed": True, "controller_ready": False,
            "mode": mode, "evidence": "UNADAPTED_BASE_REPLAY" if manifest["adaptation"] == "base" else mode,
            "observation_id": observation["observation_id"], "artifact_id": manifest["artifact_id"],
            "action": action[0].tolist(), "action_names": manifest["action_names"],
            "action_units": manifest["action_units"], "action_semantics": manifest["action_semantics"],
            "conditioning_text": prompt, "policy": "physicalai.SmolVLA", "device": manifest["device"],
            "model_load_seconds": load_seconds, "inference_seconds": time.monotonic() - started}


def engineering_base_probe(root: Path, image_path: Path, anomaly_report_path: Path | None = None) -> dict:
    """Exercise pinned pretrained weights with explicitly artificial normalization.

    This route never uses the task manifest, returns physical units, or returns a
    controller candidate. The synthetic state/statistics are probe inputs only.
    """
    global MODEL_INFERENCE_EXECUTED
    import io
    import numpy as np
    import torch
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import save_file
    from physicalai.data import Observation
    from physicalai.policies.smolvla.policy import SmolVLA
    from physicalai.policies.smolvla.pretrained_utils import fix_state_dict_keys

    import inspect
    import socket
    policy_source = Path(inspect.getfile(SmolVLA)).resolve()
    studio_root = policy_source.parents[5]
    actual_revision = subprocess.run(["git", "-C", str(studio_root), "rev-parse", "HEAD"],
                                     check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    _require(actual_revision == STUDIO_REVISION, "Engineering Studio revision mismatch")
    source_paths = [policy_source.parent / name for name in
                    ("policy.py", "model.py", "preprocessor.py", "pretrained_utils.py")]
    source_paths.append(policy_source.parent.parent / "mixins/snapflow.py")
    clean = subprocess.run(["git", "-C", str(studio_root), "diff", "--quiet", "HEAD", "--",
                            *[str(path.relative_to(studio_root)) for path in source_paths]], timeout=5)
    _require(clean.returncode == 0, "Relevant installed Studio policy source differs from pinned revision")
    source_hashes = {str(path.relative_to(studio_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    root = root.resolve()
    downloads = json.loads((root / "downloads.json").read_text())
    expected_revisions = {
        "lerobot/smolvla_base": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    }
    required_files = {"checkpoint/config.json", "checkpoint/model.safetensors"} | {
        f"backbone/{name}" for name in (
            "config.json", "added_tokens.json", "chat_template.json", "generation_config.json",
            "merges.txt", "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
            "tokenizer.json", "tokenizer_config.json", "vocab.json")}
    inventory = {str(Path(entry["path"]).relative_to(root)) for entry in downloads["downloads"]}
    _require(inventory == required_files, "Pinned engineering download inventory is incomplete")
    for entry in downloads["downloads"]:
        _require(entry["revision"] == expected_revisions.get(entry["repo"]), "Unexpected model revision")
        path = Path(entry["path"]).resolve()
        _require(path.is_relative_to(root), "Downloaded artifacts must stay inside probe directory")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        _require(actual == entry["sha256"], f"Downloaded artifact checksum mismatch: {path.name}")
        if entry["repo"] == "lerobot/smolvla_base" and path.name == "model.safetensors":
            _require(actual == "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb",
                     "Policy weights do not match the upstream pinned LFS digest")
    checkpoint, backbone = root / "checkpoint", root / "backbone"
    # Keep original downloads intact. A hard link avoids copying 907 MB of weights.
    fixture = root / "identity-fixture-checkpoint"
    fixture.mkdir(exist_ok=True)
    weight_path = fixture / "model.safetensors"
    if not weight_path.exists():
        os.link(checkpoint / "model.safetensors", weight_path)
    config = json.loads((checkpoint / "config.json").read_text())
    config["vlm_model_name"] = str(backbone)
    # The policy checkpoint includes the trained VLM weights. Requiring exact keys
    # below prevents random model parameters when skipping redundant backbone weights.
    config["load_vlm_weights"] = False
    (fixture / "config.json").write_text(json.dumps(config, indent=2))
    fixture_stats = {f"{name}.{kind}": torch.zeros(6) if kind == "mean" else torch.ones(6)
                     for name in ("observation.state", "action") for kind in ("mean", "std")}
    save_file(fixture_stats, str(fixture / "identity-fixture.safetensors"))
    (fixture / "policy_preprocessor.json").write_text(json.dumps({"steps": [{
        "registry_name": "normalizer_processor", "state_file": "identity-fixture.safetensors",
        "config": {"norm_map": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}},
    }]}))
    raw = image_path.read_bytes()
    with Image.open(io.BytesIO(raw)) as decoded:
        rgb = decoded.convert("RGB")
        source_shape = [rgb.height, rgb.width, 3]
        image_tensor = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    anomaly_report = json.loads(anomaly_report_path.read_text()) if anomaly_report_path else None
    prompt, anomaly_provenance = engineering_conditioning(hashlib.sha256(raw).hexdigest(), anomaly_report)
    load_start = time.monotonic()
    policy = SmolVLA(pretrained_name_or_path=fixture, image_key_reorder_map={"camera1": 0},
                     num_cameras=3, tokenizer_max_length=128).to("cpu").eval()
    with safe_open(str(weight_path), framework="pt") as weights:
        weight_keys = set(fix_state_dict_keys({key: None for key in weights.keys()}))
    model_keys = set(policy.model.state_dict())
    inactive_missing = validate_engineering_weight_keys(
        weight_keys, model_keys, policy.model._model._snapflow_enabled)
    for name in ("observation.state", "action"):
        _require(policy._dataset_stats[name]["mean"] == [0.] * 6 and
                 policy._dataset_stats[name]["std"] == [1.] * 6,
                 "Explicit engineering identity statistics did not reach Studio")
    token_count = len(policy._preprocessor.tokenizer(prompt, truncation=False)["input_ids"])
    _require(token_count <= 128, "Engineering prompt exceeds token budget")
    observation = Observation(state=torch.zeros(1, 6), images={"camera1": image_tensor}, task=[prompt])
    policy.reset()
    load_seconds = time.monotonic() - load_start
    started = time.monotonic()
    with torch.inference_mode():
        action = policy.select_action(observation).detach().cpu()
    MODEL_INFERENCE_EXECUTED = True
    inference_seconds = time.monotonic() - started
    _require(list(action.shape) == [1, 6] and bool(torch.isfinite(action).all()), "Invalid engineering output")
    return {
        "status": "ENGINEERING_PROBE_ONLY", "model_inference_executed": True,
        "controller_ready": False, "task_ready": False, "motion_commanded": False,
        "evidence": ("real_recorded_image_real_anomaly_synthetic_state_identity_fixture_statistics" if anomaly_report
                     else "real_recorded_image_synthetic_state_and_anomaly_identity_fixture_statistics"),
        "policy": "physicalai.SmolVLA.select_action", "device": "cpu", "torch_threads": 2,
        "host": socket.gethostname(),
        "versions": {name: version(name) for name in ("physicalai", "physicalai-train", "torch")},
        "studio_revision": actual_revision, "studio_source_hashes": source_hashes,
        "checkpoint_revisions": expected_revisions,
        "all_active_pretrained_weight_keys_match": True, "pretrained_weight_key_count": len(weight_keys),
        "inactive_missing_keys": inactive_missing, "snapflow_enabled": False,
        "state": {"values": [0.] * 6, "provenance": "synthetic_normalized_fixture", "physical_units": None},
        "normalization": "explicit identity fixture; not dataset or robot normalization",
        "image": {"path": str(image_path), "sha256": hashlib.sha256(raw).hexdigest(), "shape": source_shape,
                  "camera_slot": 0, "slot_semantics": "unverified engineering mapping", "masked_slots": [1, 2]},
        "conditioning_text": prompt, "conditioning_token_count": token_count,
        "anomaly_provenance": anomaly_provenance,
        "anomaly_stage": anomaly_report,
        "anomaly_report_sha256": hashlib.sha256(anomaly_report_path.read_bytes()).hexdigest() if anomaly_report_path else None,
        "output_vector": action[0].tolist(), "action_semantics": None, "physical_units": None,
        "model_load_seconds": load_seconds, "inference_seconds": inference_seconds,
        "inference_sample_count": 1, "denoising_steps": policy.config.num_steps,
        "latency_scope": "one cold-process select_action call; excludes import, artifact hashing, image decode, and model load",
        "captured_at_unix": time.time(),
        "limits": ["No task adaptation", "No physical state/action mapping", "No motion validation",
                   "No demonstrated defect understanding", "No autonomous sorting success"],
        "fixture_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in fixture.iterdir() if path.name != "model.safetensors"},
        "downloads": downloads["downloads"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--engineering-base-probe", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--anomaly-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-inspection", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--scaffold", type=Path, help="Write an explicitly incomplete manifest during --inspect")
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    try:
        # Import/model logs must never corrupt the single JSON response on stdout.
        with redirect_stdout(sys.stderr):
            if args.engineering_base_probe:
                _require(args.image is not None, "Engineering probe requires a recorded image")
                result = engineering_base_probe(args.engineering_base_probe, args.image.resolve(), args.anomaly_report)
            elif args.inspect:
                result = inspect_contract(args.checkpoint_inspection, args.state)
                if args.scaffold:
                    _require(args.state is not None and args.checkpoint_inspection is not None,
                             "Scaffold requires checkpoint inspection and recorded state")
                    state = json.loads(args.state.read_text())
                    scaffold = {
                        "schema_version": 1, "artifact_id": "BLOCKED-no-task-policy",
                        "checkpoint_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
                        "studio_revision": STUDIO_REVISION,
                        "studio_root": "/home/ird-demo/physical-ai-studio",
                        "runtime_versions": result["versions"], "adaptation": "base",
                        "task_id": None, "task_validation_passed": False,
                        "task_validation_evidence": None, "checkpoint_dir": None,
                        "backbone_dir": None, "calibration_id": state["calibration_sha256"],
                        "state_names": state["joints"], "state_units": state["units"],
                        "action_names": None, "action_units": None, "action_semantics": None,
                        "normalization": {"state": None, "action": None}, "cameras": None,
                        "anomaly_camera_keys": None,
                        "tokenizer_max_length": 48, "max_age_seconds": None,
                        "max_sensor_skew_seconds": None, "device": "cpu", "files": {},
                        "blockers": result["blockers"], "model_inference_executed": False,
                        "checkpoint_inspection": result,
                    }
                    args.scaffold.parent.mkdir(parents=True, exist_ok=True)
                    args.scaffold.write_text(json.dumps(scaffold, indent=2) + "\n")
                    result["scaffold_path"] = str(args.scaffold)
            else:
                _require(args.manifest is not None, "--manifest is required")
                result = run(args.manifest.resolve(), json.load(sys.stdin))
        if args.output:
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        failure = {"status": "BLOCKED", "error": f"{type(error).__name__}: {error}",
                   "model_inference_executed": MODEL_INFERENCE_EXECUTED, "controller_ready": False,
                   "task_ready": False}
        if args.output:
            args.output.write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
