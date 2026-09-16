"""Offline synthetic checks. No model, Torch, network, GPU, or robot is used."""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("training", Path(__file__).with_name("training_smolvla.py"))
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


class Preconditions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for name in ("data/meta", "base", "backbone", "library"):
            (root / name).mkdir(parents=True)
        self.config = json.loads(Path(__file__).with_name("training-config.template.json").read_text())
        self.config.update(dataset_root=str(root / "data"), repo_id="synthetic/fixture", studio_library=str(root / "library"), base_dir=str(root / "base"), backbone_dir=str(root / "backbone"), dataset_contract=str(root / "contract.json"), image_key_reorder_map={"top": 0})
        self.info = {"total_episodes": 2, "total_frames": 4, "fps": 20, "features": {"observation.state": {"shape": [1], "names": ["joint"]}, "action": {"shape": [1], "names": ["joint"]}, "observation.images.top": {"shape": [3, 2, 2]}}}
        self.stats = {key: {"mean": [0.0], "std": [1.0], "min": [-1.0], "max": [1.0]} for key in ("observation.state", "action")}
        (root / "data/meta/info.json").write_text(json.dumps(self.info))
        (root / "data/meta/stats.json").write_text(json.dumps(self.stats))
        (root / "base/config.json").write_text(json.dumps({"vlm_model_name": str(root / "backbone")}))
        (root / "base/model.safetensors").write_bytes(b"SYNTHETIC NON-MODEL FIXTURE")
        (root / "backbone/config.json").write_text("{}")
        contract = {"provenance": "recorded_real", "task_id": "synthetic-check-only", "calibration_id": "test", "synchronization_evidence": "fixture", "units_evidence": "fixture", "split_evidence": "fixture", "training_only": True, "action_semantics": "absolute_position", "state_names": ["joint"], "state_units": ["fixture_units"], "action_names": ["joint"], "action_units": ["fixture_units"], "camera_sources": {"top": "fixture_camera"}}
        (root / "contract.json").write_text(json.dumps(contract))
        self.refresh()

    def refresh(self):
        for field, path_field in (("dataset_files", "dataset_root"), ("base_files", "base_dir"), ("backbone_files", "backbone_dir")):
            root = Path(self.config[path_field])
            self.config[field] = {str(p.relative_to(root)): training.digest(p) for p in root.rglob("*") if p.is_file()}

    def test_valid_declarations_are_only_input_check(self):
        self.assertEqual(training.check(self.config)["contract"]["task_id"], "synthetic-check-only")

    def test_template_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Provide dataset_root"):
            training.check(json.loads(Path(__file__).with_name("training-config.template.json").read_text()))

    def test_changed_data_rejected(self):
        (Path(self.config["dataset_root"]) / "extra.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "cover every file"):
            training.check(self.config)

    def test_zero_and_nonfinite_stats_rejected(self):
        for value in (0.0, float("nan")):
            self.stats["action"]["std"] = [value]
            (Path(self.config["dataset_root"]) / "meta/stats.json").write_text(json.dumps(self.stats))
            self.refresh()
            with self.assertRaisesRegex(ValueError, "Invalid action std"):
                training.check(self.config)

    def test_unverified_or_mixed_data_rejected(self):
        contract_path = Path(self.config["dataset_contract"])
        original = json.loads(contract_path.read_text())
        for field, value in (("provenance", "synthetic"), ("training_only", False), ("units_evidence", None)):
            contract = deepcopy(original)
            contract[field] = value
            contract_path.write_text(json.dumps(contract))
            with self.assertRaises(ValueError):
                training.check(self.config)

    def test_mapping_and_output_rejected(self):
        for field, value in (("image_key_reorder_map", {"wrong": 0}), ("output_dir", "/tmp/run"), ("max_steps", -1)):
            config = deepcopy(self.config)
            config[field] = value
            with self.assertRaises(ValueError):
                training.check(config)


class SourceContract(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("STUDIO_LIBRARY"), "Set STUDIO_LIBRARY for pinned source API checks")
    def test_actual_keyword_arguments_exist(self):
        library = Path(os.environ["STUDIO_LIBRARY"])
        source = ast.parse(Path(__file__).with_name("training_smolvla.py").read_text())
        for class_name, relative in (("Trainer", "train/trainer.py"), ("LeRobotDataModule", "data/lerobot/datamodule.py"), ("SmolVLA", "policies/smolvla/policy.py")):
            parsed = ast.parse((library / "src/physicalai" / relative).read_text())
            cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
            params = {p.arg for p in init.args.args + init.args.kwonlyargs}
            calls = [n for n in ast.walk(source) if isinstance(n, ast.Call) and (isinstance(n.func, ast.Name) and n.func.id == class_name or isinstance(n.func, ast.Attribute) and n.func.attr == class_name)]
            self.assertTrue(calls)
            for call in calls:
                self.assertLessEqual({k.arg for k in call.keywords}, params)


if __name__ == "__main__":
    unittest.main()
