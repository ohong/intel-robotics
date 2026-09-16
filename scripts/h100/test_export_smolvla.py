"""Portable synthetic contract fixtures; these never establish real policy parity."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import json
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import export_smolvla as exporter


class ExportContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'evidence_kind': 'real', 'finalized': True, 'status': 'OFFLINE_DATA_VALIDATED',
                         'normalization': 'train episodes only; fixture label', 'train_episodes': [0],
                         'validation_episodes': [1], 'final_eval_episodes': [], 'joint_names': ['a', 'b'],
                         'episodes': [{'episode_index': 0, 'frames': 4}, {'episode_index': 1, 'frames': 2}],
                         'cameras': ['observation.images.camera1', 'observation.images.camera2']}
        self.stats = {key: {'mean': [2.25, 3.5], 'std': [.5, 1.25], 'min': [1., 2.], 'max': [4., 5.],
                            'q01': [1.1, 2.1], 'q99': [3.9, 4.9], 'count': [4]}
                      for key in ('observation.state', 'action')}
        self.native = {key: {**copy.deepcopy(value), 'shape': (2,), 'name': key.removeprefix('observation.')}
                       for key, value in self.stats.items()}
        self.native.update({f'observation.camera{i}': {'name': f'camera{i}', 'shape': (3, 480, 640)} for i in (1, 2)})
        self.config = {key: False for key in exporter.HF_OVERRIDES}
        self.config.update(num_cameras=3, tokenizer_max_length=128, image_key_reorder_map={'camera1': 0, 'camera2': 2},
                           load_vlm_weights=False, adapt_to_pi_aloha=False, use_random_input_noise=True,
                           n_action_steps=50, vlm_model_name='/old/backbone')
        self.hparams = {'config': self.config, 'dataset_stats': self.native}

    def plan(self):
        return exporter.export_plan(self.hparams, self.manifest, self.stats, Path('/new/backbone'))

    def test_two_real_cameras_three_slots_and_native_noise_preserved(self):
        config, stats, kwargs, slots = self.plan()
        self.assertEqual(slots, ['camera1', None, 'camera2'])
        self.assertEqual(config['vlm_model_name'], '/new/backbone')
        self.assertEqual(self.config['vlm_model_name'], '/old/backbone')
        self.assertTrue(kwargs['use_random_input_noise'])
        self.assertEqual(kwargs['num_cameras'], 3)
        self.assertEqual(kwargs['n_action_steps'], 50)
        self.assertEqual(stats['action']['mean'], self.stats['action']['mean'])
        self.assertEqual(set(config['input_features']), {'observation.state', 'observation.images.camera1', 'observation.images.camera2'})

    def test_actual_qc_normalization_and_camera_identity_schema(self):
        self.manifest['camera_identity_mapping'] = {key: {} for key in self.manifest.pop('cameras')}
        self.manifest['normalization'] = {'training_episodes_only': [0], 'training_rows': 4,
                                          'excluded_validation_episodes': [1], 'excluded_final_episodes': []}
        self.assertEqual(self.plan()[3], ['camera1', None, 'camera2'])
        self.manifest['normalization']['excluded_validation_episodes'] = []
        with self.assertRaisesRegex(ValueError, 'normalization split differs'):
            self.plan()

    def test_wrong_split_or_artificial_source_rejected(self):
        for change in ({'evidence_kind': 'synthetic'}, {'finalized': False}, {'train_episodes': [1]},
                       {'normalization': 'all episodes'}, {'validation_episodes': [0]}):
            with self.subTest(change=change):
                original = copy.deepcopy(self.manifest)
                self.manifest.update(change)
                with self.assertRaises(ValueError):
                    self.plan()
                self.manifest = original

    def test_missing_identity_or_wrong_statistics_rejected(self):
        for key, value in [('mean', [0., 0.]), ('std', [1., 1.]), ('q99', None)]:
            original = self.native['action'][key]
            self.native['action'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                self.plan()
            self.native['action'][key] = original
        self.stats['action']['count'] = [6]
        with self.assertRaisesRegex(ValueError, 'frame count'):
            self.plan()

    def test_nan_negative_std_and_camera_slot_errors_rejected(self):
        self.stats['action']['mean'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid statistics'):
            self.plan()
        self.setUp()
        self.config['image_key_reorder_map']['camera2'] = 0
        with self.assertRaisesRegex(ValueError, 'distinct integers'):
            self.plan()
        self.setUp()
        self.config['tokenizer_max_length'] = 48
        with self.assertRaisesRegex(ValueError, 'length 128'):
            self.plan()

    def test_all_active_and_inactive_keys_required(self):
        expected = exporter.INACTIVE_KEYS | {'_model.active.weight'}
        values = {f'model.{key}': object() for key in expected}
        values['_preprocessor.state.mean'] = object()
        self.assertEqual(set(exporter.model_weights(values, expected)), expected)
        del values['model._model.target_time_mlp_in.weight']
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            exporter.model_weights(values, expected)

    def test_provenance_requires_selected_weights_and_real_train_stats_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, run_path, result_path, manifest_path, stats_path = [root / name for name in ('checkpoint', 'run.json', 'result.json', 'manifest.json', 'stats.json')]
            library, backbone = root / 'library', root / 'backbone'
            library.mkdir()
            backbone.mkdir()
            (library / 'policy.py').write_text('source fixture')
            (backbone / 'config.json').write_text('{}')
            checkpoint.write_bytes(b'checkpoint fixture')
            exporter.dump(stats_path, self.stats)
            self.manifest['train_stats_sha256'] = exporter.digest(stats_path)
            exporter.dump(manifest_path, self.manifest)
            run = {'run_identity': 'fixture', 'config': {'training_scope': 'real_data_pipeline_probe', 'backbone_files': exporter.inventory(backbone)},
                   'studio_source_sha256': {'policy.py': exporter.digest(library / 'policy.py')},
                   'manifest_sha256': exporter.digest(manifest_path), 'train_stats_sha256': exporter.digest(stats_path)}
            result = {'run_identity': 'fixture', 'evidence_kind': 'real', 'global_step': 1,
                      'status': 'REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE', 'selected_sha256': exporter.digest(checkpoint),
                      'controller_ready': False, 'task_ready': False, 'anomalib_conditioned': False}
            exporter.dump(run_path, run)
            exporter.dump(result_path, result)
            args = (checkpoint, run_path, result_path, manifest_path, stats_path, library, backbone)
            exporter.check_provenance(*args)
            stats_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'statistics hash mismatch'):
                exporter.check_provenance(*args)
            exporter.dump(stats_path, self.stats)
            checkpoint.write_bytes(b'wrong selection')
            with self.assertRaisesRegex(ValueError, 'validation-selected'):
                exporter.check_provenance(*args)

    def test_source_contract_matches_current_loader_when_checkout_available(self):
        source = Path('/Users/ohong/dev/intel-robotics-integration/pc-context/sources/physical-ai-studio/library/src/physicalai/policies/smolvla/policy.py')
        if not source.exists():
            self.skipTest('Frozen Studio source is unavailable on this host')
        tree = ast.parse(source.read_text())
        loader = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_from_hf')
        overwritten = set()
        for node in ast.walk(loader):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == 'hf_config':
                        overwritten.add(ast.literal_eval(target.slice))
        self.assertEqual(set(exporter.HF_OVERRIDES), overwritten)


if __name__ == '__main__':
    unittest.main()
