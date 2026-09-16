"""Contract tests use disposable manifests; no ML, CUDA, or robot dependencies."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training_act import check, check_resume_packages, check_smoke_gate, check_splits, digest, final_eval_status, identity, snapshot
from verify_act_intel import check_action_shape


class ActContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        capture = patch('training_act.check_capture', return_value={'kind': 'disposable contract fixture'})
        capture.start()
        self.addCleanup(capture.stop)
        base = Path(self.temporary.name)
        self.root = base / 'dataset'
        self.root.mkdir()
        (self.root / 'fixture').write_text('DISPOSABLE CONTRACT FIXTURE; NOT ROBOT DATA')
        self.stats_path = base / 'train_stats.json'
        self.stats_path.write_text(json.dumps({key: {'mean': [0], 'std': [1], 'min': [0], 'max': [1],
                                                     'q01': [0], 'q99': [1], 'count': [4]}
                                                   for key in ('action', 'observation.state')}))
        self.manifest = {'status': 'OFFLINE_DATA_VALIDATED', 'evidence_kind': 'real', 'finalized': True,
                         'split_scope': 'grouped_evaluation',
                         'skill': 'contract-fixture-only', 'train_episodes': [0], 'validation_episodes': [1],
                         'final_eval_episodes': [2], 'episodes': [
                             {'episode_index': i, 'frames': 4, 'destination': 'fixture', 'outcome': 'complete_success',
                              'outcome_evidence': 'Disposable contract assertion, not physical evidence',
                              'specimen_id': f'fixture-specimen-{i}', 'session_id': f'fixture-session-{i}'} for i in range(3)],
                         'files': snapshot(self.root), 'joint_names': ['fixture_joint'],
                         'normalization': 'train episodes only; test', 'train_stats_sha256': digest(self.stats_path)}
        self.manifest_path = base / 'manifest.json'
        self.config = {'studio_revision': 'c4ff730fb49f84e5102d01088d52cfff1ba62854', 'manifest': str(self.manifest_path),
                       'dataset_root': str(self.root), 'skill': 'contract-fixture-only',
                       'output_dir': '/workspace/second-look-h100/runs/contract-fixture', 'batch_size': 2,
                       'max_steps': 10, 'max_seconds': 60, 'validation_every': 2, 'checkpoint_every': 2,
                       'seed': 42, 'num_workers': 0, 'precision': '32-true',
                       'policy': {'pretrained_backbone_weights': None, 'compile_model': False, 'chunk_size': 4, 'n_action_steps': 4}}
        self.save()

    def save(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def test_complete_contract(self):
        self.assertEqual(check(self.config, 'train')['root'], self.root.resolve())

    def test_adjacent_frames_cannot_override_whole_episode_split(self):
        self.manifest['validation_episodes'] = [0, 1]
        with self.assertRaisesRegex(ValueError, 'leakage'):
            check_splits(self.manifest, 'train')

    def test_final_holdout_required_for_training_but_two_episode_smoke_allowed(self):
        self.manifest['final_eval_episodes'] = []
        self.manifest['episodes'].pop()
        check_splits(self.manifest, 'smoke')
        with self.assertRaisesRegex(ValueError, 'final_eval'):
            check_splits(self.manifest, 'train')

    def test_synthetic_cannot_enter_real_training(self):
        self.manifest['evidence_kind'] = 'synthetic'
        for ep in self.manifest['episodes']:
            ep['outcome'] = 'synthetic_success'
        self.save()
        with self.assertRaisesRegex(ValueError, 'requires real'):
            check(self.config, 'train')
        check(self.config, 'software-qualification')

    def test_real_cannot_be_mislabeled_qualification(self):
        with self.assertRaisesRegex(ValueError, 'Qualification requires synthetic'):
            check(self.config, 'software-qualification')

    def test_mutated_snapshot_rejected(self):
        (self.root / 'fixture').write_text('changed after finalization')
        with self.assertRaisesRegex(ValueError, 'frozen snapshot'):
            check(self.config, 'train')

    def test_leaked_normalization_count_rejected(self):
        stats = json.loads(self.stats_path.read_text())
        stats['action']['count'] = [12]
        self.stats_path.write_text(json.dumps(stats))
        self.manifest['train_stats_sha256'] = digest(self.stats_path)
        self.save()
        with self.assertRaisesRegex(ValueError, 'training split'):
            check(self.config, 'train')

    def test_mixed_destinations_rejected(self):
        self.manifest['episodes'][2]['destination'] = 'other fixture'
        self.save()
        with self.assertRaisesRegex(ValueError, 'separate ACT'):
            check(self.config, 'train')

    def test_whole_episodes_still_reject_shared_specimens(self):
        self.manifest['episodes'][1]['specimen_id'] = self.manifest['episodes'][0]['specimen_id']
        with self.assertRaisesRegex(ValueError, 'specimen_id leaks'):
            check_splits(self.manifest, 'train')

    def test_loader_smoke_cannot_become_quality_training(self):
        self.manifest['split_scope'] = 'loader_smoke'
        check_splits(self.manifest, 'smoke')
        with self.assertRaisesRegex(ValueError, 'grouped_evaluation'):
            check_splits(self.manifest, 'train')

    def test_resume_identity_allows_budget_extension_not_skill_or_model_change(self):
        checked = check(self.config, 'train')
        initial = identity(self.config, checked)
        extended = copy.deepcopy(self.config)
        extended['max_steps'] = 100
        self.assertEqual(initial, identity(extended, checked))
        extended['policy']['chunk_size'] = 8
        self.assertNotEqual(initial, identity(extended, checked))

    def test_budgeted_training_requires_matching_real_intel_parity(self):
        checked = check(self.config, 'train')
        run_id = identity(self.config, checked)
        smoke = Path(self.temporary.name) / 'smoke-result.json'
        parity = Path(self.temporary.name) / 'intel-parity.json'
        smoke.write_text(json.dumps({'mode': 'smoke', 'evidence_kind': 'real',
                                     'status': 'EXPORTED_FOR_INTEL_VERIFICATION', 'run_identity': run_id}))
        parity.write_text(json.dumps({'evidence_kind': 'real', 'status': 'PARITY_PASSED', 'run_identity': run_id}))
        self.config.update(smoke_result=str(smoke), intel_parity_result=str(parity))
        check_smoke_gate(self.config, checked)
        parity.write_text(json.dumps({'evidence_kind': 'real', 'status': 'PARITY_PASSED', 'run_identity': 'another snapshot'}))
        with self.assertRaisesRegex(ValueError, 'match this frozen data'):
            check_smoke_gate(self.config, checked)

    def test_qualification_report_cannot_unlock_real_training(self):
        checked = check(self.config, 'train')
        smoke = Path(self.temporary.name) / 'smoke-result.json'
        parity = Path(self.temporary.name) / 'intel-parity.json'
        smoke.write_text(json.dumps({'mode': 'software-qualification', 'evidence_kind': 'synthetic'}))
        parity.write_text('{}')
        self.config.update(smoke_result=str(smoke), intel_parity_result=str(parity))
        with self.assertRaisesRegex(ValueError, 'real one-step smoke'):
            check_smoke_gate(self.config, checked)

    def test_resume_rejects_package_version_or_dependency_drift(self):
        check_resume_packages({'torch': 'same'}, {'torch': 'same'}, 'a==1\nb==2\n', 'b==2\na==1\n')
        with self.assertRaisesRegex(ValueError, 'Package versions changed'):
            check_resume_packages({'torch': 'old'}, {'torch': 'new'}, '', '')
        with self.assertRaisesRegex(ValueError, 'package freeze changed'):
            check_resume_packages({'torch': 'same'}, {'torch': 'same'}, 'a==1\n', 'a==2\n')

    def test_final_holdout_status_does_not_imply_unallocated_evidence(self):
        for kind, episodes, expected in (
            ('synthetic', [], 'NOT_APPLICABLE_SYNTHETIC'),
            ('synthetic', [2], 'NOT_APPLICABLE_SYNTHETIC'),
            ('real', [], 'NOT_ALLOCATED'),
            ('real', [2], 'UNTOUCHED'),
        ):
            with self.subTest(kind=kind, episodes=episodes):
                self.assertEqual(final_eval_status({'evidence_kind': kind, 'final_eval_episodes': episodes}), expected)

    def test_action_shape_uses_export_feature_not_legacy_chunk_default(self):
        result = check_action_shape([100, 6], (100, 6), (100, 6), 1)
        self.assertEqual(result['chunk_metadata_status'], 'LEGACY_PROPERTY_MISMATCH')
        for reference, actual in [((1, 6), (1, 6)), ((100, 6), (1, 6))]:
            with self.subTest(reference=reference, actual=actual), self.assertRaisesRegex(ValueError, 'shape differs'):
                check_action_shape([100, 6], reference, actual, 1)


if __name__ == '__main__':
    unittest.main()
