"""Portable checks for experimental SmolVLA data and weight boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import training_smolvla as runner


class WeightTests(unittest.TestCase):
    def test_only_inactive_legacy_parameters_can_be_absent(self):
        self.assertEqual(runner.verify_weight_keys({'active'}, {'active'} | runner.INACTIVE_KEYS, False), sorted(runner.INACTIVE_KEYS))

    def test_missing_active_parameter_fails(self):
        with self.assertRaisesRegex(ValueError, 'Active base weights mismatch'):
            runner.verify_weight_keys({'active'}, {'active', 'vision.weight'}, False)

    def test_unexpected_parameter_fails(self):
        with self.assertRaisesRegex(ValueError, 'Active base weights mismatch'):
            runner.verify_weight_keys({'active', 'foreign'}, {'active'}, False)

    def test_snapflow_cannot_use_untrained_legacy_parameters(self):
        with self.assertRaisesRegex(ValueError, 'SnapFlow disabled'):
            runner.verify_weight_keys({'active'}, {'active'}, True)


class PredictionDeviceTests(unittest.TestCase):
    def test_comparison_uses_same_cuda_device(self):
        runner.require_prediction_devices('cuda:0', 'cuda:0')

    def test_equal_seed_cannot_excuse_cpu_cuda_device_mismatch(self):
        for model_device, observation_device in [('cpu', 'cuda:0'), ('cuda:0', 'cpu'), ('cpu', 'cpu'), ('cuda:1', 'cuda:1')]:
            with self.subTest(model=model_device, input=observation_device), self.assertRaisesRegex(ValueError, 'CUDA device zero'):
                runner.require_prediction_devices(model_device, observation_device)


class ConditioningTests(unittest.TestCase):
    def test_exact_canonical_text_and_threshold_boundary(self):
        text = runner.annotation_text('Sort LEGO.', {'eligible': True, 'score': 59.5, 'decision': 'ANOMALOUS'}, 59.5)
        self.assertEqual(text, 'Sort LEGO.\nInspection: anomalous; score=59.5; threshold=59.5.\n')

    def test_nonfinite_and_invalid_detections_fail(self):
        for row in ({'eligible': False, 'score': None, 'decision': 'UNKNOWN'},
                    {'eligible': True, 'score': float('nan'), 'decision': 'NORMAL'},
                    {'eligible': True, 'score': 20, 'decision': 'UNKNOWN'},
                    {'eligible': True, 'score': 20, 'decision': 'ANOMALOUS'}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                runner.annotation_text('Sort LEGO.', row, 50)

    def test_same_caption_cannot_label_two_destinations(self):
        manifest = {'episodes': [{'episode_index': 0, 'destination': 'left'}, {'episode_index': 1, 'destination': 'right'}]}
        config = {'conditioning': {'mode': 'episode_task', 'claim': 'experimental_imitation',
                  'episodes': {str(i): {'text': 'Sort.', 'evidence': 'actual-source'} for i in (0, 1)}}}
        with self.assertRaisesRegex(ValueError, 'conflicting destinations'):
            runner.check_conditioning(config, manifest)

    def test_episode_override_needs_provenance(self):
        manifest = {'episodes': [{'episode_index': 0, 'destination': 'left'}]}
        config = {'conditioning': {'mode': 'episode_task', 'claim': 'experimental_imitation',
                                  'episodes': {'0': {'text': 'Move left.'}}}}
        with self.assertRaisesRegex(ValueError, 'source evidence'):
            runner.check_conditioning(config, manifest)


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'annotations.json'
        self.manifest = {'cameras': ['observation.images.low'], 'files': {'data.parquet': {'bytes': 8, 'sha256': 'f' * 64}},
                         'episodes': [{'episode_index': 0, 'frames': 3}]}
        snapshot_hash = hashlib.sha256(json.dumps(self.manifest['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.document = {'schema_version': 1, 'dataset_sha256': snapshot_hash, 'artifact_id': 'frozen-detector',
                         'camera_key': 'observation.images.low', 'threshold': 50, 'rgb_hash_format': 'uint8_HWC_RGB_contiguous',
                         'rows': [{'index': 0, 'episode_index': 0, 'frame_index': 0, 'eligible': True,
                                   'image_sha256': 'a' * 64, 'score': 40, 'decision': 'NORMAL'}]}
        self.config = {'conditioning': {'mode': 'anomalib_rows', 'annotation_manifest': str(self.path), 'artifact_id': 'frozen-detector',
                                       'threshold': 50, 'instruction': 'Sort LEGO.'}}

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self, *, for_fit=False):
        self.path.write_text(json.dumps(self.document))
        self.config['conditioning']['annotation_sha256'] = runner.digest(self.path)
        return runner.check_annotations(self.config, self.manifest, for_fit=for_fit)

    def test_valid_frozen_annotation(self):
        self.assertEqual(self.validate()['rows'][0]['index'], 0)

    def test_other_snapshot_fails(self):
        self.document['dataset_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'different recording snapshot'):
            self.validate()

    def test_duplicate_start_fails(self):
        self.document['rows'].append(copy.deepcopy(self.document['rows'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate/invalid annotation index'):
            self.validate()

    def test_invented_conditioning_text_fails(self):
        self.document['rows'][0]['text'] = 'This object is defective.'
        with self.assertRaisesRegex(ValueError, 'deployment conditioning format'):
            self.validate()

    def test_wrong_camera_or_artifact_fails(self):
        for key, value in [('camera_key', 'observation.images.other'), ('artifact_id', 'different')]:
            original = self.document[key]
            self.document[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.document[key] = original

    def test_invalid_detection_preserved_as_excluded_start(self):
        self.document['rows'][0].update(eligible=False, score=None, decision='UNKNOWN', reason='invalid_crop')
        self.assertFalse(self.validate()['rows'][0]['eligible'])

    def test_pixel_valid_unknown_target_can_decode_but_cannot_fit(self):
        self.document.update(conditioning_verified=False, target_association='UNKNOWN')
        self.assertTrue(self.validate()['rows'][0]['eligible'])
        with self.assertRaisesRegex(ValueError, 'manipulated object'):
            self.validate(for_fit=True)

    def test_row_override_cannot_weaken_verified_association(self):
        self.manifest.update(train_episodes=[0], validation_episodes=[])
        self.document.update(conditioning_verified=True, target_association='VERIFIED')
        self.document['rows'][0]['target_association'] = 'UNKNOWN'
        with self.assertRaisesRegex(ValueError, 'eligible start lacks verified'):
            self.validate(for_fit=True)

    def test_uncertain_score_cannot_be_eligible(self):
        self.document['uncertainty_band'] = [39, 61]
        with self.assertRaisesRegex(ValueError, 'uncertainty band'):
            self.validate()

    def test_episode_split_leakage_fails(self):
        manifest = {'train_episodes': [0], 'validation_episodes': [0], 'final_eval_episodes': [], 'episodes': [{'episode_index': 0}]}
        with self.assertRaisesRegex(ValueError, 'leakage'):
            runner.check_splits(manifest, 'smoke')


class PipelineProbeTests(unittest.TestCase):
    def setUp(self):
        self.config = {'training_scope': 'real_data_pipeline_probe', 'max_steps': 1,
                       'conditioning': {'mode': 'recorded_task'}}
        self.manifest = {
            'task_ready': False, 'controller_ready': False, 'split_scope': 'loader_smoke',
            'generalization_claim_permitted': False, 'joint_names': ['joint.pos'],
            'episodes': [{'outcome': 'visual_transfer_observed', 'observed_transfer': True,
                          'visual_evidence': 'sampled-contact-sheet.jpg', 'observed_destination': 'observed bowl',
                          'task_caption': 'Original recorded caption.', 'task_success': 'unknown'}],
            'robot_contract': {'joint_order': ['joint.pos'], 'state_semantics': 'measured follower joint positions',
                               'action_semantics': 'absolute joint position targets; not velocities or deltas',
                               'body_units': 'normalized calibrated position [-100,100]',
                               'gripper_units': 'normalized calibrated position [0,100]',
                               'calibration_id_provenance': 'frozen configuration evidence', 'action_ack_limit': 'targets only'},
            'camera_identity_provenance': 'frozen configuration', 'timing_limits': ['No wall-clock skew proof'],
            'camera_identity_mapping': {'observation.images.low': {'serial': 'fixture', 'recorded_rgb_shape_hwc': [10, 20, 3]}},
            'features': {'observation.images.low': {'shape': [10, 20, 3]}}
        }

    def test_truthful_visual_transfer_does_not_require_fake_success(self):
        result = runner.check_pipeline_probe(self.config, self.manifest)
        self.assertEqual(result['physical_task_success'], 'NOT VALIDATED')
        self.assertEqual(self.manifest['episodes'][0]['task_success'], 'unknown')
        self.assertEqual(runner.camera_keys(self.manifest), ['observation.images.low'])

    def test_twenty_update_limit_is_hard(self):
        self.config['max_steps'] = 21
        with self.assertRaisesRegex(ValueError, 'twenty'):
            runner.check_pipeline_probe(self.config, self.manifest)

    def test_probe_cannot_bypass_unverified_detector_association(self):
        self.config['conditioning']['mode'] = 'anomalib_rows'
        with self.assertRaisesRegex(ValueError, 'original recorded captions'):
            runner.check_pipeline_probe(self.config, self.manifest)

    def test_changed_frozen_units_fail(self):
        self.manifest['robot_contract']['body_units'] = 'radians'
        with self.assertRaisesRegex(ValueError, 'joint units'):
            runner.check_pipeline_probe(self.config, self.manifest)

    def test_dict_normalization_keeps_whole_episode_exclusions(self):
        manifest = {'train_episodes': [0, 1, 3], 'validation_episodes': [2, 4], 'final_eval_episodes': [],
                    'normalization': {'training_episodes_only': [0, 1, 3], 'training_rows': 1829,
                                      'excluded_validation_episodes': [2, 4], 'excluded_final_episodes': []}}
        runner.check_normalization(manifest, 1829)
        manifest['normalization']['training_episodes_only'] = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'episode list differs'):
            runner.check_normalization(manifest, 1829)


if __name__ == '__main__':
    unittest.main()
