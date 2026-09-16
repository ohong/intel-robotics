"""Portable boundary checks; these do not pretend to execute a model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from secondlook.anomaly import AnomalyContractError, _metadata, calibrate, validate_manifest


class ManifestContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('a', 'b', 'c'):
            (self.root / name).write_bytes(name.encode())

    def row(self, image='a', role='train', specimen='part-1', label='normal', authoritative=True):
        return {'path': image, 'role': role, 'specimen_id': specimen, 'label': label,
                'ground_truth': {'source': 'operator inspection record 1', 'authoritative': authoritative}}

    def manifest(self, rows):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps({'schema_version': 1, 'images': rows}))
        return path

    def test_accept_authoritative_normal_training(self):
        result = validate_manifest(self.manifest([self.row()]))
        self.assertEqual(result['images'][0]['role'], 'train')
        self.assertEqual(len(result['images'][0]['sha256']), 64)

    def test_specimen_overlap_rejected(self):
        with self.assertRaisesRegex(AnomalyContractError, 'Specimen overlaps'):
            validate_manifest(self.manifest([self.row(), self.row('b', 'test')]))

    def test_content_overlap_rejected_even_with_different_names(self):
        (self.root / 'b').write_bytes((self.root / 'a').read_bytes())
        with self.assertRaisesRegex(AnomalyContractError, 'Duplicate image'):
            validate_manifest(self.manifest([self.row(), self.row('b', 'test', 'part-2')]))

    def test_unverified_labels_require_smoke_mode(self):
        path = self.manifest([self.row(label='unknown', authoritative=False)])
        with self.assertRaisesRegex(AnomalyContractError, 'authoritative ground truth'):
            validate_manifest(path)
        self.assertEqual(validate_manifest(path, engineering_smoke=True)['images'][0]['label'], 'unknown')

    def test_explicit_provenance_required_even_for_smoke(self):
        row = self.row()
        del row['ground_truth']['source']
        with self.assertRaisesRegex(AnomalyContractError, 'Ground-truth source'):
            validate_manifest(self.manifest([row]), engineering_smoke=True)

    def test_anomaly_cannot_train_task_or_smoke(self):
        path = self.manifest([self.row(label='anomalous')])
        for smoke in (False, True):
            with self.assertRaises(AnomalyContractError):
                validate_manifest(path, engineering_smoke=smoke)

    def test_smoke_cannot_be_promoted_by_calibration(self):
        with patch('secondlook.anomaly._metadata', return_value={'evidence_kind': 'engineering_smoke'}):
            with self.assertRaisesRegex(AnomalyContractError, 'Smoke artifacts cannot be calibrated'):
                calibrate(self.root, 'unused')

    def test_threshold_without_validation_and_test_is_rejected(self):
        for kind in ('engineering_smoke', 'task_model_uncalibrated', 'task_model_evaluated'):
            (self.root / 'metadata.json').write_text(json.dumps({'files': {}, 'threshold': 1.0, 'evidence_kind': kind}))
            with self.assertRaisesRegex(AnomalyContractError, 'Threshold requires'):
                _metadata(self.root)

    def test_nonfinite_threshold_is_rejected(self):
        (self.root / 'metadata.json').write_text(json.dumps({'files': {}, 'threshold': float('nan')}))
        with self.assertRaisesRegex(AnomalyContractError, 'finite'):
            _metadata(self.root)

    def test_dataset_identity_survives_path_relocation(self):
        first = validate_manifest(self.manifest([self.row()]))
        (self.root / 'moved').mkdir()
        (self.root / 'moved' / 'a').write_bytes((self.root / 'a').read_bytes())
        second = validate_manifest(self.manifest([self.row('moved/a')]))
        self.assertEqual(first['dataset_sha256'], second['dataset_sha256'])
        self.assertNotEqual(first['manifest_sha256'], second['manifest_sha256'])

    def test_calibration_changes_artifact_identity(self):
        from secondlook.anomaly import _artifact_identity
        metadata = {'model_id': 'stable-model', 'threshold': None}
        first = _artifact_identity(metadata)
        metadata.update(threshold=3.0, artifact_id=first)
        self.assertNotEqual(first, _artifact_identity(metadata))

    def test_image_hash_change_is_rejected(self):
        (self.root / 'metadata.json').write_text(json.dumps({'files': {'a': 'wrong'}, 'threshold': None}))
        with self.assertRaisesRegex(AnomalyContractError, 'hash mismatch'):
            _metadata(self.root)


if __name__ == '__main__':
    unittest.main()
