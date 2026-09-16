"""Synthetic scores verify software boundaries, not LEGO detector accuracy."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from secondlook.anomaly import (AnomalyContractError, _artifact_identity, _decision,
                               _metadata, _prepare, _report, _roi, calibrate,
                               evaluate, validate_manifest)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for i, (role, label) in enumerate((('train', 'normal'), ('validation', 'normal'),
                                          ('validation', 'anomalous'), ('test', 'normal'), ('test', 'anomalous'))):
            name = f'fixture-{i}'
            (self.root / name).write_bytes(name.encode())
            rows.append({'path': name, 'role': role, 'label': label, 'specimen_id': name,
                         'session_id': f'session-{role}',
                         'ground_truth': {'source': 'synthetic software test only; no physical truth claim', 'authoritative': True}})
        self.path = self.root / 'manifest.json'
        self.path.write_text(json.dumps({'schema_version': 1, 'images': rows}))
        self.dataset = validate_manifest(self.path)
        self.metadata = {'files': {}, 'threshold': None, 'calibration': None, 'evaluation': None,
                         'evidence_kind': 'task_model_uncalibrated', 'dataset': self.dataset,
                         'config': {'image_size': 128}}
        self.save()
        self.seen = []

    def save(self):
        self.metadata['artifact_id'] = _artifact_identity(self.metadata)
        (self.root / 'metadata.json').write_text(json.dumps(self.metadata))

    def factory(self, *args, **kwargs):
        import numpy as np
        seen = self.seen
        metadata = _metadata(self.root)
        backend = kwargs.get('backend', 'openvino')
        class Fake:
            def infer(self, path):
                seen.append((backend, Path(path).name))
                score = {'fixture-1': 1., 'fixture-2': 5., 'fixture-3': 2.79, 'fixture-4': 8.}[Path(path).name]
                if backend == 'torch' and Path(path).name == 'fixture-3':
                    score += .02  # Numerically close but crosses uncertainty boundary at 2.8.
                return {'score': score, 'anomaly_map': np.zeros((2, 2)),
                        'disposition': _decision(score, metadata['threshold'], metadata.get('uncertainty_margin', 0.))[0]}
        return Fake()

    def test_calibration_reads_validation_only_and_freezes_test_protocol(self):
        with patch('secondlook.anomaly.AnomalyDetector', side_effect=self.factory):
            report = calibrate(self.root, self.path)
            self.assertEqual(self.seen, [('openvino', 'fixture-1'), ('openvino', 'fixture-2')])
            self.assertIsNone(report['evaluation'])
            self.assertEqual(report['threshold'], 3.)
            self.assertEqual(report['uncertainty_margin'], .2)
            self.assertEqual(_metadata(self.root)['evidence_kind'], 'task_model_calibrated')
            with self.assertRaisesRegex(AnomalyContractError, 'frozen'):
                calibrate(self.root, self.path)
            self.seen.clear()
            final = evaluate(self.root, self.path)['evaluation']
            self.assertEqual({name for _, name in self.seen}, {'fixture-3', 'fixture-4'})
            self.assertTrue(final['export_parity']['records'][0]['score_pass'])
            self.assertFalse(final['export_parity']['records'][0]['decision_pass'])
            self.assertFalse(final['export_parity']['pass'])
            self.assertEqual(final['calibration_id'], report['calibration']['calibration_id'])
            self.assertEqual(final['threshold'], report['threshold'])
            with self.assertRaisesRegex(AnomalyContractError, 'already recorded'):
                evaluate(self.root, self.path)

    def test_test_evaluation_requires_calibration(self):
        with self.assertRaisesRegex(AnomalyContractError, 'Calibrate'):
            evaluate(self.root, self.path)

    def test_changed_test_provenance_is_rejected_before_inference(self):
        with patch('secondlook.anomaly.AnomalyDetector', side_effect=self.factory):
            calibrate(self.root, self.path)
            (self.root / 'fixture-4').write_bytes(b'changed final test')
            with self.assertRaisesRegex(AnomalyContractError, 'unchanged images'):
                evaluate(self.root, self.path)

    def test_session_leakage_is_rejected_even_for_unique_specimens(self):
        data = json.loads(self.path.read_text())
        data['images'][3]['session_id'] = 'session-validation'
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AnomalyContractError, 'session overlaps'):
            validate_manifest(self.path)

    def test_single_session_pilot_is_explicit_and_preserves_real_session(self):
        data = json.loads(self.path.read_text())
        for row in data['images']:
            row['session_id'] = 'one-real-collection'
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AnomalyContractError, 'session overlaps'):
            validate_manifest(self.path)
        data['split_policy'] = 'specimen_only_pilot'
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AnomalyContractError, 'split_policy_reason'):
            validate_manifest(self.path)
        data['split_policy_reason'] = 'Only one real collection session; independent specimens reserved for pilot validation'
        self.path.write_text(json.dumps(data))
        result = validate_manifest(self.path)
        self.assertEqual(result['shared_sessions'], ['one-real-collection'])
        self.assertTrue(result['split_limitations'])
        self.assertEqual({row['session_id'] for row in result['images']}, {'one-real-collection'})

    def test_unseen_test_can_be_appended_after_calibration(self):
        data = json.loads(self.path.read_text())
        original_rows = data['images'][:3]
        self.path.write_text(json.dumps({**data, 'images': original_rows}))
        self.metadata['dataset'] = validate_manifest(self.path)
        self.save()
        with patch('secondlook.anomaly.AnomalyDetector', side_effect=self.factory):
            calibrated = calibrate(self.root, self.path)
            self.path.write_text(json.dumps(data))
            result = evaluate(self.root, self.path)['evaluation']
        self.assertEqual(result['calibration_id'], calibrated['calibration']['calibration_id'])
        self.assertEqual(len(result['dataset']['images']), 5)
        self.assertEqual(len(self.metadata['dataset']['images']), 3)

    def test_appended_training_or_changed_original_provenance_is_rejected(self):
        from secondlook.anomaly import _split_rows
        data = json.loads(self.path.read_text())
        (self.root / 'extra').write_bytes(b'new training image')
        data['images'].append({**data['images'][0], 'path': 'extra', 'specimen_id': 'extra'})
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AnomalyContractError, 'only appended test'):
            _split_rows(self.metadata, self.path, 'test')
        data['images'].pop()
        data['images'][0]['ground_truth']['source'] = 'changed attribution'
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AnomalyContractError, 'unchanged images/provenance'):
            _split_rows(self.metadata, self.path, 'test')

    def test_split_policy_cannot_change_after_fit(self):
        from secondlook.anomaly import _split_rows
        data = json.loads(self.path.read_text())
        data.update(split_policy='specimen_only_pilot', split_policy_reason='later relaxed policy')
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AnomalyContractError, 'policy.*unchanged'):
            _split_rows(self.metadata, self.path, 'validation')

    def test_partial_session_records_are_rejected(self):
        data = json.loads(self.path.read_text())
        del data['images'][0]['session_id']
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AnomalyContractError, 'every image'):
            validate_manifest(self.path)

    def test_uncertainty_and_confusion_are_separate(self):
        rows = [{'score': score, 'label': label, 'specimen_id': str(i)}
                for i, (score, label) in enumerate([(1., 'anomalous'), (3., 'anomalous'),
                                                  (4., 'normal'), (3., 'normal')])]
        report = _report(rows, 3., .2)
        self.assertEqual(report['missed_defects'], 1)
        self.assertEqual(report['rejected_good_parts'], 1)
        self.assertEqual(report['uncertain_defects'], 1)
        self.assertEqual(report['uncertain_good_parts'], 1)
        self.assertEqual(report['score_distributions']['normal']['median'], 3.5)

    def test_nonfinite_scores_and_invalid_fraction_cannot_calibrate(self):
        for fraction in (-1., float('nan'), 1.):
            with self.assertRaisesRegex(AnomalyContractError, 'uncertainty_fraction'):
                calibrate(self.root, self.path, uncertainty_fraction=fraction)
        with patch('secondlook.anomaly.AnomalyDetector') as detector:
            detector.return_value.infer.return_value = {'score': float('nan')}
            with self.assertRaisesRegex(AnomalyContractError, 'nonfinite score'):
                calibrate(self.root, self.path)

    def test_tampered_calibration_metadata_is_rejected(self):
        with patch('secondlook.anomaly.AnomalyDetector', side_effect=self.factory):
            calibrate(self.root, self.path)
        metadata = json.loads((self.root / 'metadata.json').read_text())
        metadata['threshold'] = 100.
        (self.root / 'metadata.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(AnomalyContractError, 'identity mismatch'):
            _metadata(self.root)

    def test_roi_bounds(self):
        for roi in ([0, 0, 0, 1], [-.1, 0, 1, 1], [0, 0, 1, float('nan')], [0, 1], ['0', 0, 1, 1]):
            with self.assertRaises(AnomalyContractError):
                _roi(roi)
        self.assertEqual(_roi(), [0., 0., 1., 1.])

    def test_neighbor_count_rejects_invalid_values_before_loading_data(self):
        from secondlook.anomaly import fit
        for count in (0, 21, 1.5, True, '9'):
            with self.assertRaisesRegex(AnomalyContractError, 'num_neighbors'):
                fit('unused', 'unused', num_neighbors=count)

    def test_roi_image_preprocessing_matches_explicit_crop(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest('Image dependencies are unavailable on this host')
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        image[2:9, 5:16] = [250, 20, 100]
        cropped, geometry = _prepare(image, 64, [.25, .2, .8, .9])
        expected, _ = _prepare(image[2:9, 5:16], 64)
        np.testing.assert_array_equal(cropped, expected)
        self.assertEqual(geometry['roi_pixels'], [5, 2, 16, 9])
        self.assertEqual(geometry['image_size'], [20, 10])


if __name__ == '__main__':
    unittest.main()
