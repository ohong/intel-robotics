"""Synthetic fixtures only: no physical camera or task-quality claims."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None

from secondlook.cv_observation import observe_frame

spec = importlib.util.spec_from_file_location('cv_capture', Path(__file__).parents[1] / 'scripts/cv_capture.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


@unittest.skipIf(np is None, 'Synthetic image checks need numpy and Pillow')
class CVTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        self.rgb[:, ::2] = [220, 130, 40]
        self.frame = SimpleNamespace(timestamp=100., sequence=5, data=self.rgb)
        self.header = SimpleNamespace(timestamp_ns=100_000_000_000, sequence=5,
                                      color_mode=0, dtype=0, width=16, height=12)
        self.calls = []
        def infer(rgb):
            self.calls.append(rgb)
            return {'score': 2., 'threshold': 3., 'disposition': 'NORMAL',
                    'anomaly_map': np.ones((3, 3)), 'latency_ms': 4., 'model_latency_ms': 3.,
                    'artifact_id': 'synthetic-fixture', 'evidence_kind': 'synthetic',
                    'frame_id': 'wrong-id', 'captured_at': 99.,
                    'localization_geometry': {'roi_normalized': [.2, .2, .8, .8]},
                    'uncertainty_band': [2.8, 3.2]}
        self.detector = SimpleNamespace(infer=infer)

    def test_invalid_frames_never_infer(self):
        for rgb, timestamp in ((self.rgb, 90.), (self.rgb, 101.), (self.rgb, float('nan')),
                               (self.rgb, True), (np.zeros((12, 16, 3), np.uint8), 100.),
                               (np.full((12, 16, 3), 40, np.uint8), 100.),
                               (np.full((12, 16, 3), np.nan), 100.),
                               (np.zeros((12, 16), np.uint8), 100.),
                               (np.zeros((0, 16, 3), np.uint8), 100.),
                               (None, 100.), ([[1], [1, 2]], 100.)):
            with self.subTest(shape=getattr(rgb, 'shape', None), timestamp=timestamp):
                result = observe_frame(self.detector, rgb, 'fixture', timestamp, now=100.1)
                self.assertFalse(result['inference_executed'])
                self.assertEqual(result['decision'], 'INVALID')
                self.assertEqual(result['anomaly_status'], 'NOT_EVALUATED')
                self.assertIsNone(result['score'])
        self.assertEqual(self.calls, [])

    def test_association_full_pixels_and_unverified_object(self):
        result = observe_frame(self.detector, self.rgb, 'original-id', 100., now=100.1)
        self.assertEqual(result['frame_id'], 'original-id')
        self.assertEqual(result['captured_at'], 100.)
        self.assertIs(self.calls[0], self.rgb)
        self.assertEqual(result['object_presence']['status'], 'UNVERIFIED')
        self.assertEqual(result['occlusion']['status'], 'UNVERIFIED')
        self.assertEqual(result['defect_visibility']['status'], 'UNVERIFIED')
        self.assertEqual(result['decision'], 'GOOD')
        self.assertEqual(result['observation_validity'], 'UNVERIFIED')
        self.assertEqual(result['anomaly_score'], 2.)
        self.assertEqual(result['model_version'], 'synthetic-fixture')
        self.assertEqual(result['localization_geometry']['roi_normalized'], [.2, .2, .8, .8])

    def test_explicit_evidence_requires_association(self):
        evidence = {'frame_id': 'other', 'validated': True, 'source': 'synthetic unit test', 'object_presence': 'PRESENT'}
        with self.assertRaises(ValueError):
            observe_frame(self.detector, self.rgb, 'fixture', 100., now=100.1, validated_evidence=evidence)
        evidence['frame_id'] = 'fixture'
        result = observe_frame(self.detector, self.rgb, 'fixture', 100., now=100.1, validated_evidence=evidence)
        self.assertEqual(result['object_presence']['status'], 'PRESENT')
        self.assertEqual(result['occlusion']['status'], 'UNVERIFIED')

    def test_physical_invalidity_never_infers_and_retains_model(self):
        self.detector.metadata = {'artifact_id': 'loaded-synthetic-fixture'}
        for field, value in (('object_presence', 'ABSENT'), ('occlusion', 'OCCLUDED'),
                             ('defect_visibility', 'NOT_VISIBLE')):
            evidence = {'frame_id': 'fixture', 'validated': True, 'source': 'synthetic evidence', field: value}
            result = observe_frame(self.detector, self.rgb, 'fixture', 100., now=100.1,
                                   validated_evidence=evidence)
            self.assertEqual(result['decision'], 'INVALID')
            self.assertEqual(result['frame_validity'], 'VALID')
            self.assertEqual(result['observation_validity'], 'INVALID')
            self.assertEqual(result['model_version'], 'loaded-synthetic-fixture')
            self.assertFalse(result['inference_executed'])
        self.assertEqual(self.calls, [])

    def test_roi_uses_same_floor_ceil_geometry_as_detector(self):
        self.assertEqual(capture.roi_box([.11, .11, .81, .81], 11, 13), (1, 1, 9, 11))

    def test_capture_preserves_exact_timestamp_and_converts_bgr(self):
        self.header.color_mode = 1
        rgb, metadata = capture.checked_rgb(self.frame, self.header, now=100.1, max_age=1.)
        np.testing.assert_array_equal(rgb, self.rgb[..., ::-1])
        self.assertEqual(metadata['capture_timestamp_ns'], 100_000_000_000)
        self.assertEqual(metadata['capture_timestamp'], 100.)
        self.assertEqual(metadata['sequence'], 5)
        self.assertEqual(metadata['publisher_color'], 'BGR')

    def test_header_mismatch_and_stale_rejected(self):
        for field, value in (('sequence', 6), ('timestamp_ns', 100_000_000_001), ('dtype', 1),
                             ('color_mode', 7), ('width', 17)):
            with self.subTest(field=field):
                original = getattr(self.header, field)
                setattr(self.header, field, value)
                with self.assertRaises(ValueError):
                    capture.checked_rgb(self.frame, self.header, now=100.1, max_age=1.)
                setattr(self.header, field, original)
        with self.assertRaises(ValueError):
            capture.checked_rgb(self.frame, self.header, now=103., max_age=1.)

    def test_rgb_png_preview_and_provenance_no_overwrite(self):
        rgb, metadata = capture.checked_rgb(self.frame, self.header, now=100.1, max_age=1.)
        provenance = {'session_id': 'synthetic-session', 'specimen_id': 'synthetic-specimen',
                      'role': 'train', 'label': 'unknown', 'camera_service': 'synthetic/publisher',
                      'evidence_kind': 'synthetic',
                      'ground_truth': {'source': 'synthetic test pattern, no defect label', 'authoritative': False}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = capture.save_capture(root, rgb, metadata, provenance, roi=[.25, .25, .75, .75], size=8)
            with Image.open(root / row['path']) as saved:
                np.testing.assert_array_equal(np.asarray(saved), self.rgb)
            self.assertEqual(row['preview_roi_pixels'], [4, 3, 12, 9])
            self.assertEqual(row['label'], 'unknown')
            self.assertFalse(row['ground_truth']['authoritative'])
            self.assertEqual(row['capture_timestamp_ns'], 100_000_000_000)
            capture.contact_sheet(root, [row])
            self.assertTrue((root / 'contact-sheet.png').is_file())
            with self.assertRaises(FileExistsError):
                capture.save_capture(root, rgb, metadata, provenance, roi=[0., 0., 1., 1.], size=8)


if __name__ == '__main__':
    unittest.main()
