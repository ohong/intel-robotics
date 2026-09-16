"""Synthetic observation-contract checks; no camera or model workload."""
from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from secondlook.anomaly import AnomalyContractError, ObservationInvalidError
from secondlook.cv_observation import observe_frame


@unittest.skipIf(np is None, 'Observation fixtures require numpy')
class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        self.rgb[:, ::2] = [220, 130, 40]
        self.detector = SimpleNamespace()

    def test_typed_preprocessing_rejection_keeps_identity_and_no_model_result(self):
        self.detector.metadata = {'artifact_id': 'loaded-synthetic-fixture'}
        for reason in ('colored foreground absent or too small',
                       'colored foreground bounding box too small',
                       'multiple comparable colored foreground components'):
            def reject(_rgb):
                raise ObservationInvalidError(reason)
            self.detector.infer = reject
            result = observe_frame(self.detector, self.rgb, 'exact-frame', 100., now=100.1)
            self.assertEqual(result['frame_id'], 'exact-frame')
            self.assertEqual(result['captured_at'], 100.)
            self.assertEqual(result['model_version'], 'loaded-synthetic-fixture')
            self.assertEqual(result['model_artifact_id'], 'loaded-synthetic-fixture')
            self.assertEqual(result['frame_validity'], 'VALID')
            self.assertEqual(result['observation_validity'], 'INVALID')
            self.assertEqual(result['decision'], 'INVALID')
            self.assertIn('PREPROCESSING_REJECTED', result['quality_flags'])
            self.assertEqual(result['decision_reason'], reason)
            self.assertFalse(result['inference_executed'])
            for field in ('score', 'anomaly_score', 'anomaly_map', 'latency_ms',
                          'model_latency_ms', 'inference_ms', 'localization_geometry'):
                self.assertIsNone(result[field])
            for field in ('object_presence', 'occlusion', 'defect_visibility'):
                self.assertEqual(result[field]['status'], 'UNVERIFIED')

    def test_genuine_detector_errors_still_propagate(self):
        for error in (AnomalyContractError('bad artifact'), RuntimeError('model failure'),
                      ValueError('malformed model output')):
            def fail(_rgb):
                raise error
            self.detector.infer = fail
            with self.subTest(error=error), self.assertRaises(type(error)) as caught:
                observe_frame(self.detector, self.rgb, 'fixture', 100., now=100.1)
            self.assertIs(caught.exception, error)

