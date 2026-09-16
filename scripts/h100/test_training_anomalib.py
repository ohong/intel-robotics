"""Disposable contract tests; no model execution or hardware access."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from training_anomalib import checked, gpu_idle, gpu_lock, parser


class AnomalibContractTests(unittest.TestCase):
    def arguments(self):
        return parser().parse_args(['--source-root', '/missing-source', '--source-sha256', 'bad',
                                   '--manifest', '/missing-manifest', '--output', '/missing-output'])

    def test_reject_nonfinite_ratio_before_loading_dependencies(self):
        args = self.arguments()
        args.sampling_ratio = float('nan')
        with self.assertRaisesRegex(ValueError, 'sampling_ratio'):
            checked(args)

    def test_reject_neighbor_bound(self):
        args = self.arguments()
        args.num_neighbors = 21
        with self.assertRaisesRegex(ValueError, 'num_neighbors'):
            checked(args)

    def test_smoke_cannot_claim_calibration(self):
        args = self.arguments()
        args.engineering_smoke = args.calibrate = True
        with self.assertRaisesRegex(ValueError, 'Smoke cannot be calibrated'):
            checked(args)

    def test_reject_source_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'secondlook').mkdir()
            (source / 'secondlook/anomaly.py').write_text('# disposable source fixture\n')
            args = self.arguments()
            args.source_root = source
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                checked(args)

    def test_existing_empty_output_is_rejected(self):
        from training_anomalib import digest
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'secondlook').mkdir()
            module = source / 'secondlook/anomaly.py'
            module.write_text('# disposable source fixture\n')
            args = self.arguments()
            args.source_root, args.source_sha256, args.output = source, digest(module), source
            with self.assertRaisesRegex(ValueError, 'Output must not exist'):
                checked(args)

    def test_lock_blocks_second_owner_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu.lock'
            with gpu_lock(path):
                with self.assertRaisesRegex(RuntimeError, 'another job'):
                    with gpu_lock(path):
                        self.fail('Second owner acquired lock')
            with gpu_lock(path):
                self.assertGreater(json.loads(path.read_text())['pid'], 0)

    def test_busy_gpu_fails_without_killing_owner(self):
        with patch('training_anomalib.subprocess.check_output', return_value='1234, unrelated-training, 12000') as command:
            with self.assertRaisesRegex(ValueError, 'active compute processes'):
                gpu_idle(0)
            self.assertEqual(command.call_count, 1)

    def test_gpu_query_failure_is_not_treated_as_idle(self):
        with patch('training_anomalib.subprocess.check_output', side_effect=subprocess.CalledProcessError(1, 'nvidia-smi')):
            with self.assertRaises(subprocess.CalledProcessError):
                gpu_idle(0)


if __name__ == '__main__':
    unittest.main()
