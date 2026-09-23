"""Disposable Arrow recording fixture; preprocessing only, no model or hardware."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from prepare_placement import prepare
from placement_contract import INSTRUCTIONS

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None


@unittest.skipIf(pa is None, 'Run with prepared Intel/H100 Arrow preprocessing environment')
class PreparationTests(unittest.TestCase):
    def test_canonical_derivative_and_train_only_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, derivative, prepared = base / 'source', base / 'dataset', base / 'prepared'
            for folder in ('meta/episodes/chunk-000', 'data/chunk-000'):
                (source / folder).mkdir(parents=True)
            names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
            feature = {'dtype': 'float32', 'shape': [6], 'names': names}
            camera = 'observation.images.camera 1 _low_'
            info = {'fps': 30, 'features': {'observation.state': feature, 'action': feature,
                                           camera: {'dtype': 'video', 'shape': [12, 16, 3]}}, 'total_tasks': 2}
            (source / 'meta/info.json').write_text(json.dumps(info))
            (source / 'meta/stats.json').write_text('{}')
            original_captions = ['Move good to blue plate', 'Move bad to pink plate']
            pq.write_table(pa.Table.from_pandas(pd.DataFrame({'task_index': [0, 1]}, index=original_captions)), source / 'meta/tasks.parquet')
            rows = []
            for eid in range(27):
                for frame in range(3):
                    rows.append({'episode_index': eid, 'frame_index': frame, 'index': len(rows), 'task_index': 0 if eid < 15 else 1,
                                 'observation.state': [float(eid)] * 6, 'action': [float(eid + 1)] * 6})
            pq.write_table(pa.Table.from_pylist(rows), source / 'data/chunk-000/file-000.parquet')
            episode_rows = [{'episode_index': eid, 'length': 3, 'tasks': [original_captions[0 if eid < 15 else 1]]} for eid in range(27)]
            pq.write_table(pa.Table.from_pylist(episode_rows), source / 'meta/episodes/chunk-000/file-000.parquet')
            shutil.copytree(source, derivative)
            sources = {'routes': {str(eid): 'BLUE' if eid < 15 else 'PINK' for eid in range(27)},
                       'environment': {'robots': [{'robot': {'type': 'SO101_Follower', 'payload': {'calibration': {'fixture': True}}}}],
                                       'cameras': [{'name': 'camera 1 (low)', 'id': 'fixture-camera', 'fingerprint': {'serial': 'fixture'},
                                                    'payload': {'width': 16, 'height': 12}}]},
                       'source_receipt': {'evidence_kind': 'synthetic fixture'}, 'visual_qc': {'synthetic': True}, 'prior_config': {}}
            args = {'dataset_root': derivative, 'output_root': prepared, 'seed': 42,
                    'derivative_copy_confirmed': True, 'visual_normalization_identity_verified': True,
                    'operator_confirmation': 'Synthetic fixture only; no physical success',
                    'h100_root': '/workspace/second-look-h100/test-fixture'}
            for key, value in sources.items():
                path = base / (key + '.json')
                path.write_text(json.dumps(value))
                args[key] = path
            with redirect_stdout(io.StringIO()):
                prepare(SimpleNamespace(**args))
            manifest = json.loads((prepared / 'placement-manifest.json').read_text())
            stats = json.loads((prepared / 'train_stats.json').read_text())
            self.assertEqual(manifest['joint_names'], names)
            self.assertEqual(manifest['normalization']['training_rows'], 51)
            expected_mean = np.mean(manifest['train_episodes'])
            np.testing.assert_allclose(stats['observation.state']['mean'], [expected_mean] * 6)
            self.assertNotEqual(expected_mean, np.mean(range(27)))
            self.assertEqual(stats['observation.state']['count'], [51])
            self.assertEqual(sorted(eid for key in ('train_episodes', 'validation_episodes', 'final_eval_episodes') for eid in manifest[key]), list(range(27)))
            tasks = pq.read_table(derivative / 'meta/tasks.parquet').to_pandas()
            self.assertEqual(tasks.index.tolist(), [INSTRUCTIONS['BLUE'], INSTRUCTIONS['PINK']])
            rewritten = pq.read_table(derivative / 'data/chunk-000/file-000.parquet').to_pylist()
            for row in rewritten:
                self.assertEqual(tasks.index[row['task_index']], INSTRUCTIONS['BLUE' if row['episode_index'] < 15 else 'PINK'])
            rewritten_episodes = pq.read_table(derivative / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
            self.assertEqual(rewritten_episodes[20]['tasks'], [INSTRUCTIONS['PINK']])
            self.assertEqual(pq.read_table(source / 'meta/tasks.parquet').to_pandas().index.tolist(), original_captions)
            self.assertEqual(json.loads((prepared / 'caption-provenance.json').read_text())['20']['original_captions'], [original_captions[1]])
            self.assertFalse(manifest['episodes'][0]['individual_success_verified'])


if __name__ == '__main__':
    unittest.main()
