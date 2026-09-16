"""Synthetic row fixtures: these are not recorded demonstrations."""
import copy
import unittest

from contracts import validate_groups, validate_rows


def fixture():
    names = ['shoulder_pan.pos', 'gripper.pos']
    info = {'codebase_version': 'v3.0', 'fps': 10, 'total_episodes': 2, 'total_frames': 6,
            'features': {key: {'shape': [2], 'names': names.copy()} for key in ('observation.state', 'action')}}
    info['features']['observation.images.synthetic'] = {'dtype': 'video'}
    episodes = [{'episode_index': eid, 'length': 3, 'dataset_from_index': eid * 3,
                 'dataset_to_index': (eid + 1) * 3} for eid in range(2)]
    rows = [{'episode_index': eid, 'frame_index': i, 'index': eid * 3 + i, 'timestamp': i / 10,
             'observation.state': [i, i * 10], 'action': [i, i * 10]} for eid in range(2) for i in range(3)]
    return info, episodes, rows


class ContractTests(unittest.TestCase):
    def test_complete_episode_split(self):
        result = validate_rows(*fixture(), [1])
        self.assertEqual(result['train_episodes'], [0])
        self.assertEqual(result['validation_episodes'], [1])

    def test_corruptions_fail(self):
        mutations = {
            'nan action': lambda i, e, r: r[0]['action'].__setitem__(0, float('nan')),
            'timestamp gap': lambda i, e, r: r[1].__setitem__('timestamp', 0.8),
            'missing frame': lambda i, e, r: r.pop(),
            'duplicate frame': lambda i, e, r: r[1].__setitem__('frame_index', 0),
            'episode boundary': lambda i, e, r: e[1].__setitem__('dataset_from_index', 2),
            'joint order': lambda i, e, r: i['features']['action'].__setitem__('names', ['gripper.pos', 'shoulder_pan.pos']),
            'constant gripper': lambda i, e, r: [row['action'].__setitem__(1, 0) for row in r],
            'empty camera': lambda i, e, r: i['features'].pop('observation.images.synthetic'),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                values = copy.deepcopy(fixture())
                mutation(*values)
                with self.assertRaises(ValueError):
                    validate_rows(*values, [1])

    def test_bad_splits_fail(self):
        for split in ([], [0, 1], [2], [1, 1]):
            with self.subTest(split=split), self.assertRaises(ValueError):
                validate_rows(*fixture(), split)

    def test_final_eval_is_excluded(self):
        info, episodes, rows = fixture()
        info.update(total_episodes=3, total_frames=9)
        episodes.append({'episode_index': 2, 'length': 3, 'dataset_from_index': 6, 'dataset_to_index': 9})
        rows.extend([{**copy.deepcopy(row), 'episode_index': 2, 'index': row['index'] + 6} for row in rows[:3]])
        report = validate_rows(info, episodes, rows, [1], final_eval_episodes=[2])
        self.assertEqual(report['train_episodes'], [0])
        self.assertEqual(report['validation_episodes'], [1])
        self.assertEqual(report['final_eval_episodes'], [2])
        with self.assertRaises(ValueError):
            validate_rows(info, episodes, rows, [1], final_eval_episodes=[1])

    def test_grouped_evaluation_rejects_shared_specimens_and_sessions(self):
        report = validate_rows(*fixture(), [1])
        labels = {str(i): {'destination': 'normal_pile', 'outcome': 'complete_success',
                           'outcome_evidence': 'Synthetic test of label validation, not actual evidence',
                           'specimen_id': 'same', 'session_id': 'same'} for i in range(2)}
        validate_groups(report, labels, 'real', 'loader_smoke')
        with self.assertRaises(ValueError):
            validate_groups(report, labels, 'real', 'grouped_evaluation')
        labels['1'].update(specimen_id='different', session_id='different')
        validate_groups(report, labels, 'real', 'grouped_evaluation')
        del labels['1']['session_id']
        with self.assertRaises(ValueError):
            validate_groups(report, labels, 'real', 'loader_smoke')


if __name__ == '__main__':
    unittest.main()
