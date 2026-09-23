"""Synthetic contract checks; these never train a model or connect a robot."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import placement_contract as contract
import training_smolvla as runner
from prepare_placement import split_routes


def fixture():
    episodes = []
    for eid in range(10):
        destination = 'BLUE' if eid < 5 else 'PINK'
        episodes.append({'episode_index': eid, 'destination': destination, 'frames': 20,
                         'task_caption': contract.INSTRUCTIONS[destination], 'outcome': 'complete_success',
                         'outcome_evidence': 'synthetic fixture attestation',
                         'placement_events': {'starts_at_actual_ACT_handoff': True, 'handoff_evidence': 'fixture/start',
                                              'release_frame': 15, 'clear_frame': 19,
                                              'release_evidence': 'fixture/release', 'clearance_evidence': 'fixture/clear'}})
    manifest = {'skill': 'inspection_to_selected_bin', 'evidence_kind': 'real', 'finalized': True,
                'task_ready': False, 'controller_ready': False, 'episodes': episodes,
                'placement_contract': {'start': 'actual_ACT_handoff_pose', 'end': 'released_block_and_cleared_bin',
                                       'handoff_reference': 'fixture/handoff', 'robot_control_owner': 'fixture/owner'},
                'robot_contract': {'state_semantics': 'measured follower joint positions', 'joint_order': list(range(6))},
                'joint_names': list(range(6)), 'cameras': ['observation.images.overhead'],
                'features': {'observation.images.overhead': {'shape': [480, 640, 3]},
                             'observation.state': {'shape': [6]}, 'action': {'shape': [6]}},
                'train_episodes': [0, 1, 2, 5, 6, 7], 'validation_episodes': [3, 8], 'final_eval_episodes': [4, 9],
                'normalization': {'training_episodes_only': [0, 1, 2, 5, 6, 7], 'training_rows': 120,
                                  'excluded_validation_episodes': [3, 8], 'excluded_final_episodes': [4, 9]},
                'split_scope': 'episode_pilot', 'generalization_claim_permitted': False}
    conditioning = {'mode': 'selected_instruction', 'claim': 'experimental_imitation',
                    'instructions': dict(contract.INSTRUCTIONS), 'decision_mapping': dict(contract.DECISIONS)}
    return manifest, conditioning


class PlacementContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.conditioning = fixture()

    def check(self):
        return contract.check_placement(self.manifest, self.conditioning)

    def test_balanced_whole_episode_pilot(self):
        self.assertEqual(self.check()['route_counts'], {'BLUE': 5, 'PINK': 5})
        self.assertFalse(self.check()['controller_ready'])

    def test_decision_boundary(self):
        for decision, destination in [('GOOD', 'BLUE'), ('BAD', 'PINK')]:
            request = contract.placement_input(decision, {'observation.images.overhead': object()}, [0] * 6)
            self.assertEqual(request['task'], contract.INSTRUCTIONS[destination])
        self.assertIsNone(contract.placement_input('UNKNOWN', {}, None))
        for value in ['normal', 'anomalous', '', None, 'BLUE']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                contract.selected_instruction(value)

    def test_missing_measured_state_or_camera_rejected(self):
        for camera, state in [({}, [0] * 6), ({'observation.images.overhead': None}, [0] * 6),
                              ({'observation.images.overhead': object()}, [0] * 5),
                              ({'observation.images.overhead': object()}, [float('nan')] * 6)]:
            with self.subTest(state=state), self.assertRaises(ValueError):
                contract.placement_input('GOOD', camera, state)

    def test_one_route_missing(self):
        self.manifest['episodes'] = self.manifest['episodes'][:5]
        with self.assertRaisesRegex(ValueError, "'PINK': 5"):
            self.check()

    def test_step_one_is_insufficient(self):
        self.manifest['skill'] = 'pickup_to_inspection'
        with self.assertRaisesRegex(ValueError, 'step-one ACT data is insufficient'):
            self.check()

    def test_invalid_evidence_and_order_fail(self):
        for key, value in [('starts_at_actual_ACT_handoff', False), ('handoff_evidence', None),
                           ('release_frame', 19), ('clear_frame', 20), ('clearance_evidence', '')]:
            manifest = copy.deepcopy(self.manifest)
            manifest['episodes'][0]['placement_events'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.check_placement(manifest, self.conditioning)

    def test_wrong_instruction_or_mapping_fails(self):
        self.conditioning['decision_mapping']['GOOD'] = 'PINK'
        with self.assertRaisesRegex(ValueError, 'mapping'):
            self.check()
        self.conditioning['decision_mapping'] = dict(contract.DECISIONS)
        self.manifest['episodes'][0]['task_caption'] = 'Move the good block.'
        with self.assertRaisesRegex(ValueError, 'instruction'):
            self.check()

    def test_episode_leakage_and_unbalanced_holdout_fail(self):
        for key, ids in [('validation_episodes', [0, 8]), ('final_eval_episodes', [3, 4])]:
            manifest = copy.deepcopy(self.manifest)
            manifest[key] = ids
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.check_placement(manifest, self.conditioning)

    def test_holdout_stats_rejected(self):
        self.manifest['normalization']['training_rows'] = 200
        with self.assertRaisesRegex(ValueError, 'normalization'):
            self.check()

    def test_native_runner_uses_strict_contract(self):
        config = {'training_scope': 'step_three_placement', 'conditioning': self.conditioning}
        self.assertEqual(runner.check_conditioning(config, self.manifest), self.conditioning)
        self.conditioning['mode'] = 'anomalib_rows'
        with self.assertRaisesRegex(ValueError, 'without anomaly scores'):
            runner.check_conditioning(config, self.manifest)

    def test_commanded_state_rejected(self):
        self.manifest['robot_contract']['state_semantics'] = 'previous commanded targets'
        with self.assertRaisesRegex(ValueError, 'measured'):
            self.check()

    def test_operator_scope_does_not_invent_frame_evidence_or_success(self):
        self.manifest['placement_contract'].update(qualification='operator_confirmed_demonstration_scope',
            operator_confirmation='User confirms all episodes FROM INSPECTION to bins',
            sampled_visual_qc='sampled video review', individual_success_verified=False)
        for episode in self.manifest['episodes']:
            episode.update(outcome='operator_confirmed_placement_demonstration', individual_success_verified=False)
            episode.pop('placement_events')
        self.assertEqual(self.check()['route_counts'], {'BLUE': 5, 'PINK': 5})
        self.manifest['episodes'][0]['individual_success_verified'] = True
        with self.assertRaisesRegex(ValueError, 'do not invent'):
            self.check()

    def test_all_27_episodes_split_once_by_route(self):
        routes = {eid: 'BLUE' if eid < 15 else 'PINK' for eid in range(27)}
        split = split_routes(routes, 42)
        self.assertEqual({key: len(value) for key, value in split.items()},
                         {'train_episodes': 17, 'validation_episodes': 5, 'final_eval_episodes': 5})
        self.assertEqual(sorted(eid for ids in split.values() for eid in ids), list(range(27)))
        for key in ('validation_episodes', 'final_eval_episodes'):
            self.assertEqual(sum(routes[eid] == 'BLUE' for eid in split[key]), 3)
            self.assertEqual(sum(routes[eid] == 'PINK' for eid in split[key]), 2)
        self.assertEqual(split, split_routes(routes, 42))


if __name__ == '__main__':
    unittest.main()
