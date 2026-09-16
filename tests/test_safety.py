import math
import unittest

from secondlook.safety import (JointLimit, ReviewBudget, RunPermit, SafetyError,
                               SafetyGate, require_fresh, validate_action)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        # Synthetic fixture units/bounds; these are NOT robot configuration.
        self.limits = {"test_joint": JointLimit(-10, 10, 2, "synthetic units")}
        self.now = 100.
        self.stops = []
        self.gate = SafetyGate(self.limits, 1, 2, self.stops.append, lambda: self.now)

    def authorize(self):
        self.gate.authorize(RunPermit("mock-session", 110, "mock.stop", "fixture"), 100)

    def test_disarmed_by_default(self):
        with self.assertRaisesRegex(SafetyError, "DISARMED"):
            self.gate.begin({"test_joint": 1}, {"test_joint": 0}, 100, 100)

    def test_requires_complete_limits_and_session(self):
        with self.assertRaises(SafetyError):
            SafetyGate({}, 1, 1, self.stops.append)
        with self.assertRaises(SafetyError):
            self.gate.authorize(RunPermit("", 110, "", ""), 100)

    def test_finite_matching_bounded_actions(self):
        cases = [({}, {"test_joint": 0}), ({"wrong": 0}, {"test_joint": 0}),
                 ({"test_joint": float("nan")}, {"test_joint": 0}),
                 ({"test_joint": 11}, {"test_joint": 0}),
                 ({"test_joint": 3}, {"test_joint": 0}),
                 ({"test_joint": 1}, {"test_joint": float("inf")}),
                 ({"test_joint": True}, {"test_joint": 0}),
                 ({"test_joint": 1}, {"test_joint": 12})]
        for action, state in cases:
            with self.subTest(action=action, state=state), self.assertRaises(SafetyError):
                validate_action(action, state, self.limits)
        self.assertEqual(validate_action({"test_joint": 1}, {"test_joint": 0}, self.limits),
                         {"test_joint": 1.0})

    def test_stale_observation_or_state_disarms(self):
        for observation_at, state_at in [(98, 100), (100, 98), (101, 100)]:
            self.authorize()
            with self.assertRaises(SafetyError):
                self.gate.begin({"test_joint": 1}, {"test_joint": 0}, observation_at, state_at)
            self.assertIsNone(self.gate.permit)
        self.assertEqual(len(self.stops), 3)

    def test_overlapping_runs_rejected(self):
        self.authorize()
        self.gate.begin({"test_joint": 1}, {"test_joint": 0}, 100, 100)
        with self.assertRaisesRegex(SafetyError, "Another run"):
            self.gate.begin({"test_joint": 1}, {"test_joint": 0}, 100, 100)
        self.gate.end()

    def test_expiry_and_supervision_loss_stop_once(self):
        for now in (103., 111.):
            self.now = 100.
            self.authorize()
            self.now = now
            with self.assertRaises(SafetyError):
                self.gate.check()
            with self.assertRaises(SafetyError):
                self.gate.check()
        self.assertEqual(len(self.stops), 2)

    def test_heartbeat_cannot_reactivate_stopped_session(self):
        self.authorize()
        self.gate.abort("mock timeout")
        with self.assertRaises(SafetyError):
            self.gate.heartbeat("mock-session")
        self.assertEqual(self.stops, ["mock timeout"])

    def test_failed_stop_latches_and_releases_owner(self):
        self.authorize()
        def broken_stop(_reason):
            raise RuntimeError("mock unavailable hardware")
        self.gate.stop = broken_stop
        with self.assertRaises(RuntimeError):
            self.gate.begin({"test_joint": 7}, {"test_joint": 0}, 100, 100)
        self.assertIsNone(self.gate.permit)
        self.assertFalse(self.gate._owner.locked())

    def test_failed_ambiguous_and_exhausted_retry_outcomes(self):
        budget = ReviewBudget(2)
        self.assertEqual(budget.assess("ambiguous"), "RETRY")
        self.assertEqual(budget.assess("failed"), "RETRY")
        self.assertEqual(budget.assess("failed"), "REVIEW")
        self.assertEqual(budget.assess("verified_success"), "REVIEW")
        self.assertEqual(ReviewBudget(1).assess("unavailable_hardware"), "REVIEW")
        self.assertEqual(ReviewBudget(1).assess("timeout"), "REVIEW")
        self.assertEqual(ReviewBudget(1).assess("verified_success"), "VERIFIED")


if __name__ == "__main__":
    unittest.main()
