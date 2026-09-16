"""Synthetic bounds only. No Studio import, device, process, or robot command."""
from dataclasses import replace
import unittest

from secondlook.safety import JointLimit, RunPermit, SafetyError
from scripts.recording.native_policy_guard import (
    CaptureStamp, GuardScope, JOINT_NAMES, JOINT_UNITS, ModelInputIdentity, NativePolicyGuard, PolicySample,
)


class NativeGuardTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.events = []
        self.writes = []
        self.scope = GuardScope(
            clock_id="fixture-boot:monotonic", camera_keys=("low", "high"),
            limits={name: JointLimit(0, 10, 2, unit)
                    for name, unit in zip(JOINT_NAMES, JOINT_UNITS)},
            max_age_s=.2, max_action_age_s=.2, max_sensor_skew_s=.1,
            heartbeat_timeout_s=.5, goal_timeout_s=2,
            min_goal_time_s=.01, max_goal_time_s=.1,
            approved_hold_procedure="fixture-only approved hold",
        )
        self.guard = self.make_guard()

    def make_guard(self, **kwargs):
        return NativePolicyGuard(
            self.scope, clock=lambda: self.now,
            cancel_policy=kwargs.get("cancel_policy", lambda r: self.events.append(("cancel", r))),
            request_approved_hold=lambda r: self.events.append(("hold", r)),
        )

    def authorize(self):
        self.guard.authorize(RunPermit("fixture-session", 110, self.scope.approved_hold_procedure,
                                       "fixture workspace"), supervision_at=self.now)

    def sample(self, **changes):
        stamp = CaptureStamp(self.now, self.scope.clock_id)
        return replace(PolicySample(JOINT_NAMES, JOINT_UNITS, [1.] * 6, stamp,
                                    {"low": stamp, "high": stamp}, True, stamp, 0,
                                    JOINT_NAMES, JOINT_UNITS,
                                    ModelInputIdentity("fixture", stamp, (("low", stamp), ("high", stamp))),
                                    0, 0), **changes)

    def write(self, action, **kwargs):
        self.writes.append((action, kwargs))

    def send(self, sample=None, action=None, **kwargs):
        self.guard.send([2.] * 6 if action is None else action,
                        sample or self.sample(), goal_time=kwargs.get("goal_time", .05),
                        sole_writer=kwargs.get("sole_writer", self.write))

    def test_disarmed_no_write_and_cannot_rearm(self):
        with self.assertRaises(SafetyError):
            self.send()
        self.assertEqual(self.writes, [])
        with self.assertRaises(SafetyError):
            self.authorize()

    def test_missing_scope_limits_units_and_stop_fail_closed(self):
        for change in ({"limits": {}}, {"approved_hold_procedure": ""}, {"clock_id": ""},
                       {"max_action_age_s": 0}, {"camera_keys": ()}):
            with self.subTest(change=change), self.assertRaises(SafetyError):
                NativePolicyGuard(replace(self.scope, **change), cancel_policy=lambda _: None,
                                  request_approved_hold=lambda _: None)

    def test_valid_send_has_exact_order_and_goal_time(self):
        self.authorize()
        self.send()
        self.assertEqual(self.writes, [((2.,) * 6, {"goal_time": .05})])
        self.assertEqual(self.events, [])

    def test_bad_shape_nan_order_units_and_warmup_never_send(self):
        cases = [({}, [1.] * 5), ({}, [[1.] * 6]), ({}, [float("nan")] * 6),
                 ({}, [True] * 6), ({"names": JOINT_NAMES[::-1]}, None),
                 ({"candidate_names": JOINT_NAMES[::-1]}, None),
                 ({"candidate_units": ("radians",) * 6}, None),
                 ({"units": ("degrees",) * 6}, None), ({"warmup_ready": False}, None),
                 ({"cameras": {}}, None)]
        for sample_changes, action in cases:
            with self.subTest(sample_changes=sample_changes, action=action):
                self.guard = self.make_guard()
                self.authorize()
                with self.assertRaises(SafetyError):
                    self.send(self.sample(**sample_changes), action)
        self.assertEqual(self.writes, [])

    def test_bounds_state_delta_and_target_delta(self):
        for state, action in (([1.] * 6, [11.] * 6), ([11.] * 6, [2.] * 6),
                              ([1.] * 6, [4.] * 6)):
            self.guard = self.make_guard()
            self.authorize()
            with self.assertRaises(SafetyError):
                self.send(self.sample(state=state), action)
        self.guard = self.make_guard()
        self.authorize()
        self.send()
        self.now += .01
        with self.assertRaises(SafetyError):
            self.send(self.sample(state=[4.] * 6, candidate_sequence=1, chunk_id=1), [5.] * 6)
        self.assertEqual(len(self.writes), 1)

    def test_stale_future_mixed_clock_missing_capture_and_skew(self):
        for stamp in (CaptureStamp(99, self.scope.clock_id), CaptureStamp(101, self.scope.clock_id),
                      CaptureStamp(100, "Unix"), None, CaptureStamp(99.85, self.scope.clock_id)):
            self.guard = self.make_guard()
            self.authorize()
            with self.assertRaises(SafetyError):
                self.send(self.sample(robot=stamp))
        self.assertEqual(self.writes, [])

    def test_repeated_capture_or_cached_action_latches(self):
        for repeated_capture in (True, False):
            self.guard = self.make_guard()
            self.authorize()
            old = self.sample()
            self.send(old)
            self.now += .01
            with self.assertRaises(SafetyError):
                self.send(replace(old, candidate_sequence=1) if repeated_capture else self.sample())
        self.assertEqual(len(self.writes), 2)

    def test_stale_candidate_and_goal_time_rejected(self):
        self.authorize()
        with self.assertRaises(SafetyError):
            self.send(self.sample(candidate_generated=CaptureStamp(99, self.scope.clock_id)))
        self.guard = self.make_guard()
        self.authorize()
        with self.assertRaises(SafetyError):
            self.send(goal_time=.2)
        self.assertEqual(self.writes, [])

    def test_watchdog_without_inference_and_late_heartbeat_cannot_revive(self):
        self.authorize()
        self.now += .6
        with self.assertRaises(SafetyError):
            self.guard.poll()
        with self.assertRaises(SafetyError):
            self.guard.heartbeat("fixture-session")
        with self.assertRaises(SafetyError):
            self.send()
        self.assertEqual([name for name, _ in self.events], ["cancel", "hold"])
        self.assertEqual(self.writes, [])

    def test_goal_timeout_despite_regular_heartbeat(self):
        self.authorize()
        for _ in range(4):
            self.now += .4
            self.guard.heartbeat("fixture-session")
        self.now += .41
        with self.assertRaisesRegex(SafetyError, "Goal timeout"):
            self.guard.poll()

    def test_callback_failure_still_attempts_hold_and_writer_failure_not_retried(self):
        def fail(_reason):
            raise RuntimeError("fixture cancellation failed")
        self.guard = self.make_guard(cancel_policy=fail)
        self.authorize()
        with self.assertRaises(SafetyError):
            self.send(action=[float("inf")] * 6)
        self.assertEqual([name for name, _ in self.events], ["hold"])
        self.assertEqual(len(self.guard.callback_errors), 1)
        self.guard = self.make_guard()
        self.authorize()
        def writer(*args, **kwargs):
            self.writes.append("attempt")
            raise OSError("fixture publication failed")
        with self.assertRaises(OSError):
            self.send(sole_writer=writer)
        with self.assertRaises(SafetyError):
            self.send(sole_writer=writer)
        self.assertEqual(self.writes, ["attempt"])


if __name__ == "__main__":
    unittest.main()
