"""Execute the INSTALLED RobotRuntime with disposable fake devices only.

Run using Studio's existing interpreter; no installation or package changes.
All limits, clocks, generated candidates, and stop acknowledgments are synthetic.
"""
from __future__ import annotations

from dataclasses import replace
import threading
import time
from types import SimpleNamespace
import unittest

from secondlook.safety import JointLimit, RunPermit, SafetyError
from scripts.recording.native_policy_guard import (
    CandidateProvenance, CaptureStamp, GuardScope, JOINT_NAMES, JOINT_UNITS,
    LocalGuardWatchdog, ModelInputIdentity, NativePolicyGuard, guarded_runtime_type,
)


class FakeRobot:
    joint_names = JOINT_NAMES

    def __init__(self):
        self.writes = []

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_observation(self):
        return SimpleNamespace(state=[1.] * 6, timestamp=time.monotonic())

    def send_action(self, action, *, goal_time):
        self.writes.append((tuple(action), goal_time))


class FakeCamera:
    def connect(self):
        pass

    def disconnect(self):
        pass

    def read_latest(self):
        return SimpleNamespace(data=None, timestamp=time.monotonic(), sequence=0)


class FakeSource:
    def __init__(self, *, blocked=False, metadata=True):
        self.blocked = blocked
        self.metadata = metadata
        self.entered = threading.Event()
        self.release = threading.Event()
        self.provenance = None

    def connect(self, **kwargs):
        pass

    def disconnect(self):
        pass

    def update(self, robot_state, camera_frames, step):
        self.entered.set()
        if self.blocked and not self.release.wait(2):
            raise RuntimeError("Fixture cancellation failed to release blocked inference")
        if self.metadata:
            # Synthetic source generation, never substituted for real queue data.
            self.provenance = CandidateProvenance(CaptureStamp(time.monotonic(), "fixture:monotonic"),
                step, JOINT_NAMES, JOINT_UNITS, True,
                ModelInputIdentity(f"fixture-{step}", CaptureStamp(robot_state.timestamp, "fixture:monotonic"),
                    tuple((key, CaptureStamp(frame.timestamp, "fixture:monotonic"))
                          for key, frame in camera_frames.items())), step, 0)
        return [2.] * 6


class NativeRuntimeFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This imports the real loop, but every device constructor below is fake.
        from physicalai.runtime import RobotRuntime
        cls.runtime_type = guarded_runtime_type(RobotRuntime)

    def setUp(self):
        self.robot = FakeRobot()
        self.source = FakeSource()
        self.events = []
        self.scope = GuardScope(
            "fixture:monotonic", ("fixture-camera",),
            {name: JointLimit(0, 10, 2, unit) for name, unit in zip(JOINT_NAMES, JOINT_UNITS)},
            max_age_s=1, max_action_age_s=1, max_sensor_skew_s=.5,
            heartbeat_timeout_s=.1, goal_timeout_s=2, min_goal_time_s=.001,
            max_goal_time_s=.1, approved_hold_procedure="fixture acknowledged hold",
        )

    def build(self, *, authorize=True, callbacks=()):
        def cancel(reason):
            self.events.append(("cancel", time.monotonic(), reason))
            self.source.release.set()
        def hold(reason):
            self.events.append(("hold", time.monotonic(), reason))
        self.guard = NativePolicyGuard(self.scope, cancel_policy=cancel, request_approved_hold=hold)
        if authorize:
            now = time.monotonic()
            self.guard.authorize(RunPermit("fixture-session", now + 3,
                                           self.scope.approved_hold_procedure, "fixture only"),
                                 supervision_at=now)
        runtime = self.runtime_type(
            robot=self.robot, action_source=self.source, fps=100,
            cameras={"fixture-camera": FakeCamera()}, callbacks=callbacks, guard=self.guard,
            candidate_provenance=lambda: self.source.provenance,
            observation_clock_id=self.scope.clock_id, state_units=JOINT_UNITS,
        )
        return runtime

    def test_real_native_loop_valid_candidate_reaches_fake_send_once(self):
        with self.build() as runtime:
            self.assertEqual(runtime.run(duration_s=.01), 1)
        self.assertEqual(self.robot.writes, [((2.,) * 6, .03)])

    def test_final_callback_corruption_never_reaches_fake_send(self):
        class LastTransform:
            def on_action_ready(self, **kwargs):
                return [float("nan")] * 6
        with self.build(callbacks=[LastTransform()]) as runtime:
            with self.assertRaises(SafetyError):
                runtime.run(duration_s=.01)
        self.assertEqual(self.robot.writes, [])

    def test_missing_authorization_and_candidate_metadata_reject(self):
        for authorize, metadata in ((False, True), (True, False)):
            self.source = FakeSource(metadata=metadata)
            with self.build(authorize=authorize) as runtime:
                with self.assertRaises(SafetyError):
                    runtime.run(duration_s=.01)
                if not authorize:
                    self.assertFalse(self.source.entered.is_set())
        self.assertEqual(self.robot.writes, [])

    def test_missing_limits_reject_before_runtime_construction(self):
        self.scope = replace(self.scope, limits={})
        with self.assertRaises(SafetyError):
            self.build()
        self.assertEqual(self.robot.writes, [])

    def test_local_watchdog_cancels_blocked_inference_no_late_publish(self):
        self.source = FakeSource(blocked=True)
        with self.build() as runtime:
            watchdog = LocalGuardWatchdog(self.guard, .01)
            started = time.monotonic()
            watchdog.start()
            try:
                with self.assertRaises(SafetyError):
                    runtime.run(duration_s=.01)
            finally:
                watchdog.close()
            self.assertTrue(self.source.entered.is_set())
            self.assertLess(time.monotonic() - started, 1)
        self.assertEqual([event[0] for event in self.events], ["cancel", "hold"])
        self.assertLess(self.events[1][1] - started, 1)
        self.assertEqual(self.robot.writes, [])

    def test_second_thread_cannot_use_guarded_send_or_run(self):
        self.source = FakeSource(blocked=True)
        errors = []
        with self.build() as runtime:
            def run():
                try:
                    runtime.run(duration_s=.01)
                except SafetyError as error:
                    errors.append(str(error))
            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(self.source.entered.wait(1))
            with self.assertRaisesRegex(SafetyError, "Another run"):
                runtime.run(duration_s=.01)
            with self.assertRaisesRegex(SafetyError, "does not own"):
                runtime._resilient_send([2.] * 6)
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertEqual(self.robot.writes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
