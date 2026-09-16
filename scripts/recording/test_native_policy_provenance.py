"""Installed native SyncExecution/PolicySource tests; fake model/devices only."""
from dataclasses import replace
import time
from types import SimpleNamespace
import unittest

import numpy as np
from physicalai.runtime import RobotRuntime, SyncExecution
from physicalai.runtime.execution.async_execution import AsyncExecution
from physicalai.runtime.smoothers import LerpSmoother, ReplaceSmoother

from secondlook.safety import JointLimit, RunPermit, SafetyError
from scripts.recording.native_policy_guard import (
    CaptureStamp, GuardScope, JOINT_NAMES, JOINT_UNITS, NativePolicyGuard,
    PolicySample, guarded_runtime_type,
)
from scripts.recording.native_policy_provenance import make_sync_provenance_source


class FakeModel:
    chunk_size = 1  # Deliberately misleading legacy getter; actual output is 3.

    def reset(self):
        pass

    def predict_action_chunk(self, observation):
        assert not any(key.startswith("__secondlook") for key in observation)
        self.last_array = np.full((3, 6), 2., dtype=np.float32)
        return self.last_array


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.model = FakeModel()
        self.source = self.make_source()
        self.bus = SimpleNamespace(emit_metrics=lambda *a, **k: None,
                                   emit_inference=lambda *a, **k: None)
        self.source.connect(bus=self.bus, session_id="synthetic")
        self.addCleanup(self.source.disconnect)

    def make_source(self, **changes):
        args = dict(model=self.model, execution=SyncExecution(request_threshold=0.),
                    smoother=ReplaceSmoother(), clock_id="fixture", camera_keys=("cam",),
                    expected_chunk_length=3, action_names=JOINT_NAMES, action_units=JOINT_UNITS,
                    clock=lambda: self.now)
        args.update(changes)
        return make_sync_provenance_source(**args)

    def inputs(self, stamp=None):
        stamp = self.now if stamp is None else stamp
        return (SimpleNamespace(state=np.ones(6), timestamp=stamp, images=None),
                {"cam": SimpleNamespace(data=np.zeros((4, 4, 3), np.uint8), timestamp=stamp)})

    def guard(self, clock=None):
        scope = GuardScope("fixture", ("cam",),
            {name: JointLimit(0, 10, 2, unit) for name, unit in zip(JOINT_NAMES, JOINT_UNITS)},
            max_age_s=.5, max_action_age_s=.5, max_sensor_skew_s=.1,
            heartbeat_timeout_s=2, goal_timeout_s=5, min_goal_time_s=.01,
            max_goal_time_s=.1, approved_hold_procedure="fake hold")
        guard = NativePolicyGuard(scope, clock=clock or (lambda: self.now),
            cancel_policy=lambda _: None, request_approved_hold=lambda _: None)
        guard.authorize(RunPermit("fake", self.now + 10, "fake hold", "fake workspace"),
                        supervision_at=self.now)
        return guard

    def send(self, guard, action, provenance, writes):
        current = CaptureStamp(self.now, "fixture")
        sample = PolicySample(JOINT_NAMES, JOINT_UNITS, [1.] * 6, current, {"cam": current},
            provenance.warmup_ready, provenance.generated, provenance.sequence,
            provenance.names, provenance.units, provenance.model_input,
            provenance.chunk_id, provenance.action_index)
        guard.send(action.tolist(), sample, goal_time=.03,
                   sole_writer=lambda *a, **k: writes.append(a))

    def test_explicit_sync_replace_required(self):
        for changes in ({"execution": AsyncExecution()}, {"execution": object()},
                        {"smoother": LerpSmoother()}, {"execution": None},
                        {"expected_chunk_length": None}, {"action_names": JOINT_NAMES[::-1]},
                        {"action_units": ("radians",) * 6}):
            with self.subTest(changes=changes), self.assertRaises(SafetyError):
                self.make_source(**changes)

    def test_chunk_identity_generation_and_input_times_survive_dequeue(self):
        self.source.update(*self.inputs(), 0)
        first = self.source.candidate_provenance
        self.now += .1
        self.source.update(*self.inputs(), 1)
        second = self.source.candidate_provenance
        self.assertEqual(first.generated, second.generated)
        self.assertEqual(first.model_input, second.model_input)
        self.assertEqual(first.chunk_id, second.chunk_id)
        self.assertEqual((first.action_index, second.action_index), (0, 1))
        self.assertGreater(second.sequence, first.sequence)
        self.assertEqual(second.model_input.robot.seconds, 10.)
        self.assertEqual(second.generated.seconds, 10.)
        with self.assertRaises(AttributeError):
            second.model_input.observation_id = "rewrite"

    def test_replayed_array_cannot_be_enqueued(self):
        self.source.update(*self.inputs(), 0)
        with self.assertRaises(SafetyError):
            self.source.action_queue.push_chunk(self.model.last_array)

    def test_new_chunk_gets_new_identity_without_rewriting_previous(self):
        self.source.update(*self.inputs(), 0)
        first = self.source.candidate_provenance
        for step in (1, 2, 3):
            self.now += .1
            self.source.update(*self.inputs(), step)
        new = self.source.candidate_provenance
        self.assertGreater(new.chunk_id, first.chunk_id)
        self.assertGreater(new.sequence, first.sequence)
        self.assertEqual(new.action_index, 0)
        self.assertNotEqual(new.model_input.observation_id, first.model_input.observation_id)
        self.assertEqual(first.model_input.robot.seconds, 10.)
        self.assertEqual(first.generated.seconds, 10.)
        self.assertEqual(new.model_input.robot.seconds, self.now)

    def test_nonfinite_chunk_rejected_before_enqueue(self):
        class NonfiniteModel(FakeModel):
            def predict_action_chunk(self, observation):
                return np.full((3, 6), np.nan)
        source = self.make_source(model=NonfiniteModel())
        source.connect(bus=self.bus, session_id="nonfinite")
        self.addCleanup(source.disconnect)
        with self.assertRaisesRegex(SafetyError, "finite"):
            source.update(*self.inputs(), 0)
        self.assertEqual(source.action_queue.remaining, 0)

    def test_recent_generation_from_old_input_rejected(self):
        self.now = 20.
        action = self.source.update(*self.inputs(stamp=10.), 0)
        provenance = self.source.candidate_provenance
        self.assertEqual(provenance.generated.seconds, 20.)
        writes = []
        with self.assertRaisesRegex(SafetyError, "original model input"):
            self.send(self.guard(), action, provenance, writes)
        self.assertEqual(writes, [])

    def test_fresh_captures_cannot_refresh_stale_queued_chunk(self):
        self.source.update(*self.inputs(), 0)
        self.now += .6
        action = self.source.update(*self.inputs(), 1)
        writes = []
        with self.assertRaises(SafetyError):
            self.send(self.guard(), action, self.source.candidate_provenance, writes)
        self.assertEqual(writes, [])

    def test_empty_queue_retains_identity_and_reused_action_rejected(self):
        action = self.source.update(*self.inputs(), 0)
        original = self.source.candidate_provenance
        guard, writes = self.guard(), []
        self.send(guard, action, original, writes)
        self.now += .1
        with self.assertRaises(SafetyError):
            self.send(guard, action, original, writes)
        self.assertEqual(len(writes), 1)
        queue = self.source.action_queue
        queue.pop()
        queue.pop()
        last = queue.last_provenance
        self.assertIsNone(queue.pop())
        self.assertEqual(queue.last_provenance, last)

    def test_actual_shape_required_not_legacy_chunk_getter(self):
        wrong = self.make_source(expected_chunk_length=1)
        wrong.connect(bus=self.bus, session_id="wrong-shape")
        self.addCleanup(wrong.disconnect)
        with self.assertRaisesRegex(SafetyError, "chunk shape"):
            wrong.update(*self.inputs(), 0)

    def test_real_native_loop_with_native_policy_source_and_execution(self):
        source = self.make_source(clock=time.monotonic)
        self.now = time.monotonic()
        guard = self.guard(clock=time.monotonic)
        writes = []
        class Robot:
            joint_names = JOINT_NAMES
            def connect(self): pass
            def disconnect(self): pass
            def get_observation(self):
                return SimpleNamespace(state=np.ones(6), timestamp=time.monotonic(), images=None)
            def send_action(self, action, *, goal_time): writes.append(tuple(action))
        class Camera:
            def connect(self): pass
            def disconnect(self): pass
            def read_latest(self):
                return SimpleNamespace(data=np.zeros((4, 4, 3), np.uint8), timestamp=time.monotonic())
        runtime = guarded_runtime_type(RobotRuntime)(robot=Robot(), action_source=source,
            fps=100, cameras={"cam": Camera()}, guard=guard,
            candidate_provenance=lambda: source.candidate_provenance,
            observation_clock_id="fixture", state_units=JOINT_UNITS)
        with runtime:
            self.assertEqual(runtime.run(duration_s=.03), 3)
        self.assertEqual(writes, [(2.,) * 6] * 3)
        self.assertEqual(source.candidate_provenance.action_index, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
