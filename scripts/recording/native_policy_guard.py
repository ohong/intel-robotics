"""Staged pre-send seam for one native runtime writer; imports no hardware SDK.

This module is not wired into Studio. Configuration and authorization have no
defaults. Callbacks request cancellation/approved hold in the existing owner;
they must never create a second robot writer or disconnect a robot.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from secondlook.safety import JointLimit, RunPermit, SafetyError, SafetyGate, require_fresh, validate_action

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_UNITS = ("normalized_-100_100",) * 5 + ("normalized_0_100",)


@dataclass(frozen=True)
class CaptureStamp:
    seconds: float
    # Must identify the Intel host boot and its monotonic clock, not Unix time.
    clock_id: str


@dataclass(frozen=True)
class ModelInputIdentity:
    observation_id: str
    robot: CaptureStamp
    cameras: tuple[tuple[str, CaptureStamp], ...]


@dataclass(frozen=True)
class PolicySample:
    names: tuple[str, ...]
    units: tuple[str, ...]
    state: Sequence[float]
    robot: CaptureStamp
    cameras: Mapping[str, CaptureStamp]
    warmup_ready: bool
    candidate_generated: CaptureStamp
    candidate_sequence: int
    candidate_names: tuple[str, ...]
    candidate_units: tuple[str, ...]
    model_input: ModelInputIdentity
    chunk_id: int
    action_index: int


@dataclass(frozen=True)
class GuardScope:
    clock_id: str
    camera_keys: tuple[str, ...]
    limits: Mapping[str, JointLimit]
    max_age_s: float
    max_action_age_s: float
    max_sensor_skew_s: float
    heartbeat_timeout_s: float
    goal_timeout_s: float
    min_goal_time_s: float
    max_goal_time_s: float
    approved_hold_procedure: str


def _positive(value: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _vector(values: Sequence[float], label: str) -> dict[str, float]:
    # Deliberately require a flat Python vector. SDK adapters must explicitly
    # convert numpy shape (6,), never silently flatten a batch or action chunk.
    if not isinstance(values, (list, tuple)) or len(values) != 6 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in values
    ):
        raise SafetyError(f"{label}: expected exactly six finite scalar values")
    return dict(zip(JOINT_NAMES, values, strict=True))


class NativePolicyGuard:
    """One-shot authorization with a latched fault and no reset/rearm method.

    Integration must call poll from a local watchdog even during inference.
    The injected sole-writer send must be nonblocking and bounded; this Python
    seam cannot safely terminate a hung native motor call or a dead process.
    """

    def __init__(self, scope: GuardScope, *, cancel_policy: Callable[[str], None],
                 request_approved_hold: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic):
        if not isinstance(scope, GuardScope):
            raise SafetyError("An explicit scope is required")
        if not scope.clock_id or not scope.approved_hold_procedure.strip():
            raise SafetyError("Same-clock provenance and approved hold procedure are required")
        if not scope.camera_keys or len(set(scope.camera_keys)) != len(scope.camera_keys):
            raise SafetyError("Exact unique camera keys are required")
        if set(scope.limits) != set(JOINT_NAMES):
            raise SafetyError("Exactly the six installed SO-101 joint limits are required")
        for i, name in enumerate(JOINT_NAMES):
            limit = scope.limits[name]
            limit.validate()
            if limit.unit != JOINT_UNITS[i]:
                raise SafetyError("Limit units must match the normalized installed contract")
            # These are representation bounds, not authorized workspace limits.
            if limit.minimum < (-100 if i < 5 else 0) or limit.maximum > 100:
                raise SafetyError("Configured bounds exceed normalized representation")
        bounds = (scope.max_age_s, scope.max_action_age_s, scope.max_sensor_skew_s, scope.heartbeat_timeout_s,
                  scope.goal_timeout_s, scope.min_goal_time_s, scope.max_goal_time_s)
        if not all(_positive(v) for v in bounds) or scope.min_goal_time_s > scope.max_goal_time_s:
            raise SafetyError("Explicit finite positive timing bounds are required")
        if not callable(cancel_policy) or not callable(request_approved_hold):
            raise SafetyError("Both existing-owner fault callbacks are required")
        self.scope = scope
        self.clock = clock
        self.cancel_policy = cancel_policy
        self.request_hold = request_approved_hold
        self.reason: str | None = None
        self.callback_errors: list[str] = []
        self._lock = threading.RLock()
        self._started: float | None = None
        self._last_stamps: dict[str, float] = {}
        self._last_action: dict[str, float] | None = None
        self._last_candidate_sequence = -1
        self._last_chunk: tuple[int, int, CaptureStamp, ModelInputIdentity] | None = None
        self._gate = SafetyGate(dict(scope.limits), scope.max_age_s, scope.heartbeat_timeout_s,
                                self._fault, clock)

    def _fault(self, reason: str) -> None:
        if self.reason is not None:
            return
        self.reason = reason
        # Latch first and attempt both signals even if cancellation fails.
        for callback in (self.cancel_policy, self.request_hold):
            try:
                callback(reason)
            except Exception as error:
                self.callback_errors.append(f"{type(error).__name__}: {error}")

    def authorize(self, permit: RunPermit, *, supervision_at: float) -> None:
        with self._lock:
            if self.reason is not None or self._started is not None:
                raise SafetyError("Guard cannot rearm; a new supervised scope is required")
            if permit.stop_procedure != self.scope.approved_hold_procedure:
                raise SafetyError("Permit must name the exact approved hold procedure")
            self._gate.authorize(permit, supervision_at)
            self._started = self.clock()

    def heartbeat(self, session_id: str) -> None:
        with self._lock:
            # A late heartbeat must not revive a timed-out supervisor.
            self.poll()
            self._gate.heartbeat(session_id)

    def reject(self, reason: str) -> None:
        """Latch a missing integration prerequisite without publishing anything."""
        with self._lock:
            self._gate.abort(reason)
            self._fault(reason)

    def poll(self) -> None:
        with self._lock:
            if self.reason is not None:
                raise SafetyError(f"Latched fault: {self.reason}")
            self._gate.check()
            if self._started is None or self.clock() - self._started >= self.scope.goal_timeout_s:
                self._gate.abort("Goal timeout")
                raise SafetyError("Goal timeout")

    def send(self, action: Sequence[float], sample: PolicySample, *, goal_time: float,
             sole_writer: Callable[..., None]) -> None:
        """Validate immediately before the sole writer; no retries or clamping.

        This seam belongs AFTER every native on_action_ready transform. The
        sample must be the exact capture used by the action source this tick.
        """
        with self._lock:
            try:
                self.poll()
                if sample.names != JOINT_NAMES or sample.units != JOINT_UNITS:
                    raise SafetyError("State/action joint order or units mismatch")
                if sample.candidate_names != JOINT_NAMES or sample.candidate_units != JOINT_UNITS:
                    raise SafetyError("Policy candidate joint order or units mismatch")
                if sample.warmup_ready is not True:
                    raise SafetyError("Policy warmup is not ready")
                if not _positive(goal_time) or not self.scope.min_goal_time_s <= goal_time <= self.scope.max_goal_time_s:
                    raise SafetyError("Command goal_time outside approved bounds")
                if set(sample.cameras) != set(self.scope.camera_keys):
                    raise SafetyError("Camera set mismatch")
                stamps = {"robot": sample.robot, **{f"camera:{k}": v for k, v in sample.cameras.items()}}
                now = self.clock()
                candidate = sample.candidate_generated
                if not isinstance(candidate, CaptureStamp) or candidate.clock_id != self.scope.clock_id:
                    raise SafetyError("Candidate requires original generation timestamp provenance")
                if type(candidate.seconds) not in (int, float):
                    raise SafetyError("Invalid candidate timestamp")
                require_fresh(candidate.seconds, now, self.scope.max_action_age_s, "candidate")
                original = sample.model_input
                if not isinstance(original, ModelInputIdentity) or not original.observation_id:
                    raise SafetyError("Original model-input identity is required")
                if len(original.cameras) != len(self.scope.camera_keys) or set(dict(original.cameras)) != set(self.scope.camera_keys):
                    raise SafetyError("Original model-input camera set mismatch")
                input_stamps = [original.robot, *(stamp for _, stamp in original.cameras)]
                for stamp in input_stamps:
                    if not isinstance(stamp, CaptureStamp) or stamp.clock_id != self.scope.clock_id:
                        raise SafetyError("Original model-input clock provenance mismatch")
                    require_fresh(stamp.seconds, now, self.scope.max_age_s, "original model input")
                    if stamp.seconds > candidate.seconds:
                        raise SafetyError("Candidate predates its model input")
                if max(s.seconds for s in input_stamps) - min(s.seconds for s in input_stamps) > self.scope.max_sensor_skew_s:
                    raise SafetyError("Original model-input sensor skew exceeds bound")
                if type(sample.chunk_id) is not int or sample.chunk_id < 0 or type(sample.action_index) is not int or sample.action_index < 0:
                    raise SafetyError("Explicit chunk and action indices are required")
                if self._last_chunk is not None:
                    chunk, index, generated, identity = self._last_chunk
                    if sample.chunk_id < chunk or (sample.chunk_id == chunk and (
                        sample.action_index <= index or candidate != generated or original != identity
                    )):
                        raise SafetyError("Reused chunk action or changed original provenance")
                if type(sample.candidate_sequence) is not int or sample.candidate_sequence <= self._last_candidate_sequence:
                    raise SafetyError("Candidate sequence did not advance; cached action forbidden")
                for key, stamp in stamps.items():
                    if not isinstance(stamp, CaptureStamp) or stamp.clock_id != self.scope.clock_id:
                        raise SafetyError("Missing source timestamp or mixed clock provenance")
                    if type(stamp.seconds) not in (int, float):
                        raise SafetyError("Invalid capture timestamp")
                    require_fresh(stamp.seconds, now, self.scope.max_age_s, key)
                    if stamp.seconds <= self._last_stamps.get(key, float("-inf")):
                        raise SafetyError(f"{key}: capture timestamp did not advance")
                times = [stamp.seconds for stamp in stamps.values()]
                if max(times) - min(times) > self.scope.max_sensor_skew_s:
                    raise SafetyError("Sensor capture skew exceeds approved bound")
                targets, state = _vector(action, "action"), _vector(sample.state, "state")
                targets = validate_action(targets, state, self._gate.limits)
                if self._last_action is not None:
                    validate_action(targets, self._last_action, self._gate.limits)
                self.poll()
                # Keep validation and this bounded publication serialized with
                # watchdog faults; no policy send may follow a latched fault.
                sole_writer(tuple(targets[name] for name in JOINT_NAMES), goal_time=goal_time)
                self._last_stamps = {key: stamp.seconds for key, stamp in stamps.items()}
                self._last_action = targets
                self._last_candidate_sequence = sample.candidate_sequence
                self._last_chunk = (sample.chunk_id, sample.action_index, candidate, original)
            except Exception as error:
                self._gate.abort(str(error))
                # DISARMED failures also latch: an attempted send cannot later
                # be silently enabled by delayed authorization.
                self._fault(str(error))
                raise


@dataclass(frozen=True)
class CandidateProvenance:
    generated: CaptureStamp
    sequence: int
    names: tuple[str, ...]
    units: tuple[str, ...]
    warmup_ready: bool
    model_input: ModelInputIdentity
    chunk_id: int
    action_index: int


class LocalGuardWatchdog:
    """Local poller, not a supervision-heartbeat producer or physical stop."""

    def __init__(self, guard: NativePolicyGuard, period_s: float):
        if not _positive(period_s) or period_s > guard.scope.heartbeat_timeout_s / 4:
            raise SafetyError("Watchdog period must be at most one quarter of heartbeat timeout")
        self.guard = guard
        self.period_s = period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="native-policy-guard", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                self.guard.poll()
            except SafetyError:
                return

    def start(self) -> None:
        self.guard.poll()
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=1)
        if self._thread.is_alive():
            raise SafetyError("Watchdog callback did not return within its shutdown bound")


def guarded_runtime_type(native_runtime_type):
    """Build a staged subclass without importing SDKs or constructing devices.

    Uses audited private hooks from Studio c4ff730 / PhysicalAI 8e4021703.
    Only a disposable native-loop fixture has exercised this integration.
    """
    class GuardedRuntime(native_runtime_type):
        def __init__(self, *args, guard: NativePolicyGuard,
                     candidate_provenance: Callable[[], CandidateProvenance],
                     observation_clock_id: str, state_units: tuple[str, ...], **kwargs):
            if observation_clock_id != guard.scope.clock_id or state_units != JOINT_UNITS:
                raise SafetyError("Verified observation clock and state units are required")
            super().__init__(*args, **kwargs)
            self._guard = guard
            self._candidate_provenance = candidate_provenance
            self._observation_clock_id = observation_clock_id
            self._state_units = state_units
            self._guard_observation = None
            self._guard_run_lock = threading.Lock()
            self._guard_writer_thread = None

        def run(self, **kwargs):
            if not self._guard_run_lock.acquire(blocking=False):
                raise SafetyError("Another run owns the guarded writer")
            self._guard_writer_thread = threading.get_ident()
            try:
                self._guard.poll()
                return super().run(**kwargs)
            finally:
                self._guard_writer_thread = None
                self._guard_run_lock.release()

        def _read_observation(self):
            observation = super()._read_observation()
            self._guard_observation = observation
            return observation

        def _resilient_send(self, action):
            # This override executes after ALL native action-ready callbacks.
            # It intentionally replaces retry-on-send with a latched failure.
            try:
                if self._guard_writer_thread != threading.get_ident():
                    raise SafetyError("Caller does not own the guarded runtime writer")
                if self._guard_observation is None:
                    raise SafetyError("Current tick observation is unavailable")
                provenance = self._candidate_provenance()
                if not isinstance(provenance, CandidateProvenance):
                    raise SafetyError("Original policy candidate provenance is unavailable")
                robot, cameras = self._guard_observation
                # ndarray.tolist preserves invalid rank for the guard to reject.
                values = action.tolist() if hasattr(action, "tolist") else action
                state = robot.state.tolist() if hasattr(robot.state, "tolist") else robot.state
                sample = PolicySample(
                    tuple(self.robot.joint_names), self._state_units, state,
                    CaptureStamp(robot.timestamp, self._observation_clock_id),
                    {key: CaptureStamp(frame.timestamp, self._observation_clock_id)
                     for key, frame in cameras.items()}, provenance.warmup_ready,
                    provenance.generated, provenance.sequence, provenance.names, provenance.units,
                    provenance.model_input, provenance.chunk_id, provenance.action_index,
                )
                self._guard.send(values, sample, goal_time=self._goal_time,
                                 sole_writer=self.robot.send_action)
            except Exception as error:
                self._guard.reject(str(error))
                raise

    return GuardedRuntime
