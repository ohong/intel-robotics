"""Portable safety checks. These checks do not establish physical collision safety.

No robot SDK or motion transport is imported here. The live application has no
motion backend. Test controllers exercise only the supervision contract.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping


class SafetyError(ValueError):
    pass


def require_fresh(timestamp: float, now: float, max_age: float, label: str) -> None:
    if not all(math.isfinite(x) for x in (timestamp, now, max_age)) or max_age <= 0:
        raise SafetyError(f"{label}: invalid freshness configuration")
    age = now - timestamp
    if age < 0 or age > max_age:
        raise SafetyError(f"{label}: stale or future timestamp")


@dataclass(frozen=True)
class JointLimit:
    minimum: float
    maximum: float
    max_step: float
    unit: str

    def validate(self) -> None:
        if (not self.unit or not all(math.isfinite(x) for x in
                (self.minimum, self.maximum, self.max_step)) or
                self.minimum >= self.maximum or self.max_step <= 0):
            raise SafetyError("Explicit valid joint limits and units are required")


def validate_action(action: Mapping[str, float], state: Mapping[str, float],
                    limits: Mapping[str, JointLimit]) -> dict[str, float]:
    if not limits or set(action) != set(limits) or set(state) != set(limits):
        raise SafetyError("Action, state, and configured joint sets must match exactly")
    result = {}
    for name, limit in limits.items():
        limit.validate()
        target, position = action[name], state[name]
        if (isinstance(target, bool) or isinstance(position, bool) or
                not isinstance(target, (int, float)) or
                not isinstance(position, (int, float)) or
                not math.isfinite(target) or not math.isfinite(position)):
            raise SafetyError(f"{name}: action and state must be finite numbers")
        if not limit.minimum <= position <= limit.maximum:
            raise SafetyError(f"{name}: measured state outside configured bounds")
        if not limit.minimum <= target <= limit.maximum:
            raise SafetyError(f"{name}: action outside configured bounds")
        if abs(target - position) > limit.max_step:
            raise SafetyError(f"{name}: action exceeds configured step bound")
        result[name] = float(target)
    return result


@dataclass(frozen=True)
class RunPermit:
    session_id: str
    expires_at: float
    stop_procedure: str
    workspace: str


class SafetyGate:
    """One in-process owner; deployments must also enforce one process owner.

    A local watchdog can call check() while a supported backend executes. This
    generic gate does not create or certify a backend or physical stop procedure.
    """

    def __init__(self, limits: Mapping[str, JointLimit], max_age: float,
                 supervision_age: float, stop: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic):
        if not limits:
            raise SafetyError("Physical limits are not configured")
        for limit in limits.values():
            limit.validate()
        if any(not math.isfinite(x) or x <= 0 for x in (max_age, supervision_age)):
            raise SafetyError("Freshness bounds must be positive and finite")
        self.limits = dict(limits)
        self.max_age = max_age
        self.supervision_age = supervision_age
        self.stop = stop
        self.clock = clock
        self.permit: RunPermit | None = None
        self.last_supervision = float("-inf")
        self._owner = threading.Lock()
        self._stopped = True

    def authorize(self, permit: RunPermit, supervision_at: float) -> None:
        if self._owner.locked():
            raise SafetyError("Cannot change permit during a run")
        if (not permit.session_id or not permit.stop_procedure or not permit.workspace or
                not math.isfinite(permit.expires_at) or permit.expires_at <= self.clock()):
            raise SafetyError("A complete, unexpired supervised permit is required")
        require_fresh(supervision_at, self.clock(), self.supervision_age, "supervision")
        self.permit = permit
        self.last_supervision = supervision_at
        self._stopped = False

    def heartbeat(self, session_id: str) -> None:
        if self.permit is None or self.permit.session_id != session_id or self._stopped:
            raise SafetyError("No matching active supervised session")
        self.last_supervision = self.clock()

    def check(self) -> None:
        try:
            if self._stopped or self.permit is None:
                raise SafetyError("DISARMED")
            if self.clock() >= self.permit.expires_at:
                raise SafetyError("Run authorization expired")
            require_fresh(self.last_supervision, self.clock(), self.supervision_age,
                          "supervision")
        except SafetyError as error:
            self.abort(str(error))
            raise

    def abort(self, reason: str) -> None:
        # Latch before invoking the backend, even when its stop function fails.
        already_stopped = self._stopped
        self._stopped = True
        self.permit = None
        if not already_stopped:
            self.stop(reason)

    def begin(self, action: Mapping[str, float], state: Mapping[str, float],
              observation_at: float, state_at: float) -> dict[str, float]:
        if not self._owner.acquire(blocking=False):
            raise SafetyError("Another run already owns the controller")
        try:
            self.check()
            require_fresh(observation_at, self.clock(), self.max_age, "observation")
            require_fresh(state_at, self.clock(), self.max_age, "robot state")
            return validate_action(action, state, self.limits)
        except Exception:
            try:
                self.abort("Rejected action or unavailable state")
            finally:
                self._owner.release()
            raise

    def end(self) -> None:
        if self._owner.locked():
            self._owner.release()


class ReviewBudget:
    """A bounded retry decision, never an autonomous success claim."""

    def __init__(self, max_retries: int):
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise SafetyError("Retry budget must be a nonnegative integer")
        self.remaining = max_retries
        self.closed = False

    def assess(self, outcome: str) -> str:
        if self.closed:
            return "REVIEW"
        if outcome == "verified_success":
            self.closed = True
            return "VERIFIED"
        if outcome not in ("ambiguous", "failed"):
            self.closed = True
            return "REVIEW"
        if self.remaining == 0:
            self.closed = True
            return "REVIEW"
        self.remaining -= 1
        return "RETRY"
