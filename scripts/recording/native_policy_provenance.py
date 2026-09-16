"""Isolated synchronous/replacement-only native PolicySource prototype.

No device constructors, runtime connections, or installed-source edits. Explicit
native SyncExecution and ReplaceSmoother instances are required; defaults are
never silently replaced. Async/RTC/blended provenance needs a broader API change.
"""
from __future__ import annotations

from collections import deque
import time
import uuid

from secondlook.safety import SafetyError
from scripts.recording.native_policy_guard import (
    CandidateProvenance, CaptureStamp, JOINT_NAMES, JOINT_UNITS, ModelInputIdentity,
)

_IDENTITY_KEY = "__secondlook_original_model_input__"


def make_sync_provenance_source(*, model, execution, smoother, clock_id: str,
                                camera_keys: tuple[str, ...], expected_chunk_length: int,
                                action_names: tuple[str, ...], action_units: tuple[str, ...],
                                clock=time.monotonic):
    """Use the installed source/execution, retaining provenance beside queue data.

    The trusted caller verifies the host monotonic clock and observation schema.
    This prototype intentionally rejects embedded robot camera inputs, which need
    an explicit camera mapping. The returned source owns no hardware.
    """
    import numpy as np
    from physicalai.runtime import PolicySource, SyncExecution
    from physicalai.runtime.execution.queue import ChunkedActionQueue
    from physicalai.runtime.smoothers import ReplaceSmoother

    if type(execution) is not SyncExecution or type(smoother) is not ReplaceSmoother:
        raise SafetyError("Explicit native SyncExecution and ReplaceSmoother are required")
    if not clock_id or not camera_keys or len(set(camera_keys)) != len(camera_keys):
        raise SafetyError("Verified clock and exact unique camera keys are required")
    if type(expected_chunk_length) is not int or expected_chunk_length <= 0:
        raise SafetyError("Explicit artifact-verified chunk length is required")
    if action_names != JOINT_NAMES or action_units != JOINT_UNITS:
        raise SafetyError("Artifact action names and units must match the exact guard contract")

    class ProvenanceModel:
        def __init__(self):
            self.pending = None
            self.chunk = 0

        def reset(self):
            self.pending = None
            model.reset()

        def predict_action_chunk(self, observation):
            if self.pending is not None:
                raise SafetyError("Previous generated chunk was not consumed")
            identity = observation.get(_IDENTITY_KEY)
            if not isinstance(identity, ModelInputIdentity):
                raise SafetyError("Original model-input provenance is missing")
            # Metadata never enters model token/image preprocessing.
            inputs = {key: value for key, value in observation.items() if key != _IDENTITY_KEY}
            result = model.predict_action_chunk(inputs)
            if not isinstance(result, np.ndarray) or result.shape != (expected_chunk_length, 6):
                raise SafetyError("Returned chunk shape differs from the verified artifact contract")
            if result.dtype.kind not in "fi" or not np.isfinite(result).all():
                raise SafetyError("Generated action chunk must contain only finite numeric values")
            stamp = CaptureStamp(clock(), clock_id)
            self.pending = (result, self.chunk, stamp, identity)
            self.chunk += 1
            return result

    wrapped_model = ProvenanceModel()

    class ProvenanceQueue(ChunkedActionQueue):
        def __init__(self):
            super().__init__(smoother=smoother)
            self._metadata = deque()
            self.last_provenance = None
            self._next_sequence = 0

        def push_chunk(self, chunk, offset=0):
            # SyncExecution passes the exact freshly returned array. Reject
            # copies, replayed arrays, and async offsets instead of guessing.
            pending = wrapped_model.pending
            if offset != 0 or pending is None or pending[0] is not chunk:
                raise SafetyError("Unpaired, replayed, or offset chunk is forbidden")
            _, chunk_id, generated, identity = pending
            with self._lock:
                self._deque.clear()
                self._metadata.clear()
                for index, action in enumerate(chunk):
                    self._deque.append(action.copy())
                    self._metadata.append(CandidateProvenance(
                        generated, self._next_sequence, action_names, action_units, True,
                        identity, chunk_id, index,
                    ))
                    self._next_sequence += 1
                wrapped_model.pending = None

        def pop(self):
            with self._lock:
                if not self._deque:
                    self._consecutive_holds += 1
                    self._total_holds += 1
                    # Keep previous provenance so native cached-last fallback
                    # cannot pretend to be a fresh action at the send guard.
                    return None
                self._consecutive_holds = 0
                self._total_pops += 1
                self.last_provenance = self._metadata.popleft()
                return self._deque.popleft()

        def clear(self):
            with self._lock:
                self._deque.clear()
                self._metadata.clear()
                self.last_provenance = None
                self._consecutive_holds = 0

        def reset(self):
            self.clear()
            with self._lock:
                self._total_holds = 0
                self._total_pops = 0

    queue = ProvenanceQueue()

    class ProvenanceSource(PolicySource):
        @property
        def candidate_provenance(self):
            return queue.last_provenance

        def _to_model_input(self, robot_obs, camera_frames):
            if getattr(robot_obs, "images", None) or set(camera_frames) != set(camera_keys):
                raise SafetyError("Explicit external-camera mapping is required")
            values = super()._to_model_input(robot_obs, camera_frames)
            values[_IDENTITY_KEY] = ModelInputIdentity(
                uuid.uuid4().hex, CaptureStamp(robot_obs.timestamp, clock_id),
                tuple((key, CaptureStamp(camera_frames[key].timestamp, clock_id)) for key in camera_keys),
            )
            return values

    return ProvenanceSource(model=wrapped_model, execution=execution, action_queue=queue)
