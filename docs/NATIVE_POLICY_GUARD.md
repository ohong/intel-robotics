# Native policy guard — staged, hardware-free

**Status: DIGITAL/MOCK VERIFIED against the installed native loop. Not deployed or armed.**
This work does not block demonstration recording. It changes no services,
installed packages, calibration, torque, or robot ownership.

## Installed source audit

Read through `intel-robot` on September 15, 2026. Library base:
`/home/ird-demo/physical-ai-studio/application/backend/.venv/lib/python3.13/site-packages/physicalai`.

- `runtime/core.py:390–417`: observation → policy update → action callbacks → send → tick telemetry.
- `runtime/_callback_bus.py:72–102`: action-ready exceptions propagate and prevent the current send.
  Callbacks run in order; a later callback can change a previously checked action.
  Tick callbacks run after sending, and their errors are logged rather than stopping the run.
- `runtime/action_sources/policy.py:147–202`: warmup happens within update. When its queue
  is empty, the policy returns its last action. Fresh images do not prove a fresh policy result.
- `robot/transport/_shared_robot.py:580–640`: state reads can return a cached observation.
  Sending publishes an absolute target with `goal_time`; disconnect does not stop motors.
- `robot/so101/so101.py:79–95,509`: state capture timestamps use `time.monotonic()`.
  UVC `_omnicamera.py:354,380,389` and RealSense `_camera.py:276` also use host
  `time.monotonic()`. These are capture/read-completion times, not sensor exposure
  timestamps. `capture/transport/_header.py:105,197` preserves seconds through
  integer nanoseconds. Live boot/source identity still needs verification.

## Hook choice

Two approaches were considered:

1. Add an action-ready callback last. This is small, but it has only action/step inputs.
   It cannot independently establish the current observation, queue provenance, or actual
   outgoing `goal_time`. Callback ordering also remains a bypass risk.
2. Guard the existing writer immediately before sending, after every action transform.
   This needs a narrow runtime integration but can validate the exact outgoing command.

The staged adapter chooses option 2. `NativePolicyGuard.send` accepts one bounded,
existing-owner send callable. It opens no transport and creates no second writer.
`guarded_runtime_type` subclasses the installed loop and intercepts `_resilient_send`.
It captures the same tick's observations through `_read_observation`, then checks
the final transformed command. These private hooks are version-specific.
No runtime monkey-patch or automatic installation is provided. Only the fixture
constructs this subclass, with fake devices. Native `connect()` remains capable
of energizing real devices; this send guard does not make connection safe.
`run()` checks the guard before source initialization/update. The caller must still
validate full authorization **before** native `connect()` or Studio session preconnect.
Source objects, callbacks, and mutable observation buffers are trusted in this fixture;
it does not establish isolation against code that mutates captures or bypasses the seam.

```text
same-tick captures + policy candidate provenance
                         |
policy → all transforms → guard → existing sole writer
                          ^
             local watchdog / supervision heartbeat
```

## Enforced contract

The adapter reuses `secondlook.safety` for finite values, joint bounds, state-to-target
step limits, authorization expiry, and heartbeat age. It additionally requires:

- Exactly six scalar values, with independently declared state and action order:
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- Explicit units: `normalized_-100_100` for the first five and `normalized_0_100`
  for the gripper. These are adapter contract labels. Representation ranges are
  **not** approved motion limits. All six tighter physical bounds and step limits
  must be supplied; none are invented here.
- A finite target within bounds, bounded change from measured state, and bounded
  change from the previous accepted target. No clipping, retries, homing, or torque release.
- Exact camera keys; original robot/camera capture timestamps that advance on each
  accepted command; bounded age and sensor skew; explicit warmup readiness.
- Original candidate generation time and an increasing candidate sequence. Different
  actions in one inference chunk may share generation time, but each needs its own
  sequence. Repeating the queue's cached last action is rejected. Do not relabel a
  cached output as newly generated just because a runtime tick occurred.
- Immutable original model-input identity and capture timestamps, plus chunk ID and
  action index. A recently generated result from an old model input is rejected.
  Reusing a chunk action or rewriting its original metadata is rejected.
- Explicit command `goal_time` bounds, total goal timeout, permit expiry, local
  supervision heartbeat timeout, and the exact approved hold procedure.

All timestamps and the guard's clock must use the same Intel boot's monotonic clock.
Identify that clock in `clock_id`. Never pass Unix timestamps from the JSON policy
bridge into this guard without an independently verified conversion and provenance.
Missing capture timestamps or queue provenance means no authorization for live use.

## Faults and integration limits

Faults latch before callbacks run. Cancellation and the approved hold request are
each attempted once, even if cancellation raises. Callback errors remain recorded;
an attempted hold is **not** evidence that the arm holds. Subsequent policy sends
are rejected. A new guard and new supervised scope are needed after a fault.

Callbacks must be bounded, nonblocking requests to the **existing** controller owner.
They must not create another writer, disconnect hardware, release torque, or home.
Stock `PolicySource.disconnect()` can wait for an execution lock; do not use it as
an assumed bounded watchdog callback. The approved physical response is still unknown.

A local watchdog must call `poll()` independently of inference. Heartbeats must come
from the approved local supervision mechanism, not an SSH keepalive or browser traffic.
`LocalGuardWatchdog` implements independent polling; the caller explicitly starts it.
It does not generate heartbeats or establish the supervision source. The polling
period must be at most one quarter of the heartbeat timeout. Python scheduling is
not a hard real-time guarantee.
It serializes checking and publication; the send callable must be bounded and
nonblocking. A hung native call, Python process failure, or missed watchdog scheduling
requires an independently verified local controller stop mechanism. This module does
not provide or certify one. Returning from `send()` proves publication only, not arrival.

Before production integration, the owner must provide real limits/authorization,
camera clock proof, candidate queue metadata, warmup evidence, the local watchdog,
and tested approved fault callbacks. The runtime must supply the exact current-tick
captures and put this guard after all transforms, covering every policy send attempt.
It must preserve the single runtime session and prevent other clients from writing.
Strictly advancing camera timestamps can reject a control tick faster than capture;
coordinate the policy send cadence instead of weakening the freshness check silently.

### Exact remaining native API gaps

- `ActionQueue.pop()` returns only an array. `push_chunk()` and its smoother retain
  arrays, not generation times or contributing observation identities. `PolicySource`
  returns its cached last array on queue underrun. Queue counters cannot reconstruct
  original inference provenance, especially after smoothing. The subclass requires
  `CandidateProvenance` from the actual source and rejects `None`. The fixture supplies
  synthetic provenance only. No tick timestamp is substituted for real inference data.
- Warmup state is a private source flag. A live adapter must expose and verify it;
  inferring readiness from a connected socket is insufficient.
- Native `RobotRuntime.stop()` allows the current tick to finish and send. The fixture
  therefore latches the pre-send guard independently; stopping the native loop alone
  is not the required cancellation mechanism.
- The subclass rejects a second run and calls into its guarded send seam from another
  thread. It does **not** prevent direct access to the robot object or another process.
  Backend `SessionNameLock` serializes runtime ownership, but WebSocket clients can
  submit control commands without a per-client command lease. The low-level owner
  accepts action-channel samples without a sender capability check. One hardware owner
  is not exclusive authorization of command producers. Closing these bypasses is a
  production integration blocker; the fixture makes no global exclusivity claim.

## Verification

`python3 -m unittest discover -s tests -p test_native_policy_guard.py -v`

**11 mock tests passed.** They cover default disarm, missing scope, shape/order/units,
nonfinite values, state and target bounds, target deltas, stale/future/mixed clocks,
capture skew, repeated captures, cached candidates, warmup, command timing, total
timeout, supervision loss, non-revival, failing cancellation, and send failures without
retry. All limits and observations are synthetic fixtures. No live policy or robot
was executed, and no physical stop success is claimed.

### Installed native-loop fixture

Fixture source: `scripts/recording/test_native_runtime_fixture.py`.
Disposable staging directory on Intel:
`/home/ird-demo/second-look/recording-tools/native-guard-fixture`.

Executed with the existing Studio `.venv/bin/python`:
`python -m scripts.recording.test_native_runtime_fixture`.
**6 tests passed** after the provenance extension. The real installed `RobotRuntime.run()` drove
fake robot, camera, and action-source objects. No physical device class was constructed.

Verified: one valid fake publication; a final callback's NaN never reaches fake send;
missing authorization, limits, or original candidate metadata reject; a local watchdog
requests fixture cancellation and fixture hold while inference blocks, and the released
inference result never publishes; a second run/thread cannot use the guarded send seam.
The watchdog test asserts completion under one second with a 100 ms synthetic heartbeat
timeout and 10 ms poll period. This is bounded software test evidence, not measured
robot stop latency. No runtime fixture process remains running.

### Bounded provenance prototype

`scripts/recording/native_policy_provenance.py` adds an isolated native source/model/
queue adapter. Its constructor requires explicit native `SyncExecution` and
`ReplaceSmoother` objects. Async, RTC, and Lerp instances are rejected. The installed
defaults remain unchanged. Supporting blended or asynchronous results still requires
a broader execution/queue/smoother provenance API; this prototype does not claim that.

The artifact owner must supply verified action order, units, camera keys, clock
identity, and expected chunk length. Output shape and finite values are checked before
enqueue. Actual array rows determine chunk size; the legacy model getter is not trusted.
The synthetic fixture intentionally reports getter length 1 while returning 3 rows.
This reflects the reported compatibility risk, not a task-policy validation claim.

The source records an immutable ID and original sensor timestamps with each model
input. The model wrapper removes this metadata before inference, then pairs the exact
returned array with completion time and chunk ID. The replacement queue copies each
action with immutable metadata and its index/sequence. Dequeue never assigns a new
generation or capture timestamp. Re-enqueueing an unpaired/replayed array is rejected.
An empty queue retains the prior metadata, allowing the final guard to reject native
cached-last fallback. Fresh current observations cannot refresh an old action chunk.

The native fake-policy path now runs real installed `PolicySource`, `SyncExecution`,
and `RobotRuntime` together. Its three fake publications preserve one chunk's original
input identity and generation time through the final-send guard. All model outputs,
joint limits, and physical acknowledgments remain synthetic. The wrapper trusts the
selected model to return newly computed output; it cannot detect undisclosed caching
inside model code. Live model semantics and the approved artifact remain separate gates.

Verification command in the same disposable staging directory:
`python -m scripts.recording.test_native_policy_provenance`.
**10 tests passed**, covering exact constructor/shape contracts, immutable metadata,
replacement chunks, nonfinite outputs, array replay, stale generated actions, fresh
generation from old inputs, reused actions, and the complete native fake-policy loop.
The earlier 6 native-loop and 11 local guard tests also pass. No installed files were
edited, no model weights were loaded, and no controller/device was attached.
