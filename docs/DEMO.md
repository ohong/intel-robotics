# Demonstration notes

## Current honest demonstration

This shows the deployed inspection pilot. At 17:59:59 PDT, camera was LIVE and detector INVALID, with no inference. The real overlay was last inspected at 17:43 PDT; present current state accurately. The autonomous challenge is incomplete.

1. Open [Second Look](http://127.0.0.1:8088).
2. Identify release `a9ed13af508b6165`, Studio low camera 1, and the real shared RGB feed.
3. Show the score and anomaly overlay when the frozen crop accepts the scene. Show INVALID when multiple comparable components prevent inference.
4. Explain the genuine H100 fit and Intel OpenVINO GPU.0 FP32 deployment. All 28 validation conversion comparisons passed.
5. Show the validation table in [BUILD_EVIDENCE.md](BUILD_EVIDENCE.md): one defective view was missed and four views were UNKNOWN.
6. Show live policy BLOCKED and controller DISARMED/backend UNAVAILABLE, with zero physical trials.
7. Present the separate [SmolVLA recorded replay](POLICY.md) as engineering evidence. State and normalizers are synthetic, action semantics are unverified, and it is not connected to live control.

The model uses one-session repeated views. There is no unseen final test, calibrated defect probability, or autonomous outcome. A valid pixel crop does not verify physical presence, occlusion, or defect visibility. The current pilot model-only medians are 106.14 ms on Torch CPU, 63.98 ms on OpenVINO CPU, and 14.68 ms on OpenVINO GPU FP32. See the benchmark method and limits in [BUILD_EVIDENCE.md](BUILD_EVIDENCE.md). Earlier CPU/GPU/NPU smoke measurements use a different model.

## Prepared physical session

The **Demonstrations** task owns the existing NEED HANDS request and all robot commands. Use that task for this session; do not create a second controller. After the user reported camera 3 disconnected and requested recording stop, Demonstrations verified the software was already STOPPED at 17:56:16 PDT. Runtime metadata was null, no StudioSession or robot owner remained, and ttyACM0/1 had no owners. Logs show last-subscriber loss at 17:52:27 triggered hold/finalize; owners exited idle at 17:53:04. No agent stop, reconnect, or robot command was sent. Physical pose and torque remain UNKNOWN; this does not authorize physical changes.

> **NEED HANDS — about 10 minutes after the approved safe pause**
> **Do:** With Demonstrations, confirm the arm's physical state and reachable approved stop procedure. Identify the allowed workspace, limits, defect criteria, and two destination piles. Confirm camera 3 and place one identified block inside the ROI, with every other block outside it. Supply defect/bin ground truth and unseen specimens, then supervise only the bounded authorized session.
> **Done when:** Demonstrations confirms the pause and physical scope, records one bounded authorization, and the labeled unseen specimens are ready.
> **Next:** The agent captures frozen-model evaluation, records the prepared demonstrations, checks data, and completes software work within the approved scope. Failures and interventions remain in the record.

Do not auto-start recording or control. Do not move objects or change the physical setup before the approved safe pause. Do not reconnect, home, or release torque to infer a safe state. Reconfirm authorization if workspace, scope, or supervision changes.

## Still required for a judged physical result

- An unseen-specimen evaluation with the frozen detector and honest misses/abstentions.
- Task-compatible VLA integration with verified camera mapping, state/action units, normalizers, and physical action semantics.
- One validated command owner, local stop behavior, and bounded recovery under the approved workspace and supervision.
- Real successful and failed trials with object/destination evidence, cycle time, and intervention counts.

Five closed episodes, totaling 3249 frames, are copied to immutable Intel snapshot `/home/ird-demo/second-look/recording-snapshots/pilot-five-20260916T004831Z/dataset`. The adjacent snapshot receipt verifies source hashes before/after copying and copied hashes; no writable file descriptors remained. Structural/video checks passed for both native camera streams: finite six-value state/actions, valid episode boundaries, 30 fps, all 3249 frames per camera, and no out-of-range values, black frames, or long freezes. One adjacent low-camera frame repeats in episode 0. The best state/action mean absolute error occurs at a four-frame lag; this diagnostic does not change the data or prove synchronized capture.

**Detector-conditioned sorting training is BLOCKED.** The frozen detector annotations cover all 3249 frames: 2614 are INVALID with no inference; all 635 actual predictions are ANOMALOUS across both caption classes. There are zero verified target-associated starts. Every row has `target_association=UNKNOWN` and `conditioning_verified=false`; the crop often sees a pile or remnant rather than the picked block. Finite scores and candidate eligibility do not authorize training.

H100 is authorized only for a bounded genuine SmolVLA **original-caption** real-data software/export probe: start with one CUDA step, at most 20 updates. Tag every result `REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE`; task/controller readiness stays false. Original captions are actual conditioning inputs in this separate probe. It does not establish integrated Anomalib–VLA sorting or authorize deployment.

Train-only statistics and the exploratory manifest are complete for train episodes 0/1/3 (1829 rows), with validation 2/4 and no final test. The real native ACT loader passed a small-profile, two-camera smoke: one CPU update, 81 finite gradient tensors, gradient norm 214.2813, output shape `[2,4,6]`, zero reload error, and zero caption-mutation error. ACT ignores text. This is real-data software evidence only; task/controller readiness remains false.

The earlier synthetic ACT qualification used no real task episodes. Its artifact remains `UNSAFE_NOT_TASK_TRAINED`. The completed archive is a frozen detector checkpoint; it excludes these five episodes and VLA weights. The agent owns digital work; the operator supplies physical facts, actions, and supervision.
