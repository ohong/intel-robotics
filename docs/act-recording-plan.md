# ACT recording, environment setup, and evaluation

## Scope and authority

This is a task-specific addendum to the existing Physical AI Challenge repository, not a replacement kickoff. Read `AGENTS.md`, `PROJECT_SPEC.md`, `docs/CHALLENGE_DAY.md`, `docs/REMOTE_ENVIRONMENT.md`, and current agent state. Preserve ongoing application work and coordinate with the H100-readiness thread.

The supplied **Physical AI Challenge Technical Brief**, pages 1–3, requires integrated SO-101 execution, Anomalib defect detection, Studio/VLA reasoning, and optimized Intel deployment. Page 4 rewards measured integration and reliability. It does not prescribe ACT, an episode count, or a benchmark that must precede training. The task-specific objects, defect definition, and operating conditions are absent from that general brief. Retrieve the actual event rules before recording task demonstrations; ask for a photo or concise clarification only when authoritative material is inaccessible digitally.

Everything below is an implementation recommendation. Live runtime inspection determines actual configuration. No connection, sensor, recording, checkpoint, or training test has been executed by the author of this document.

## Outcome and ownership

Prepare a working recording environment, collect and validate demonstrations through short physical requests, establish a small evaluation protocol, and hand a frozen dataset plus reproducible configuration to the training owner. Continue through digital checks and repairs rather than returning software homework.

The operator supplies physical setup, object placement, leader-arm demonstrations, and unavoidable access/approval. Codex owns discovery, Studio setup, recorder controls where tools support them, data validation, manifests, splits, transfers, training integration, results, and debugging. Do not require the operator to annotate frames, select camera indices, tune hyperparameters, or interpret tracebacks. When local recording controls are inaccessible through tools, prepare the exact minimal click/key action instead of pretending they were operated.

Keep robot motion disarmed during discovery. One agent/process owns robot commands and recording resources. Obtain one bounded supervised teleoperation authorization after confirming the physical workspace and stop procedure; do not ask for approval of every in-scope command. Scope changes and autonomous-policy trials need their own appropriate authorization. Never assume SSH disconnect or a recording keyboard shortcut constitutes an emergency stop.

## Known event context, not a new installation recipe

- Existing SSH alias: `intel-robot`, if configured. Earlier verified connection: `ird-demo@10.36.254.246:22`, hostname `NUC16GDKX76`. Inspect newer runtime notes before using an old DHCP address. Do not SSH to yourself from a remote Codex session.
- Event sheet names Conda environments `hack_lerobot` and `hack_physical_ai`; discover interpreter paths and Studio's actual process environment.
- Event sheet names calibration IDs `hack_follower` and `hack_leader`. They are not necessarily Studio display names or serial-port names.
- Preserve installed packages, working calibration, motor configuration, drivers, and existing processes. Do not merge Conda environments or rerun generic stack installation.
- Intel's native Studio documentation gives `http://localhost:3000`; inspect actual listeners and reuse the running service. Verify backend communication as well as page loading. [S1]
- H100 access is handled by the parallel readiness thread; use its verified transfer method and an owned directory under `/workspace`. Never copy private keys or restart/reconfigure the node.

## Gate 1 — minimal evaluation specification, in parallel with setup

Spend approximately 10 minutes defining a task-specific test protocol, not building benchmark infrastructure. Save it in the existing evaluation document or `docs/EVAL_PROTOCOL.md`.

Record the permitted initial state, what observation should produce which response, measurable completion conditions, timeout, failure categories, and permitted recovery. A proposed pick-and-place success condition is: the object is released entirely in the designated destination, remains there for two seconds, and the arm clears it, within a timeout chosen before scoring, without a human rescue. Adapt this to the real rules; it is not an organizer requirement.

For a normal/defective sorting task, a small starting matrix is 2 object conditions × 3 permitted presentation positions × 2 orientations = 12 trial setups. This is a recommended pilot size, not statistical proof or a prescribed benchmark. Use different actual arrangements from the demonstrations and reserve specimens where available. Do not fabricate a second orientation or defect category when the task does not allow one.

Maintain three distinct roles:

1. Training episodes fit the policy and training-only normalization statistics.
2. Whole held-out validation episodes and development rollouts guide checkpoint selection and repair.
3. Fresh final physical trials measure the frozen system. Do not tune on these and still call them untouched test evidence.

Split by complete episode, and by specimen/session where relevant; never put adjacent frames from one trajectory into train and validation. Preserve frame provenance when extracting images for Anomalib. New demonstrations of the same physical specimen are not evidence of generalization to unseen specimens.

Record completion count/attempts, correct response, grasp and placement outcomes, intervention count, drops/contacts, cycle time, model/decision latency, and defect false accepts/false rejects. Include unsuccessful and aborted attempted trials under a predefined counting rule. Offline action loss is a diagnostic, not proof of autonomous task completion.

Use a compatible, already-safe checkpoint as a baseline only if one exists. Never command an untrained or unknown policy merely to obtain a baseline. Teleoperation is a collection/control check, not autonomous performance. Standardized simulator benchmarks may serve other purposes, but are not a prerequisite for this task-specific ACT run. [S8]

## Gate 2 — physical and software recording environment

Prepare the software first, then issue one compact physical-setup request. Ask the operator to clear the follower workspace of laptops, food, loose cables, and other obstacles; secure the supplied bases using the organizer-approved setup; position allowed objects/destinations; and confirm the stop procedure. Do not assume clamping, electrical safety, camera visibility, or stop behavior from the photographs.

Discover actual cameras. The photographs do not establish that a suitable recording camera is attached. If none is usable, prepare everything else and request mounting/connecting the supplied camera. A still phone photograph or unsynchronized phone video is not a complete ACT demonstration.

Reuse or create an environment such as `second-look-v1` in Physical AI Studio:

| Setting | Required configuration |
|---|---|
| Follower | Correct SO-101, resolved device mapping, supplied `hack_follower` calibration where compatible |
| Teleoperator | Correct leader, resolved device mapping, supplied `hack_leader` calibration where compatible |
| Camera(s) | Real live views covering relevant gripper contact, object, and destination; stable logical feature names |
| Capture | Match the actual checkpoint requirements and sustainable sensor/recorder rate |
| Scene | Repeatable, permitted initial state and fixed camera/base arrangement |

The documented Studio path is Robots, Cameras, then Environments; an environment binds follower, leader, and cameras. A common documented capture setting is 640×480 at 30 FPS, not a mandate. Do not degrade small-defect imagery just to match a policy image size; evaluate defect visibility independently. [S2]

Inspect existing device settings and scripts before invoking discovery helpers. LeRobot provides `lerobot-find-cameras`, but camera identifiers can change after reconnects. Avoid competing camera opens and robot connections. Do not guess `/dev/ttyACM0` or camera index 0. Verify each actual view and device mapping. [S4]

Record installed versions, camera keys, frame dimensions, observed frame rate, color order, joint ordering/units, gripper convention, commanded-action representation, robot IDs, calibration provenance, preprocessing, and dataset format in a capture manifest. Freeze this observation/action contract across capture, H100 training, and Intel deployment. Match existing configurations rather than imposing arbitrary replacement names.

## Gate 3 — two real smoke episodes before bulk collection

Prepare a local dataset with uploads disabled by default. Reuse the installed Studio recorder rather than building a new recording UI. If Studio cannot record correctly, use the installed LeRobot recorder as a bounded fallback and prove its export is compatible with the application's loader.

In the documented Studio workflow: create/select dataset and environment; choose Add episode; verify live views; then start, perform, and accept/discard each demonstration. Entering recording/teleoperation can enable following; do not enter it before physical readiness. UI video review is distinct from commanding `lerobot-replay`, which physically moves the robot. [S3, S5]

For one allowed manipulation behavior, record two short but complete successful demonstrations. Use an agreed start state; perform a controlled approach, grasp, lift/transfer, release, and task-appropriate clearance; stop before the manual scene reset. Do not record long waits or hands repositioning the object as part of the successful behavior. Preserve failed attempts separately with honest labels rather than quietly including them in the successful imitation set.

Validate both episodes before asking for many more:

- Decode and inspect all camera streams across each episode, including the grasp and final state.
- Confirm nonempty, time-aligned observations, measured follower state, recorded action targets, episode boundaries, and gripper values with plausible variation.
- Check for freezes, severe timestamp gaps, missing frames, NaNs, unit/order mismatches, and truncated output.
- Load a minibatch using the exact intended ACT trainer, exercise forward/backward computation, and save/reload a smoke checkpoint. This is a software check, not a learned-policy success claim.
- Validate any adapter/export path used between Studio, the H100 trainer, and Intel inference. The shared model name ACT does not guarantee identical checkpoint formats across implementations.

LeRobot datasets contain videos plus state/action tables and indexing/configuration metadata, not just MP4s. Current Studio documents LeRobot v3 export, but inspect the installed version. Transfer a consistent finalized export with all its metadata; never copy only videos or train from files the recorder is still writing. [S3, S6]

## Gate 4 — collect useful training data

Start with one unambiguous, repeatable manipulation skill. If the task has distinct accept/reject behaviors, preserve skill labels and either separate the datasets/policies or verify explicit goal conditioning in the installed implementation. Do not mix indistinguishable observations with contradictory target actions and expect task text to fix them.

Standard ACT consumes visual/robot-state inputs and predicts action chunks; adding a natural-language dataset caption does not itself make the standard policy language-conditioned. ACT can be the movement component, but it does not by itself establish the brief's VLA reasoning requirement. Preserve the genuine Studio/VLA integration and document how perception-dependent action selection reaches the controller. [S7, S9; brief p.3]

Data plan, adapted to remaining time:

- Two successful smoke episodes establish record/load compatibility, not performance.
- An initial 10–20 clean training episodes may support an early pipeline/checkpoint experiment while collection continues; they are not a promise of reliable manipulation.
- Aim toward roughly 50 successful demonstrations for the first constrained skill, plus a separate held-out validation set. Studio and LeRobot recommend about 50 as a starting point; difficulty and variation determine actual needs. [S3, S5]

Initially vary only allowed object positions/orientations modestly, keeping the camera and base arrangement fixed. Use smooth consistent demonstrations and show the entire grasp and release. Expand variation in response to observed failures rather than collecting unrelated motions. Coordinate with the main agent before collecting a second skill or changing the scene.

Have the agent track coverage and issue brief requests, e.g.:

> NEED HANDS — place the sample at the highlighted start position and step clear. When the recording indicator is ready, demonstrate the approved transfer with the leader. I will record, inspect, accept/reject, and queue the next example.

The operator may need to identify the physical specimen's authoritative condition when the agent cannot infer ground truth independently. Never use the detector's own prediction as the evaluation label.

## Separate image samples for Anomalib

Maintain a separate image-data view alongside the motion episodes. With a normal-only baseline such as PatchCore, fit the normal feature bank on known-good training objects. Reserve normal and defective observations for threshold selection and independent evaluation; do not feed defective examples into the normal reference set. [S10]

The agent can extract suitable inspection frames from finalized episodes or capture stills through the real sensor. Ensure crops, lighting, pose distribution, and defect visibility match intended inspection. Preserve object/session/episode provenance, deduplicate near-identical frames, and keep data splits consistent across derived crops. A long video of one object is not many independent specimens. Request pixel masks only when localization evaluation requires them and credible annotation is available.

## Gate 5 — training handoff and first evaluation

Coordinate with the H100-readiness owner. Supply a finalized, versioned dataset snapshot, hashes, split manifest, training-only preprocessing/normalization, feature schema, selected trainer version, and exact output artifact requirements. Keep new recordings separate until the next explicit snapshot. Use a supported export/snapshot boundary, not a live copy of buffered video/Parquet files. [S6]

Resolve whether this is actual fine-tuning from a compatible ACT policy checkpoint or new task-policy training. A pretrained image backbone alone does not constitute an existing task-trained ACT policy. Record checkpoint provenance and do not assume an arbitrary downloaded ACT matches this arm/camera/action setup. [S7, S9]

Prefer the existing Studio-compatible trainer/export route if it works. Do not independently switch frameworks while the main app targets another loader. Studio supports ACT and exposes model-training controls, but clicking a local training button is not evidence that the job runs on the remote H100. Verify the actual execution host. [S8]

Save early checkpoints, inspect held-out loss and predicted action ranges, and verify native/exported inference compatibility with actuators disabled. Then coordinate one bounded supervised physical trial. Collect corrective demonstrations for the dominant observed failure, retrain, and rerun development cases. Keep final scored trials separate.

Do not block first training on completion of benchmark software. Do not block data collection on unrelated dashboard work. The first useful milestone is two verified real episodes; the next is a compatible trained artifact and a measured supervised physical trial.

## Deliverables and completion report

Produce or update the existing equivalents of:

- `docs/REMOTE_ENVIRONMENT.md`: actual environment/robot/camera mapping and verified recorder controls.
- `docs/EVAL_PROTOCOL.md`: success, timeout, trial design, intervention/failure definitions, split policy.
- `docs/DATASET_STATUS.md`: accepted counts by skill and split, defects/coverage, snapshot IDs, verification results, remaining gaps.
- Capture/training configuration and validation tools in the main agent's agreed paths; no duplicate application subsystem.

Keep raw footage, credentials, large datasets, and checkpoint binaries out of Git unless explicitly authorized; commit manifests and reproducible configuration. No automatic public Hub upload. Report what passed, what remains unverified, the current physical readiness state, and one concrete physical action needed next. Do not invent successful tests or keep the robot armed while waiting indefinitely.

## Primary references

The links below describe upstream behavior, not proof of the installed versions. Verify commands and UI labels against the event image.

- **[S1] Intel hackathon resources:** native Studio launch and runtime context. `https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html`
- **[S2] Studio environment setup:** robot/camera/environment configuration. `https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/04-environment-setup.md`
- **[S3] Studio recording datasets:** recording, acceptance, review, export. `https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/05-recording-datasets.md`
- **[S4] LeRobot cameras:** discovery and frame access. `https://huggingface.co/docs/lerobot/en/cameras`
- **[S5] LeRobot real-world imitation learning:** calibration IDs, collection, local storage, policy evaluation. `https://huggingface.co/docs/lerobot/en/il_robots`
- **[S6] LeRobotDataset v3:** multimodal schema, metadata, finalization. `https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3`
- **[S7] LeRobot ACT:** inputs, training, policy overview. `https://huggingface.co/docs/lerobot/en/act`
- **[S8] Studio training / PhysicalAI library:** training and deployment interfaces. `https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/06-training-policies.md` and `https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/README.md`
- **[S9] LeRobot ACT implementation:** actual inputs, action queue, training loss. `https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py`
- **[S10] Anomalib PatchCore:** normal feature-bank fitting and anomaly outputs. `https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html`
