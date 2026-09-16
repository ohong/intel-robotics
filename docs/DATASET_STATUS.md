# Recording dataset status

**FIVE REAL EPISODES — structural/video and native ACT loader checks passed; detector-conditioned sorting training is BLOCKED.**
Updated September 15, 2026, after full detector annotation and real-data software checks. Recording remains stopped; no hardware commands accompanied these checks.

Studio now names dataset `093df5db-9ecb-416a-95d6-4efe3752442b`
`Sort blocks into bins`, in environment `collect data`
(`470c317b-5022-479c-af27-c8d3aeef91fe`). The live API reports five episodes,
3,249 frames, and two recorded task captions. The previous empty-dataset status is
superseded by these read-only checks; collection occurred outside this agent's control.

## Frozen pilot and quality checks

The Intel snapshot is
`/home/ird-demo/second-look/recording-snapshots/pilot-five-20260916T004831Z/dataset`.
Its adjacent `snapshot-receipt.json` records seven file hashes. No source files had
writable handles before or after copying. Source hashes before and after copying
matched the copy. Only copied files were made read-only; the original was untouched.

| Episode | Original caption | Frames | Sampled visual evidence |
|---|---|---:|---|
| 0 | Pick up a good piece. | 770 | Red block transferred into pink/coral bowl |
| 1 | Pick up a good piece. | 502 | Red block transferred into pink/coral bowl |
| 2 | Pick up a good piece. | 916 | Cyan block transferred into pink/coral bowl after an earlier approach |
| 3 | Pick up a defective piece. | 557 | Red block transferred into light-blue/gray bowl |
| 4 | Pick up a defective piece. | 504 | Marked yellow block transferred into light-blue/gray bowl |

All numeric rows are finite, with matching six-joint state/action order, valid
normalized ranges, episode boundaries, and nominal 30 fps timestamps. Both AV1
streams decode all 3,249 frames with gap-free presentation timestamps. Snapshot
hashes remained unchanged. One adjacent low-camera frame pair is identical in
episode 0; there are no black frames or long exact freezes.

The action/state diagnostic favors a four-frame lag in every episode. This is
not measured sensor latency and does not justify shifting labels. Episode 1 has
a 22.27-unit elbow target jump at frame 2. Its maximum gripper target/state
difference is 47.25 normalized units. Stored actions are targets, not motor acknowledgments.

Eight sampled image times per episode show transfers and released blocks. These
samples do not establish defect ground truth, complete task success, or specimen
identity. Episode 2 includes an earlier approach before its later lift. Failed
or uncertain attempts remain preserved in the full snapshot.

The exploratory split is whole episodes: train **0, 1, 3**; validation **2, 4**;
final test **none**. Specimen and session identities are unknown, so this split
cannot support independence or generalization claims. Train-only statistics and the real native ACT loader smoke are complete. These software checks do not resolve target association or verify sorting success.

Reports live beside the snapshot under `qc-readonly/`: `numeric-report.json`,
`qc-report.json`, `visual-review.json`, and five episode contact sheets.

## Latest stop verification

At 17:56:16 PDT, a read-only runtime metadata probe returned no session. Process
inspection found no runtime or robot owner, and both serial ports had no owners.
Logs show last-subscriber hold and dataset finalization at 17:52:27, followed by
both robot owners exiting at 17:53:04. The API still returned all five episodes.
Recording had already stopped; this agent sent no stop, motion, or reconnect command.
Physical pose and motor torque remain unknown. Software inactivity does not prove torque-off.

## Capture contract

| Input | Configuration | Read-only preflight evidence |
|---|---|---|
| Camera 1, low | USB index 4; serial `344223022426`; RGB 1920×1080, configured 30 fps | 31 distinct frames; estimated 29.96 fps; maximum observed frame age 41.94 ms |
| Camera 2, high | RealSense SDK serial `243622060187`; RGB 1280×720, configured 30 fps | 31 distinct frames; estimated 27.12 fps; maximum observed frame age 36.13 ms |
| Follower | Serial `5B79018445`; saved ACM1; runtime uses stable by-id path | Supplied `hack_follower` calibration preserved |
| Leader | Serial `5B79018575`; saved ACM0; runtime uses stable by-id path | Supplied `hack_leader` calibration preserved |

Runtime configuration uses normalized joint units and 30 fps. The manifest stores
calibration values, source hashes, camera identities, and the supplied calibration-file hashes.

Preflight sampled real camera publishers, separately. It does not prove synchronized
episode capture. Stored absolute camera exposure timestamps are absent. There is
**no proven inter-camera or camera-to-joint skew**. Configured fps is not measured synchronization.

## Detector annotations and blocked sorting training

**Detector-conditioned sorting training is BLOCKED.** The frozen detector annotations cover all 3249 frames: 2614 are INVALID with no inference; all 635 actual predictions are ANOMALOUS across both caption classes. There are zero verified target-associated starts. Every row has `target_association=UNKNOWN` and `conditioning_verified=false`; the crop often sees a pile or remnant rather than the picked block. Finite scores and candidate eligibility do not authorize training.

Actual annotation inputs preserve original RGB pixel hashes and the recorded dataset time basis. Physical presence, occlusion, and defect visibility remain UNVERIFIED. INVALID rows retain null scores. The first/last candidate indices are bounds, not proof of a continuous valid window. No label is inferred from the model's scores.

Full Mac annotations: `/Users/ohong/dev/intel-robotics/artifacts/cv/recording-annotations82/annotations.json`.
SHA256: `823e8289dc092fa6b85ed2142738c47f753c89bae27f34a124d31c228a011e38`.
Adjacent `coverage.json` explains target-association and distribution limits.

## Completed data-quality and loader checks

Train-only statistics use 1829 rows from episodes 0,1,3. Image statistics use installed LeRobot `aggregate_stats` over only the selected training-episode estimates. Validation episodes 2,4 do not contribute. This estimate does not establish synchronized capture or independent specimens/sessions.

Paths below are relative to the immutable snapshot's sibling `qc-readonly/` directory:

| Artifact | SHA256 / result |
|---|---|
| `train_stats.json` | `521dda17961afadbe4d2e702ac7406878965f2ae594e11bfdc5c002c38948912` |
| `exploratory-training-manifest.json` | `62271742d55072a61710601ec00bab8118a784718a92c92e5ede9af8bb0ab013` |
| `native-act-real-smoke/smoke-result.json` | PASS: actual native loader, both cameras, one CPU update, 81 finite gradient tensors, gradient norm 214.2813, output `[2,4,6]`, reload error 0, caption-mutation error 0 |

The real ACT smoke used a small model profile; it is not a trained task policy. ACT ignores text, so zero caption-mutation error does not verify instruction following. Task/controller readiness remains false. Snapshot source hashes remained unchanged. Earlier synthetic ACT results remain separate historical software evidence.

Sibling `qc-handoff-initial.tar.gz`: 3266762 bytes, SHA256 `744532091de88d99a850f8905a40af906e4b76b69b134b78c817bfc1e30cad77`. It excludes the real ACT report, which was created later. Do not infer that report is inside this frozen package.

Verified Mac copies: `/Users/ohong/dev/intel-robotics/artifacts/recording/pilot-five-20260916T004831Z/qc-handoff-initial.tar.gz` and sibling `native-act-real-smoke.json`. The separate ACT receipt SHA256 is `485cf9f03e3f15a812d82640fcad1b5b2b958b083df27613cd58dea2c2e0c558`.

## Authorized SmolVLA software probe

H100 is authorized only for a bounded genuine SmolVLA **original-caption** real-data software/export probe: start with one CUDA step, at most 20 updates. Tag every result `REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE`; task/controller readiness stays false. Original captions are actual conditioning inputs in this separate probe. It does not establish integrated Anomalib–VLA sorting or authorize deployment.

The authorized private H100 transfer occurred, and the one-step CUDA probe started at 18:10 PDT. Completion is not yet verified.

The probe uses the original good/defective task captions as conditioning, with train episodes 0,1,3 and validation 2,4. This is distinct from any proposed deployment prompt. A future integrated sorting policy must condition on the generic mission and valid target-associated detector output, without ground-truth class leakage. Current annotations do not meet that requirement.

`normal-route` and `reject-route` remain historical proposed ACT skills, not confirmed task rules. No tested ACT caption change selects a route. Recorded environment versions: Studio `c4ff730fb49f84e5102d01088d52cfff1ba62854`, application `0.1.0`; PhysicalAI `0.1.2.dev70+g8e4021703`; LeRobot `0.6.0`; Torch `2.11.0+xpu`; Lightning `2.6.5`.

## Next physical session

Demonstrations remains the sole control owner. After the approved safe pause, confirm physical state, camera 3, defect criteria, bin rules, workspace/stop limits, and bounded authorization. Place one identified block inside the inspection ROI and keep all other blocks outside it. Then capture target-associated observations and demonstrations within that scope. No automatic recording, reconnect, or motion follows the software checks.

## Recording controls — DEPLOYED; NOT PHYSICALLY EXERCISED

The deployed patch adds these behaviors:

- **Pause following** requests hold when connected, including before dataset readiness.
- **Resume following** requires an explicit click after dataset readiness.
- Hold prevents Start through the button, form, and hotkey. Accept can finish an active episode.
- Opening or reconnecting a recording route requests hold. The inference branch remains unchanged.
- Status shows **Holding position (torque on)** or **Following leader**. No motion shortcut is added.

Eight mocked callback tests passed. The lead reported passing TypeScript, focused
ESLint, and focused Prettier checks in isolated `studio-ui-check`. After verifying no
active serial owners, the lead ran a guarded `fuser` check, patch dry-run, backups,
and successful apply. The native dev UI hot-compiles; no service restart occurred.
Runtime-page rendered inspection remains pending physical authorization. The controls
have not been physically exercised.

Backup: `/home/ird-demo/studio-ui-check/backup/recording-controls`. Deployed source
SHA256 values are recorded in the capture manifest.

### Runtime fault before deployment

On September 15 at 16:22:39 PDT, follower owner `79778` stopped after five read failures.
Its disconnect hold failed with `comm-6`. Runtime `94649` exited at 16:22:49 after
45 seconds idle. Leader owner `79777` exited at 16:22:59 after 10 seconds idle.

**Physical state is UNKNOWN.** No agent motion or reconnect command was issued.
No active serial owner does not prove torque-off or physical safety. The lead
verified unchanged calibration hashes.

**Opening a runtime can initialize servos and enable torque. Hold keeps torque on;
it is neither disarmed nor an emergency stop.** The patch does not change connection initialization.

## Evidence

### Follower communication fault

At 16:22:04, the runtime lost its last subscriber and requested hold. At 16:22:39,
the follower owner exited after five read failures. Its final hold also failed.
The installed `scservo_sdk/scservo_def.py:23` defines `COMM_RX_TIMEOUT = -6`:
no status packet arrived. This does not identify whether power, wiring, a motor,
or another communication fault caused the timeout. Stable USB serial paths still
existed; the kernel journal returned no entries for 16:21–16:24.

Logs are preserved under `artifacts/recording/runtime-fault/`. Physical pose and
torque remain unknown. No reconnect was attempted. The operator must confirm the
approved stop state and visually check follower power and cable connections before
an authorized reconnect. Do not manipulate live wiring or move the arm to test it.

- [Capture manifest](../artifacts/recording/capture-manifest.json): full identities, versions, hashes, and limits.
- [Environment](../artifacts/recording/environment.json) and [runtime](../artifacts/recording/runtime.yaml): corrected camera 2 at 1280×720.
- [Preflight](../artifacts/recording/preflight/capture-check.json): real camera-only observations.
- [Synthetic loader](../artifacts/recording/act-loader-synthetic/manifest.json) and [ACT smoke](../artifacts/recording/act-loader-synthetic/smoke-result.json): software evidence only.
- [UI patch source](../artifacts/recording/ui-fix/recording-follow-controls.patch) and [Studio notes](STUDIO_TELEOP.md): controls and calibration provenance.
