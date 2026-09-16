# Second Look build evidence

## Running pilot — 15 September 2026

Second Look was verified on Intel at [127.0.0.1:8088](http://127.0.0.1:8088). Source `2aa5149c57507428b28341312c1c9fd3aaaeecf2` produced release `a9ed13af508b6165`. All 121 tests passed on that exact Intel release, with zero skips. Receipt: original Mac checkout `artifacts/verification/intel-checks-a9ed13af508b6165.txt`.

The application attaches to Studio low camera 1, `physicalai/camera/UVCCamera/4/frame`, at 1920×1080 RGB. It runs actual Anomalib PatchCore through OpenVINO GPU.0 FP32. Camera and real overlay were inspected at 17:30 and again at 17:43 PDT. The later browser check showed the cropped anomaly overlay and experimental validation counts. During those checks, the live scene alternated between inference and INVALID when the crop detects multiple comparable components. Invalid observations clear previous scores/maps. Physical presence, occlusion, and defect visibility remain UNVERIFIED.

At the last app check, policy was BLOCKED, controller DISARMED/backend UNAVAILABLE, and physical trials 0. The autonomous challenge is incomplete. The confirmed task is two-pile LEGO sorting; exact defect criteria, destination rules, workspace limits, stop procedure, and bounded motion authorization remain unresolved.

### Frozen detector and training evidence

| Item | Verified pilot value |
|---|---|
| Intel artifact | `/home/ird-demo/second-look/artifacts/models/lego-h100-v3-82fdd2d1` |
| Artifact ID | `82fdd2d1db11d43aeb8376ee4b61cf4f85bd28f0102c4ed48d2ec41ac5d8a6c0` |
| Metadata SHA256 | `c29cfb2aa9e6446e4f78cc8e4b7a9cb7cc745e293e171e400cb326465f8671ec` |
| Mac complete model | `/Users/ohong/dev/intel-robotics/artifacts/models/h100-patchcore-v3` |
| Model | PatchCore, Wide ResNet50-2 layers 2/3, 256 px input, nine neighbors, 0.1 coreset, seed 17 |
| H100 work | Genuine CUDA feature extraction and coreset fitting: 3.2476 s; total 116.22 s including setup, export, and calibration |
| Input contract | Parent ROI `[900,620,1400,1060]`, then dominant-color foreground crop; one colored block |
| Unsupported scenes | Gray/white objects and multiple comparable components; no validated physical occlusion filter |
| Decision rule | Threshold 59.548561096191406; uncertainty half-width 1.3926738739013673 |

Thirty operator-labeled images cover six specimens: four good and two defective. All images come from one session. Fifteen normal images from three good specimens formed the fit set. Validation used five views of the fourth good specimen and nine views of two defective specimens. One hand-occluded view was excluded before fitting.

| Validation truth | NORMAL | ANOMALOUS | UNKNOWN |
|---|---:|---:|---:|
| Good: five views of one specimen |2|0|3|
| Defective: nine views of two specimens |1|7|1|

**One defective view was missed; four of 14 views were UNKNOWN.** Repeated views are not independent physical trials. Validation selected the decision threshold; it is not an unseen final test. No final-test generalization or adequate sorting quality is established. Raw scores are not probabilities, and heatmaps use a per-image color scale.

All 28 Intel comparisons passed: Torch CPU versus OpenVINO CPU and GPU FP32 across 14 validation images, including score, map, and decision parity. Maximum score error was 0.000175477; maximum map error was 0.000301361. Image/model hashes were checked, and the artifact remained unchanged. Conversion agreement does not establish detection quality.

Receipts: `scripts/h100/evidence/anomalib-h100-v3.json`, `anomalib-return-v3.json`, and `anomalib-intel-parity-v3.json`. [H100_READINESS.md](H100_READINESS.md) records the frozen source, data identity, fit command, export, and return. No matched CPU/H100 training speedup is claimed.

### Current pilot model latency

At 17:43 PDT, the frozen pilot artifact ran on one known validation image with identical preprocessing, batch one, three warmups, and ten timed model calls.

| Runtime | Median | p95 | Load / compile |
|---|---:|---:|---:|
| Torch CPU FP32, four threads |106.14 ms|122.83 ms|4104.09 ms|
| OpenVINO CPU FP32, 12 threads / one stream |63.98 ms|67.48 ms|175.74 ms|
| OpenVINO GPU.0 FP32 |14.68 ms|17.35 ms|368.65 ms|

CPU and GPU score/map/decision parity passed against Torch at `atol=0.1, rtol=0.01`; model hashes remained unchanged. This is model-only latency, excluding capture, preprocessing, scheduling, browser rendering, and robot motion. The live app and Studio remained active, with uncontrolled contention; the benchmark used process niceness 10. Ten calls on one image do not establish general performance or task accuracy. The current pilot has no NPU measurement.

Receipt: original checkout `artifacts/verification/pilot-benchmark.json`. Earlier 128 px smoke-model measurements below remain separate.

### Other component evidence and limits

- Genuine Studio SmolVLA consumed a real detector result and the same hash-bound recorded image on Intel CPU. State and identity normalizers were synthetic; action semantics remain unverified. The 26 policy tests passed. This is recorded replay, separate from the live pilot; task/controller readiness remains false.
- Synthetic ACT completed one CUDA update, export/reload, and Intel parity. That earlier synthetic qualification remains `UNSAFE_NOT_TASK_TRAINED`; it used no real task episodes.
- At 17:46 PDT, read-only Studio GET reported five episodes in dataset `093df5db-9ecb-416a-95d6-4efe3752442b`, **Sort blocks into bins**. Three have good-piece task strings; two have defective-piece task strings. That API read checked metadata only; subsequent structural/visual checks are below. Receipt: original checkout `artifacts/verification/dataset-final-read.json`.
- Five closed episodes, totaling 3249 frames, are copied to immutable Intel snapshot `/home/ird-demo/second-look/recording-snapshots/pilot-five-20260916T004831Z/dataset`. The adjacent snapshot receipt verifies source hashes before/after copying and copied hashes; no writable file descriptors remained. Structural/video checks passed for both native camera streams: finite six-value state/actions, valid episode boundaries, 30 fps, all 3249 frames per camera, and no out-of-range values, black frames, or long freezes. One adjacent low-camera frame repeats in episode 0. The best state/action mean absolute error occurs at a four-frame lag; this diagnostic does not change the data or prove synchronized capture.
- **Detector-conditioned sorting training is BLOCKED.** The frozen detector annotations cover all 3249 frames: 2614 are INVALID with no inference; all 635 actual predictions are ANOMALOUS across both caption classes. There are zero verified target-associated starts. Every row has `target_association=UNKNOWN` and `conditioning_verified=false`; the crop often sees a pile or remnant rather than the picked block. Finite scores and candidate eligibility do not authorize training.
- Train-only statistics and the exploratory manifest are complete for train episodes 0/1/3 (1829 rows), with validation 2/4 and no final test. The real native ACT loader passed a small-profile, two-camera smoke: one CPU update, 81 finite gradient tensors, gradient norm 214.2813, output shape `[2,4,6]`, zero reload error, and zero caption-mutation error. ACT ignores text. This is real-data software evidence only; task/controller readiness remains false.
- H100 is authorized only for a bounded genuine SmolVLA **original-caption** real-data software/export probe: start with one CUDA step, at most 20 updates. Tag every result `REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE`; task/controller readiness stays false. Original captions are actual conditioning inputs in this separate probe. It does not establish integrated Anomalib–VLA sorting or authorize deployment.
- Full annotations: `/Users/ohong/dev/intel-robotics/artifacts/cv/recording-annotations82/annotations.json`, SHA256 `823e8289dc092fa6b85ed2142738c47f753c89bae27f34a124d31c228a011e38`; adjacent `coverage.json` records the association and distribution limits. See [DATASET_STATUS.md](DATASET_STATUS.md) for training-statistics, manifest, ACT, and QC-package receipts.
- Native command guard checks passed 27 tests using provenance and native fake fixtures. The guard is not deployed. Global command-producer exclusivity remains unresolved.
- The Demonstrations task owns robot control. Historical snapshot at 17:32: native PID 238125 owned ttyACM1 and PID 238126 owned ttyACM0, both started 17:29:56 under parent 237658. This was an independent owner change, superseded by the stopped-session verification below. Integration issued no robot or reconnect command and left that activity untouched. Physical pose/torque remain UNKNOWN after the earlier 16:22 follower motor 6 timeout.

The original artifact checkout is `/Users/ohong/dev/intel-robotics`; active source is `/Users/ohong/dev/intel-robotics-integration`. Final snapshot `artifacts/verification/final-pilot-verification.json` confirms service PID 219757, camera HTTP 200, exact-ID overlay 200, wrong-ID overlay 503, and `/api/arm` 404. Sequence advanced 186921→186944; GPU inference was FRESH/UNKNOWN within the uncertainty band. The receipt preserves an earlier owner-assertion failure and the independent serial-owner change. The completed detector checkpoint archive is `/Users/ohong/dev/intel-robotics/artifacts/deliverables/second-look-pilot-a9ed13af508b6165.tar.gz`: 300084158 bytes, SHA256 `87e0aee1dfc0ebf3138f27c2f5f9e1bd38313c629774b4363718a7b1138b3de2`. All 3815 members passed readback hash verification. Adjacent `pilot-bundle.json` is authoritative. Packaged source is `5d72c38881d5fc15785cf36021c618c3a8a4dade`; deployed runtime remains `2aa5149c57507428b28341312c1c9fd3aaaeecf2`. The archive excludes the five real episodes and VLA weights. Its frozen docs can predate this completion receipt.

### Recording stop and current limits

After the user reported camera 3 disconnected and requested recording stop, Demonstrations verified the software was already STOPPED at 17:56:16 PDT. Runtime metadata was null, no StudioSession or robot owner remained, and ttyACM0/1 had no owners. Logs show last-subscriber loss at 17:52:27 triggered hold/finalize; owners exited idle at 17:53:04. No agent stop, reconnect, or robot command was sent. Physical pose and torque remain UNKNOWN; this does not authorize physical changes.

At 17:59:59 PDT, read-only API inspection confirmed release `a9ed13af508b6165`, camera LIVE, detector INVALID, policy BLOCKED, controller DISARMED, and zero autonomous physical trials. The foreground was absent or too small; no inference ran, and the score was null. Receipt: original checkout `artifacts/verification/post-recording-stop-app.json`. The latest API check does not establish physical correctness or a new valid overlay.

## Historical scope below

The following records preserve earlier engineering checks. They apply to their named releases, cameras, smoke models, and timestamps. They do not describe the current LEGO pilot, its accuracy, or current hardware ownership. The older archive `second-look-inspection-59ed3e663ea96b02.tar.gz` is a smoke-model baseline: 186059310 bytes, SHA256 `1cabbef7d57fdeb0085f87bc1013899d0c8dca99431e7c9fe21652b59d91609b`.

## Chosen design

| Approach | Benefit | Cost / limitation |
|---|---|---|
| Full Studio application runtime | Existing recording, training, policy and robot-session tools | Its normal robot connection can configure motors and torque; live use requires a defined supervised workspace. |
| Small Intel-local service plus Studio Python policy boundary | Reuses installed environments, keeps motor access explicit, and checks observation/model contracts | Integration code needs explicit camera, state, normalization, and action mappings. |

The current baseline uses the second approach. It does not relabel the supervisor as a VLA.

```text
Intel PC
  Studio camera owner -> shared frames + capture time + sequence
                                    |
                        attach-only Studio subscriber
                                    |
                        Anomalib PatchCore -> OpenVINO
                                    |
                         score + anomaly map
                                    |
                        policy input contract: BLOCKED
                        (no compatible task checkpoint)
                                    |
                        Second Look robot output: DISABLED

Mac: source, tests, SSH deployment, loopback browser tunnel
Studio Python environment: actual policy API/contract inspection
```

Anomalib runs in the existing `intel_dev_env`. Studio stays in its existing backend virtual environment. Sponsor packages, calibration, drivers, and authentication remain unchanged.

The operator subsequently configured all cameras in Studio. Direct camera access would compete with those feeds. We compared Studio's preview WebSocket with its shared-memory subscriber API. The WebSocket has no capture timestamp and can apply camera settings. The selected attach-only subscriber preserves real capture timestamps and sequence numbers, and cannot create or reconfigure a publisher. Studio remains the physical camera owner.

## Historical engineering baseline ledger

| Check | Status | Evidence and limits |
|---|---|---|
| Fresh public-key SSH | VERIFIED | User `ird-demo`, host `NUC16GDKX76`; existing host trust retained. |
| Real camera capture | HARDWARE VERIFIED | Devices 4 and 10 returned 640×480 setup views. Device 12 returned near-black pixels after warmup. No task labels. |
| Read-only follower inspection | HARDWARE VERIFIED | Six motor positions read. Stored calibration matches all six motors. All torque flags were already enabled; no changes sent. `artifacts/discovery/follower-state.json`. |
| Anomalib fitting/export/inference | DIGITAL VERIFIED ON INTEL | Actual PatchCore, two unlabeled setup views, 128×128 input, ten-entry coreset. Threshold unset and disposition UNKNOWN. |
| Model runtime comparison | DIGITAL VERIFIED ON INTEL | Torch CPU and OpenVINO CPU/GPU/NPU tested; default GPU FP16 failed map parity. See `anomaly-smoke-benchmark.json`. |
| Application tests | DIGITAL VERIFIED ON INTEL | All 82 Python tests passed on release `59ed3e663ea96b02` in the installed Intel environment, with no skips. They cover detector/data contracts, invalid observations, crop geometry, runtime failures, policy contracts, shared camera transport, overlay identity, and mocked action safeguards. |
| Live application/API | HARDWARE + DIGITAL VERIFIED | Changing camera observations, actual GPU.0 FP32 inference, UNKNOWN classification, and zero physical trials. Wrong overlay observation ID returns HTTP 503; no arm endpoint exists. |
| Invalid-result browser recovery | DIGITAL / REPLAY VERIFIED | A disposable loopback fixture changed a displayed result to INVALID. Browser inspection confirmed the previous image and score disappeared and model timing became unavailable. The fixture was stopped. See `artifacts/verification/invalid-browser-verification.json`. |
| Missing camera | DIGITAL FAULT VERIFIED | A separate disposable process with an invalid camera path returned UNAVAILABLE and HTTP 503. Controller stayed DISARMED. The process was terminated. |
| Single runtime owner | DIGITAL VERIFIED | A second process using the same camera-session lock exited before opening the camera. |
| Clean process restart | DIGITAL VERIFIED | Restarted the owned service; a new process recovered camera and detector without robot access. |
| Initial Studio policy boundary | DIGITAL VERIFIED | Installed Studio preprocessing and observation APIs executed using the read-only robot state. Wrong camera mapping was rejected. This initial check did not run model inference; the later genuine VLA replay is recorded below. |
| Studio shared-frame probe | HARDWARE + DIGITAL VERIFIED | Three real 1280×720 RGB frames with advancing sequences and 1.3–2.3 ms capture age. Attached to the existing publisher without camera reconfiguration. |
| Studio shared-camera application | HARDWARE + DIGITAL VERIFIED | Release `ddf40f82a512ba0f` receives real 1920×1080 camera 1 frames through the attach-only subscriber. Shared sequences advance; Studio camera/serial process owners remained unchanged. See `artifacts/verification/shared-api-verification.json`. |
| Current camera/observation integration | HARDWARE + DIGITAL VERIFIED | Release `59ed3e663ea96b02` uses the high camera 2 publisher at 1280×720. Actual GPU FP32 inference reports valid pixels and unverified physical presence/visibility. The confirmed sorting instruction appears in the browser. |
| Delayed overlay retention | REPAIRED / LIVE VERIFIED | Release14690 rejected a 1.5-second delayed request while capture age remained below3seconds. Release59ed retains bounded images through the same freshness window; three live delayed requests passed with exact IDs. Failed and repaired results remain in `artifacts/verification/overlay-retention-regression.json` and `latest-api-verification.json`. |
| H100 infrastructure | DIGITAL VERIFIED / TASK TRAINING BLOCKED | CUDA computation, synthetic optimizer/checkpoint resume, bidirectional hashes, and native ACT/Trainer imports passed. A full-size synthetic ACT update, exact checkpoint reload, and Torch/OpenVINO export passed. The returned native and OpenVINO FP32 artifacts passed Intel CPU parity at1e-4 tolerance. Native max error2.67e-5; OpenVINO max error3.81e-5. No real task training yet. See `H100_READINESS.md`. |
| VLA model inference | DIGITAL / RECORDED-IMAGE REPLAY VERIFIED | Genuine pinned Studio SmolVLA executed on Intel CPU. A real detector score from the same hash-bound recorded image entered the model's language input. State and normalizers were synthetic fixtures; the finite six-value output has no verified physical semantics. Task and controller readiness remain false. See `POLICY.md`. |
| Defect accuracy | NOT TESTED | Pilot operator-labeled images are available; fitting, calibration, and unseen-specimen evaluation are in progress. |
| Autonomous manipulation/outcome | NOT TESTED | Zero motion commands or physical trials from this build. |

## Historical smoke-model reproduction and limits

Model artifact: `/home/ird-demo/second-look/artifacts/models/scene-smoke`. Its metadata records source image hashes, package versions, preprocessing, coreset settings, model hashes, and provenance. The complete local reference copy is under `pc-context/local/models/scene-smoke`.

The comparison uses batch one, identical preprocessed input, three warmups, and ten synchronous measurements. Compilation and loading are reported separately. The reported latencies exclude image acquisition, physical motion, and outcome verification. Conversion parity is numerical agreement, not task accuracy.

Raw scores are not probabilities. Heatmap colors use a per-image scale. Model provenance and missing thresholds remain visible in the application.

### Measured model latency

| Runtime | Median model call | Conversion parity | Deployment |
|---|---:|---|---|
| Torch CPU FP32 | 20.0–20.7 ms across comparison runs | Reference | Baseline |
| OpenVINO CPU FP32 | 13.19 ms | Passed | Available fallback |
| OpenVINO GPU default FP16 | 1.78 ms | **Failed anomaly-map parity** | Rejected |
| OpenVINO GPU FP32 | 7.36 ms | Passed | Selected |
| OpenVINO NPU FP16 | 4.10 ms | Passed configured tolerance | Benchmark only |

Parity tolerance was `atol=0.1, rtol=0.01`. Both reference scenes also passed CPU/GPU FP32 map parity. This small numerical check establishes neither defect accuracy nor generalization. The live application includes capture, preprocessing, scheduling, and rendering overhead; its timing is not the benchmark timing.

### Stored evidence

- `docs/anomaly-smoke-benchmark.json`: complete latency samples, precision, device, parity, versions, and limits.
- `docs/anomaly-parity-two-scenes.json`: second-scene conversion check.
- `artifacts/verification/api-verification.json`: live API checks with release identity.
- `artifacts/verification/missing-camera-result.json`: camera-unavailable fault result.
- `artifacts/verification/studio-contract.json`: actual installed Studio contract probe; `model_inference_executed=false`.
- `artifacts/verification/studio-shared-probe.json`: actual attach-only frame probe, including capture timestamps and sequences.
- `artifacts/discovery/follower-state.json`: read-only positions, calibration comparison, and torque observations.

Raw captures, model weights, and machine-local evidence stay outside version control. Immutable deployed source manifests and model metadata identify the corresponding artifacts.

### Genuine VLA engineering replay

At 16:45 PDT, the existing Anomalib/OpenVINO GPU FP32 detector produced raw score79.97917, UNKNOWN, and a null threshold. Native Studio SmolVLA consumed that exact inspection text and the same recorded image. Detector model time was13.213ms; VLA inference was4.239seconds after8.992seconds of loading. These are single calls, not a warmed benchmark.

Evidence: `artifacts/policy/smolvla-same-image-inference.json`, its separate detector report and map, and the pinned download manifest. The image is real; state and normalization are synthetic. This replay is separate from the live dashboard and never sends robot commands. It proves software integration, not defect understanding or task success. The earlier probe with synthetic anomaly metadata is preserved separately.

Software disarm means this application sends no motor commands. During the earlier read-only probe, all six torque flags were enabled and left unchanged. Later, the operator's Studio follower process failed after five communication-read errors, followed by idle teardown. The separate robot-control task is diagnosing that failure. The earlier idle teardown left no serial owners at that snapshot; physical position and torque were unknown. A physical safe state has not been established by this build.

The detector route check used Intel. The separate H100 task prepares task-policy training. Its synthetic CUDA test is infrastructure evidence, not a trained robot policy. Task-specific fitting/training remains blocked on authoritative labels and physical examples.
