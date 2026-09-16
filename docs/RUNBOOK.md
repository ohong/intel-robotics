# Second Look runbook

## Running pilot

The application was last verified at [127.0.0.1:8088](http://127.0.0.1:8088) through the Mac loopback tunnel. Intel host `intel-robot` (`NUC16GDKX76`) runs service PID 219757, release `a9ed13af508b6165` under `/home/ird-demo/second-look`. Source revision is `2aa5149c57507428b28341312c1c9fd3aaaeecf2`.

The pilot attaches to Studio low camera 1 at 1920×1080 RGB and uses the immutable H100-fitted LEGO artifact with OpenVINO GPU.0 FP32. See [BUILD_EVIDENCE.md](BUILD_EVIDENCE.md) for the one missed defective view, four abstentions, and missing unseen final test.

Second Look sends no robot commands. Its software disarm does not establish physical pose or torque. The separate Demonstrations task owns all robot control and the pending physical request. After the user reported camera 3 disconnected and requested recording stop, Demonstrations verified the software was already STOPPED at 17:56:16 PDT. Runtime metadata was null, no StudioSession or robot owner remained, and ttyACM0/1 had no owners. Logs show last-subscriber loss at 17:52:27 triggered hold/finalize; owners exited idle at 17:53:04. No agent stop, reconnect, or robot command was sent. Physical pose and torque remain UNKNOWN; this does not authorize physical changes.

**Detector-conditioned sorting training is BLOCKED.** The frozen detector annotations cover all 3249 frames: 2614 are INVALID with no inference; all 635 actual predictions are ANOMALOUS across both caption classes. There are zero verified target-associated starts. Every row has `target_association=UNKNOWN` and `conditioning_verified=false`; the crop often sees a pile or remnant rather than the picked block. Finite scores and candidate eligibility do not authorize training.

H100 is authorized only for a bounded genuine SmolVLA **original-caption** real-data software/export probe: start with one CUDA step, at most 20 updates. Tag every result `REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE`; task/controller readiness stays false. Original captions are actual conditioning inputs in this separate probe. It does not establish integrated Anomalib–VLA sorting or authorize deployment. See [DATASET_STATUS.md](DATASET_STATUS.md) for complete annotation, statistics, manifest, and real ACT loader evidence. None of these steps enables live control.

At 17:59:59 PDT, read-only API inspection confirmed release `a9ed13af508b6165`, camera LIVE, detector INVALID, policy BLOCKED, controller DISARMED, and zero autonomous physical trials. The foreground was absent or too small; no inference ran, and the score was null. Receipt: original checkout `artifacts/verification/post-recording-stop-app.json`. The last browser overlay inspection was 17:43 PDT. Do not reconnect a robot session to restore a camera.

## Inspect without changing the service

Run from the Mac integration worktree. Do not switch or overwrite the original user checkout.

```sh
cd /Users/ohong/dev/intel-robotics-integration
python3 scripts/run_checks.py
python3 scripts/remote.py status
python3 scripts/remote.py check
python3 scripts/remote.py pull --path artifacts/runtime
```

`status` reports the actual launch under `unit`: check `revision`, `camera_service`, and `anomaly_artifact`. Its top-level expected values still use smoke defaults; they do not identify the active pilot. `check` runs only deployed `run_app.py --help`. It does not verify inference or start hardware.

The existing Mac tunnel is PID 16010. If no tunnel owns local 8088, this command opens a loopback-only tunnel:

```sh
python3 scripts/remote.py tunnel --local-port 8088 --remote-port 8088
```

The foreground tunnel ends with Ctrl-C. The Intel service survives SSH disconnect.

## Exact pilot launch

Studio must already publish `physicalai/camera/UVCCamera/4/frame`. Second Look attaches only; it cannot open, reconfigure, or create that publisher. Missing shared frames produce UNAVAILABLE after bounded retries.

The remote `current` symlink must identify the verified owned release, and `runtime/armed.json` must explicitly contain `{"armed": false}`. The frozen model must exist at the path below. These software guards do not certify physical safety.

```sh
python3 scripts/remote.py start --port 8088 \
  --studio-camera-service physicalai/camera/UVCCamera/4/frame \
  --anomaly-artifact /home/ird-demo/second-look/artifacts/models/lego-h100-v3-82fdd2d1
python3 scripts/remote.py status
```

**Always pass both pilot arguments.** Plain `start` defaults to high camera 2 and the old `scene-smoke` artifact. An already-running service with different release, camera, artifact, interpreter, port, or instruction arguments is rejected. Do not treat that mismatch as a successful pilot launch.

The helper creates `second-look-app.service` with Intel Python 3.11, the current release ID, OpenVINO GPU `f32`, a runtime evidence file, and an exclusive app lock. The camera helper runs in Studio's existing Python environment. The service is transient and is not configured for reboot startup.

After launch, inspect the browser and actual `unit` arguments. Confirm advancing observations, real camera 1 pixels, the expected artifact, and cleared score/map on INVALID input. The scene may be invalid under the frozen crop; do not change model or threshold to make a presentation look successful. Physical presence, occlusion, and defect visibility remain unverified even when inference succeeds.

To stop only the owned Second Look service and subscriber:

```sh
python3 scripts/remote.py stop
```

This leaves Studio owners untouched. It is not a robot stop procedure.

## Reproduce and update the source

The release's source manifest records Git revision, dirty state, every selected source hash, and the 16-character release identity. Docs are excluded from source transfer. A new manifest from an edited worktree can produce a new identity; it does not replace evidence for the tested release.

```sh
python3 scripts/remote.py manifest
python3 scripts/remote.py deploy --dry-run
python3 scripts/remote.py deploy --stage-only
```

Staging transfers an explicit allowlist, checks hashes, and preserves existing releases and artifacts. It excludes docs, model weights, captures, credentials, caches, and environment files. The model is separately installed and verified; deployment does not create it.

For an intentional source update, verify the staged source and checks first. Then stop Second Look, activate with `deploy`, and use the exact pilot launch above. Activation requires explicit local and remote disarmed guards. Deployment alone does not replace a running process.

```sh
python3 scripts/remote.py stop
python3 scripts/remote.py deploy
python3 scripts/remote.py start --port 8088 \
  --studio-camera-service physicalai/camera/UVCCamera/4/frame \
  --anomaly-artifact /home/ird-demo/second-look/artifacts/models/lego-h100-v3-82fdd2d1
```

Do not open a Studio robot environment while restoring a camera feed. Robot connection can energize hardware and belongs to the authorized Demonstrations session.

## Artifacts and receipts

- Complete Mac model: `/Users/ohong/dev/intel-robotics/artifacts/models/h100-patchcore-v3`.
- Artifact ID: `82fdd2d1db11d43aeb8376ee4b61cf4f85bd28f0102c4ed48d2ec41ac5d8a6c0`.
- `metadata.json` SHA256: `c29cfb2aa9e6446e4f78cc8e4b7a9cb7cc745e293e171e400cb326465f8671ec`. Metadata also records hashes for Torch and OpenVINO model files.
- Fitting reproduction and transfer receipts: [H100_READINESS.md](H100_READINESS.md), `scripts/h100/evidence/`. Retraining may create a different bank; use the frozen artifact to reproduce these validation decisions.
- Current pilot model benchmark: original checkout `artifacts/verification/pilot-benchmark.json`; see [BUILD_EVIDENCE.md](BUILD_EVIDENCE.md) for method and limits.
- Exact Intel test result: original checkout `artifacts/verification/intel-checks-a9ed13af508b6165.txt`.
- Verified final snapshot: original checkout `artifacts/verification/final-pilot-verification.json`.
- The completed detector checkpoint archive is `/Users/ohong/dev/intel-robotics/artifacts/deliverables/second-look-pilot-a9ed13af508b6165.tar.gz`: 300084158 bytes, SHA256 `87e0aee1dfc0ebf3138f27c2f5f9e1bd38313c629774b4363718a7b1138b3de2`. All 3815 members passed readback hash verification. Adjacent `pilot-bundle.json` is authoritative. Packaged source is `5d72c38881d5fc15785cf36021c618c3a8a4dade`; deployed runtime remains `2aa5149c57507428b28341312c1c9fd3aaaeecf2`. The archive excludes the five real episodes and VLA weights. Its frozen docs can predate this completion receipt. The archive root contains `source/`, `bundle-manifest.json`, and `README.txt`. Treat `source/` as the extracted project root. It contains docs, the detector model, labeled detector images, and selected evidence. `source/source-manifest.json` records the exact live runtime source.

The older `second-look-inspection-59ed3e663ea96b02.tar.gz` reproduces the previous smoke baseline. It does not reproduce this LEGO pilot. `pull` copies allowed runtime evidence without overwriting existing local files. Raw data and model artifacts require their recorded explicit transfer paths.

No launch, parser check, test suite, conversion comparison, or software guard proves autonomous success. Remaining physical prerequisites and the prepared session are in [DEMO.md](DEMO.md).
