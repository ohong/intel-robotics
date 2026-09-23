# Execution state

## Latest placement wrap — September 23, 2026

- Source: `main` in `/Users/ohong/dev/intel-robotics`. The placement source was reconciled earlier; `docs/PLACEMENT_SMOLVLA_STATUS.md` records the final scope and open gates. The `codex/placement-smolvla` branch and worktree are being retired.
- Local evidence: ignored `artifacts/placement/` was copied from the branch and compared byte-for-byte. The frozen Intel dataset and H100 transfer archive remain outside Git.
- Current remote access: `intel-robot` at the last known address timed out. H100 transfer/job state cannot be verified today. Do not infer continued training from the old transfer receipt.
- Owned training or robot-control processes started by this wrap: none. No placement checkpoint, Intel inference benchmark, or physical placement result is established.
- Next milestone, if resumed: restore trusted Intel access, verify H100 archive and job/lock state, run H100 SmolVLA smoke and fit, then measure each route offline and verify Intel inference before supervised motion.
- Original challenge deadline has passed; no active event time budget is known. The older sections below preserve historical evidence and may describe an earlier pause or stale service PIDs.

Updated September 15, 2026. Inspection pilot only; autonomous challenge incomplete.

## Paused by the user

The user requested a pause until more episodes are collected and training is authorized again. Stop requests were sent to the training and offline-QC owners. Preserve checkpoints, reports, and snapshots. Do not start more training, analysis, or deployment. Leave the user’s recording and hardware services unchanged.

New verified snapshots on Intel, under `second-look/recording-snapshots/`: `good-blue-ten-20260916T015528Z` (10 episodes, 6774 frames) and `inspection-six-20260916T015529Z` (6 episodes, 4930 frames). Both have three cameras; QC is incomplete. Keep these separate from the older five episodes. Camera placement changed, and the newer good-to-blue caption conflicts with the older observed good-to-coral route. Defect/bin rules remain unconfirmed.

The original-five SmolVLA probe completed one CUDA update and saved checkpoints. Native reload comparison failed; the training owner identified a CPU/CUDA noise mismatch as the suspected cause. No passing corrected comparison or deployable policy is established. Preserve the failed receipt; no further updates are authorized during this pause.

## Clock and source

- Original six-hour start, cutoff, and remaining time: UNKNOWN; no reset. Public judging: September 16.
- Active source: `main` in `/Users/ohong/dev/intel-robotics`, consolidated September 15 from `codex/second-look` (application, docs, tests) and `codex/h100-readiness` (`scripts/h100/`, `docs/H100_READINESS.md`). Runtime source `2aa5149c57507428b28341312c1c9fd3aaaeecf2` produced release `a9ed13af508b6165`; later docs do not change deployment.
- `artifacts/` and `pc-context/` are no longer tracked in Git; they are ignored and curated separately. Documents below still cite paths inside them, and those files remain on the machines named, not in this repository. `codex/placement-smolvla` is NOT merged; it holds uncommitted placement work in its own worktree.
- Original checkout `/Users/ohong/dev/intel-robotics` remains user `main` at `0e3a1c8`; artifacts remain there. Intel: `intel-robot`, `ird-demo@NUC16GDKX76`, root `/home/ird-demo/second-look`.

## Last verified services

- App: <http://127.0.0.1:8088>, `second-look-app.service`, PID 219757 verified 17:41 PDT; Mac tunnel PID 16010. At 17:59:59 API check: camera LIVE, detector INVALID/no inference/null score, policy BLOCKED, controller DISARMED/backend UNAVAILABLE, autonomous physical trials 0. Last real overlay browser check 17:43.
- Attach-only low camera 1: `physicalai/camera/UVCCamera/4/frame`, 1920×1080 RGB. Last camera owners: 81425/video4, 91174/video10.
- Studio ports 3000/7860 tunnel PID 11092. CV shutter 8091: `secondlook-cv-shutter-live.service`, Mac mirror PID 33929/tunnel PID 32892.
- After camera 3 disconnection/stop request, Demonstrations verified recording already STOPPED 17:56:16: no runtime/robot or serial owners. Automatic hold/finalize 17:52:27; owners exited 17:53:04. No agent stop/reconnect/motion command. Physical pose/torque UNKNOWN.

## Model and evidence

- Frozen Intel model: `/home/ird-demo/second-look/artifacts/models/lego-h100-v3-82fdd2d1`; genuine H100 PatchCore fit, OpenVINO GPU.0 FP32. Single colored block required in ROI `[900,620,1400,1060]`; physical presence/occlusion/visibility remain UNVERIFIED.
- Validation: one defect miss and four UNKNOWN among 14 repeated views from one session; no unseen final test. Exact-release Intel tests 121/121, zero skips; conversion comparisons 28/28 passed. Model-only GPU median 14.68 ms; method/limits in [BUILD_EVIDENCE.md](docs/BUILD_EVIDENCE.md).
- Original-checkout evidence: `artifacts/verification/{intel-checks-a9ed13af508b6165.txt,final-pilot-verification.json,post-recording-stop-app.json,pilot-benchmark.json}`. Complete model: `artifacts/models/h100-patchcore-v3`.
- Verified detector archive: `artifacts/deliverables/second-look-pilot-a9ed13af508b6165.tar.gz`, 300084158 bytes, 3815 members; adjacent `pilot-bundle.json` supplies SHA/source. Excludes five real episodes/VLA weights; frozen docs predate later updates. Launch: [RUNBOOK.md](docs/RUNBOOK.md).

## Data and active software probe

- Five episodes/3249 frames: structural/video, train-only statistics, manifest, and real native ACT small-profile loader/update/reload checks complete. Transfers observed; defect/bin correctness, synchronization, specimen/session independence unverified. Details: [DATASET_STATUS.md](docs/DATASET_STATUS.md).
- Immutable Intel snapshot: `/home/ird-demo/second-look/recording-snapshots/pilot-five-20260916T004831Z/dataset`. Verified Mac backup: `/Users/ohong/dev/intel-robotics/artifacts/recording/pilot-five-20260916T004831Z/dataset`; seven files, 99132116 bytes, all source SHA values match. Adjacent `mac-snapshot-verification.json` is the receipt.
- **Detector-conditioned sorting training BLOCKED:** 3249 annotations contain 2614 INVALID/no-inference rows and 635 predictions, all ANOMALOUS across both captions. Zero target-associated starts; all `target_association=UNKNOWN`, `conditioning_verified=false`. Crop often sees another block/pile. Finite scores do not authorize training. Full evidence: original `artifacts/cv/recording-annotations82/{annotations,coverage}.json`.
- H100 owns only an original-caption genuine SmolVLA software/export probe: one-update job launched 18:10 PDT; no CUDA update verified yet; maximum 20 updates. Tag `REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE`, task/controller false. Captions condition this separate probe. Train 0/1/3; validation 2/4; no final test. This is not integrated Anomalib–VLA success. Prior recorded-image SmolVLA replay and synthetic ACT remain separately scoped evidence.

## Owner, blockers, next milestone

- **Demonstrations** task `01a0a734-59c0-7961-917f-203c9392eb18` is sole robot-control owner. No automatic recording, reconnect, or motion.
- Next: finish the bounded software probe; independently establish approved safe pause, camera 3 facts, defect/bin rules, workspace/stop limits, and bounded supervision authorization through Demonstrations. Present one identified block in the ROI, all others outside; capture target-associated data and unseen evaluation within that scope.
- No task-ready policy/controller or autonomous result. Guard fixtures do not establish global command exclusivity or physical safety. Tools stopping does not promise continued execution.
