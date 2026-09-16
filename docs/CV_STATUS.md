# Computer vision status

Updated September 15, 2026. Owner: CV task.

## Current result

A genuine Anomalib 2.6.0 PatchCore pilot is fitted, exported, and active in the application on port 8088, using Camera 1 Low. It is explicitly experimental. Robot control remains disarmed; this result does not establish autonomous sorting readiness.

- Captured 30 original RGB photos of six operator-labeled physical blocks: four good and two defective, five views each. All originals and provenance are saved on Intel and mirrored to the Mac.
- Frozen fitting dataset includes 29 images. One image with a hand overlapping the inspection area was excluded before scoring; its original remains saved.
- Normal memory bank uses 15 images from three good blocks. Validation uses five images of the fourth good block and nine images of the two defective blocks. Specimens do not cross these splits. All images share one collection session, so session-independent performance is unmeasured.
- Final candidate uses Wide ResNet50-2 layers 2/3, 256-pixel input, foreground cropping inside a fixed inspection region, sampling ratio 0.1, nine neighbors, and seed 17. H100 CUDA feature extraction and memory-bank fitting were measured; the bank contains 1,536 reference features. No defective image entered that bank.

## Frozen validation results

Threshold: 59.548561096191406. An uncertainty interval around that threshold is preserved in model metadata. Scores are anomaly distances, not probabilities.

| True label | Predicted good | Predicted defective | Uncertain |
|---|---:|---:|---:|
| Good: 5 images | 2 | 0 | 3 |
| Defective: 9 images | 1 | 7 | 1 |

There is one missed defective image and four uncertain images. These are repeated views of only three validation specimens, not independent final-test trials. Binary threshold counts without abstention are TP 8, FP 0, FN 1, TN 5; F1 is 0.9412 on validation only.

The remaining miss is `capture-2174b6af124e42f0929315e8bdde0826.png`, an underside-facing red block. Visible black markings survive the actual crop and resize, while the studs are hidden by pose. Its score is 55.6651. The physical defect definition and pixel-level localization ground truth are not established by these images; labels were not changed to hide the failure.

## Verified export and deployment

- Frozen artifact identity: `82fdd2d1db11d43aeb8376ee4b61cf4f85bd28f0102c4ed48d2ec41ac5d8a6c0`.
- Deployed immutable Intel directory: `/home/ird-demo/second-look/artifacts/models/lego-h100-v3-82fdd2d1`.
- Hash-verified Mac copy: `/Users/ohong/dev/intel-robotics/artifacts/models/h100-patchcore-v3`.
- All 14 validation images passed PyTorch CPU versus OpenVINO CPU and GPU FP32 comparison: 28 score, map, and exact-decision checks. Maximum score difference was 0.000175477; maximum map difference was 0.000301361. Tolerances were atol 0.1 and rtol 0.01. Image hashes and unchanged artifact hashes were verified.
- Receipt: `artifacts/cv/lego-pilot-v1/parity-h100-v3.json`. Missed-view review: `artifacts/cv/lego-pilot-v1/missed-review/`.
- Application owner activated Camera 1 Low publisher `physicalai/camera/UVCCamera/4/frame`, RGB 1920x1080. Fixed parent ROI is `[0.46875, 0.5740740740740741, 0.7291666666666666, 0.9814814814814815]`; effective foreground crop and RGB/bilinear/ImageNet preprocessing come from frozen metadata.
- Application shows experimental status, validation limits, and explicit invalid-observation reasons. Multiple comparable foreground components produce an invalid observation with no score or stale map. This filter does not prove full occlusion or physical defect visibility.
- Application owner reported 121 Intel tests passed with no skips. This is software verification, not a measured task success rate.

## Remaining work and boundaries

No unseen final-test images have been collected: the last checked capture journal still contains the original 30 images and six specimens. Present new known-good and known-defective blocks in the inspection region, with the relevant surface visible, after coordination with the robot owner. Preserve specimen IDs and operator labels. Do not tune the frozen model using final-test results.

Run future evaluation on a separate copy because evaluation adds metadata; the deployed artifact must remain immutable. The current pilot is suitable for experimental testing, not autonomous good-part acceptance. All robot motion and physical setup remain with the controller owner. The capture page on port 8091 and Mac mirror remain separate from the application.

Reproduction and interface commands are documented in `docs/ANOMALY_COMPONENT.md` and `docs/CV_CAPTURE.md`. H100 training and return evidence is maintained by the H100 task. No further model tuning is planned before unseen testing.

---

## Historical progress notes

The entries below describe earlier states. Statements that fitting or task data were pending are superseded by the current status above.

# LEGO detector status

Updated September 15, 2026, 16:13 PDT. Owner: CV task.

## Verified

- Intel host `NUC16GDKX76`; Mac owns source. No robot commands or camera reconfiguration.
- Installed Anomalib 2.6.0, OpenVINO 2026.3.0, Torch 2.10.0+xpu. Existing direct PatchCore implementation reused.
- Bounded checkpoint inspection of project models, Hugging Face cache, and Downloads found only the existing engineering smoke detector. The cached Wide ResNet is a generic backbone, not a LEGO detector.
- Two fresh Studio shared-memory frames saved through attach-only subscribers. All labels remain unknown, and these scouting images are excluded from task training/evaluation.
- Camera 1 clips the foreground blocks. Camera 2 sees seven blocks at the lower-right; some blocks show dark markings. Their meaning is unconfirmed.
- Full-frame 128-pixel resizing loses important visual detail. The saved crop comparison preserves the markings at 128/256 pixels. This is visibility scouting, not confirmation that the actual defect is visible.
- Existing engineering-smoke PatchCore ran on a fresh camera-2 image on Intel. Torch and OpenVINO CPU/GPU FP32 score and map parity passed at atol=0.1, rtol=0.01. Maximum map errors: CPU 0.0000687, GPU 0.0000839. Threshold remains null and decision UNKNOWN.

## Measured model-call timing

Same preprocessed input, batch 1, three warmups, ten timed calls. Excludes preprocessing and capture. Concurrent venue workloads were not controlled.

| Backend | Median ms | Compile/load ms |
|---|---:|---:|
| Torch CPU, four threads | 31.98 | 3770.18 |
| OpenVINO CPU FP32 | 13.34 | 110.80 |
| OpenVINO GPU.0 FP32 | 5.59 | 226.22 |

Raw evidence: `artifacts/cv/scout/openvino-parity.json`, `camera1/2.png`, corresponding JSON provenance, `preprocessing-comparison.png`. Remote copy: `/home/ird-demo/second-look/artifacts/cv/scout-20260915`.

## Design decision

A fixed single-part inspection region requires consistent presentation but avoids an additional object detector and preserves defect pixels. Detecting and cropping every block in a pile supports freer placement but requires separate localization and occlusion validation. Start with a fixed region only if the permitted task layout supports it. Freeze crop and preprocessing before collecting final evaluation data. Do not train the current seven-block scene as a known-good example.

## Physical dependencies

The Demonstrations task owns physical requests and all robot/recording control. It is obtaining the exact defect definition, known specimen labels, and allowed presentation. No parallel physical request is pending from this task.

Need normal-only training specimens, held-out validation normal/defective specimens, and separate final-test normal/defective specimens. Preserve specimen and capture-session groups. If only a small set exists, report that limit; do not disguise neighboring frames as independent specimens.

## Remaining work

Capture labeled examples; check actual defect visibility after the frozen crop/resize; fit normal references; calibrate on validation only; freeze and evaluate final test; compare Torch/OpenVINO decisions; deliver the resulting artifact to the application owner. Until then no measured missed-defect rate, good-part rejection rate, or task-ready classifier exists.

## Completed component handoff — 16:20 PDT

- `scripts/cv_capture.py`: bounded attach-only lossless capture, exact publisher metadata, quality statistics, crop/resize previews, durable capture journal.
- `scripts/cv_manifest.py`: merges complete captures without changing labels or split assignments; rejects content/specimen/session leakage.
- `secondlook/anomaly.py` and CLI: fixed ROI, normal-only task fitting, validation-only frozen calibration, separate final-test evaluation, raw score distributions, errors and uncertainty counts, Torch/OpenVINO score/map/decision parity.
- `secondlook/cv_observation.py`: frame association and invalid-input checks. Unknown physical predicates remain UNVERIFIED. Explicit absent/occluded/unviewable observations skip inference.
- All 34 focused tests passed on Intel with its supplied environment. Tests use synthetic fixtures/mock scores for split and threshold semantics; they do not measure LEGO accuracy.
- New capture CLI passed on the existing Camera 2 publisher. Fresh capture-to-wrapper-to-real-OpenVINO-CPU inference passed. Result: frame validity VALID, observation validity UNVERIFIED, anomaly score 89.14217, decision UNCERTAIN. The artifact remains the prior engineering-smoke model.
- A separate synthetic black-frame check returned INVALID with no model inference. The unchanged model was not fitted again.
- Full live interface evidence is under `artifacts/cv/interface-live-check/` on both Mac and Intel. `new-interface-parity.json` records score, map, and UNKNOWN-decision parity on CPU/GPU FP32 with the revised code.
- Component source staged at `/home/ird-demo/second-look/cv-component-source`. Application owner integrates and deploys; this task did not restart the app, camera publisher, or robot session.
- Run instructions: `docs/CV_CAPTURE.md` and `docs/ANOMALY_COMPONENT.md`.

No task detector artifact, threshold, final-test accuracy, or validated task-specific object/occlusion filter exists yet. These require the pending physical labels and allowed inspection setup. No CV-owned process remains running.

## User capture browser — 16:39 PDT

User requested Camera 1 Low shutter and automatic Mac copies. Live page: http://127.0.0.1:8091 . Dedicated Intel service `secondlook-cv-shutter-live.service` subscribes attach-only to `physicalai/camera/UVCCamera/4/frame`. Source is `scripts/cv_capture_server.py` and `secondlook/capture_static/`. Original 1920x1080 RGB PNGs plus explicit operator labels, stable specimen IDs, frame timestamps, hashes, and unassigned split provenance save under `/home/ird-demo/second-look/artifacts/cv/lego-capture-20260915`.

Mac mirror PID33929 runs `scripts/cv_capture_mirror.py`; tunnel PID32892 forwards8091. Copies land in `/Users/ohong/dev/intel-robotics/artifacts/cv/lego-capture-20260915`. The browser displays hash-verified Mac copy count. Mirror log/runtime records are under `artifacts/cv`. All processes are capture/storage only.

Three Intel server tests passed. Browser shutter saved one Unknown test frame in a separate `shutter-ui-check` collection; mirror verified its SHA256 and browser showed Mac copies1. This test is excluded from user data. Production opened at0 captures with a working live feed and Mac sync. No new model fit or robot command occurred. Keep collection session identity intact; assign specimen-based splits after capture and disclose shared-session limits.
