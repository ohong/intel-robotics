# Anomaly component

`secondlook/anomaly.py` uses the installed Anomalib `PatchcoreModel`. Its pretrained
Wide ResNet backbone extracts reference embeddings. Anomalib's k-center coreset
selects the memory bank. The model exports through `openvino.convert_model`.
Both the score and localization map come from the actual model.

We use the direct tensor API instead of Lightning `Engine`. Engine provides more
training orchestration, but adds datamodule and trainer lifecycle requirements.
The direct API fits this bounded reference-bank model and makes split handling
explicit. There is no replacement model, scripted score, or fake inference.

## Python contract

```python
from secondlook.anomaly import AnomalyDetector
model = AnomalyDetector('/path/to/artifact', device='CPU')
result = model.infer(rgb_uint8_image)  # RGB, not OpenCV BGR
```

`infer` also accepts an image path. It returns `score`, `threshold`, `disposition`,
`anomaly_map`, `latency_ms`, `model_latency_ms`, `backend`, `device`, `artifact_id`,
and `evidence_kind`. It also returns `uncertainty_band`, `decision_reason`, and
`localization_geometry`. The map is a two-dimensional NumPy array at model input size.
Resize it into `localization_geometry.roi_pixels`, not across the full frame.
The pixel box is `[left, top, right, bottom]`, with exclusive right/bottom bounds.
`image_size` and `map_size` use `[width, height]`. The default ROI covers the full image. A raw
score is a distance, **not a calibrated probability**. Larger values indicate
more difference from the reference bank. No ground-truth label enters inference.

An unset threshold always returns `UNKNOWN`. Engineering smoke artifacts can
never be calibrated. Task artifacts need authoritative normal training examples,
validation examples before assigning a disposition. `calibrate` selects a threshold
from validation images only, then freezes that threshold and its uncertainty band.
`evaluate` separately consumes final-test images with the frozen settings. Neither
calibration nor evaluation changes the normal memory bank. Evaluated means measured;
it does not certify adequate task quality or successful export parity.

Calibration maximizes validation F1. Equal F1 prefers the widest observed score gap,
then the higher threshold. The default uncertainty half-width is 5% of the validation
score range. Scores inside the closed band return `UNKNOWN`. This is an explicit
abstention heuristic, not confidence calibration. An unset threshold also returns
`UNKNOWN`, with `decision_reason: "uncalibrated"`.

Observation validity is a separate application boundary. An anomaly distance cannot
prove that a part is visible, isolated, in focus, or correctly presented. The capture
and application wrapper handles structural/freshness checks and preserves unverified
physical predicates. Invalid observations must not become defect classifications.

## Manifest

Use one JSON file containing training and validation splits before fitting.
Include existing reserved tests, or append newly collected final-test specimens later:

```json
{
  "schema_version": 1,
  "images": [
    {
      "path": "images/part-001.jpg",
      "role": "train",
      "specimen_id": "part-001",
      "session_id": "train-session-001",
      "label": "normal",
      "ground_truth": {
        "source": "Actual operator inspection record or authoritative dataset record",
        "authoritative": true
      }
    }
  ]
}
```

Roles are `train`, `validation`, and `test`. Labels are `normal`, `anomalous`, and
`unknown`. All views of the same physical specimen must share a specimen ID.
A specimen cannot cross splits. If any row supplies `session_id`, every row must
supply it, and sessions cannot cross splits. Missing session records remain supported
for existing artifacts, but reports explicitly state the independence limitation.
Duplicate image bytes are rejected, including
renamed copies. Content checks cannot detect mislabeled specimen identities or
near-duplicate images; correct specimen records are still required.

### Explicit single-session pilot

The default `split_policy` is `"specimen_and_session"`. For a pilot with one actual
collection session, explicitly add these top-level fields before fitting:

```json
{
  "split_policy": "specimen_only_pilot",
  "split_policy_reason": "One actual collection session; separate specimens reserved for pilot validation"
}
```

This policy requires a true `session_id` on every row. It allows that session to
cross splits while still rejecting specimen overlap and duplicate image content.
Reports list shared sessions and state that session-independent generalization is
unmeasured. Never invent separate session IDs to bypass the default guard.

Reserve some known-good specimens for validation. Fitting all known-good specimens
would leave no independent normal examples for measuring rejected-good errors.
The specimen-only pilot preserves that measurement, with the shared-session limitation.
The policy and its reason cannot change after fitting.

For a future final test, keep every original row unchanged and append only `test`
rows to a combined manifest. Paths may relocate; image hashes, labels, specimen and
session IDs, split roles, and ground-truth provenance must match. New test specimens
and image contents must be disjoint from training and validation. Under the default
policy, test sessions must also be disjoint. Calibration must finish before evaluation.
Evaluation records the complete extended manifest and frozen calibration identity.
Neither appending test data nor evaluating it modifies the fitted memory bank.

Smoke manifests use `label: "unknown"`, `authoritative: false`, and a source
explaining the capture. Smoke is explicit through `--engineering-smoke`.
Known anomalous images cannot become training references even in smoke mode.
The manifest, image hashes, model hashes, configuration, and installed package
versions are recorded in `metadata.json`. Image paths resolve relative to the
manifest. Preserve the complete dataset paths during calibration.

## Commands

Run with the Intel environment Python; do not install into sponsor environments.

```sh
python scripts/anomaly_cli.py validate --manifest data.json
python scripts/anomaly_cli.py fit --manifest data.json --output artifacts/model
python scripts/anomaly_cli.py calibrate --manifest data.json --artifact artifacts/model
python scripts/anomaly_cli.py evaluate --manifest data.json --artifact artifacts/model
python scripts/anomaly_cli.py infer --artifact artifacts/model --image sample.jpg --map-output map.npy
python scripts/anomaly_cli.py benchmark --artifact artifacts/model --image sample.jpg --devices CPU --output benchmark.json
```

Fitting defaults to 128 square pixels, a 2% coreset, one nearest neighbor, four
Torch CPU threads, and a fixed seed. The image limit is 64 by default. These
bounded defaults establish a runnable route; they do not establish task accuracy.
`--num-neighbors` selects the installed PatchCore score-weighting neighborhood size
(integer 1 through 20; default 1). Values above one use its weighted-score route.
Preprocessing uses RGB, an optional fixed normalized ROI, PIL bilinear square resizing,
and ImageNet normalization. Supply `fit --roi LEFT TOP RIGHT BOTTOM` to choose a crop.
All coordinates must be in `[0, 1]`; right/bottom must exceed left/top. Pixel rounding
uses floor for left/top and ceil for right/bottom. The crop is recorded in model
configuration and applied consistently for fitting, export, inference, and evaluation.
Choose the crop from physical geometry and training/validation observations before
final testing. Inspect the actual resized crop to confirm the defect is still visible.
The complete preprocessing is identical for Torch and OpenVINO.

### Optional foreground crop

`fit --foreground-crop` tightens the model input around a dominant colored component
inside the fixed ROI. The default remains disabled, preserving existing artifacts.
This addresses background and shadow variation. Color determines the bounding box,
not the defect label. PatchCore still produces every anomaly score and decision.

The stored configuration uses saturation `(max(R,G,B)-min(R,G,B))/max(R,G,B) > 0.25`
and maximum channel value `> 50`. Eight-connected components identify the largest
colored region. It must cover at least 1% of the parent ROI, with both box sides
at least 20 pixels. A second component larger than 25% of the largest is ambiguous.
Missing, tiny, or ambiguous foreground raises `ObservationInvalidError` before inference.
This subclasses `AnomalyContractError`; application validity handling must catch this
specific subclass. Configuration, artifact, and model errors remain runtime faults.

The bounding box receives padding of `ceil(0.08 * longest_side)` pixels, clamped to
the parent ROI. The model receives every original RGB pixel within the resulting
rectangle, including black marks and holes. No foreground mask replaces image pixels.
`localization_geometry.roi_pixels` reports the effective box in full-frame coordinates;
`parent_roi_pixels` retains the fixed ROI. Component geometry is also recorded.

This crop assumes one sufficiently saturated object. Gray/white blocks, occlusion,
multiple blocks, or a black mark that splits the colored region can cause rejection.
Inspect actual training/validation crop examples before choosing it. Compare the
fixed-ROI and foreground-crop models on validation data only, then freeze the choice
before collecting final-test results. The same crop runs during fit, export, inference,
calibration, evaluation, and backend parity checks.

`fit` performs export automatically; no separate export command is needed. It refuses
nonempty output directories. `calibrate --uncertainty-fraction 0.05` selects a band
using validation data only. The fraction must be in `[0, 0.5]`; zero disables the band.
Calibration cannot run twice on one artifact. `evaluate` requires calibration and
cannot run twice either. Preserve its saved report; do not repeatedly tune against
the final test. Any later model selection needs a new experiment and new held-out data.

Calibration and evaluation reports include raw score distributions by label, image
and specimen counts, binary threshold confusion, missed defects, rejected good parts,
and uncertain normal/defective counts. Reports retain per-image paths/hashes and
decisions. Multiple images of one specimen are not independent physical trials.
The evaluation records dataset identity and the exact frozen calibration identity.
Changed original image content or label/split provenance is rejected before inference.
Adding unseen final-test rows after calibration is supported through the combined
manifest described above; adding training or validation rows after fitting is rejected.

Artifacts include `model.pt`, `model.xml`, `model.bin`, and `metadata.json`.
Export stores FP32 weights. Compilation may select a different device precision;
benchmark results report the actual inference precision hint and execution devices.
Model load and compilation timing are separate from warmed inference timing.
Each benchmark checks score, map, and decision parity against Torch under identical
inputs. Final evaluation repeats those comparisons on every held-out image using
OpenVINO CPU FP32. Numeric tolerances do not excuse different decisions. Pass/fail
appears separately for scores, maps, and decisions. Use `evaluate --parity-atol 0.1
--parity-rtol 0.01` to record explicit tolerances. GPU/NPU benchmarks remain separate;
CPU parity does not certify an untested device or precision.
Unsupported devices produce explicit errors. Device availability alone is not a
workload acceleration result. Benchmark outputs never claim defect accuracy.

These tools use image files only. They do not open cameras, robot ports, or motion
sessions. Use the lead agent's resource coordination before running heavy jobs.

## Verified engineering smoke, 15 September 2026

Ran on Intel using Anomalib 2.6.0, OpenVINO 2026.3.0, Torch 2.10.0+xpu,
and two real unlabeled room/table captures. Fitting selected a `[10, 1536]`
reference memory bank. Export and score/map inference succeeded. The threshold
remains unset, and inference returns `UNKNOWN`.

Artifact: `/home/ird-demo/second-look/artifacts/models/scene-smoke`.
Artifact ID: `1e89b319c3f0f6bd60b0ca176d2527febc10d7d873ed8d5f36315adf599e487e`.
Dataset identity excludes image paths, so relocating data preserves it. A separate
`model_id` identifies weights/configuration. Calibration changes `artifact_id`.

Each backend used three warmups and ten timed calls on the same preprocessed
reference image. Torch used four threads; OpenVINO used device default `LATENCY`
settings. These figures measure model execution, excluding preprocessing and
robot work. Compilation is reported separately in the linked JSON.

| Actual execution | Precision | Median ms | Map max absolute error | Parity |
|---|---|---:|---:|---|
| Torch CPU, first comparison | FP32 | 20.044 | 0 | reference |
| OpenVINO CPU | FP32 | 13.193 | 0.02286 | pass |
| OpenVINO GPU.0, default | FP16 | 1.782 | 1.03920 | **fail** |
| OpenVINO NPU | FP16 | 4.096 | 0.13899 | pass |
| OpenVINO GPU.0, explicit FP32 | FP32 | 7.364 | 0.02049 | pass |

Parity requires NumPy `allclose` on both score and every map pixel, with absolute
tolerance 0.1 and relative tolerance 0.01. NPU passes because this combines absolute
and relative tolerance; it does not require a maximum absolute error below 0.1.
These tolerances establish a conversion smoke check, not task-quality acceptance.
The benchmark image is a training reference. No independent defect accuracy or
physical task success was measured. CPU remains the default supported route.

For explicit GPU FP32: `AnomalyDetector(path, device='GPU', precision='f32')` or
`--device GPU --precision f32`. Default GPU FP16 failed this map-parity check.

Detailed evidence: [anomaly-smoke-benchmark.json](anomaly-smoke-benchmark.json).
Remote raw logs and reports: `/home/ird-demo/second-look/component-anomaly/`.
Twelve portable contract tests passed on Mac. Model execution was verified on
Intel; no sponsor environment packages, camera sessions, or robot state changed.

### Two-scene parity follow-up

Both existing real scene captures passed score and every-pixel map comparisons
against Torch on OpenVINO CPU FP32 and GPU FP32. Both are training references,
not held-out observations. No additional latency benchmark was performed.

| Input | CPU maximum map error | GPU FP32 maximum map error |
|---|---:|---:|
| camera-4.jpg | 0.02286 | 0.02049 |
| camera-10.jpg | 0.03069 | 0.04570 |

GPU FP32 is therefore a supported candidate for the app's smoke default, with CPU
as fallback. NPU remains benchmark-only pending task data. The default GPU FP16
failure remains in the original benchmark evidence. All tested dispositions
remain `UNKNOWN`, with null thresholds.

The deployed application selects GPU FP32. CPU remains the CLI default.

To reproduce this fit on the Intel PC without replacing the deployed artifact,
use a new output directory:

```sh
cd /home/ird-demo/second-look/current
/home/ird-demo/miniforge3/envs/intel_dev_env/bin/python scripts/anomaly_cli.py fit \
  --manifest /home/ird-demo/second-look/component-anomaly/smoke-manifest.json \
  --output /home/ird-demo/second-look/artifacts/models/scene-smoke-reproduction \
  --engineering-smoke
```

The source manifest and both reference images remain on the Intel PC. Copies
also remain in the Mac's ignored `artifacts` directory. Model metadata contains
their hashes. These inputs reproduce the engineering check, not a task dataset.

Evidence: [anomaly-parity-two-scenes.json](anomaly-parity-two-scenes.json).
