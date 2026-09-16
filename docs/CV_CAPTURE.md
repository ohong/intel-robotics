# Capture and observation contract

## Scope

The capture tool reads an existing Intel Physical AI Studio publisher. It never creates or configures a camera.
It uses Studio Python and `SharedCamera.from_publisher(..., overwrite_settings=False, zero_copy=False)`.
It imports no robot API. Studio must already own the physical camera.

Two approaches were considered: saving JPEGs from the app, or reading the Studio publisher directly.
Direct subscription preserves lossless pixels, original timestamps, and publisher metadata. It requires Studio's Python environment.

## Bounded capture on Intel

Run from the project root on Intel. Replace session and specimen identifiers with observed provenance.
These names are identifiers, not defect labels. Keep one physical specimen per invocation.

```sh
python3 scripts/cv_capture.py \
  --service physicalai/camera/RealSenseCamera/243622060187/frame \
  --output artifacts/cv/scouting-session-01/specimen-01 \
  --session-id scouting-session-01 --specimen-id specimen-01 \
  --role train --label unknown \
  --ground-truth-source 'Scouting capture; defect ground truth not established' \
  --count 12 --duration 30 --interval 1
```

The launcher starts `/home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python`.
Use `--studio-python` only when the installed Studio environment differs. No environment installation occurs.
The output directory must not exist. Capture stops at the count or deadline.
Failure returns exit code 2 and preserves a partial manifest when cleanup can run.
A journal, `captures.jsonl`, records each saved frame immediately. A forced termination may leave only this journal and saved images.
The launcher also enforces a process timeout covering duration, connection timeout, and ten seconds of cleanup.

Each capture stores:

- A lossless full RGB PNG with SHA-256.
- Exact publisher sequence and nanosecond timestamp, plus the original floating-point timestamp.
- Camera service, session, specimen, split role, explicit label, and ground-truth source and authority.
- Raw, cropped, and resized previews, plus a labeled contact sheet.
- Descriptive brightness, clipping, spatial variation, and neighbor-difference statistics.

`capture_timestamp` uses the publisher's monotonic clock. `saved_at_utc` is save time, not capture time.
The subscriber validates header/frame timestamp equality, sequence, dimensions, color encoding, dtype, and freshness.
Repeated frames are skipped; regressed or inconsistent timestamp/sequence pairs fail.

`--roi L T R B` configures normalized preview bounds. Left/top round down; right/bottom round up.
`--preview-size 256` controls the square resized preview. These match detector geometry and resize conventions.
The manifest always references full RGB source images. The detector owns the inference crop, so pixels are never cropped twice.
Default previews show the complete image. A preview ROI is not automatically a validated inspection area.

## Labels and splits

Use `unknown` and a truthful source while scouting. Never infer a label from an anomaly score.
Use `--authoritative` only with actual authoritative label evidence and its source.
Known anomalous samples cannot enter the training reference bank.

Assign all observations of the same specimen to one role. Assign a capture session to one role.
Use separate sessions and specimens for validation and test. Neighboring frames do not establish independent generalization.

Merge complete per-specimen captures before fitting:

```sh
python3 scripts/cv_manifest.py merge \
  --inputs artifacts/cv/scouting-session-01/specimen-01/manifest.json \
           artifacts/cv/scouting-session-01/specimen-02/manifest.json \
  --output artifacts/cv/scouting-manifest.json --engineering-smoke
```

The merge preserves labels and provenance, rejects incomplete captures and existing output, and validates cross-capture leakage.
The validator checks specimen, session, and image-content overlap. Unknown labels require explicit `--engineering-smoke`.
Paths remain relative to the merged manifest. Move the complete dataset tree together to preserve them.
A combined task manifest requires a training split and authoritative ground truth. A standalone held-out capture is not a complete training manifest.

## Observation interface

```python
from secondlook.cv_observation import observe_frame
result = observe_frame(detector, full_rgb_uint8, frame_id, captured_at,
                       now=local_monotonic_time, max_age=1.0)
```

The wrapper preserves frame identity and capture time. It rejects malformed, stale, future, nonfinite, near-black, and spatially uniform frames.
Near-black means every channel is at most 2. Uniform means every pixel has the same RGB value.
These conservative signal checks cannot confirm an object or visible defect.

| Field | Meaning |
| --- | --- |
| `frame_id`, `observation_id` | Same original observation identity |
| `captured_at`, `capture_timestamp` | Same original monotonic capture timestamp |
| `frame_validity` | `VALID` or `INVALID` pixel/identity/freshness checks |
| `observation_validity` | `INVALID`, `UNVERIFIED`, or `VALID`, including physical evidence |
| `quality_flags`, `quality_stats` | Reasons and descriptive measurements |
| `object_presence`, `occlusion`, `defect_visibility` | `{status, source}`; default `UNVERIFIED` |
| `anomaly_status`, `disposition` | Model disposition; invalid input uses `NOT_EVALUATED` / `UNKNOWN` |
| `anomaly_score`, `score` | Same raw score, or null when inference is skipped |
| `decision` | `GOOD`, `DEFECT`, `UNCERTAIN`, or `INVALID` |
| `anomaly_map`, `localization_geometry` | Detector map and ROI geometry, or null |
| `model_version`, `artifact_id`, `model_artifact_id` | Loaded artifact identity |
| `inference_ms`, `latency_ms`, `model_latency_ms`, `processing_ms` | Detector and wrapper timings in milliseconds |
| `uncertainty_band`, `decision_reason` | Detector threshold policy and reason when available |
| `inference_executed` | Whether this observation entered the model |

`GOOD` and `DEFECT` describe model score decisions. They do not establish physical object validity or authorize action.
A valid pixel frame with unchecked physical predicates has `observation_validity=UNVERIFIED`.
An invalid observation never enters inference. Its score and map remain null; loaded artifact identity remains available.
Model errors raise clearly rather than producing an invented result.

Explicit validated evidence requires matching `frame_id`, `validated=true`, and a nonempty `source`.
Accepted values are `PRESENT`/`ABSENT`, `CLEAR`/`OCCLUDED`, and `VISIBLE`/`NOT_VISIBLE`, respectively.
Absent objects, occluded objects, or invisible inspection features make the observation invalid and skip inference.
This evidence does not include defect ground-truth labels. The wrapper passes pixels alone to `detector.infer`.

## Verification

```sh
uv run --no-project --with numpy --with pillow python -m unittest \
  tests/test_cv_capture.py tests/test_cv_manifest.py -v
```

Twelve focused tests passed on Mac using synthetic image/byte fixtures.
They cover invalid-input inference blocking, metadata association, explicit physical evidence, exact ROI rounding,
lossless RGB saving, provenance, no overwrite, and manifest merge leakage checks.
These tests establish software behavior only. Live capture, task visibility, and task defect quality require separate evidence.
