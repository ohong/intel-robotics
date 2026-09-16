# Native Studio SmolVLA pilot

This runner fine-tunes the actual pretrained vision-language-action model. Its
outputs remain experimental. Native checkpoint validation does not establish
Intel deployment parity, correct physical sorting, or controller readiness.

Two loading paths were compared. Studio's pretrained constructor loads base
statistics before replacing them in `setup()`. The chosen route builds native
SmolVLA with explicit training statistics, then loads every active pretrained
weight. This avoids the pretrained statistics fallback. Four inactive legacy
SnapFlow parameters may be absent; SnapFlow stays disabled. Any other missing or
unexpected parameter fails before fitting. The backbone weights are already
inside the policy checkpoint, so `load_vlm_weights=false` avoids a second download.

The runner reuses the exercised ACT loader and capture-contract helpers, but
never substitutes ACT for SmolVLA. It acquires the same `.gpu-job.lock` and rejects
an occupied H100. It imports no robot or camera control implementation.

## Required inputs

Fill `smolvla-config.template.json` from inspected evidence. Nulls are intentional
blockers. `base_files` and `backbone_files` map every relative filename to
`{"bytes": number, "sha256": "..."}`. Model symlinks may resolve only within the
owned `/workspace/second-look-h100` directory.

The frozen recording manifest must have `OFFLINE_DATA_VALIDATED`, real provenance,
closed files, observed outcome evidence, exact file hashes, camera keys, capture
configuration, and explicit whole-episode train/validation lists. Statistics come
from training episodes only. No final-evaluation loader is constructed. Explicit
`training_scope="developmental_observed_transfer"` also accepts evidenced
`observed_transfer` outcomes without relabeling them as task successes.
`loader_smoke` splits remain experimental and do not establish generalization.

For the integrated pilot, select `conditioning.mode="anomalib_rows"`. The annotation
manifest must contain:

```text
schema_version: 1
artifact_id: frozen detector identity
camera_key: exact observation.images.* key
threshold: frozen finite number
rgb_hash_format: uint8_HWC_RGB_contiguous
dataset_sha256: SHA256 of canonical sorted compact JSON of recording manifest files
rows:
  index, episode_index, frame_index, eligible,
  image_sha256, score, decision, reason (required when excluded)
```

Alternatively, `snapshot_receipt_sha256` may replace `dataset_sha256` when both
annotation and conditioning config pin that receipt hash. The config also pins
the complete annotation manifest SHA-256, artifact ID, and threshold.

A row's `index` is the original global LeRobot row index. Scores and source image
hashes must come from the actual frozen detector and actual decoded frames.
`NORMAL`/`ANOMALOUS` and lowercase equivalents are accepted; equality with the
threshold means anomalous. Unknown/invalid detections cannot start action chunks.
Rows with `eligible=false` remain part of future action targets. The start-window
filter never truncates the underlying episode or renumbers its future actions.
Every train/validation episode must retain at least one eligible start. Pixel-valid
scores are insufficient for fitting: the annotation must declare
`conditioning_verified=true` and `target_association="VERIFIED"`. Eligible rows
may not override those with an unverified association. A detector crop containing
unrelated stationary pieces does not establish the manipulated object's condition.

The generic mission instruction comes from the config. The exact conditioning is:

```text
{instruction}
Inspection: {lowercase_decision}; score={score:.6g}; threshold={threshold:.6g}.
```

There is a final newline. Recorded class captions remain QA metadata. They never
enter this conditioning branch. An optional annotation `text` must match exactly.
Every loaded eligible frame is checked against its RGB content hash. The runner
rejects token truncation. Explicit `recorded_task` and evidenced `episode_task`
branches remain available for separate experimental imitation; they cannot claim
Anomalib integration and reject identical captions for conflicting destinations.

## Execution

Use the prepared training interpreter and stage both `training_smolvla.py` and
its existing `training_act.py` helper beside it. Run a new smoke output first:

```sh
python training_smolvla.py --config actual-config.json --mode decode-check --decode-output decoder-parity.json
python training_smolvla.py --config actual-config.json --mode check
python training_smolvla.py --config actual-smoke-config.json --mode smoke
python training_smolvla.py --config actual-train-config.json --mode train
```

The no-model `decode-check` compares native H100 loader RGB hashes against the
first, middle, and last annotated frames of each train/validation episode. It
requires snapshot and annotation integrity but skips outcome and object-association
fitting gates. A decoder pass cannot unlock unknown physical association.

Smoke performs one optimizer update and one held-out validation minibatch. Training
uses all eligible starts from the validation episodes. Validation seeds the
flow-matching noise/time sequence identically each epoch and restores training RNG
state afterward. Training retains random initial flow noise. Every compared
prediction uses the same separate forked CPU/CUDA seed, including the caption
sensitivity check; changed sampling noise cannot masquerade as language use. `val/loss` selects one best checkpoint. `last.ckpt` and recovery
checkpoints retain optimizer/scheduler state. A time bound can stop before any
validation; that run fails selection and retains recovery evidence only.

Resume with `--mode train --resume /absolute/path/to/last.ckpt`. The data,
annotation, camera mapping, learning-rate schedule, package freeze, and source
identity must match. Only `max_steps`, `max_seconds`, and output location are
excluded from the identity hash; checkpoints must still reside in that output.
The wall-clock bound applies cooperatively during fit. Reload checks add work.

The runner compares native predictions before and after saving `last.ckpt`, checks
finite selected-checkpoint actions, and changes task text while holding real
images/state fixed. A nonzero caption effect proves functional language use. It
does not prove a correct decision, placement, or detector quality. Both readiness
flags remain false. The result records exact selected/recovery hashes and leaves
final-evaluation episodes untouched.

## Local validation

The portable tests cover active-weight completeness, inactive-key restrictions,
episode leakage, annotation identity, camera identity, detector threshold equality,
canonical task text, conflicting captions, excluded rows, and finite scores.
They do not execute CUDA, decode real episodes, or establish training success.


## Bounded real-data pipeline probe

`training_scope="real_data_pipeline_probe"` accepts the frozen exploratory
recording owner's manifest directly. It preserves `visual_transfer_observed`,
unknown task success, observed destinations, recorded captions, nullable specimen
and session identity, and the stated timing limits. It does not fabricate the
older ACT success/capture contract. The config must pin `manifest_sha256`.

This scope permits only `conditioning.mode="recorded_task"`, with at most twenty
total optimizer updates. One smoke update is sufficient for the requested software
path check. The loader must reproduce each manifest's original `task_caption`.
Frozen `robot_contract` and `camera_identity_mapping` supply source-backed metadata;
no live capture manifest is required. Dictionary-form train-only normalization
must match the exact train IDs, row count, and excluded held-out episodes.

Its result status is always `REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE`. This is an
experimental software/gradient/checkpoint result on real recorded inputs. Recorded
“good” and “defective” captions are not verified defect labels. It does not bypass
the separate detector-to-manipulated-object association gate.


### Same-device checkpoint comparison

Lightning teardown can move the fitted policy to CPU. Equal seeds do not align
CPU and CUDA Gaussian samples. The runner explicitly moves the fitted reference
and each reload to CUDA device zero before comparison, checks the observation's
device, and gives each call a fresh observation copy. It preserves the original
strict output tolerances. `run.json` saves the post-fit device, completed step,
gradient evidence, and recovery-checkpoint hash before prediction checks, so a
verification failure does not erase the completed-update evidence.
