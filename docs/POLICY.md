# Physical AI Studio policy bridge

## Current result

**Genuine pretrained VLA inference executed on the Intel CPU. Task execution remains BLOCKED.**

At **16:45 PDT**, a recorded image passed through the real Anomalib/OpenVINO
scene-smoke detector and then native Studio `SmolVLA.select_action` on
`NUC16GDKX76`. The detector returned score **79.97917175292969**, disposition
**UNKNOWN**, and **null threshold**. The VLA consumed that exact inspection
metadata and the same image, verified by SHA-256. It returned a finite six-value
vector. Detector model inference took **13.213 ms** on GPU.0 FP32; VLA inference
took **4.239 seconds** on two CPU threads. Each stage ran once.

State and normalization statistics remained explicit synthetic fixtures.
Camera-role and physical-action semantics remain unverified. The result declares
`ENGINEERING_PROBE_ONLY`, `controller_ready=false`, and `task_ready=false`.
No robot command was sent. This verifies the actual Anomalib-to-language-to-VLA
data path. It does not prove defect understanding, calibrated classification,
policy quality, or physical sorting.

The earlier **16:40 PDT** probe used synthetic anomaly metadata. Its result is
preserved separately; it is not relabeled as the combined replay.

Evidence: [combined replay](../artifacts/policy/smolvla-same-image-inference.json),
[real detector result](../artifacts/policy/smolvla-same-image-anomaly.json),
[initial synthetic-metadata probe](../artifacts/policy/smolvla-engineering-inference.json),
[download manifest](../artifacts/policy/smolvla-downloads.json), and
[architecture check](../artifacts/policy/architecture-check.json).
These local evidence artifacts are intentionally ignored by Git.

The adapter uses the genuine installed `physicalai.policies.smolvla.SmolVLA`
class. A valid artifact takes the supported `select_action(Observation)` route.
There is no scripted action fallback. Missing artifacts produce `BLOCKED`.
The adapter never opens a robot, camera, serial port, or gym.

The installed Studio revision is `c4ff730fb49f84e5102d01088d52cfff1ba62854`.
Its Python environment contains `physicalai 0.1.2.dev70+g8e4021703`,
`physicalai-train 0.1.0`, and `torch 2.11.0+xpu`.

### Initial base inspection and remaining task limits

We inspected [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
at revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
Its card describes a base model intended for task-specific fine tuning.
The initial inspection downloaded only small configuration and normalization
files. It did not execute a model. A later, separate engineering probe downloaded
the full 906,712,520-byte pinned policy and executed it with explicitly artificial
fixture normalization. The original task-normalization blockers remain unchanged.

The checkpoint has six state and action dimensions. Its image features are
`observation.images.camera1`, `camera2`, and `camera3`.
The normalizer files contain three embodiment-specific action statistics:
`so100`, `so100-blue`, and `so100-red`. They contain **no state statistics**.
Several action means exceed 100. The observed follower reports five joint values
in `range_m100_100` and gripper values in `range_0_100`.
Six matching dimensions do not establish compatible joint order or units.

The installed loader can substitute identity statistics for missing state
statistics. It can select the first matching action statistics.
Our worker rejects missing or ambiguous saved statistics before loading weights.
It also checks the statistics actually resolved by Studio against the manifest.

Remote source evidence lives under
`/home/ird-demo/second-look/artifacts/policy/base-inspection/`.
The contract inspection records SHA-256 hashes for every downloaded file.

## Architecture decision

Two supported routes were considered:

| Route | Benefit | Cost |
|---|---|---|
| Studio backend and REST service | Existing application orchestration | More services and potential hardware-manager ownership |
| Installed Studio policy library | Direct supported inference API; isolated dependencies | Second Look must validate the model contract |

We selected the library route. It keeps the application in `intel_dev_env` and
runs Studio imports in the existing Studio interpreter. Neither environment is modified.

```text
App in intel_dev_env
  task + Anomalib result + image + measured state
                |
        validated JSON request
                v
Studio .venv -> SmolVLA -> action candidate
                |
        validated JSON response
                v
        evidence / operator view
        controller_ready = false
```

The worker runs offline. It uses local, hashed checkpoint and backbone files.
Its stdout contains one JSON object; model logs go to stderr.
Each request has a subprocess timeout. A timeout returns no candidate.
This first bridge reloads the model per request. It is suitable for contract
verification and replay; sustained control latency remains unverified.
A persistent worker should follow measured need and a valid task policy.

## Python interface

```python
from secondlook.policy import StudioPolicyBridge, PolicyContractError

bridge = StudioPolicyBridge(
    python="/home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python",
    manifest_path="/home/ird-demo/second-look/artifacts/policy/manifest.json",
    timeout_seconds=60,
)
status = bridge.status()
# Only after a complete, verified artifact exists:
# candidate = bridge.infer(observation, mode="replay")
```

`status()` only validates the manifest schema. It does not report model readiness.
`infer()` raises `PolicyContractError` when its contract fails.
An output candidate includes `model_inference_executed`, `controller_ready`,
artifact and observation identities, action names/units, conditioning text,
execution device, model load time, and inference time.
`controller_ready` is always false. A separate supervised controller must check
state freshness, motion bounds, command ownership, and the approved stop behavior.

## Input contract

All timestamps are Unix seconds from the Intel PC. Replay uses original timestamps
and checks sensor association/skew; live also checks age before and after inference.
The application must preserve real timestamps, not restamp old images as current.

| Field | Requirement |
|---|---|
| `schema_version` | Integer 1 |
| `observation_id`, `task_id`, `task` | Explicit association and instruction; task ID matches manifest |
| `provenance` | `real` or `recorded_real`; synthetic state is rejected by the task interface |
| `calibration_id` | Matches manifest; use inspected calibration hash |
| `captured_at` | Observation time |
| `state` | `names`, `units`, finite `values`, `captured_at`; exact manifest ordering |
| `images` | Exact camera-key set; each has absolute `path`, SHA-256, `source_id`, `shape`, `color`, `captured_at` |
| `anomaly` | Same observation ID; finite score/threshold, decision, model ID, real inference evidence, capture time, and exact `source_images` map |

The manifest declares `anomaly_camera_keys`: the nonempty, unique subset of
configured cameras actually consumed by the detector. `anomaly.source_images`
must contain exactly those keys. Each entry must contain exactly `source_id` and
`sha256`, matching that camera's input metadata. The worker also hashes the actual
image file before constructing the policy input. An unchanged observation ID or
timestamp cannot authorize swapping the inspected image for another image.
A detector using one camera must not claim that it inspected every VLA camera.

Example association for a detector using the `top` camera:

```json
{
  "source_images": {
    "top": {"source_id": "configured-physical-camera-id", "sha256": "64-character-image-sha256"}
  }
}
```

Image `shape` is `[height,width,3]`. Color is explicitly RGB.
The worker decodes and hashes the actual file. It produces float32 tensors
`[1,3,height,width]` in `[0,1]`; Studio performs padded resize and visual normalization.
State is float32 `[1,joint_count]` in the dataset's verified units.
Task is a one-element string list in `Observation`.

The exact conditioning text is:

```text
{task}
Inspection: {decision}; score={score}; threshold={threshold}.
```

Anomalib results therefore enter the actual language input. This does not prove
that a base checkpoint understands defects or uses those results correctly.
The worker checks tokenizer length before inference and refuses truncation.
Scores remain raw model scores, not calibrated probabilities.

`StudioPolicyBridge.infer()` accepts only `replay` and `live`.
The generic JSON worker also rejects `engineering_probe` before importing models.
Synthetic state and anomaly metadata are rejected by both task modes.
`replay` requires recorded real state and actual detector output.
`live` requires real synchronized inputs, a task-adapted manifest, and pinned task
validation evidence. A base checkpoint can never enter live inference.

## Artifact manifest

The manifest pins Studio commit, installed package versions, checkpoint revision,
checkpoint/backbone directories, and every local artifact's SHA-256.
Checkpoint `vlm_model_name` must reference the pinned local backbone directory.
Changing that path changes `config.json`; preserve the upstream config separately
and hash the final local file. The worker does not fetch missing assets.

The manifest must also declare:

- State and action joint ordering, units, and absolute-position semantics.
- Dataset-derived state/action mean and standard deviation, with no default values.
- Exact camera names, physical source identity, source shape, and complete slot ordering.
- Explicit `anomaly_camera_keys` naming the detector input cameras; no implied camera associations.
- Actual calibration identity, task identity, validation evidence, and adaptation status.
- Chosen freshness/skew bounds, tokenizer limit, and local CPU or XPU device.

The current scaffold deliberately leaves unknown fields `null`. It fails validation.
No numeric coordinates, model statistics, action mapping, or camera role are invented.
No fitting entrypoint is provided without a defined task and actual demonstrations.
Supported Studio training remains available once those inputs exist.

## Non-motion verification

The lead deploys the files. Run this in the installed Studio interpreter:

```sh
/home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python \
  /home/ird-demo/second-look/current/scripts/policy_worker.py --inspect \
  --checkpoint-inspection /home/ird-demo/second-look/artifacts/policy/base-inspection \
  --state /home/ird-demo/second-look/artifacts/discovery/follower-state.json \
  --scaffold /home/ird-demo/second-look/artifacts/policy/manifest.scaffold.json
```

This imports genuine Studio classes, constructs an `Observation` from saved real
state, verifies camera slot mapping/rejection, and reads saved normalizer tensors.
It does not instantiate SmolVLA weights or call `select_action`.
The recorded state and earlier camera image are not synchronized.
This check reports `model_inference_executed=false` and remains `BLOCKED`.

Portable tests:

```sh
python3 -m unittest discover -s tests -p test_policy_contract.py -v
```

These tests use synthetic fixtures. They cover state mapping, camera identity,
normalization, stale/future/skewed inputs, base-policy live rejection, provenance,
observation association, timeouts, malformed actions, and artifact path containment.
They do not establish policy quality or physical success.

### Initial contract verification — September 15, 2026

The exact `inspect_contract` function ran through the installed Studio interpreter
using read-only SSH execution. It returned exit code 0.

- Genuine `SmolVLA`, `Observation`, and `SmolVLAPreprocessor` imports passed.
- Saved real state produced a float32 observation with shape `[1,6]`.
- The installed preprocessor resolved `images.camera1`, `camera2`, and `camera3` in order.
- The installed preprocessor rejected a mismatched camera key.
- Actual saved tensors confirmed absent state statistics and three action-statistic groups.
- Model inference remained unexecuted; the result remained `BLOCKED`.

Studio emitted existing Qwen docstring and decorator deprecation warnings during
import. They did not stop the check. The complete worker redirects import logs
to stderr to preserve its stdout JSON protocol.

Artifact hashes from that check:

| Artifact | SHA-256 |
|---|---|
| `config.json` | `650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d` |
| `policy_preprocessor.json` | `7683d648280a72f0d51e869fb8dd1596beed90b7f84849330cf1bc26634a4bd6` |
| Normalizer and unnormalizer safetensors | `490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8` |
| `policy_postprocessor.json` | `2b78bb742065288df2ec63b0ee35f97a1f6950171cf2272f7e34b1fe3873b17b` |
| Saved `follower-state.json` | `e6c5c4b3fcbb5489f246dbbcd8cf0535eadd18b189bbaac0b226475ed180e30c` |

Fourteen portable policy tests passed during this initial inspection. The later
engineering run verified pretrained model loading and inference. Task conditioning
effectiveness and physical controller integration remain untested.

## Updated task and practical VLA routes

The operator now confirms defective versus non-defective LEGO sorting into two
piles. Recordings and physical operation belong to the separate Demonstrations
task. The H100 task owns ACT training. This policy work does not use that GPU.

| Route | Evidence from installed source | Decision |
|---|---|---|
| Native Studio SmolVLA | Real text tokenizer, image/state inputs, learned continuous actions; supported pretrained loader | Selected for bounded engineering inference and later task adaptation |
| Studio's LeRobot SmolVLA wrapper | Supported named policy; can consume LeRobot model artifacts | Credible alternative if training produces LeRobot artifacts; avoid changing environments now |
| Native Pi0 / Pi0.5 / GR00T | Installed policy implementations, but different model assets and preparation | No demonstrated time or integration benefit for this build |
| ACT plus existing dispatcher | ACT consumes images/state; the dispatcher chooses a skill outside that learned policy | Useful control baseline; does not itself establish language-conditioned VLA compliance |

An ACT skill can be part of a compliant larger architecture if a genuine VLA
interprets task/perception and supplies action intent. Our existing dispatcher
does not prove that connection. A separate SmolVLA engineering probe also does
not make an unrelated ACT physical demonstration an integrated VLA system.

### Task-adapted SmolVLA prerequisites

Demonstrations must preserve camera images, real state, commanded actions, joint
order, units, calibration identity, episode boundaries, timing, and task text.
Record the correct physical response for both defect classes. Preserve failures
and interventions. Do not infer class labels from an unvalidated detector.

Before fitting, resolve the exact camera feature names and semantic slot mapping.
The installed SmolVLA preprocessor supports explicit mapping and masked unused
slots. This supports two real cameras without inventing a third physical view.
Compute state/action mean and standard deviation from training episodes only.
Keep evaluation specimens or episodes separate; neighboring frames are not
independent validation samples.

The training task text must use the same conditioning format as deployment,
including the relevant perception result. Merely supplying inspection text to a
base model at inference does not train this behavior. During live operation,
perception comes from actual Anomalib output; evaluation labels stay outside
that inference path.

The installed native SmolVLA `setup()` replaces pretrained normalization with
`train_dataset.stats` when fine tuning an initialized model. The shipped
`library/configs/physicalai/smolvla.yaml` uses `LeRobotDataModule`. Its PushT
example is a template, not the LEGO task or an executable project recipe.
The next fitting owner must use the actual recorded dataset and selected
checkpoint. No duplicate H100 job is launched here.

A deployment artifact needs trained weights, dataset statistics, tokenizer and
backbone config, camera map, physical action semantics, data revision, and
held-out results. The current strict bridge consumes a local Hugging Face
checkpoint directory. Native Lightning `.ckpt` output needs an explicit,
verified load/export adapter; renaming it is insufficient.

### Engineering-only inference boundary

The separate `--engineering-base-probe` route uses one recorded real image,
six synthetic zero state values, and explicit identity fixture statistics.
Without `--anomaly-report`, it uses synthetic anomaly text. With that argument,
it validates and uses real uncalibrated same-image Anomalib metadata. These statistics are newly written, labeled probe inputs;
they are never presented as the missing robot dataset statistics.
The image occupies slot zero, with remaining slots masked. This camera mapping
is explicitly unverified. The result is `ENGINEERING_PROBE_ONLY`, with an
uninterpreted `output_vector`, null physical units/semantics, and both
`controller_ready=false` and `task_ready=false`.
The task bridge rejects this result status. Task execution remains blocked.

The probe skips a redundant 2 GB VLM weight download because the 906,712,520-byte
policy checkpoint includes its VLM weights. It checks every active weight key.
The installed Studio model adds four target-time parameters absent in the older
base checkpoint. Its source bypasses those parameters when SnapFlow is disabled;
the probe permits exactly those inactive keys and records them. Any other missing
or unexpected key blocks inference. Task inference retains its stricter check.

The pinned policy weight digest from Hugging Face's LFS metadata is
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.
Backbone/tokenizer revision is `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
The worker verifies artifacts before loading. It uses offline mode and two CPU
threads. No camera, robot, or application service is opened or restarted.


### Completed engineering probe — September 15, 16:40 PDT

The pinned policy download finished and matched upstream SHA-256
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.
All 500 active pretrained parameter keys matched the installed model.
The only missing keys were the four documented inactive SnapFlow parameters.
Relevant Studio source matched the pinned Git revision; source hashes are saved.

The recorded image was 1920 × 1080 RGB, with SHA-256
`4ab108fcd59536b1d1a4b3779001891c5d8771d6c1ada71b393c92b60998ce68`.
The complete LEGO instruction and synthetic inspection text occupied 31 tokens.
The returned vector was finite with shape `[1,6]`; it has no established physical
units or action semantics. No controller received it.

The model-load measurement was 8.963 seconds. The single `select_action` call
was 4.227 seconds. Inference timing excludes import, artifact hashing, image
decoding, and model loading. No warmed distribution, accelerator comparison,
physical cycle time, or sorting success was measured.

The portable suite passed **17 tests** at this first probe milestone. Download and inference supervisors
exited successfully; their ownership markers were removed. No owned model process
remains running. The root application and Studio services were not restarted.

Reproduce this engineering-only run on Intel using the staged, isolated runner:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python \
  /home/ird-demo/second-look/artifacts/policy/probe-code/scripts/policy_worker.py \
  --engineering-base-probe /home/ird-demo/second-look/artifacts/policy/engineering-base \
  --image /home/ird-demo/second-look/artifacts/cv/scout-20260915/camera1.png \
  --output /home/ird-demo/second-look/artifacts/policy/engineering-base/inference-result.json
```

The worker enforces offline mode and the CPU thread limit internally. The
executed supervisor additionally used priority 10 and a 180-second timeout.
Remote result and logs remain under `artifacts/policy/engineering-base/`.
The task manifest remains incomplete and blocked; this fixture does not replace it.


### Completed combined replay — September 15, 16:45 PDT

Both model stages used the identical saved RGB image from the first probe.
The detector read its pixels from the exact hashed image bytes. The VLA worker
checked the detector report against the image SHA-256 before constructing text.
It refused a different image, nonfinite score, invented threshold, or false
classified disposition. These allowances exist only in the engineering route;
the strict task bridge remains blocked on uncalibrated metadata.

| Stage | Real or fixture | Measured result |
|---|---|---|
| Image | Real recorded, unlabeled | Same 1920 × 1080 RGB SHA-256 as first probe |
| Anomalib model | Real scene-smoke PatchCore exported to OpenVINO | GPU.0 FP32; raw score 79.97917175292969 |
| Decision | Real uncalibrated detector disposition | UNKNOWN; threshold null |
| Robot state | Synthetic engineering fixture | Six zeros; physical units unset |
| State/action statistics | Synthetic identity fixture | Not robot or dataset statistics |
| VLA | Real pinned pretrained Studio SmolVLA | Finite `[1,6]` output; no physical semantics |
| Controller | Unused | No motion command; readiness false |

Actual conditioning text:

```text
Sort LEGO into defective and non-defective piles.
Inspection: UNKNOWN; anomaly score=79.97917175292969; threshold=null (uncalibrated).
```

This text occupied **51 tokens** under the pinned tokenizer. The engineering
limit was 128, so no instruction or inspection metadata was truncated.

Detector compile/load took **277.291 ms**. Its model call took **13.213 ms**;
preprocessing plus model call took **25.087 ms**. VLA load took **8.992 seconds**;
its single model call took **4.239 seconds**. These separate stage measurements
exclude subprocess orchestration and do not measure a continuous physical loop.

The combined report embeds the real detector result and its source/artifact
identities. The detector map is saved as
`artifacts/policy/smolvla-same-image-anomaly-map.npy`. Model logs remain alongside
the reports. The original synthetic-metadata result remains unchanged.

For a replay, append these arguments to the earlier engineering invocation:

```text
--anomaly-report /home/ird-demo/second-look/artifacts/policy/engineering-base/same-image-anomaly.json
--output /home/ird-demo/second-look/artifacts/policy/engineering-base/same-image-vla-inference.json
```

Use the combined output path instead of the first probe output path.
At the combined replay milestone, the portable suite passed **18 tests**. Both additional model calls exited;
no downloader or policy process remains running. No new fitting, camera access,
serial access, controller access, or root application restart occurred.


## Contract review fixes

The generic task route now requires image-content association, in addition to
observation identity and time. The detector camera set is declared explicitly in
the manifest. Its source-image map must match that set, each physical camera ID,
and each input content hash. Missing, additional, or swapped sources fail closed.
Actual image bytes remain checked by the worker before policy inference.

Two engineering boundaries were compared. A shared mode with a different response
would require every caller to distinguish engineering output from task candidates.
Rejecting that mode at the task interface is simpler and preserves the already
separate engineering CLI. We selected rejection: the task bridge never launches a
worker for `engineering_probe`, and direct generic-worker requests also fail.
Failures report `BLOCKED`, `task_ready=false`, and no action. The dedicated
`--engineering-base-probe` route and its saved reports remain unchanged.

**26 portable policy tests pass** after these fixes. New regressions cover image
swaps with identical IDs/times, exact detector camera/hash mapping, declared
camera subsets, and both bridge and direct-worker engineering-mode rejection.
This review used no remote connection, model execution, or hardware call.
Earlier measured reports remain preserved as historical evidence.
