# ACT capture and training contract

## Decision and verified scope

Use the installed **native Studio ACT** implementation. It preserves the application's
training and Lightning checkpoint route. Direct `lerobot.policies.act.ACTPolicy` is an
alternative, but its configuration, processors, and checkpoint format differ. We do not
switch implementations because both have the name ACT.

Verified interpreter on the Intel PC:

```text
/home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python
physicalai 0.1.2.dev70+g8e4021703
lerobot 0.6.0
torch 2.11.0+xpu
lightning 2.6.5
pyarrow 25.0.0
```

The selected route is the route in Studio's `backend/src/training/job.py`:

```text
finalized LeRobot v3 dataset
  -> lerobot.datasets.lerobot_dataset.LeRobotDataset
  -> physicalai.data.lerobot.LeRobotDataModule
  -> physicalai.policies.ACT
  -> physicalai.train.trainer.Trainer
  -> model.ckpt -> native ACT.load_from_checkpoint
```

The CPU software test passed with synthetic images and numeric rows: two episodes,
eight frames each, one optimizer step, 81 finite gradient tensors, checkpoint reload,
and identical reloaded predictions. The reduced smoke configuration uses four action
steps and 64×64 input images. This proves the installed software path. It does not prove
task learning, native-default training performance, exported inference, or robot safety.
The synthetic contact sheet was visually inspected; it contains generated color ramps.

Evidence: `artifacts/recording/act-loader-synthetic/manifest.json`, `smoke-result.json`,
`train_stats.json`, and contact sheets. These are generated local artifacts and may be
ignored by Git. Source paths and hashes appear in the smoke report. The unsafe synthetic
checkpoint remains under the Intel PC's `second-look/recording-tools/` test directory.

## ACT does not use task captions

The native model builds input tokens from robot state and image features. Its action
head predicts joint-position chunks. Dataset captions are carried by the loader, but
the ACT model does not encode them. Changing the caption on the same synthetic minibatch
produced exactly the same predicted chunk after checkpoint reload.

Therefore, capture **normal-to-normal-pile** and **defective-to-reject-pile** as separate
skills, datasets, and ACT policies. A caption does not resolve conflicting destinations.
The VLA/perception selector remains a separate required integration. ACT alone does not
establish the challenge's VLA requirement.

## Capture contract

The first registered real dataset is:

```text
/home/ird-demo/.local/share/physicalai/datasets/093df5db-9ecb-416a-95d6-4efe3752442b
```

It was empty when tooling was prepared. No real episode has passed this validator yet.
Use actual finalized metadata as the authority for feature keys, shapes, FPS, and names.
Do not infer completed capture from registration or an empty directory.

| Field | Contract and provenance |
|---|---|
| Format | Installed LeRobot v3.0: all `meta/`, `data/`, and `videos/` files together |
| State | Measured follower positions, one six-value `observation.state` vector |
| Actions | Absolute position targets, one six-value `action` vector; not deltas or velocities |
| Joint order | `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`; freeze actual exported names including suffixes |
| Units | Calibrated Studio driver uses normalized body positions [-100,100] and gripper [0,100], not radians or degrees |
| Gripper | Preserve installed calibration and direction; verify physical open/closed endpoints during authorized supervision |
| Camera color | RGB; loader emits float CHW tensors in [0,1] |
| Camera keys | Current runtime names: `camera 2 _high_`, `camera 1 _low_`; freeze the actual exported `observation.images.*` keys |
| Current camera settings | High: 1280×720 at 30 FPS; low: 1920×1080 at 30 FPS, as reported by the recording owner after configuration repair |
| Calibration | Preserve follower serial 5B79018445 and leader serial 5B79018575; archive/hash the capture configuration |
| Native ACT preprocessing | Aspect-preserving bilinear resize and symmetric zero padding; policy default input size 384×384 |
| Native ACT output | Default chunk length 100, six position targets per action; freeze final training configuration explicitly |

The normalized units are established by `backend/src/robots/catalog/so101.py` and the
installed `physicalai/robot/so101/so101.py`. The episode files must still match them.
The recording owner refreshed `artifacts/recording/environment.json` and `runtime.yaml`
after the camera repair. Finalized dataset metadata remains the capture authority.

The recorder pairs its latest camera samples with state/action records and stores
`frame_index / fps` timestamps. Consistent table timestamps do **not** measure camera-state
skew, dropped sensor updates, or true wall-clock capture frequency. The validator decodes
every indexed frame, checks temporal indexing, and flags long identical image runs.
It cannot reconstruct missing sensor timestamps. Keep this limitation in the training handoff.

Source inspection establishes the intended row pairing: the native loop reads follower
state and cameras, computes the target, attempts to send it, then records that same
pre-action observation with the target. This is not a measured response to that target.
`_resilient_send` can exhaust retries and skip a send while the tick still names the
target `action_sent`. That field is not a motor acknowledgment. For real pilot episodes,
retain the bounded runtime logs, reject episodes with communication or writer failures,
and inspect measured motion and images. Dataset-only checks cannot prove motor acceptance
or rule out every cached sensor sample. The stock browser stream also omits tick capture
timestamps and most lifecycle fault details.

## Finalization, labels, and splits

Stop recording and use the recorder's supported save/finalization boundary before validation.
The lead owns that operation. These tools never connect to Studio's runtime, cameras, or robots.
`--finalized` records the lead's declaration. Before/after file hashes detect concurrent
changes but cannot prove no other writer exists. Never validate or transfer a live writer's dataset.

Use whole episodes for three disjoint sets:

1. `train_episodes`: fit weights and normalization statistics.
2. `validation_episodes`: select checkpoints and guide development.
3. `final_eval_episodes`: excluded from all fitting, normalization, and checkpoint selection.

The first two smoke episodes can use one train and one validation episode, with an empty
final-evaluation set. This is a loader check, not a final evaluation plan. Allocate final
episodes in a later frozen snapshot. Untouched final **physical trials** remain separate;
held-out recorded demonstrations only measure offline prediction behavior.

For real data, provide `--episode-labels labels.json`:

```json
{
  "0": {
    "destination": "normal_pile",
    "outcome": "complete_success",
    "outcome_evidence": "Reference to operator confirmation or inspected final release/clearance",
    "specimen_id": "actual specimen identifier",
    "session_id": "actual collection session identifier"
  }
}
```

This example is a schema, not a label for an existing demonstration. Supply one entry per
actual episode. Real labels require a destination, `complete_success`, outcome evidence,
`specimen_id`, and `session_id`.
The successful imitation dataset rejects failed/aborted attempts. Preserve those attempts
separately with honest labels. Reserve specimens/sessions when possible; whole-episode
splitting alone does not prove independence of physical specimens. Never use detector
predictions as authoritative ground truth.

Default `--split-scope loader_smoke` permits shared specimens/sessions for the first two
diagnostic episodes. Its manifest explicitly forbids evaluation-quality claims. For
`--split-scope grouped_evaluation`, the validator requires distinct specimen and session
IDs across train, validation, and final-evaluation sets. The training owner must reject
`loader_smoke` for quality claims. Empty final-evaluation sets never establish final results.

## Statistics workaround: do not use the unmodified Studio split path

Source inspection found that `LeRobotDataModule._ensure_quantile_stats()` calculates missing
quantiles over all episodes before its split. LeRobot metadata also carries dataset-wide
statistics. A held-out episode must not influence fitted preprocessing.

The validator computes numeric mean/std/min/max and exact q01/q99 from **training rows only**.
For each camera it computes RGB pixel mean/std/min/max from decoded **training frames only**.
It saves `train_stats.json` and its SHA-256 in the manifest. Final-evaluation and validation
frames never contribute. The smoke entry point replaces the reader's in-memory `meta.stats`
before constructing the datamodule from a prebuilt dataset. Numeric quantiles are already
present, so the datamodule does not calculate all-episode quantiles. No installed source or
dataset files are modified.

H100 training must retain this adapter and statistics route. Construct independent train
and validation readers with the manifest's explicit episode lists. Give both readers the
same training-only statistics. Never pass `val_split` and allow a fresh random split.
Do not construct a final-evaluation dataloader during fitting or checkpoint selection.
`smoke_act.py` contains the exercised loading sequence.

## Commands and handoff interface

Run from the scoped Intel directory `/home/ird-demo/second-look/recording-tools` with the
interpreter above. On the Mac, source lives in `scripts/recording/`.

```sh
STUDIO_PY=/home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python
"$STUDIO_PY" validate_dataset.py /path/to/finalized/dataset \
  --output /path/to/new-validation-evidence \
  --skill normal-to-normal-pile --evidence-kind real --finalized \
  --episode-labels /path/to/actual-labels.json \
  --validation-episodes 1
"$STUDIO_PY" smoke_act.py /path/to/new-validation-evidence/manifest.json \
  --output /path/to/new-smoke-evidence
```

For later snapshots, supply `--final-eval-episodes 10,11` and the chosen validation IDs.
IDs are examples only. All three sets must be disjoint, with nonempty train and validation.
For a copied frozen dataset, `smoke_act.py --dataset /workspace/actual-snapshot` overrides
the source path while requiring all original file hashes to match.

`validate_dataset.py` rejects empty/incomplete metadata, mismatched counts and boundaries,
misordered/duplicate frames, time gaps, nonfinite values, joint-order changes, missing image
streams, wholly dark streams, and grippers without variation. Its default cap is 20,000
frames. Every indexed RGB frame is decoded. It emits eight sampled contact-sheet rows per
episode, including endpoints. Inspect real sheets and episode review before accepting success.

Manifest fields consumed by training:

```text
status = OFFLINE_DATA_VALIDATED
evidence_kind = real                 # synthetic is never a task-training handoff
finalized = true                    # owner declaration plus unchanged file hashes
split_scope = loader_smoke | grouped_evaluation
skill, features, fps, joint_names
train_episodes, validation_episodes, final_eval_episodes
episodes[].destination, outcome, outcome_evidence, specimen_id, session_id
files[relative_path] = {bytes, sha256}
train_stats_sha256                   # hash of sibling train_stats.json
```

`smoke_act.py` runs one CPU minibatch, checks finite nonzero gradients, saves and reloads
a native checkpoint, and compares outputs. It also verifies that changing captions has
no effect. Default `--profile small` limits CPU cost. `--profile studio` retains native
architecture sizes; neither profile downloads backbone weights. Both are software checks,
and their checkpoints must never command a robot.

The full training owner must freeze the actual policy parameters, package/source revisions,
data hashes, training statistics, seed, and initialization provenance. A pretrained image
backbone is not a task-trained ACT policy. Return the native checkpoint plus exact config,
normalization data, checkpoint-selection evidence, and export artifacts. Verify native and
exported inference without actuators before any separately authorized autonomous trial.

## Current checks and remaining gates

- Passed: five local synthetic fixture tests, including final-evaluation exclusion and split-scope guards.
- Passed: installed Studio synthetic v3 decode, train-only statistics, one native ACT step,
  finite gradients, native checkpoint reload, and caption-invariance check.
- Passed: synthetic contact-sheet visual inspection.
- Not tested: real episodes, measured sensor alignment, successful grasp/release,
  H100 training, native-default-size smoke, export parity, or autonomous execution.
- No packages were installed. No services, hardware connections, or sponsor configuration changed.
