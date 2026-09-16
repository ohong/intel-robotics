# H100 Studio training preparation

**Prepared; task training has not executed.** No real synchronized demonstrations,
verified task action contract, or task-ready weights are available in this work.
The scripts do not use a robot, camera, controller, or simulated gym.

## Supported route

These two routes use the same Studio trainer:

| Entry point | Benefit | Cost |
|---|---|---|
| `physicalai fit --config config.json --fit.ckpt_path checkpoint.ckpt` | Existing Studio CLI and configuration parser | Extra parser layer for artifact and normalizer checks |
| `physicalai.train.Trainer.fit(policy, datamodule=data, ckpt_path=...)` | Direct input checks, local logs, explicit resume | Small Python launcher |

We select the direct API. This is Studio's `SmolVLA`, `LeRobotDataModule`, and
`Trainer`, not the independent LeRobot training CLI. Studio creates the training
dataset during datamodule construction. Its policy `setup()` replaces base
normalizers with actual training dataset statistics. The launcher checks those
resolved state/action mean and standard-deviation values before the first batch.
Studio's own callback sets the policy's action time offsets on the dataset.

Source inspected at Studio commit `c4ff730fb49f84e5102d01088d52cfff1ba62854`:

- `library/src/physicalai/train/trainer.py`
- `library/src/physicalai/cli/fit.py` and `library/docs/how-to/training/cli.md`
- `library/src/physicalai/policies/smolvla/policy.py`
- `library/src/physicalai/data/lerobot/datamodule.py`
- `library/configs/physicalai/smolvla.yaml`
- `library/tests/unit/policies/test_smolvla.py`

## Isolated setup

Execute on the H100 host. Keep everything below `/workspace/second-look-h100`.
The bootstrap uses existing `uv`, `git`, Python, and CUDA Torch wheels. It creates
an isolated venv with access to existing packages; additions only enter that venv.
It constrains Torch and torchvision to their existing versions. A dependency
conflict fails resolution instead of replacing the sponsor environment.

```sh
bash scripts/h100/training-setup.sh
```

Studio requires Python >=3.12,<3.15. The bootstrap installs pinned Studio
`physicalai-train[smolvla]` and runtime commit
`8e4021703ef43387a835c6647b993cecc069ca85`. Key dependencies include
`lerobot[dataset]==0.6.0`, Lightning, `transformers>=5.5,<5.6`,
`safetensors>=0.4.3,<1`, and `num2words>=0.5.14,<0.6`.
Studio also requires OpenVINO >=2026.3; this is a package requirement, not evidence
that the training workload uses OpenVINO. No model weights are downloaded by setup.

The bootstrap records the `uv` version, exact resolved package freeze, and genuine
API import/signature report. Installation and import success are separate from a
SmolVLA forward/backward pass. The package freeze captures resolution; it is not
a previously tested universal lockfile. Consult the lead's readiness record for
actual H100 execution results and any resolver changes.

## Required training inputs

Copy the two JSON templates outside the repository and fill them from evidence.
`null` values deliberately fail preflight. The initial numeric training parameters
are conservative engineering defaults, not measured convergence settings.

1. Supply a complete, local LeRobot dataset compatible with installed LeRobot 0.6.
2. Supply recorded real, synchronized image/state/action demonstrations and task text.
3. Record the task, calibration, action ordering, units, and absolute-position semantics.
4. Record camera physical identities and map each dataset camera suffix to slots 0–2.
5. Supply dataset-derived state/action mean, std, min, and max in `meta/stats.json`.
6. Separate held-out specimens/sessions before exporting the training dataset and statistics.
7. Supply local base weights from `lerobot/smolvla_base` revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
8. Supply the complete local backbone/tokenizer and its pinned provenance.
9. Set the base copy's `config.json` `vlm_model_name` to that local backbone directory.
10. Record every dataset, base, and backbone file's SHA-256 in the configuration maps.

Map keys are file paths relative to the corresponding root. Values are SHA-256
hex strings. Preserve the original upstream configuration and its provenance
outside the modified base directory. The launcher checks bytes against the
supplied manifests; it cannot establish that a declaration or physical label is
true. Do not use placeholder evidence to make these checks pass.

`meta/info.json` must include exact state/action feature names and dimensions.
The state/action names and units in the contract must come from the recorded
system. Matching six dimensions alone does not establish compatibility.
Camera mapping uses suffixes such as `top`, not `observation.images.top`.
Unknown camera roles and missing normalization statistics must be resolved first.

The contract's `synchronization_evidence`, `units_evidence`, and `split_evidence`
identify the actual capture/inspection records. `camera_sources` maps dataset
camera suffixes to physical source identifiers. `provenance` must be
`recorded_real`; `training_only` must be true. These are evidence declarations,
not software certification of the physical facts.

No automatic random validation split is enabled. Studio computes missing quantile
statistics across its pre-split dataset. A training-only export avoids using
held-out observations in these statistics. Later evaluate separate specimens
and task outcomes. Training loss is not autonomous manipulation success.

## Check, launch, and resume

The check uses the standard library and does not import Studio or initialize CUDA:

```sh
python3 scripts/h100/training_smolvla.py --config /workspace/second-look-h100/training-config.json --check
```

After real prerequisites exist, launch in the isolated interpreter. The console
log also captures messages outside Python logging:

```sh
set -o pipefail
CUDA_VISIBLE_DEVICES=0 /workspace/second-look-h100/training-env/bin/python \
  scripts/h100/training_smolvla.py \
  --config /workspace/second-look-h100/training-config.json \
  2>&1 | tee /workspace/second-look-h100/training-console.log
```

Resume the same run with the unchanged configuration and environment:

```sh
CUDA_VISIBLE_DEVICES=0 /workspace/second-look-h100/training-env/bin/python \
  scripts/h100/training_smolvla.py \
  --config /workspace/second-look-h100/training-config.json \
  --resume /workspace/second-look-h100/runs/first-adaptation/checkpoints/last.ckpt
```

Resume restores Lightning model, optimizer, scheduler, and training progress via
`Trainer.fit(ckpt_path=...)`. Only checkpoints from that run's directory are
accepted. They must be trusted local artifacts. The step limit is the total step
ceiling, including previously completed steps. The time limit applies to each
invocation. Lightning checks its time limit during training; input loading and
checkpoint writes are outside a hard wall-clock guarantee. A supervisor should
bound the whole process if a strict external deadline applies.

The run requires one visible H100 and rejects existing GPU compute processes.
This is an immediate occupancy check, not a cross-user scheduling lock. Keep the
lead as the sole owner of allocated GPU jobs. Runtime networking is disabled for
Hugging Face and Transformers. All application cache/temp/output paths stay under
`/workspace/second-look-h100`.

Outputs include `run.json`, `studio.log`, CSV training metrics,
`dataset-stats.json`, periodic full `.ckpt` files, and `last.ckpt`.
An interrupted or failed run does not become success evidence. Checkpoint cadence
bounds lost progress on a crash; it does not guarantee a save on forced process
termination. CUDA kernels can remain nondeterministic despite the recorded seed.

## Executed offline checks

Seven standard-library checks passed on Mac: valid synthetic declarations,
missing prerequisites, changed artifact sets, invalid statistics, unverified
provenance/mixed splits, camera/output bounds, and constructor keyword compatibility
against the pinned source. Both templates remain intentionally unusable for real
training. Shell syntax and Python syntax passed. The blank template exits 2 with
`BLOCKED: Provide dataset_root`.

```sh
STUDIO_LIBRARY=/path/to/pinned/physical-ai-studio/library \
  python3 scripts/h100/training_test.py -v
```

These checks do not load actual weights, decode real demonstrations, perform a
SmolVLA training step, test resume on SmolVLA, measure policy quality, export an
Intel inference artifact, or establish physical safety/success. Those gates remain
open until actual dependencies and task inputs permit execution.
