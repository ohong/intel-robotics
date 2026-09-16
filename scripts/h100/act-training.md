# Studio ACT training and offline Intel handoff

## Design

Use Studio's native ACT, LeRobot loader, Lightning trainer, and export methods.
The alternative is LeRobot's ACT trainer plus checkpoint conversion. The native route
avoids an extra checkpoint format boundary and matches the installed Intel loader.

Standard ACT consumes images and measured joint state. It does not use task captions
to select destinations. Each physical destination requires its own dataset and run.
The VLA/perception selector remains a separate application component.

```text
finalized single-skill snapshot + capture evidence
                    |
       whole episode and specimen/session split
            /                    \
 training only                validation       final holdout (untouched)
 weights + statistics          selection
            \                    /
              selected native ACT checkpoint
                  |                  |
             Torch export       OpenVINO FP32 export
                  \                  /
             saved-observation Intel parity
```

## Entry points

Start from `act-config.template.json`. Dataset, manifest, skill, output, and capture
paths must be set from real evidence. Nothing uploads to the Hub. All GPU writes stay
under `/workspace/second-look-h100`; the script locks our GPU-job file and rejects an
already occupied GPU before initializing CUDA.

Use the existing prepared Studio interpreter. These commands do not install packages:

```sh
python -B training_act.py /absolute/config.json --mode check
python -B training_act.py /absolute/smoke-config.json --mode smoke
python -B training_act.py /absolute/train-config.json --mode train
python -B training_act.py /absolute/train-config.json --mode train \
  --resume /workspace/second-look-h100/runs/OWN_RUN/last.ckpt
```

Each new run needs a new output directory. Smoke does one CUDA optimizer step with
finite-gradient checks, validation, and exact checkpoint reload comparison. Training
uses the configured step/time budget and selects the smallest validation `val/loss`.
Recovery checkpoints do not substitute for a validated selected checkpoint. Resume
restores the optimizer and rejects changed data, source, packages, initialization, or model
configuration. Step and time budgets may increase explicitly.

Real smoke also exports both formats. Copy its outputs to Intel and run the parity
entrypoint before budgeted training. Set `smoke_result` and `intel_parity_result` in
the training config to those reports. Training checks their real evidence label and
matching source/data/model identity. A new snapshot or architecture needs a new smoke.

The task policy starts untrained. Its ResNet18 image backbone uses the locally staged
official ImageNet weights, with SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
This is not fine-tuning an existing task policy. The launcher loads that backbone
after Studio creates the model. Checkpoints retain `pretrained_backbone_weights=null`
so Intel reloads require no backbone download.

## Frozen data and capture gates

The recording owner supplies `manifest.json` and adjacent `train_stats.json`. Every
data file and the statistics file must match their frozen hashes. The relocated
`dataset_root` may differ from the source path, but the file content may not.

Real smoke accepts `loader_smoke` splits. Training requires `grouped_evaluation`,
separate whole train/validation/final episodes, and disjoint specimen/session groups.
Final episodes are never loaded during fitting, checkpoint selection, or export.
Image and joint normalization always come from training episodes only. Passing a
prebuilt LeRobot dataset avoids Studio's all-episode quantile calculation.

Set `capture_manifest` to the recording owner's refreshed capture JSON and
`capture_runtime` to its matching runtime YAML. The YAML is hashed, never executed.
The check binds camera keys/resolutions, follower calibration, normalized joint order,
and completed episode count. The capture JSON must be refreshed after actual collection;
the current zero-episode preparation record cannot pass real training.
Its episode count describes this one skill snapshot, not a global count across skills.

Frame-index timestamps do not establish camera-to-joint sensor skew. This limit remains
in capture provenance. Offline action loss does not establish physical task success.

## Software qualification only

When no real demonstrations exist, the separate mode below accepts **only synthetic**
data. It performs one optimizer step and the same native/OpenVINO export path. It never
converts synthetic evidence into task-training evidence:

```sh
python -B training_act.py /absolute/synthetic-config.json --mode software-qualification
```

Its completed result is `UNSAFE_NOT_TASK_TRAINED`. The manifest must come from the
recording validator with `evidence_kind=synthetic`, finalized=true, and synthetic
episode outcome labels. Use the standard architecture to qualify that architecture;
an explicitly smaller `policy` configuration qualifies only that smaller model.

## Intel verification

Copy the run's `result.json`, `export-torch/`, `export-openvino/`, and
`export-replay.npz` together. Use the Intel Studio backend interpreter:

```sh
python -B verify_act_intel.py /absolute/copied-run \
  --output /absolute/new-parity-evidence.json --device CPU
```

The checker verifies transfer hashes, loads both exports through Studio
`InferenceModel`, and compares their outputs to the saved reference. Native inference
uses CPU. OpenVINO first uses FP32 weights and an explicit FP32 inference hint. GPU or
NPU runs require their own `--device` check and fresh output file. Actual execution
devices and cold/warm inference timing are reported. Device support and output parity
must be measured; enumeration alone is insufficient.

The initial fixture contains one saved validation observation. This proves only a
narrow software boundary. No robot, controller, camera, or runtime instance is created.
Synthetic qualification remains unsafe for robot execution even after parity passes.

## Budget and evidence

`max_steps` and `max_seconds` bound training cooperatively. The time limit applies to
Lightning fitting, not cold imports, checkpoint serialization, export, or Intel
verification. Cold imports and fitting/export time are recorded separately. Use an
external process supervisor for a hard whole-process deadline, with time reserved for
checkpoint saves and cleanup. No event deadline is inferred from this configuration.

The run records its seed, package freeze, selected Studio source hashes, full dataset
manifest, capture/calibration hashes, backbone provenance, checkpoint hashes, validation
score, and export hashes. The final offline holdout and all physical trials remain
unmeasured until explicitly evaluated after checkpoint freeze.

Local contract tests:

```sh
python3 -m unittest discover -s scripts/h100 -p test_training_act.py -v
```

These tests use disposable metadata fixtures. They do not load ACT or provide physical
evidence. Actual CUDA, export, and Intel results must come from the runtime reports.
