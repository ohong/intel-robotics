# Remote H100 training guide

Use the existing project trainers. Train on H100; run deployment inference on Intel.
This guide describes the workflow, not permission to resume paused training.

**Status reference:** [placement status](PLACEMENT_SMOLVLA_STATUS.md) records prepared data, but no completed placement fine-tune. Intel SSH timed out on September 23. Recheck access and jobs before starting. Historical receipts are in [H100 readiness](H100_READINESS.md).

## 1. Choose the right model and data

| Component | Purpose | Trainer | H100 environment |
| --- | --- | --- | --- |
| Anomalib PatchCore | Detect defects from images | `scripts/h100/training_anomalib.py` | `vision-env` |
| Native Studio ACT | Pickup-to-inspection manipulation | `scripts/h100/training_act.py` | `training-env` |
| Native Studio SmolVLA | Inspection-to-selected-bin placement | `scripts/h100/training_smolvla.py` | `training-env` |

- **Anomalib:** fit on normal images; calibrate with normal and defective validation images. Split by physical specimen/session when possible. Preserve labels, image hashes, camera identity, and crop settings. Do not reuse an old region of interest after camera changes.
- **ACT:** use a single consistent skill/destination per dataset. ACT does not use task captions. Follow the [ACT data contract](ACT_TRAINING_CONTRACT.md).
- **SmolVLA:** use `training_scope=step_three_placement` and `conditioning.mode=selected_instruction`. GOOD selects BLUE; BAD selects PINK; UNKNOWN makes no placement request. Use the exact instructions in the [placement workflow](../scripts/h100/placement-training.md), not historical anomaly-score prompts.

Freeze closed recordings before processing. Verify camera keys, frame timing, six-joint order, absolute action targets, and observed outcomes. Preserve originals. Split whole episodes into training, validation, and final evaluation; compute normalization from training episodes only. Never invent success labels or mix recordings with conflicting bin mappings.

The prepared placement dataset has 27 episodes and a 17/5/5 split. Reuse its manifest and receipts if unchanged. For new data, use `prepare_placement.py` on a **derivative copy**; it rewrites task metadata. Its required inputs and qualification rules are in the placement workflow. Templates contain deliberate missing values and cannot launch unchanged.

## 2. Verify access and GPU ownership

Run the following commands from the **Mac repository root**. Remote commands go through Intel; the GPU private key stays there.

```sh
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes intel-robot 'hostname; id -un'
python3 scripts/h100/transport.py --timeout 90 exec \
  'hostname; nvidia-smi; df -h /workspace'
```

If access fails, follow [Remote environment](REMOTE_ENVIRONMENT.md). Do not bypass host verification or copy credentials. `transport.py` reads Intel's `~/gpu_connect.txt`; do not print that file.

Keep source, data, environments, caches, logs, and checkpoints under `/workspace/second-look-h100`. Reuse `training-env` and `vision-env`; do not rebuild the base environment or restart the node. Network storage persistence across node restarts is not established.

Check active compute processes and coordinate ownership before each launch. The trainers also check the shared `.gpu-job.lock`. Never delete that lock or kill another job. A timeout can leave a remote process running: inspect its log/process before retrying.

## 3. Stage a frozen run

Replace uppercase placeholders such as `RUN` with unique, verified values. These examples stage the current source without changing an earlier run:

```sh
tar -cf /tmp/h100-RUN.tar scripts/h100 secondlook
scp /tmp/h100-RUN.tar intel-robot:/home/ird-demo/second-look/h100-readiness/h100-RUN.tar
python3 scripts/h100/transport.py push \
  /home/ird-demo/second-look/h100-readiness/h100-RUN.tar \
  /workspace/second-look-h100/h100-RUN.tar
python3 scripts/h100/transport.py exec \
  'mkdir -p /workspace/second-look-h100/code/RUN && tar --no-same-owner -xf /workspace/second-look-h100/h100-RUN.tar -C /workspace/second-look-h100/code/RUN'
```

`push` reads its source **on Intel**, not the Mac. Package datasets/configs separately and transfer them the same way. Compare archive SHA-256 on both machines before extraction; verify the extracted dataset against its manifest. Keep credentials out of every archive. Avoid `--overwrite` for run artifacts.

Set absolute H100 paths in the configuration. Pin the dataset/manifest, training statistics, Studio revision, base weights, tokenizer/backbone files, seed, camera mapping, budgets, and a new output directory. Copy all `scripts/h100/` helpers together. Reuse the pinned assets under `models/smolvla-assets`; do not substitute another trainer with the same model name.

## 4. Check, smoke-test, then train

This Mac shell helper runs commands from the owned H100 directory:

```sh
h100() {
  python3 scripts/h100/transport.py --timeout 3600 exec \
    "cd /workspace/second-look-h100 && $1"
}
```

### SmolVLA placement

Use separately prepared smoke and training configs with different `output_dir` values:

```sh
h100 'training-env/bin/python -B code/RUN/scripts/h100/training_smolvla.py --config /workspace/second-look-h100/configs/placement-smoke.json --mode check'
h100 'training-env/bin/python -B code/RUN/scripts/h100/training_smolvla.py --config /workspace/second-look-h100/configs/placement-smoke.json --mode smoke > logs/RUN-smoke.log 2>&1'
h100 'training-env/bin/python -B code/RUN/scripts/h100/training_smolvla.py --config /workspace/second-look-h100/configs/placement-train.json --mode train > logs/RUN-train.log 2>&1'
```

Create `logs/` before redirecting. Run each command only after the previous check passes. Smoke must prove a real CUDA update, finite gradients, and checkpoint save/reload parity. Both compared models must use the same CUDA device and noise seed. The old five-episode probe failed this check and is not deployable.

Choose `max_steps`, `max_seconds`, batch size, and checkpoint/validation intervals from measured smoke throughput. Allow extra time for cold imports and checkpoint verification. Imports from network storage can take several minutes. `-B` avoids bytecode writes.

### ACT

Use the same check → smoke → train sequence with `training_act.py`. Its config is **positional**, unlike SmolVLA:

```sh
h100 'training-env/bin/python -B code/RUN/scripts/h100/training_act.py /workspace/second-look-h100/configs/act-smoke.json --mode smoke'
```

### Anomalib PatchCore

Pin `SOURCE_SHA256` to the staged `secondlook/anomaly.py`. Set `TRAIN_COUNT` from the manifest. Run this first with `--check`; remove that flag only after validation succeeds:

```sh
h100 'vision-env/bin/python -B code/RUN/scripts/h100/training_anomalib.py --source-root /workspace/second-look-h100/code/RUN --source-sha256 SOURCE_SHA256 --manifest /workspace/second-look-h100/datasets/DETECTOR/manifest.json --output /workspace/second-look-h100/vision/runs/RUN --image-size 256 --sampling-ratio .1 --num-neighbors 9 --max-images TRAIN_COUNT --max-seconds 600 --calibrate --check'
```

These model settings match the earlier pilot, not a universal optimum. Add `--roi X0 Y0 X1 Y1` (normalized coordinates) and `--foreground-crop` only when verified for the current scene. Use the same preprocessing during inference. PatchCore builds a normal-feature memory bank; it has no optimizer checkpoint to resume. Its runner verifies CUDA fitting, reload, and OpenVINO export parity.

## 5. Monitor and resume

Read logs and the run's `progress.json`, `run.json`, `result.json`, and metrics. A checkpoint file alone does not prove a successful run.

```sh
h100 'tail -60 logs/RUN-train.log'
h100 'training-env/bin/python -B code/RUN/scripts/h100/training_smolvla.py --config /workspace/second-look-h100/configs/placement-train.json --mode train --resume /workspace/second-look-h100/runs/RUN/last.ckpt > logs/RUN-resume.log 2>&1'
```

Use the actual output path from the config. Resume only the same run, preserving source, data, packages, seed, and configuration identities. ACT also accepts `--resume`. Select weights using validation evidence, not final-test results. Placement training reports each route separately; final evaluation happens after selection and must not guide tuning.

## 6. Return artifacts and verify Intel inference

1. Export the validation-selected policy using the [SmolVLA export workflow](../scripts/h100/smolvla-export.md). Preserve actual camera slots, training normalizers, tokenizer settings, and `hf_constructor_kwargs`.
2. Bundle weights, preprocessing, manifests, package/source versions, config, metrics, logs, and exact launch/resume commands. Keep resumable checkpoints on H100 too.
3. Return the bundle to Intel and verify its complete SHA-256:

```sh
python3 scripts/h100/pull_resume.py \
  /workspace/second-look-h100/exports/RUN.tar \
  /home/ird-demo/second-look/h100-readiness/RUN.tar \
  --sha256 BUNDLE_SHA256 --timeout 1800
```

For large SmolVLA weights, the export workflow supports a lossless delta against the pinned base. Reconstructed weights must match the full target hash.

4. Run [offline Intel policy verification](placement-intel-verification.md), including both routes and latency. For Anomalib, compare Torch and OpenVINO scores, maps, and decisions on the same validation images.
5. Record results and remaining gates in `AGENT_STATE.md`. Keep the controller disarmed until the robot-control owner starts an approved supervised test. Training loss and software parity are not physical success.
