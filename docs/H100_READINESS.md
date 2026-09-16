# H100 readiness

Status: **PARTIAL**. Investigation: 2026-09-15 22:49:28–23:08:30 UTC, within the 20-minute initial timebox.

Worktree: `/Users/ohong/dev/intel-robotics-h100`; branch `codex/h100-readiness`.
Changes are limited to this document and `scripts/h100/`. The main kickoff task owns
application integration and all robot control.

## Current execution status

Infrastructure and synthetic ACT qualification: **PARTIAL**. Real ACT training:
**BLOCKED — zero finalized real demonstration episodes**. Real-image Anomalib fitting
on H100 is now complete; see the final section for measured pilot results. No task-trained ACT artifact
is available. The active manipulation route is separate normal-bin and reject-bin
Studio ACT skills; Anomalib and genuine VLA reasoning remain with their existing owners.

The ACT execution section below supersedes the initial SmolVLA preparation notes.
Use `scripts/h100/act-training.md` and `training_act.py` for current manipulation work.
Cold trainer imports, full-size synthetic CUDA/export, checksum-verified artifact
return, and native/OpenVINO Intel CPU adapter parity now pass. Synthetic weights must
never command a robot.

## Verified infrastructure

| Check | Result |
|---|---|
| Access from Mac execution context | PASS: Mac → `intel-robot` → supplied direct SSH endpoint |
| Connection discovery | Parsed `~/gpu_connect.txt` on Intel; key stayed there; existing host trust required |
| Remote identity | Linux, user `root`, hostname `55a9e1e03378` |
| GPU | NVIDIA H100 80GB HBM3; 81,559 MiB reported VRAM |
| Initial utilization | 0%, 1 MiB allocated; no compute processes |
| Driver | 570.124.06 |
| Python | `/usr/bin/python3`, 3.12.3 |
| Base PyTorch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| CUDA computation | PASS: 256×256 FP32 GPU matrix multiplication compared with CPU; max absolute error 0.000041961669921875 |
| Synthetic checkpoint | PASS: Linear(8,2), seed 17, three AdamW steps, save/reload equality, one resumed step |
| Probe resource usage | 1.27 seconds after imports; 68,169,216 peak allocated CUDA bytes |
| Transfer | PASS: SCP upload and download between Intel and H100, matching SHA-256 |
| Workspace write | PASS under `/workspace/second-look-h100`; only own temporary transfer fixture removed |

The synthetic probe is **not** a Studio/SmolVLA trainer test, task dataset test, or robot result.
Checkpoint remains at
`/workspace/second-look-h100/checkpoints/cuda-fixture-20260915T225311Z.pt`.
SHA-256: `0a325c33ed16fe8033b7d1dcad7ca50684e5d9a571e5e54eed73837b4060613c`.
Its JSON report is in the same task root under `logs/`.
The initial transfer fixture SHA-256 was
`c86444cc294ba40dceb51ff576f075e35a980fca0c498aba3a55456a6a65ac68`.

## Storage facts and limits

`/workspace` is a read/write FUSE network-volume mount. `df -h` reported a 559 TB
filesystem with 201 TB available. This is a backend filesystem figure, **not a
verified allocation or quota for our account**. No paid capacity was provisioned.
A write/read/checksum test passed. Storage persistence across lifecycle changes was
not tested; the node was never restarted, stopped, or reconfigured.

Runpod documents network volumes as independent persistent storage, normally mounted
at `/workspace` for Pods. That supports the interpretation of the observed mount,
but does not establish this event volume's retention agreement or quota.
[Runpod network volumes](https://docs.runpod.io/storage/network-volumes).

All new GPU files, caches, logs, environments, and checkpoints use
`/workspace/second-look-h100/`. Existing environments and jobs remain untouched.

## Initial investigation: trainer and input dependencies (historical)

The main kickoff task confirmed the proposed policy route is Studio SmolVLA:

- Physical AI Studio commit `c4ff730fb49f84e5102d01088d52cfff1ba62854`.
- PhysicalAI runtime commit `8e4021703ef43387a835c6647b993cecc069ca85`.
- Base model `lerobot/smolvla_base`, revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
- Studio library package `physicalai-train`; LeRobot 0.6.0; Transformers >=5.5,<5.6.

The detector already has an Intel engineering-smoke artifact; no H100 detector
training was requested or duplicated.

The initial H100 environment lacks the Studio, LeRobot, Lightning, and Transformers
trainer packages. The isolated installer resolved and installed 133 packages, then
reached its 360-second bound (exit 124) during final verification. It stopped its
owned process group. A separate 45-second API import check also timed out (exit124)
without an import result. Trainer import readiness remains unverified. The base
CUDA PyTorch installation is preserved; isolated installation may duplicate its wheels.

Exact task-training dependencies still missing:

1. Recorded, synchronized task demonstrations in the selected LeRobot dataset format.
2. Verified joint order, absolute action units, calibration association, camera-source
   mapping, and actual training-data state/action statistics.
3. Separate held-out specimens/episodes and authoritative defect criteria. The main task
   reports LEGO sorting into defective/non-defective piles; criteria and pile geometry remain missing.
4. Complete pinned SmolVLA weights plus its VLM backbone/tokenizer files locally cached.
   Existing base-inspection files contain only configuration and normalizers.

No full training launched. No task-ready checkpoint or physical success is claimed.

The main task confirmed that a separate Studio task owns ACT recording and first-two-demo
loader validation. This report covers SmolVLA preparation and shared infrastructure;
it does not claim ACT loader compatibility.

## Scripts and tested entry points

- `transport.py`: runs commands and transfers individual files through Intel. It reads
  the supplied destination/key only on Intel, verifies H100 identity, keeps host checks
  enabled, restricts GPU transfer paths to our directory, and refuses overwrites by default.
- `cuda_probe.py`: bounded synthetic CUDA, optimizer, checkpoint-save/reload/resume check.
- `training-setup.sh`: creates a separate venv under the task directory, exposes the
  sponsor packages, constrains Torch/torchvision versions, and installs the pinned Studio
  and PhysicalAI code. Records uv version, package freeze, and actual import signatures.
- `training_smolvla.py`: uses the supported `physicalai.train.Trainer.fit` API, with
  validated local data/model inputs, seed, time/step limits, CSV metrics, run provenance,
  full Lightning checkpoints, and `ckpt_path` resume. It refuses competing GPU jobs.
- `training-config.template.json `, `training-contract.template.json `: intentionally
  incomplete templates; null inputs are blockers, not invented task data.
- `training_test.py`: seven focused input/API checks passed on both Mac and H100 against the pinned Studio source.

Two trainer interfaces were considered: the `physicalai fit` CLI and the direct Studio
Trainer API. The API was selected because it permits explicit data-contract checks and
resume provenance while using the same trainer. No alternate training architecture was introduced.

Use these commands from this worktree on the Mac (or after selective integration into
the main checkout). `push` sources and `pull` destinations are **on Intel**, not the Mac.
Directory bundles must first be archived into a single file. Extract with
`tar --no-same-owner` on this FUSE volume; restoring archive ownership is unsupported. Transfer operations do not
copy the Intel private key or change H100 authentication.

```sh
# Tested command execution; no GPU computation.
python3 scripts/h100/transport.py exec 'hostname; id -un'

# Tested source upload (substitute a new destination if the file exists).
python3 scripts/h100/transport.py push \
  /home/ird-demo/second-look/h100-readiness/cuda_probe.py \
  /workspace/second-look-h100/code/cuda_probe.py

# Tested artifact return; returned checkpoint SHA-256 matched the GPU original.
python3 scripts/h100/transport.py pull \
  /workspace/second-look-h100/checkpoints/cuda-fixture-20260915T225311Z.pt \
  /home/ird-demo/second-look/h100-readiness/cuda-fixture-returned.pt
```

Those exact destination files now exist. A repeat intentionally fails rather than
replacing them; choose a new filename, or pass `--overwrite` after the subcommand only
when replacement is intended. The CUDA fixture return hash is recorded above.

### Dataset/code upload and checkpoint return forms

These forms are prepared; actual task datasets/checkpoints do not yet exist.
For a new code bundle from the Mac:

```sh
tar -cf /tmp/h100-scripts-next.tar -C scripts h100
scp /tmp/h100-scripts-next.tar intel-robot:second-look/h100-readiness/scripts-next.tar
python3 scripts/h100/transport.py push \
  /home/ird-demo/second-look/h100-readiness/scripts-next.tar \
  /workspace/second-look-h100/code/scripts-next.tar
python3 scripts/h100/transport.py exec \
  'tar --no-same-owner -xf /workspace/second-look-h100/code/scripts-next.tar -C /workspace/second-look-h100/code'
```

This archive/extraction route was tested with the current scripts. Use a new bundle
name for each revision. For the future dataset and trained run:

```sh
# Dataset archive already created on Intel by the collection owner.
python3 scripts/h100/transport.py --timeout 600 push \
  /home/ird-demo/second-look/h100-readiness/dataset.tar \
  /workspace/second-look-h100/datasets/dataset.tar

# Extract only a trusted dataset bundle supplied by the collection owner.
python3 scripts/h100/transport.py exec \
  'mkdir -p /workspace/second-look-h100/datasets/task && tar --no-same-owner -xf /workspace/second-look-h100/datasets/dataset.tar -C /workspace/second-look-h100/datasets/task'

# Bundle a completed run with its configuration, logs, and checkpoints.
python3 scripts/h100/transport.py exec \
  'test ! -e /workspace/second-look-h100/checkpoints/trained-run.tar && tar -cf /workspace/second-look-h100/checkpoints/trained-run.tar -C /workspace/second-look-h100/runs first-adaptation'

# Return the complete run bundle after training and export have finished.
python3 scripts/h100/transport.py --timeout 600 pull \
  /workspace/second-look-h100/checkpoints/trained-run.tar \
  /home/ird-demo/second-look/h100-readiness/trained-run.tar
```

Transfer checksums must be compared before consuming a new dataset or trained artifact.
The training script requires complete per-file SHA-256 manifests for dataset, base model,
and backbone. Keep returned normalization, camera mapping, training contract, code/package
versions, and checkpoint together. A Lightning `.ckpt` is a resume artifact; it is **not**
a drop-in replacement for the main application's validated inference bundle. The main
agent still owns policy export, inference compatibility, and controller validation.

### Training launch and resume (prepared; not run)

The scripts are staged at `/workspace/second-look-h100/code/h100/`.
First fill the config and contract from the actual recorded task data. Stage them as
`/workspace/second-look-h100/configs/train.json ` and the configured contract path.
The default template is intentionally rejected with `BLOCKED: Provide dataset_root`.

```sh
# Run on H100 after resolving the isolated setup and missing inputs.
/workspace/second-look-h100/training-env/bin/python \
  /workspace/second-look-h100/code/h100/training_smolvla.py \
  --config /workspace/second-look-h100/configs/train.json --check

# Start only after the integration owner chooses the run and GPU is free.
/workspace/second-look-h100/training-env/bin/python \
  /workspace/second-look-h100/code/h100/training_smolvla.py \
  --config /workspace/second-look-h100/configs/train.json \
  > /workspace/second-look-h100/logs/train-first-adaptation.log 2>&1

# Resume the same run/config, preserving optimizer and scheduler state.
/workspace/second-look-h100/training-env/bin/python \
  /workspace/second-look-h100/code/h100/training_smolvla.py \
  --config /workspace/second-look-h100/configs/train.json \
  --resume /workspace/second-look-h100/runs/first-adaptation/checkpoints/last.ckpt \
  >> /workspace/second-look-h100/logs/train-first-adaptation.log 2>&1
```

From the Mac, send a quoted command above with `transport.py exec`, choosing an adequate
`--timeout`. If transport times out, inspect the run before retrying; do not launch a
second trainer blindly. No scheduled/background training job is created by these scripts.

## Bounded setup and handoff

The first resolver attempt could not locate the CUDA-suffixed Torch version on PyPI.
The tested correction selects the official CUDA 12.8 Torch package index with
`uv pip install --torch-backend cu128`. It installs only in the isolated venv.
The retry prepared 133 packages and completed package installation, but its final
verification exceeded the setup bound. Logs remain under our workspace:

- `logs/training-setup.log`: first resolver failure.
- `logs/training-setup-retry.log`: bounded retry, exit 124.
- `training-requirements.freeze.txt`: verified 134-line resolved package record.
- `training-imports-final.json ` / `logs/training-imports-final.err`: separate import check.

To finish/recheck setup later, using the warmed workspace cache:

```sh
python3 scripts/h100/transport.py --timeout 660 exec \
  'timeout 600 bash /workspace/second-look-h100/code/h100/training-setup.sh > /workspace/second-look-h100/logs/training-setup-next.log 2>&1'
```

This is environment setup, not a training launch. Do not interpret an incomplete
import report as success. No full training or robot control was performed.

Final resource check at 23:07 UTC: GPU utilization 0%, memory 1 MiB, no compute
processes. The owned installer process IDs no longer existed. Base Torch remained
2.8.0+cu128 and torchvision remained 0.23.0+cu128. The returned synthetic checkpoint
is on Intel at `/home/ird-demo/second-look/h100-readiness/cuda-fixture-returned.pt`.

Remaining digital gate: complete cold Studio imports in the isolated environment and
then test the actual trainer with complete local model assets and real demonstrations.
Package installation alone does not prove runtime compatibility. The two bounded
imports produced no traceback, so the cause is not established; do not label them
an API incompatibility. Full task training remains dependent on the recording owner.

Resolved versions from the isolated freeze: LeRobot 0.6.0, Lightning 2.6.6,
Transformers 5.5.4, OpenVINO 2026.3.1, Torch 2.8.0+cu128, torchvision 0.23.0+cu128.
Studio source and PhysicalAI runtime revisions are pinned above. These are installed
package records, not verified successful trainer imports.


## ACT execution phase — 2026-09-15 23:12 UTC onward

The user authorized actual ACT training, validation-based checkpoint selection,
and Intel deployment handoff. The main application owner retains genuine VLA task
reasoning; the vision owner retains Anomalib. ACT is a manipulation component,
not the VLA. The recording task retains sole robot/recorder ownership.

Fresh recording-owner confirmation: **zero real episodes**. A local-only Studio
pilot dataset exists, but no finalized task trajectories or task-ready ACT weights
are available. Synthetic fixture/checkpoint files are software-test artifacts only.
Actual task training cannot start from those files.

Destination design: train separate good-bin and reject-bin ACT skills. Installed
ACT does not consume the caption to select a destination; changing caption text
produced identical actions in the recording owner's synthetic interface test.
We will not mix conflicting destinations into one unconditioned ACT model.

The recording owner is extending the validator with `final_eval_episodes`,
real/finalized provenance and explicit episode destination/outcome evidence.
Whole-episode train/validation/final-evaluation partitions must be disjoint.
Two episodes can test a loader; a three-way task split needs additional real data.
Normalization uses training episodes only. Final evaluation does not select checkpoints.

The initial clean import probe progressed through Torch and dependency imports,
but the diagnostic process exited 139 while emitting a timed stack trace. Its trace
showed bytecode writes on the network volume. A bounded retry disables bytecode writes
and verbose per-module logging. No base environment was reinstalled or changed.

ACT runtime import result: **PASS**. Clean `python -B` imported Torch in 17.87 seconds
and Studio ACT plus Trainer in 172.64 seconds (exit 0). Nonfatal upstream Qwen
docstring/deprecation messages appeared during eager policy imports. No code or
base-environment replacement was needed. This supersedes the earlier import timeout.

The official torchvision ResNet18 ImageNet backbone was downloaded to
`/workspace/second-look-h100/cache/torch/hub/checkpoints/resnet18-f37072fd.pth`.
SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`;
46,830,571 bytes. This initializes vision features, not a trained robot task.

### ACT CUDA and export qualification (synthetic only)

The default Studio ACT architecture (384×384 input, 100-action chunk) completed
one CUDA optimizer step with an ImageNet-initialized ResNet18 backbone. There were
153 finite gradient tensors. Native checkpoint save/reload produced identical outputs
(maximum absolute error 0). Native Torch and OpenVINO FP32 exports completed.

This used generated images/numeric rows from the recording owner's synthetic fixture,
revalidated with the current validator. It is **UNSAFE_NOT_TASK_TRAINED** and must
never command a robot. The synthetic validation loss is not task-quality evidence.

- GPU run: `/workspace/second-look-h100/runs/software-qualification-v1`.
- Fit/checkpoint portion: 20.38 seconds; export/reload portion: 24.78 seconds.
  These include setup/I/O and do not establish sustained real-data throughput.
- Selected checkpoint SHA-256: `c9d46a038f7400a11325849781b5344b5204bd4be92d58729c78e271c4fc830e`.
- Intel verification bundle SHA-256: `446a9ae0743d6b171f0bc16daadfef5cbb8a201d9277c9371bef0ecbdbd9cbfb`.
- Saved source/data/model identity: `7132803c901963bf853b607981b40f56b12c760def595661e897e7329bcf9744`.

The latest launcher adds real capture/group checks and requires a matching early-real-
checkpoint Intel parity report before a budgeted task run. Sixteen local contract
tests pass. The latest launcher correctly labels synthetic final evaluation as
`NOT_APPLICABLE_SYNTHETIC`; the historical v1 result said `UNTOUCHED` despite an empty
final split. No final evaluation was performed. ACT alone does not satisfy the genuine VLA requirement; that remains
with the main application owner.

### Intel handoff status

The inference-only bundle is 344,432,640 bytes; optimizer checkpoints remain on H100.
The first SCP return measured about 70–80 KiB/s. A 4 MiB SSH byte-stream probe
matched its checksum but measured 60.35 KiB/s. IPQoS changes and parallel probes did
not improve throughput. The resumable repair preserved 34,728,960 bytes and verified
the preserved prefix. The final return completed with matching whole-file SHA-256.
The resumed phase took 543.37 seconds including setup and final verification.
Throughput improved after competing downloads ended; this does not isolate rsync's effect.

Intel artifact directory:
`/home/ird-demo/second-look/h100-readiness/software-qualification-v1`.
Intel parity report:
`/home/ird-demo/second-look/h100-readiness/act-intel-parity-v2.json `.
The native Torch and OpenVINO FP32 exports both passed through actual Studio
`InferenceModel` on the Intel Core Ultra X7 358H. No camera/controller was constructed.

| Adapter | Max absolute error | Warm median, 10 calls | Actual device |
|---|---:|---:|---|
| Native Torch | 0.0000267029 | 82.27 ms | CPU |
| OpenVINO FP32 | 0.0000381470 | 72.59 ms | CPU |

Tolerance was `atol=1e-4, rtol=1e-4`, with two CPU threads and one Torch interop thread.
One synthetic saved observation was replayed repeatedly under concurrent venue workloads;
these are software timings,
not control-loop latency or task-quality measurements. Intel versions: Torch 2.11.0+xpu,
OpenVINO 2026.3.0, PhysicalAI 0.1.2.dev70+g8e4021703. Lightning 2.6.5 warned about the
H100 2.6.6 checkpoint version but loaded successfully. Both exports remain
**UNSAFE_NOT_TASK_TRAINED**.

Machine-readable receipts live in `scripts/h100/evidence/`: CUDA qualification,
Intel parity, complete artifact return, bounded transport probes, and the pinned
SmolVLA weight download. The main owner separately verified genuine SmolVLA inference;
this ACT work does not replace that component.

Actual task training remains **BLOCKED** on finalized demonstrations and physical task
facts. The recording owner reports zero real episodes and owns the consolidated
physical collection request. The user's metadata question asks for the defect
definition, physical bin mapping, and final deadline. The H100 has no active compute
process after qualification (1 MiB used, 0% utilization at the final compute check).

### Reproducible ACT commands

From the Mac worktree, check a finalized real skill config before launching its smoke:

```sh
python3 scripts/h100/transport.py --timeout 600 exec \
  '/workspace/second-look-h100/training-env/bin/python -B /workspace/second-look-h100/code/h100/training_act.py /workspace/second-look-h100/configs/act-normal-smoke.json --mode smoke'
```

The config path above is the agreed future location, not an existing real dataset.
Fill it from `act-config.template.json ` and the recording validator's frozen outputs.
Use a separate reject-bin config and snapshot. After the corresponding real early
checkpoint passes Intel parity, record its smoke/parity reports in the training config:

```sh
python3 scripts/h100/transport.py --timeout 2400 exec \
  '/workspace/second-look-h100/training-env/bin/python -B /workspace/second-look-h100/code/h100/training_act.py /workspace/second-look-h100/configs/act-normal-train.json --mode train'

python3 scripts/h100/transport.py --timeout 2400 exec \
  '/workspace/second-look-h100/training-env/bin/python -B /workspace/second-look-h100/code/h100/training_act.py /workspace/second-look-h100/configs/act-normal-train.json --mode train --resume /workspace/second-look-h100/runs/act-normal-train/last.ckpt'
```

The template's 2,000-step/1,800-second limits are proposed bounds, not a measured
real-data budget. Set the actual budget after real smoke throughput and the deadline
are known. Reserve time for exports, return transfer, and controller-owned trials.

The exact completed synthetic artifact return uses a resumable command:

```sh
python3 scripts/h100/pull_resume.py \
  /workspace/second-look-h100/checkpoints/act-qualification-v1-intel.tar \
  /home/ird-demo/second-look/h100-readiness/act-qualification-v1-intel.tar \
  --sha256 446a9ae0743d6b171f0bc16daadfef5cbb8a201d9277c9371bef0ecbdbd9cbfb \
  --timeout 1800
```

Rerun this only for an incomplete transfer. It reuses `.part`, checks the expected
whole-file hash, and refuses to overwrite an existing final artifact. Use each real
run's own bundle path and measured SHA-256 when returning trained weights. Small
uploads still use `transport.py push INTEL_ARCHIVE GPU_ARCHIVE`; package frozen data
and code on Intel first. Transfer methods never upload to a public dataset/model hub.

### Physical validation handoff

The next real-data gate belongs to the Demonstrations task: authoritative defect
criteria, physical normal/reject bin mapping, safe recording authorization, and
completed demonstrations. Its first two episodes qualify the loader, not a three-way
evaluation. Larger snapshots need whole-episode/specimen/session split evidence.

After validated training and offline Intel loading, the controller owner conducts
approved supervised trials from `docs/EVAL_PROTOCOL.md`. Record defect false positives
and false negatives, grasp success, correct-bin placement, interventions, and cycle
time. No physical success, real holdout score, or task-trained checkpoint is claimed.


Repeat the exact Intel offline adapter check with a **new output file**:

```sh
ssh intel-robot 'timeout 300 /home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python -B /home/ird-demo/second-look/h100-readiness/act-tools/verify_act_intel.py /home/ird-demo/second-look/h100-readiness/software-qualification-v1 --output /home/ird-demo/second-look/h100-readiness/act-intel-parity-repeat.json --device CPU'
```

The pinned public SmolVLA base weights are also cached on H100 at
`/workspace/second-look-h100/models/smolvla-base/smolvla-base.safetensors`.
Size 906,712,520 bytes; SHA-256
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`,
matching upstream LFS metadata for revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
This is a public pretrained model cache, not a task adaptation. It was not returned
through the slow SSH link; the main owner completed its own Intel download.

Latest H100 script staging/help check passed at 23:51 UTC. GPU was idle, 1 MiB used,
0% utilization. All owned qualification and transfer processes completed; no budgeted
real training process was launched. The trainer is ready to consume qualified inputs,
but missing physical data prevents a trained task policy and measured task outcome.


### Action chunk metadata and capture limitations

Live replay confirms native and OpenVINO outputs have shape `[100,6]`; manifests agree.
The legacy Studio `InferenceModel.chunk_size` getter nevertheless reports 1 because
SinglePass runner metadata omits that property. No action rows were lost. The revised
verifier checks actual/reference shapes against the declared action output feature and
records the property discrepancy. The guarded v2 check passed on both backends. Existing model exports remain immutable.

The controller owner must not rely on that getter for automatic scheduling. Future
`RTCExecution ` auto-sizing otherwise defaults to 50; configure 100 explicitly or fix
its manifest fallback before using RTC. Do not add unsupported constructor arguments
to SinglePass. The tested `predict_action_chunk` path returns the full 100-row output.

The recording owner clarified that each row pairs pre-action state/images with the
same tick's requested target. Recorded action targets are not motor acknowledgments:
Studio can emit a target after failed send retries. Pilot QA must retain communication
logs, inspect measured movement and decoded images, and reject write/read failures.
Stock recordings also lack source timestamps needed to prove absolute camera/state
skew. These limits cannot be repaired by training on the current empty dataset.


## Real-image Anomalib H100 run — 2026-09-16

The user authorized H100 fitting on the newly collected LEGO photographs. This is
separate from ACT manipulation training, which still lacks real demonstrations.
The CV owner retains model selection and the main owner retains deployment.

Frozen pilot: 29 usable images from six physical specimens in one capture session.
Fifteen normal images from three good specimens train the memory bank. Validation
contains five views of a fourth good specimen and nine views of two defective
specimens. One hand-occluded image was excluded before scoring. No final-test images
exist yet. The explicit `specimen_only_pilot` policy discloses shared-session limits.

Dataset identity: `afc9facb33ed0ea06673b0eeb09c2e369ddfd47bcce839989ad56a06c39bccd6`.
Original manifest SHA-256:
`66c476b8525e00ae58a903d54b5a01a86af933fe9b16dd9e2a7853b88462b1ee`.
The 66,099,200-byte archive matched SHA-256
`5e5e02b5a940f6122a3c14d398fd9c01b0ef757ed3c4e4af9ad735f9bef17bd1` on H100.

Anomalib 2.6.0 and timm 1.0.29 match Intel. The isolated `vision-env` uses
Torch 2.8.0+cu128 and OpenVINO 2026.3.1. Base and ACT environments were preserved.
The actual PatchCore import and public pretrained backbone initialization passed.

The existing canonical fitter is CPU-only. Calling it on H100 would not prove CUDA
training. The scoped `training_anomalib.py` instead imports frozen CV validation,
preprocessing, model, calibration, and artifact helpers, moves model/images/coreset
to CUDA, and checks their actual device and finite values. It moves the fitted model
back to CPU for reload and OpenVINO export. PatchCore constructs a normal reference
bank; it does not perform optimizer steps.

The CV owner chose this bounded candidate: Wide ResNet50-2, layers2/3, 256px,
10% coreset, nine neighbors, seed17, fixed parent ROI and foreground crop. It follows
existing Intel v1/v2 experiments. CUDA random sampling and package differences can
produce different banks; do not claim identical results or a controlled speedup.

Frozen source `secondlook/anomaly.py`:
`63806e6f55164cfc68a7cc1fe8d44b78da2368454e7a679a454e6819f39b4d82`.
Eight launcher contract tests and the actual frozen-manifest preflight passed.

```sh
python3 scripts/h100/transport.py --timeout 780 exec 'timeout 720 /workspace/second-look-h100/vision-env/bin/python -u -B /workspace/second-look-h100/vision/training_anomalib.py --source-root /workspace/second-look-h100/vision/source --source-sha256 63806e6f55164cfc68a7cc1fe8d44b78da2368454e7a679a454e6819f39b4d82 --manifest /workspace/second-look-h100/vision/dataset/lego-pilot-v1/manifest.json --output /workspace/second-look-h100/vision/runs/patchcore-256-foreground-n9 --image-size 256 --sampling-ratio .1 --num-neighbors 9 --max-images 15 --max-seconds 600 --roi 0.46875 0.5740740740740741 0.7291666666666666 0.9814814814814815 --foreground-crop --calibrate > /workspace/second-look-h100/vision/logs/patchcore-256-foreground-n9.log 2>&1'
```

Use a new output/log name for any intentional repeat; the runner refuses overwrite.
An interrupted PatchCore fit restarts from the frozen normal data. There is no optimizer
state to resume. All caches, source snapshots, images, logs, and weights remain inside
`/workspace/second-look-h100`. Credentials remain on Intel.


### Completed real-image H100 result

Status: **FIT COMPLETE; PILOT QUALITY LIMITED**. CUDA feature extraction and k-center
fitting completed in 3.2476 seconds. All 15 normal images contributed; finite feature
and embedding checks passed on CUDA, and the final memory bank was `[1536,1536]`
on `cuda:0`. Peak allocated CUDA memory was 291,263,488 bytes. Total run 116.22 seconds
includes cold imports, model setup, export, and validation calibration; no CPU/H100
speedup is claimed from these unmatched environments.

Saved Torch checkpoint reload on CPU passed. OpenVINO FP32 score/map parity on one
training image passed at `atol=0.1,rtol=0.01`. This is a software check; Intel-side
validation replay is separately required. Model export and threshold calibration
completed without error. H100 returned to idle, 1 MiB / 0% utilization.

Artifact ID: `82fdd2d1db11d43aeb8376ee4b61cf4f85bd28f0102c4ed48d2ec41ac5d8a6c0`.
Threshold 59.5485610962; uncertainty half-width 1.3926738739.

| Validation truth | Classified good | Classified defective | Uncertain |
|---|---:|---:|---:|
| Good: 5 views of 1 specimen |2|0|3|
| Defective: 9 views of 2 specimens |1|7|1|

The binary threshold alone gives TP8 / FP0 / FN1 / TN5, F1=0.9412. That ignores abstention;
the table reports the actual frozen decision rule. One defective view is wrongly
accepted. Four views abstain. Do not equate the image counts with independent blocks
or claim adequate physical task performance. The vision owner received all per-image
scores and retains model selection. No test set was inferred on or used for tuning.

Evidence: `scripts/h100/evidence/anomalib-h100-v3.json ` contains the complete run,
package/code versions, source/data identities, CUDA measurements, calibration scores,
and artifact hashes. The fitted artifact is under
`/workspace/second-look-h100/vision/runs/patchcore-256-foreground-n9`.

Artifact return command (218,245,120-byte complete bundle):

```sh
python3 scripts/h100/pull_resume.py /workspace/second-look-h100/vision/patchcore-256-foreground-n9.tar /home/ird-demo/second-look/h100-readiness/patchcore-256-foreground-n9.tar --sha256 7aa83030d8cb3211d40db35812d6e195ff4f7d1cb2823a252a0ed688d3659ea5 --timeout 1200
```

The bundle contains Torch weights, OpenVINO graph/weights, preprocessing/calibration
metadata, package/code evidence, and events. It contains no credentials or image pixels.


### Returned artifact and handoff

The full 218,245,120-byte bundle returned to Intel with the expected whole-file
SHA-256. All three model files passed their individual metadata hashes after extraction.
Intel path: `/home/ird-demo/second-look/h100-readiness/patchcore-256-foreground-n9`.
Return receipt: `scripts/h100/evidence/anomalib-return-v3.json`.

The vision owner accepted this candidate only for experimental unseen-block testing,
subject to all 14 validation-image Torch/OpenVINO CPU/GPU FP32 score, map, and decision
parity on Intel. The main application owner retains activation. No service restart,
robot motion, or final-test inference was performed by this H100 task. The remaining
known defect miss is `capture-2174b6af124e42f0929315e8bdde0826.png`, score 55.6651.
Its markings remain in the input crop; preprocessing did not remove the evidence.


Intel parity subsequently **PASSED**: the vision owner compared all 14 validation
images using Torch CPU versus OpenVINO CPU/GPU FP32. All 28 score/map/decision comparisons
passed, with maximum score error 0.000175477 and map error 0.000301361. Image and model
hashes were checked; the artifact remained unchanged. Receipt:
`scripts/h100/evidence/anomalib-intel-parity-v3.json`.

The complete model also exists on the Mac at
`/Users/ohong/dev/intel-robotics/artifacts/models/h100-patchcore-v3`.
All three model-file hashes were verified after the Mac copy. Training, artifact
return, and offline Intel verification are complete. The next step is an experimental
unseen-block test under the vision/application owners, without retuning on those
results. The current model still has one false acceptance and four abstentions in
validation. No autonomous physical success is established.

## Genuine SmolVLA continuation — September 16 UTC

**PARTIAL: assets and recordings staged; no real-data policy fitting yet.**
The main application owner requested a genuine Studio SmolVLA continuation after
five closed demonstrations became available. This is separate from the completed
Anomalib fit above and the older synthetic ACT qualification.

The pinned SmolVLA base (906,712,520 bytes) and matching tokenizer/backbone files
are staged under `/workspace/second-look-h100/models/smolvla-assets`. The original
weights remain unchanged. Local paths disable duplicate backbone downloads.
Receipt: `scripts/h100/evidence/smolvla-assets-h100.json`.

The immutable five-episode snapshot contains 3,249 rows and two AV1 camera streams.
Its 99,174,400-byte archive passed SHA-256 verification on H100 before extraction
under `/workspace/second-look-h100/datasets/pilot-five`. Transfer receipt:
`scripts/h100/evidence/smolvla-dataset-staging.json`. Source recording captions
remain unchanged. The proposed developmental split uses episodes 0/1/3 for fitting
and 2/4 for validation. No untouched final-evaluation set exists yet.

Demonstrations reported finite rows, valid episode boundaries, and complete video
decodes. Offline visual review found transfers into two bowls. It did not establish
ground-truth defect labels, complete task success, or independent specimens/sessions.
The vision owner also found multiple or stationary pieces inside the frozen detector
ROI. A finite, decisive score therefore does not establish association with the
demonstrated grasp target. Pixel eligibility alone cannot approve conditioned fitting.

The native runner and no-model decoder check are documented in
`scripts/h100/smolvla-training.md`. The intended prompt uses the deployed generic
mission and actual same-frame detector outputs, never the captured class caption as
perception. Fitting remains gated on truthful demonstration metadata, exact decoded
image alignment, and verified target association. Native checkpoint export also needs
Intel adapter parity before allocating a longer training budget. Neither a staged
checkpoint nor a passing loader can establish controller readiness.

### Paused by user — September 16, about 01:57 UTC

No further training, export, parity, or transfer work will start until requested.
No owned H100 training job remains active. The original probe exited with an error;
the subsequent read-only check found no GPU compute process. No robot/session changes
were made. New ten-episode and six-episode collections remain separate and unused here.

One real CUDA optimizer update completed on the original five-episode snapshot.
It used original recorded captions, seed 17, batch 2, train episodes 0/1/3, and held-out
episodes 2/4. Training loss was 0.6423024535. One validation minibatch loss was
2.4695670605. There is no baseline comparison or policy-quality claim.

The post-training reload comparison FAILED: all 600 values differed, maximum absolute
error 51.1205139160. Source inspection identified a likely CPU/CUDA sampling mismatch:
Lightning teardown can move the fitted policy to CPU, whereas the reloaded policy was
explicitly on CUDA. Commit `3846b5d` forces and verifies matching CUDA devices and saves
early evidence. Its 49 combined tests passed; the fix has NOT been run on H100.

Preserved H100 state:

- Run/checkpoints: `/workspace/second-look-h100/runs/smolvla-real-probe-v1`.
- Selected checkpoint: `checkpoints/selected-step=000001.ckpt`; recovery: `last.ckpt`.
- Log: `/workspace/second-look-h100/logs/smolvla-real-probe-v1.log`.
- Exact config: `/workspace/second-look-h100/configs/smolvla-real-probe-v1.json`.
- Frozen source used: `/workspace/second-look-h100/code/smolvla-probe-v1/scripts/h100`.

The new HF exporter and lossless weight-delta helper are committed (`5b6c1ff`), and the
real-observation extraction helper is committed (`3197c07`). Actual export, CPU parity,
H100 decoder parity, and Intel artifact return for this policy remain unrun. The
checkpoint is **REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE**, with failed verification.

Detector-conditioned fitting remains blocked independently: 2,614 of 3,249 video frames
were invalid; the other 635 all scored anomalous, including normal-caption episodes.
None has verified target association. The completed Anomalib model above is unchanged.
