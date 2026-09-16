# Native SmolVLA export

`export_smolvla.py` exports the **validation-selected** native Studio `.ckpt`.
It preserves every active and inactive model tensor. It uses the recorded
training-only state/action statistics, never identity substitutes.

## Design choice

A native `.ckpt` adapter avoids conversion, but requires another deployment load
path and carries Lightning training state. An HF-style directory fits the current
policy worker and separates inference weights from optimizer state. This helper
uses the latter approach, then compares native and HF configuration and tensors.

Studio overwrites several HF configuration fields from constructor defaults.
Consumers **must pass `export.json.hf_constructor_kwargs`** when constructing
`SmolVLA(pretrained_name_or_path=checkpoint_dir, **kwargs)`. These kwargs preserve
random input noise, all three camera slots, tokenizer length 128, denoising
settings, and action queue length. Actual camera feature keys remain the two
recorded cameras; `camera_slots` explicitly includes the masked empty slot.

## Export

Use the existing pinned environment and Studio source. The command requires
already-local assets and offline flags. It never installs packages or changes
the process environment. The exporter uses CPU for loading and parity.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python export_smolvla.py \
  --checkpoint /workspace/second-look-h100/runs/RUN/checkpoints/SELECTED.ckpt \
  --run /workspace/second-look-h100/runs/RUN/run.json \
  --result /workspace/second-look-h100/runs/RUN/result.json \
  --manifest /workspace/second-look-h100/datasets/pilot-five-qc/qc-readonly/exploratory-training-manifest.json \
  --stats /workspace/second-look-h100/datasets/pilot-five-qc/qc-readonly/train_stats.json \
  --studio-library /workspace/PINNED-STUDIO/library \
  --backbone-dir /workspace/second-look-h100/models/smolvla-assets/BACKBONE \
  --output /workspace/second-look-h100/exports/NEW-NAME \
  --observation /workspace/second-look-h100/evidence/REAL-OBSERVATION.json
```

The uppercase path components are placeholders. Read the completed `run.json`
and `result.json` for actual paths. Existing output directories are refused.

Optional `--deployment-backbone-dir /absolute/path/on/intel` writes the destination
path after local validation. Its inventory must match `export.json.backbone_files`
at the destination. This option does not assert destination availability.

The optional parity observation has this schema:

```json
{
  "evidence_kind": "real",
  "observation_id": "recorded-episode-and-frame-id",
  "episode_index": 2,
  "frame_index": 0,
  "timestamp": 0.0,
  "state": [0, 0, 0, 0, 0, 0],
  "task": "Exact recorded task text",
  "images": {
    "camera1": {"path": "camera1.png", "sha256": "ACTUAL_FILE_SHA256", "raw_rgb_sha256": "ACTUAL_SOURCE_RGB_SHA256"},
    "camera2": {"path": "camera2.png", "sha256": "ACTUAL_FILE_SHA256", "raw_rgb_sha256": "ACTUAL_SOURCE_RGB_SHA256"}
  }
}
```

This schema example contains placeholders, not valid observation evidence. Use
actual recorded state, nominal timestamp, episode/frame identity, and image bytes.
The raw hash covers contiguous uint8 HWC RGB bytes, separately from the PNG hash. Paths can be relative to the JSON file.
Parity compares complete action chunks with identical CPU RNG seeds. It restores
caller RNG state and rejects truncated prompts and nonfinite predictions.
Without `--observation`, runtime parity is explicitly `NOT TESTED`.

`export.json` records selected-checkpoint SHA, source and package provenance,
training manifest/statistics SHA, model key coverage, constructor settings,
artifact SHA inventory, and the parity result. Pipeline probes retain
`REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE`, with readiness flags false.

## Lossless smaller transfer

The Intel PC already has the pinned base safetensors file. Create a transfer
bundle containing only changed/new tensor bytes and the exact target header:

```sh
python smolvla_weight_delta.py create \
  --base /path/to/pinned/base/model.safetensors \
  --target /path/to/export/model.safetensors \
  --output /path/to/NEW-delta.zip
```

Transfer that ZIP plus the small export directory files, excluding the full
`model.safetensors`. On Intel:

```sh
python smolvla_weight_delta.py reconstruct \
  --base /path/to/pinned/base/model.safetensors \
  --delta /path/to/NEW-delta.zip \
  --output /path/to/export/model.safetensors
```

Both commands require the pinned upstream base SHA
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.
Reconstruction must reproduce the full target file SHA and byte count. The
helper does not quantize or alter tensors. Transfer savings depend on how many
weights training changed; it reports actual sizes instead of promising a ratio.

## Evidence limits

Portable tests use synthetic contract and byte fixtures. They establish schema
rejection and exact transfer mechanics. They do not establish real native/HF
inference parity. That check requires a completed real training artifact and
`--observation`. No export or parity result establishes physical task success.
