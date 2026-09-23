# Offline placement checkpoint verification on Intel

Status: verifier contract tests passed. Model loading, H100 reference inference,
Intel inference, and physical placement remain **NOT TESTED** by this change.
The verifier opens recorded PNGs and model files only. It never opens a camera,
serial port, robot owner, or controller. It never trains a model.

Two approaches were considered. Exporting immediately to OpenVINO adds conversion
and precision differences before native compatibility is known. Native Studio
loading uses the existing supported policy constructor and preserves its exact
preprocessing. This verifier uses native CPU FP32 first. OpenVINO remains a
separate measured follow-up if the installed exporter supports this checkpoint.

## Required artifacts

Use the validation-selected checkpoint, not the last recovery checkpoint.

1. Export its native HF directory with `export_smolvla.py`: `export.json`,
   `config.json`, `model.safetensors`, `policy_preprocessor.json`, and
   `training-normalization.safetensors`. Preserve the original files and receipt.
2. Supply the matching local backbone files and pinned Studio source. The verifier
   checks their inventories against `export.json`. Intel can reconstruct weights
   using `smolvla_weight_delta.py`; the resulting target SHA must match.
3. Supply the frozen placement manifest. It must identify measured follower state,
   six-joint order, validation episodes, every camera, and the training split.
4. Use `extract_smolvla_observation.py` on H100 to save one real validation frame,
   its measured state, and all recorded cameras, including the wrist camera.
   Supply the exact episode, frame, and global index. Extraction checks the frozen
   dataset before and after decoding. Transfer its entire output directory.
5. Produce the H100 CPU reference below. Transfer its JSON receipt and preserve
   its SHA independently. It contains actual complete action chunks, sampled noise
   hashes, exact prompts, input/model identities, package versions, and timings.

The extractor supports one through three actual mapped cameras. Two-camera old
recordings retain the third masked native slot; three-camera recordings use all
three slots. Camera identity may be top-level or under `capture_provenance`.

## Generate the reference on H100

Use the prepared pinned H100 environment. All H100 output stays under `/workspace`.
The uppercase shell variables below are caller-supplied paths and verified hashes.
They are placeholders, not evidence that artifacts already exist.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/h100/export_smolvla.py \
  --checkpoint "$SELECTED_CKPT" --run "$RUN_JSON" --result "$RESULT_JSON" \
  --manifest "$MANIFEST" --stats "$TRAIN_STATS" \
  --studio-library "$STUDIO_LIBRARY" --backbone-dir "$BACKBONE" \
  --output "$NEW_EXPORT_DIR" --observation "$OBSERVATION_JSON"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/h100/verify_smolvla_intel.py reference \
  --export-dir "$NEW_EXPORT_DIR" --export-sha256 "$EXPORT_JSON_SHA256" \
  --backbone-dir "$BACKBONE" --studio-library "$STUDIO_LIBRARY" \
  --manifest "$MANIFEST" --observation "$OBSERVATION_JSON" \
  --output "$NEW_REFERENCE_JSON" --seed 20260915 --threads 4 --repeats 5
```

Both reference and verification use CPU-generated noise. Equal CUDA and CPU seeds
alone cannot establish equal noise. This helper hashes the actual native CPU
noise and refuses parity when hashes differ. It restores the caller's RNG and
sampling method after each action chunk. It does not modify noise or tensor values.

Both exact instructions use the same recorded cameras, measured state, and seed:

| Decision | Instruction |
|---|---|
| GOOD | Pick up the block from the inspection area and place it in the blue bin. |
| BAD | Pick up the block from the inspection area and place it in the pink bin. |
| UNKNOWN | No instruction, policy call, or placement action. |

UNKNOWN checks the existing instruction selector. It does not test a live detector
or robot controller. Different outputs for BLUE and PINK prove functional
instruction sensitivity only; they do not prove the correct bin is reached.

## Verify on Intel

Read-only inspection found this existing environment on September 15, 2026:

- Python: `/home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python`
- Studio: `/home/ird-demo/physical-ai-studio/library`
- Packages: torch `2.11.0+xpu`, transformers `5.5.4`, safetensors `0.8.0`, lightning `2.6.5`.

The native `SmolVLA` class import also passed in this interpreter. Import printed
unrelated Qwen3VL documentation/deprecation diagnostics and exited zero. No model
was constructed. These checks are not model-load evidence. Use an owned pinned Studio
copy when source hashes differ; do not edit sponsor installations. `hack_lerobot`
has another torch version and no default `physicalai` import, so it is not the
first native-loading choice.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python \
  scripts/h100/verify_smolvla_intel.py verify \
  --export-dir "$INTEL_EXPORT_DIR" --export-sha256 "$EXPORT_JSON_SHA256" \
  --backbone-dir "$INTEL_BACKBONE" --studio-library "$PINNED_INTEL_STUDIO_LIBRARY" \
  --manifest "$MANIFEST" --observation "$OBSERVATION_JSON" \
  --reference "$H100_REFERENCE_JSON" --reference-sha256 "$REFERENCE_SHA256" \
  --output "$NEW_INTEL_RESULT_JSON" --seed 20260915 --threads 4 --repeats 5
```

The exact same verifier file must run on both hosts. CPU FP32 comparison defaults
to `atol=1e-4`, `rtol=1e-4`, with absolute/relative errors saved. Investigate failures;
do not loosen tolerances simply to pass. Changed noise hashes mean sampling is
incomparable and require a separate diagnosis.

The helper relocates only the backbone path in a temporary local config. Large
weights remain read-only symlinks to the verified artifact. The source export is
unchanged. Every route loads a fresh policy: cold load and first action chunk are
separate from five warmed complete chunks. Filesystem cache may already be warm.
Timing includes native preprocessing, postprocessing, and noise fingerprinting;
it is not observation-to-command or physical cycle latency.

Each result reports six-joint finite output, per-joint minima/maxima, and values
outside recorded training ranges. These ranges describe demonstrations and do not
certify safe joint limits. No range flag or parity pass enables motion. Readiness
flags stay false. Failed inference writes a failure receipt with any completed
route evidence. Preflight validation errors exit before inference starts.

## Portable checks

```sh
python3 scripts/h100/verify_smolvla_intel.py --self-test
```

The standard-library tests cover changed hashes, path traversal, measured-state
provenance, validation-only sampling, action shape/nonfinite/tolerance rejection,
UNKNOWN no placement, route text, and timing summaries. They use synthetic
contract fixtures and run no model. They do not prove runtime compatibility.
