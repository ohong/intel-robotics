# Step-three SmolVLA placement status

Updated September 23, 2026. This note closes the `codex/placement-smolvla` worktree. It does not claim a trained placement policy.

## Integrated work

`main` already contains the placement preparation, strict instruction contract, native SmolVLA trainer changes, and Intel inference verifier from the branch. The source files match; `main` also preserves the trainer's executable bit. Existing ACT and Anomalib tooling remains on `main`.

The branch's placement receipts are copied to local `artifacts/placement/`. That directory is intentionally ignored by Git because it includes recorded camera images and machine-specific data. The frozen Intel source and derivative remain at `/home/ird-demo/second-look/placement-v1/{source,dataset}`. The 222,894,080-byte transfer archive is at `/home/ird-demo/second-look/h100-readiness/placement-data-v1.tar`, with SHA-256 `bf5a1919b0aa00eff7607afd00d74cd527fdd3265ae3ce8fef38ff551bbe3a49`. An Intel receipt records that archive; it does not prove an H100 copy completed.

## Verified preparation

- The closed Studio placement recording has 27 episodes and 16,981 frames: 15 BLUE episodes and 12 PINK episodes. The source copy matched its file hashes.
- The derivative retains all episode frame counts and 30 FPS. Seven videos decode at 512 × 288, H.264 CRF 18, GOP 15. Original captions remain in `caption-provenance.json`; only derivative task metadata uses the two exact bin instructions.
- Whole-episode splits are 17 train, five validation, and five final evaluation. Both destinations occur in each split. State/action normalization uses training episodes only. Manifest SHA-256: `943b996460a5d836410a5809abeb1ffe02835b8a43570c53101bf81a7bc3583c`.
- Five sampled BLUE and five sampled PINK recordings show transfer into the named bin and terminal gripper clearance. The samples do not establish uninterrupted success for every episode or exact ACT handoff joint-state parity.
- The native Studio code path tokenizes task text and consumes camera images and measured follower state. The placement contract maps GOOD → BLUE, BAD → PINK, and UNKNOWN → no placement request. Anomalib supplies the decision; SmolVLA is the planned placement policy.
- Portable placement/SmolVLA tests, an Intel Arrow builder test, the H100 native import, and an idle H100 inventory passed during preparation. See `artifacts/placement/` for the dated receipts. These checks do not establish training or deployment success.

## Open gates

No step-three H100 fine-tune, selected checkpoint, route-specific offline loss, Intel model parity/latency result, or supervised physical placement result is recorded in this work. A one-step older SmolVLA probe failed reload parity and is not deployable.

Intel SSH to the last known address `10.36.254.246` timed out on September 23. The current H100 transfer and job state therefore remain unverified. Before resuming training, re-establish the trusted Intel connection, verify the transfer archive hash on H100, inspect active GPU jobs and the shared lock, then use the frozen manifest. Keep all training under `/workspace/second-look-h100`; use Intel only for deployment inference. Physical trials remain with the existing robot-control owner and require current supervised scope and stop procedure.

The prepared commands and artifact checks are in `scripts/h100/placement-training.md` and `docs/placement-intel-verification.md`. The current code is preparation for a language-conditioned policy, not an autonomous placement result.
