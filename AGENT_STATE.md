# Execution state

**Paused ("on ice") September 23, 2026 by the user.** No event clock applies. The original challenge judging (September 16) has passed. Nothing is running that this project owns: no training job, no robot-control process, no app service started by the wrap-up session.

Honest state: inspection pilot only. The detector runs on real camera frames, the placement policy is untrained, the controller is DISARMED, and there are zero autonomous physical trials. Do not relabel this.

## Where everything lives

| What | Where | Notes |
|---|---|---|
| Source and docs | GitHub `ohong/intel-robotics`, branch `main` (public) | The only branch. Every earlier branch was merged into `main` or is preserved in the archive bundle. |
| Bulk artifacts (`artifacts/`) | HF dataset `ohong/intel-robotics-dataset` (public) | This mirrors the ignored `artifacts/` tree, excluding `artifacts/runtime/` scratch logs. Restore: `hf download ohong/intel-robotics-dataset --repo-type dataset --local-dir .` |
| Full git history | HF dataset `ohong/intel-robotics-archive` (private), `intel-robotics-all.bundle` | Retired branches are kept as tags `archive/pre-cleanup/{codex/placement-smolvla,codex/placement-reconcile,codex/second-look,codex/h100-readiness,backup-pre-strip,main}`, next to the older `backup/*` tags. `SHA256SUMS` covers every file. |
| Git LFS objects | Same private archive, `git-lfs-objects.tar` | This includes the pilot deliverable `second-look-pilot-a9ed13af508b6165.tar.gz` (300084158 bytes, LFS oid `87e0aee1…`) and an older 275835296-byte tarball (`03b71d65…`). Older branches reference them as LFS pointers. |
| `pc-context/` | Same private archive, `pc-context.tar` | It holds Intel PC discovery notes. Git ignores it. |
| Placement source dataset | Intel PC `/home/ird-demo/.local/share/physicalai/datasets/a65a330b-6de7-4d01-85c5-9882037642b5` (695 MiB) | **Only copy.** Back it up to HF when Intel access returns. |
| Placement derivative | Intel PC `/home/ird-demo/second-look/placement-v1/{source,dataset,prepared,code}` | 27 episodes, 16,981 frames, 512×288. |
| Placement H100 bundle | Intel `/home/ird-demo/second-look/h100-readiness/placement-data-v1.tar` (222,894,080 B, SHA-256 `bf5a1919b0aa00eff7607afd00d74cd527fdd3265ae3ce8fef38ff551bbe3a49`) | Intended H100 target: `/workspace/second-look-h100/placement-v1/placement-data-v1.tar`. The H100 copy is **not confirmed**. |
| Frozen detector on Intel | `/home/ird-demo/second-look/artifacts/models/lego-h100-v3-82fdd2d1` | A copy is also on HF under `artifacts/models/h100-patchcore-v3/`. |
| Other Intel snapshots | `/home/ird-demo/second-look/recording-snapshots/{pilot-five-20260916T004831Z,good-blue-ten-20260916T015528Z,inspection-six-20260916T015529Z}` | The pilot-five snapshot is also on HF. The other two are on Intel only, and their QC is incomplete. |

To restore the full history on a new machine:

```bash
hf download ohong/intel-robotics-archive --repo-type dataset --local-dir intel-robotics-archive
git clone intel-robotics-archive/intel-robotics-all.bundle intel-robotics
cd intel-robotics && git remote set-url origin git@github.com:ohong/intel-robotics.git
git fetch --tags ../intel-robotics-archive/intel-robotics-all.bundle 'refs/tags/*:refs/tags/*'
tar -xf ../intel-robotics-archive/git-lfs-objects.tar -C .git
tar -xf ../intel-robotics-archive/pc-context.tar
hf download ohong/intel-robotics-dataset --repo-type dataset --local-dir .
```

## Placement facts (from the retired `codex/placement-smolvla` worktree)

- The dataset has 27 operator-confirmed inspection-to-bin demonstrations. Episodes 0–14 are BLUE (GOOD) and episodes 15–26 are PINK (BAD). UNKNOWN makes no placement request.
- The split is whole-episode: 17 train, 5 validation, and 5 final evaluation. Normalization uses only the training episodes. Manifest SHA-256: `943b996460a5d836410a5809abeb1ffe02835b8a43570c53101bf81a7bc3583c`.
- Local receipts are in `artifacts/placement/`, which is on HF. The full record is in [docs/PLACEMENT_SMOLVLA_STATUS.md](docs/PLACEMENT_SMOLVLA_STATUS.md).
- There is no fine-tune, no checkpoint, no Intel parity or latency result, and no physical placement result.

## Detector facts

- PatchCore was fitted on the H100 and runs on OpenVINO GPU.0 FP32 on the Intel PC. The median model call is 14.68 ms, against 106.14 ms on Torch CPU. Torch vs OpenVINO parity passed 28/28. Release `a9ed13af508b6165` was built from source `2aa5149c57507428b28341312c1c9fd3aaaeecf2`, and 121/121 Intel tests passed at that release.
- Validation: 14 repeated views of three specimens, all from one session. One defect was missed and four views were UNKNOWN. There is no unseen final test.
- The original five episodes are **unusable for detector-conditioned training**. Of 3249 frames, 2614 are INVALID, and the other 635 are all ANOMALOUS with no link to the target block. The older SmolVLA probe on them failed native reload and is tagged `REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE`.

## Blockers when resuming

1. On September 23, SSH to `intel-robot` (`ird-demo@10.36.254.246`, `NUC16GDKX76`) timed out. The H100 is reachable only through Intel (`scripts/h100/transport.py`), so its state is unknown. Do not infer that training continued.
2. Physical pose and torque of the SO-101 are unknown. Motion needs a fresh bounded authorization that states the workspace, limits, and stop procedure (see `AGENTS.md`).
3. Defect criteria are not authoritative yet, and unseen specimens are needed.

## Next milestone

Restore Intel access. Then verify the placement bundle hash on the H100, check the GPU lock, and run a SmolVLA placement smoke fit and then a full fit ([H100_TRAINING_GUIDE](docs/H100_TRAINING_GUIDE.md)). After that, verify the policy on Intel and wire the closed loop. The ordered plan is in the README "Next steps" section and in [docs/DEMO_READY_TASKS.md](docs/DEMO_READY_TASKS.md) (T3–T9).

## Test status at pause

On the Mac, `uv run pytest` gives 160 passed and 1 failed. The failure is `tests/test_voice.py::LiveFalRoundTrip`, which calls the live fal.ai API: it needs network access and `FAL_KEY`, and in the sandbox it timed out. Tests in `scripts/recording/` need Intel's `physicalai` package and are excluded through `testpaths`.
