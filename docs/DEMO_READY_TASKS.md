# Demo-ready orchestrator task list

> **Archived plan (written for the September 16 judging).** The project was paused on September 23, 2026. The rubric gap, target architecture, and T3–T9 tasks remain the best next-step plan, but the branch/worktree map below is historical: those branches were merged or deleted, and everything now lives on `main` plus the backups listed in [AGENT_STATE.md](../AGENT_STATE.md).

**Audience:** one general/orchestrator agent. Execute; do not return a software to-do list to the operator. Spawn focused workers where noted. One lead owns Intel hardware, robot commands, and integration.

**Judging date:** 16 September 2026 (calendar only; cutoff time UNKNOWN). Public judging is today. Compress everything after T2 if time is short. Freeze voice, UI polish, and reinspection until a live inspect→decide→place cycle exists.

**Success for the judged demo:** one integrated closed loop on the Intel PC: Studio camera → Anomalib defect decision → Physical AI Studio VLA action intent → SO-101 placement, with OpenVINO on a real supported workload, honest evidence, and a rehearsed explanation. Independent component demos do not score the 25-point end-to-end category.

**Current honest state (do not relabel):** inspection pilot only. Last verified Intel app (release `a9ed13af508b6165`, source `2aa5149c57507428b28341312c1c9fd3aaaeecf2`): camera LIVE, detector often INVALID, policy BLOCKED, controller DISARMED / backend UNAVAILABLE, autonomous physical trials **0**. Second Look still sends **no robot commands**.

Label every result `REAL`, `RECORDED_REAL`, `REPLAY`, `SYNTHETIC`, or `MOCK`. Never invent trial success, NPU acceleration, or VLA reasoning text.

---

## Rubric gap

| Criteria | Pts | Status | Gap that costs the points |
|---|---:|---|---|
| End-to-End Physical AI Solution | 25 | NOT TESTED as a loop | Camera and detector run; VLA is a separate replay; SO-101 is unused by this app. Judges will see independent demos. |
| Defect Detection with Anomalib | 20 | PARTIAL (DIGITAL + limited REAL) | Frozen PatchCore + OpenVINO GPU FP32 exists. Detector output is **not consumed** by a robotics workflow. No unseen-specimen test. 1 defect miss + 4 UNKNOWN / 14 same-session views. Physical presence UNVERIFIED. |
| VLA & Physical AI Studio Integration | 20 | PARTIAL probe only | One Intel CPU SmolVLA call on a recorded image with **synthetic** state/normalizers (`ENGINEERING_PROBE_ONLY`, `task_ready=false`). Live app never constructs `StudioPolicyBridge`. Placement training prepared, not fitted. |
| OpenVINO & Core Ultra Series 3 | 20 | PARTIAL | Detector GPU.0 FP32 is real (median 14.68 ms model-only). Current pilot has **no NPU measurement**. ACT OpenVINO is synthetic (`UNSAFE_NOT_TASK_TRAINED`). No VLA OpenVINO path. |
| Robotic Execution & Reliability | 10 | NOT TESTED | Zero Second Look motion. Safety/guard code is mock-verified and **not deployed**. Recovery from uncertain detections is unproven on hardware. |
| Innovation & Technical Demonstration | 5 | PARTIAL | Operator view exists. Proposed differentiator (bounded reinspection) is unimplemented. Voice instruction matching is in-progress and is **not** a VLA. Demo script currently confesses incompleteness. |

**Priority:** first four categories = 85 points. Build the integrated baseline before voice, reinspection, or presentation extras.

---

## Do not redo (already evidenced)

Reuse these. Do not refit, re-download, or rebuild them unless a hash check fails.

- Intel attach-only Studio camera path: `physicalai/camera/UVCCamera/4/frame`, 1920×1080 RGB.
- Frozen LEGO PatchCore: Intel `/home/ird-demo/second-look/artifacts/models/lego-h100-v3-82fdd2d1`; Mac `artifacts/models/h100-patchcore-v3`. Artifact ID `82fdd2d1…`. OpenVINO GPU FP32; 28/28 Torch vs OpenVINO CPU/GPU parity on 14 validation images.
- App, tests, deploy helpers on `main` (merged from `codex/second-look` + `codex/h100-readiness`).
- H100 env under `/workspace/second-look-h100`, SmolVLA assets, transport scripts.
- Safety gate + native policy guard (digital/mock only). Guard is **not** live.
- Recording-control Studio UI patch (deployed, not physically exercised).
- Five-episode snapshot `pilot-five-20260916T004831Z` (structural/video + small ACT loader). **Do not** use it for detector-conditioned VLA training: 2614/3249 INVALID, remaining predictions all ANOMALOUS, `target_association=UNKNOWN`.
- Public bulk data: [huggingface.co/datasets/ohong/intel-robotics-dataset](https://huggingface.co/datasets/ohong/intel-robotics-dataset). Now includes `artifacts/models/h100-patchcore-v3/` (`model.pt`, `model.bin`, `model.xml` + metadata), pushed via `hf upload`, alongside the 30 lego-capture-20260915 images and lego-pilot-v1 validation evidence already there. Git ignores weights/recordings (commit `4007320` and later). Do not commit bulky files; push new bulk artifacts to this HF repo instead.
- Placement prep on `codex/placement-smolvla` (unmerged, **not a runnable app**): 27 operator-confirmed inspection-to-bin episodes (15 BLUE / 12 PINK, 16981 frames). Derivative 512×288 at `/home/ird-demo/second-look/placement-v1/`. Manifest SHA `943b9964…`, bundle SHA `bf5a1919…` (222,894,080 B) at `/home/ird-demo/second-look/h100-readiness/placement-data-v1.tar`. H100 extract **not** confirmed in the transfer receipt. Intel verifier code exists; **no model load**. `task_ready=false`. Sampled QC 5+5 stills: transfer visible, `individual_success_verified: false`.
- Original-five SmolVLA probe **did** complete one CUDA step (train loss 0.6423, val 2.4696) then **failed native reload** (max abs err 51.12). Tag `REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE`. `codex/h100-readiness` commit `3846b5d` adds a same-device CUDA reload comparison (Mac tests only; **not** re-run on H100). Use that fix for T4; do not ship the failed probe.

---

## Branch and worktree map

Work from Mac source; run perception/control on Intel (`intel-robot`, `ird-demo@NUC16GDKX76`, `/home/ird-demo/second-look`). Exactly one process may own robot commands. Do not SSH into a session that is already on the Intel PC.

| Ref | Path | Keep / merge / ignore |
|---|---|---|
| `main` `b15f420` | GitHub HEAD; also `/Users/ohong/dev/intel-robotics` when checked out | Canonical docs + inspection app. Missing placement + voice. |
| `oh-voice-control` (current) | `/Users/ohong/dev/intel-robotics` | Voice STT/TTS + uncommitted hold-to-talk UI. Instruction-only; never motion. Park or finish **after** T5, not before. |
| `codex/placement-smolvla` `8d87667` | `/Users/ohong/.codex/worktrees/d2be/intel-robotics` | **Highest-value unmerged work, not a runnable app** (no `secondlook/`). Copy source only: `scripts/h100/prepare_placement.py`, `placement_contract.py`, `placement-training.md`, `verify_smolvla_intel.py`, `docs/placement-intel-verification.md`, placement tests. 27-episode Intel dataset `a65a330b-6de7-4d01-85c5-9882037642b5`. Do not commit `artifacts/`. |
| `codex/second-look` `cf518d7` | `/Users/ohong/dev/intel-robotics-integration` | Already merged into `main` as `c5d9c38`. Do not treat as a second writer. |
| `codex/h100-readiness` `22ed0d5` | `/Users/ohong/dev/intel-robotics-h100` | Already merged as `0952791`. H100 scripts live on `main`. |
| `oh-integrate-all` `adaaccb` | `.claude/worktrees/project-purpose-goals-d27502` | Stale snapshot. Ignore. |
| `backup-pre-strip`, `feat/project-purpose-goals-d27502` | local only | History / init. Do not merge; they still carry bulky blobs. |

`AGENT_STATE.md` on `main` is stale (pause + old SHA). Placement worktree `AGENT_STATE.md` is the later training-authorization record. Refresh `AGENT_STATE.md` after T0.

**Caption conflict (do not pool silently):**

- Older five episodes: good → coral/pink bowl; defective → light-blue/gray bowl.
- Newer `good-blue-ten-*`: good → blue; QC incomplete; camera placement changed.
- Placement-v1: GOOD → **blue** bin, BAD → **pink** bin, UNKNOWN → no placement. User asked to train SmolVLA on these 27 first.

---

## Target architecture (already chosen; implement it)

Do not invent a scripted if/else VLA. Do not send Anomalib localization as a grasp pose.

```
Studio cameras (existing publishers)
        → Second Look attach-only subscriber
        → Anomalib PatchCore (OpenVINO GPU FP32) → GOOD | BAD | UNKNOWN
                │
                ├ UNKNOWN → no policy call; approved safe pause / review
                └ GOOD/BAD → exact Studio instruction
                      GOOD: "Pick up the block from the inspection area and place it in the blue bin."
                      BAD:  "Pick up the block from the inspection area and place it in the pink bin."
                → Studio SmolVLA.select_action(Observation)  [cameras + measured follower state]
                → NativePolicyGuard (freshness, bounds, sole writer)
                → existing Studio/SO-101 owner
        OpenVINO: keep detector GPU FP32; measure NPU; export VLA/ACT only if parity holds
```

UNKNOWN must not place. Detector scores are distances, not probabilities.

---

## Execution rules for the orchestrator

1. Read `AGENTS.md`, `PROJECT_SPEC.md`, `docs/CHALLENGE_DAY.md`, this file, then `docs/RUNBOOK.md` / `docs/EVAL_PROTOCOL.md` / `docs/NATIVE_POLICY_GUARD.md`.
2. Detect host. Prefer Mac-owned source + Intel execution. Do not overwrite a newer tree. Do not start a second robot owner.
3. Motion stays disarmed until T2 records workspace, stop procedure, and one bounded supervised authorization. Then the **same** owner runs in-scope commands without per-joint approval.
4. Spawn workers for parallel digital work (QC, tests, OpenVINO benches, docs). Never spawn two agents that can send robot commands or occupy the H100 at once. Check `.gpu-job.lock` and GPU occupancy before any H100 job.
5. Preserve sponsor envs, calibration, drivers. Keep GPU files under `/workspace`. Never store secrets in git, logs, or argv.
6. NEED HANDS only for physical actions, inaccessible facts, or auth. Batch them.
7. After each task, write compact `AGENT_STATE.md`: remaining time, revision, owned processes, artifacts, next task, blockers.

---

## Tasks

Execute in order unless a task says it can run in parallel. Skip a task only when its **Done when** is already proven with a dated receipt.

### T0 — Recon, clock, and single source of truth

**Rubric:** unblocks all 100. **Spawn:** 2–3 read-only explorers (Intel inventory, H100 inventory, git/worktree merge plan). **NEED HANDS:** none unless SSH key/passphrase is missing.

**Execute**

1. Confirm host, `ssh intel-robot 'whoami; hostname; date'`, H100 reachability via existing `scripts/h100/transport.py`. Do not restart GPU nodes.
2. Inventory live processes: Studio, `second-look-app.service`, serial owners on ttyACM0/1, camera publishers, GPU jobs. Do not kill unrelated owners.
3. Hash-check the frozen detector and placement dataset paths on Intel. Confirm the 27-episode bundle location and whether the H100 copy finished (`placement-data-v1.tar` SHA `bf5a1919b0aa00eff7607afd00d74cd527fdd3265ae3ce8fef38ff551bbe3a49`).
4. Recover submission cutoff, demo duration, and submission format from event materials / organizer pages. Write facts into `docs/CHALLENGE_DAY.md` with source and time. UNKNOWN is allowed; guesses are not.
5. Merge `codex/placement-smolvla` **source** (`scripts/h100/placement_*.py`, `prepare_placement.py`, `placement-training.md`, `docs/placement-intel-verification.md`, tests) onto a working branch from `main`. Leave `oh-voice-control` unmerged until T8. Do not copy `artifacts/` or `pc-context/` into git.
6. Rewrite `AGENT_STATE.md` for this session.

**Done when:** one working tree owns integration; Intel/H100 reachability is current; CHALLENGE_DAY has dated facts or explicit UNKNOWN; no second robot owner.

### T1 — Challenge-Day rules that the demo will be scored on

**Rubric:** 25 + 20 Anomalib + 10 robot (wrong defect/bin mapping zeros the loop). **Spawn:** one docs researcher for any published kickoff sheet; lead records operator facts. **NEED HANDS:** photo or one precise answer if the handout is only physical.

**Execute**

Fill `docs/CHALLENGE_DAY.md` and the pending table in `docs/EVAL_PROTOCOL.md` from an authoritative source, not from model scores:

- Exact defect definition and acceptable normal variation
- Allowed LEGO set / specimen rules
- Bin geometry and acceptance (blue/pink vs older coral mapping)
- Workspace, fixtures, camera arrangement, operating variation
- Reinspection / human-review / retry rules
- Per-episode timeout and trial count
- Demo duration and submission path

Until a field is confirmed, keep the placement-v1 mapping (GOOD→blue, BAD→pink, UNKNOWN→no move) labeled **operator-reported**, not organizer-certified.

**Done when:** each scoring-relevant field is CONFIRMED with source/time, or explicitly UNKNOWN with the provisional value the demo will use.

### T2 — Safety authorization (gate for all motion and recording)

**Rubric:** 10 robot + makes 25 legal. **Spawn:** none. Lead only. **NEED HANDS:** required.

Prepared request (reuse/update `docs/DEMO.md`; do not create a second controller):

> **NEED HANDS — ~10 min after the approved safe pause**
> **Do:** Confirm arm pose and reachable approved stop. Identify workspace limits, defect criteria, blue and pink bins, and camera 3. Place **one** identified block in the inspection ROI; keep every other block outside it. Stay present for the bounded session.
> **Done when:** pause, scope, stop, and one bounded authorization are recorded.
> **Next:** T3 evaluation + T4/T5 software inside that scope.

**Execute**

1. Inspect scripts before any helper that might connect, calibrate, or torque-on (`docs/STUDIO_TELEOP.md`). Opening a Studio runtime can energize motors; hold is **not** an e-stop.
2. Record workspace, joint limits, stop procedure, command owner id, and authorization expiry in `runtime` evidence. Deploy `NativePolicyGuard` only after this permit exists.
3. Confirm camera publishers still match the placement recordings (three cameras on newer sets). Do not reconnect to “fix” a camera if that opens a robot session.

**Done when:** `HARDWARE VERIFIED` pause + authorization receipt exists. Physical pose/torque is no longer UNKNOWN, or is UNKNOWN with an explicit reason not to move.

### T3 — Make Anomalib demo-scorable and downstream-consumed

**Rubric:** 20 Anomalib (quality + **consumed by robotics**). **Spawn:** vision worker for eval/crops; integration worker for the decision→instruction adapter. Parallel with T4 after T2.

**Execute**

1. Unseen-specimen evaluation of frozen `lego-h100-v3-82fdd2d1` **without retuning**. Report misses, false accepts, UNKNOWN, specimen/session overlap. Status today: experimental 14-view validation only; `unseen_block_test=PENDING`.
2. Confirm ROI `[900,620,1400,1060]` still matches the live cameras after the placement change. If newer recordings miss the crop, collect a small labeled set (authoritative labels, not model self-labels) and refit only if the frozen model cannot see the block. Prefer adapting presentation (one block in ROI) over refitting.
3. Implement the live adapter: detector disposition → GOOD/BAD/UNKNOWN → exact placement instruction or no-op. Keep evaluation labels out of the inference path. UNKNOWN must not call the policy.
4. Show score, map, disposition, device, and “not a probability” on the operator view (already mostly there). Wire the same disposition into the policy request.
5. Optional if time: NPU bench of **this** artifact (not the old 128 px smoke). Keep GPU FP32 if NPU fails parity.

**Done when:** a live or recorded-real frame produces GOOD/BAD/UNKNOWN that the policy adapter consumes; unseen eval table exists with honest limits; detector no longer dead-ends at `policy BLOCKED` for a valid isolated block.

### T4 — Task-adapted Studio SmolVLA (placement-v1)

**Rubric:** 20 VLA. **Spawn:** H100 trainer (one); Intel verifier after return. Coordinate GPU lock with any prior probe owner.

**Reuse:** `codex/placement-smolvla` prep. Intel dataset `/home/ird-demo/.local/share/physicalai/datasets/a65a330b-6de7-4d01-85c5-9882037642b5`. Prepared `/home/ird-demo/second-look/placement-v1/`. Split 17/5/5 (BLUE 9/3/3, PINK 8/2/2). Base `lerobot/smolvla_base` `c83c3163b8ca9b7e67c509fffd9121e66cb96205` already on H100.

**Do not** train on the original five captioned “pick up a good/defective piece” episodes. Do not condition on ground-truth class text as a substitute for Anomalib. Keep the failed probe receipt. Original Studio captions (do not feed these to the live detector path): BLUE `"Move the good block to the blue plate"`; PINK `"Move the bad block with black scratches kto the pink plate"` (typo in source). Canonical training text is the pick-up-and-place instruction pair above.

Split seed 42, `episode_pilot` (same session → no independent generalization): train `[0,1,2,3,4,5,9,10,11,15,16,18,19,21,23,25,26]`; val `[7,8,13,20,22]`; final `[6,12,14,17,24]`.

**Execute**

1. Confirm H100 idle and lock free. Finish bundle transfer if needed; `tar --no-same-owner`.
2. Merge the `3846b5d` CUDA-device reload comparison into the placement runner before smoke. `placement_contract.py` + `training_smolvla.py --mode check` then `--mode smoke` (one CUDA step, native reload with **hashed CPU-generated noise**, instruction-sensitivity check BLUE vs PINK). Abort if reload error repeats the 51-scale failure.
3. Bounded `--mode train`. Select checkpoint by validation loss, then report BLUE/PINK val and untouched final holdout. Tag offline metrics as **not** physical success. Individual episode success is still `individual_success_verified: false` unless QC proves it.
4. Export with `export_smolvla.py`. Return checkpoint, preprocessor, train-only stats, camera/joint map, `export.json`.
5. Intel native CPU verify per `docs/placement-intel-verification.md` (`verify_smolvla_intel.py`). Both instructions, same held-out cameras + **measured** follower state (no synthetic stats). Record latency. OpenVINO export only if native parity passes and time remains.
6. Reject live use unless `task_adapted` evidence is complete: camera keys, 6-D joint order/units, gripper convention, normalizers, `controller_ready` still false until T5.

**Done when:** a hash-pinned Intel artifact runs `select_action` on a real recorded placement observation for both instructions, with measured state, `task_ready` for **offline** use, and an explicit “not physically validated” label.

### T5 — Close the live loop (the 25-point task)

**Rubric:** 25 end-to-end + remaining 20 VLA (integration across perception, reasoning, manipulation). **Spawn:** integration worker for wiring; tests worker for failure-path suite. Lead owns the process that can send commands.

**Execute**

1. Construct `StudioPolicyBridge` from `InspectionRuntime` when the T4 artifact exists. Pass live image(s), Anomalib disposition, T3 instruction, measured follower state, capture timestamps.
2. Put `NativePolicyGuard` on the **existing** Studio send path (`docs/NATIVE_POLICY_GUARD.md`). One writer. Reject stale/malformed/out-of-limit actions, overlapping commands, expired permit, lost heartbeat. SSH disconnect must not be the stop mechanism.
3. UNKNOWN / INVALID observation → no send; approved pause/review. Exhausted `ReviewBudget` → REVIEW, counted separately from autonomous success.
4. Outcome check: not “pickup area empty.” Record destination evidence (camera on the target bin, or other sensor the layout actually provides). Preserve object identity from inspect to place.
5. Expand tests beyond mock units to **replay + mock controller** covering PROJECT_SPEC §5: empty/stale, ambiguous detection, invalid action, unavailable hardware, timeout, failed outcome, exhausted retries. Today `InspectionRuntime` never calls `SafetyGate`, `ReviewBudget`, or the policy bridge.
6. Deploy a new release with `scripts/remote.py`. Always pass the pilot camera + detector paths; defaults are still high-camera / `scene-smoke`. Confirm `/api/status` shows policy ≠ BLOCKED on a valid isolated block, controller status truthful (DISARMED until armed permit; never claim hardware torque from the software flag).

**Done when:** on Intel, one supervised dry cycle (or first authorized live cycle) traces a single observation id through detector → instruction → VLA chunk → guard → (disarmed log or authorized send). Receipt JSON with revision, devices, timings. If still disarmed, the receipt must show the command **would** have been the sole writer and was blocked by the permit — that is digital closed-loop evidence, not a physical point.

### T6 — OpenVINO / Core Ultra measurements for the **deployed** workloads

**Rubric:** 20 optimization. **Spawn:** one bench worker. Parallel with T5 digital tests; do not disturb an armed session.

**Execute**

1. Re-measure the **live** detector artifact: Torch CPU, OpenVINO CPU, GPU FP32, NPU if the IR supports it. Separate load/compile vs warmed inference. Current 14.68 ms GPU number is model-only, one image, venue contention.
2. Measure VLA/action inference device and latency on Intel (CPU today; XPU/OpenVINO only with parity).
3. Record observation-to-command delay and full physical cycle time separately once T7 runs.
4. Write a 6–8 line “workload placement” card for the demo: what runs where and why GPU FP32 was kept (map parity; GPU FP16 failed on the smoke model).

**Done when:** a dated table names actual devices, precision, parity, and what was **not** measured. No claim that device enumeration equals application acceleration.

### T7 — Supervised physical trials (reliability)

**Rubric:** 10 robot + converts T5 from digital to HARDWARE VERIFIED. **Spawn:** none. Same command owner as T2/T5.

**Execute** (inside the T2 permit; reconfirm if layout/supervision changes)

1. Isolated-block presentations: at least one GOOD and one BAD if specimens exist; keep failures.
2. Record per trial: observation, instruction, detector output, VLA action, robot state, controller result, physical outcome, cycle time, revision, intervention.
3. Uncertain detector → pause/review (or organizer-allowed reinspection only). Count interventions.
4. Do not home, torque-release, or reconnect to infer safety after a fault. Use the approved local stop.

**Done when:** a trial log exists with ≥1 attempted REAL cycle, including failures. If zero successful placements, the log still scores honesty; do not hide misses. Repeatability numbers only if more than one comparable trial happened.

### T8 — Operator view, voice, and one differentiator

**Rubric:** 5 innovation + demo clarity. **Spawn:** UI worker. **Only after T5 exists** (or in parallel on a branch that cannot arm the robot).

**Execute**

1. Operator view must show: real camera, Anomalib overlay/score, instruction, actual VLA/action (not invented reasoning), controller status, outcome, timing, live/replay/mock badges.
2. Finish or park `oh-voice-control`: hold-to-talk currently uncommitted in `secondlook/static/*`. Voice may set the **mission** string only. It is not manipulation and not a stop. Requires `FAL_KEY` in `/home/ird-demo/.config/secondlook/fal.env` mode 0600 — never in git.
3. Bounded reinspection (spec differentiator) **only if** T7 is stable and Challenge-Day allows a second viewpoint. Otherwise explain UNKNOWN→pause as the recovery behavior.

**Done when:** a browser pass on the Intel-tunneled app (`127.0.0.1:8088`) matches live status; voice either works with a key or is hidden when `UNAVAILABLE`.

### T9 — Clean launch, demo rehearsal, submission packet

**Rubric:** all categories (presentation of architecture, placement, optimization, observed results). **Spawn:** docs worker for the one-page talk track; lead does the clean launch.

**Execute**

1. Clean launch from `docs/RUNBOOK.md` on Intel: disarmed `runtime/armed.json`, correct camera, frozen detector, T4 policy artifact, loopback-only HTTP. Exact-release tests green on that identity (historical 121/121 is the **old** release; voice/placement tests are extra).
2. Rewrite `docs/DEMO.md` as a 3-minute talk: architecture diagram, why Studio library + attach-only camera, why GPU FP32, detector limits, VLA instruction split, one live cycle or an honest “digital closed loop, physical pending” beat. Judges will ask about workload placement.
3. Submission: follow T1 format. Likely: running app address, this repo (private), HF dataset, detector archive receipt, policy checksums. Do not upload weights to GitHub.
4. Idle-safe handoff: app up, controller disarmed, launch commands, evidence-backed status, only remaining NEED HANDS listed.

**Done when:** a stranger can launch from the runbook, the demo script matches the live UI, and `AGENT_STATE.md` lists running PIDs, revision, and what is still NOT TESTED.

---

## Suggested spawn plan (time-boxed)

```
T0 recon ──┬── explorer: Intel processes/artifacts
           ├── explorer: H100 lock/datasets
           └── explorer: merge placement source (no robot)

T1 facts  (lead + optional web/docs worker)

T2 NEED HANDS  (lead only; blocks T5 send and T7)

After T0+T1:
  T3 vision+adapter     ║  T4 H100 train/export   ║  T6 benches (idle GPU/NPU)
         \                    /
          \                  /
           T5 wire loop + tests (lead integrates)
                    |
                   T7 physical trials (lead)
                    |
              T8 UI/voice  →  T9 demo/submit
```

If the deadline is hours, not a day: **T0, T1, T2, T4 smoke+best checkpoint, T3 adapter, T5 digital loop, T6 detector table, T9 talk.** Skip extra data collection, NPU, voice, and reinspection. A truthful digital closed loop plus the existing live detector beats another disconnected training probe.

---

## NEED HANDS batch (prepare once)

Do not drip-feed the operator. One session should cover:

1. Safe pause, stop procedure, workspace/bin confirmation, camera 3, one block in ROI.
2. Authoritative defect labels for unseen eval specimens (not model predictions).
3. Supervision for T7 trials.
4. Any organizer handout photo if T1 cannot find the rules digitally.
5. Browser sign-in only if fal.ai or HF auth is required and no existing key works.

---

## Explicit non-goals

- Do not merge `backup-pre-strip` or other pre-strip branches (multi-GB blobs).
- Do not call a caption router or generic LLM a VLA.
- Do not treat empty pickup area, overlay screenshots, or `nvidia-smi` as success.
- Do not pool the five-episode coral route with placement-v1 blue/pink data.
- Do not run `fireconnect`, homing, or torque-off as a recovery strategy.
- Do not claim the paused original-five SmolVLA probe is deployable.
- Do not start training if GPU occupancy or `.gpu-job.lock` shows another owner.
