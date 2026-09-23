# Step-three SmolVLA placement

Anomalib selects a decision. SmolVLA receives the corresponding instruction, recorded cameras, and measured follower joint positions.

| Decision | Instruction |
| --- | --- |
| GOOD | Pick up the block from the inspection area and place it in the blue bin. |
| BAD | Pick up the block from the inspection area and place it in the pink bin. |
| UNKNOWN | No placement request; retain the existing controller's approved safe pause. |

`placement_input()` prepares inputs only. The existing robot-control owner must validate observation/state freshness, handoff readiness, workspace limits, and authorization. Importing these helpers does not open a port or send commands.

## Recording and qualification

Inspect existing recordings before scheduling collection. A blue/pink caption alone does not establish a successful placement demonstration. Step-one ACT pickup-to-inspection episodes are insufficient.

1. Have the existing robot-control owner pause safely and retain exclusive command ownership.
2. Confirm the actual ACT handoff pose against its measured joint state and camera view.
3. Record the selected exact instruction with all deployed camera views, measured follower states, and action targets.
4. Start each episode at that handoff, with the block in the actual inspection position.
5. Finish each episode after the block is released in the selected bin and the arm clears it.
6. Retain failures and interventions in the source recording; qualify successful episodes with explicit evidence.
7. If qualified data is missing, request five pilot demonstrations per route in one bounded supervised session.

The agent configures software, recording controls, destination captions, and data processing. The operator supplies demonstrations and otherwise unavailable physical facts. Do not execute an uninspected recording helper: it may connect, calibrate, or energize the robot.

For each accepted episode record `episode_index`, `destination` (`BLUE` or `PINK`), `frames`, the exact `task_caption`, `outcome: complete_success`, and `outcome_evidence`. Add `placement_events` containing:

```json
{
  "starts_at_actual_ACT_handoff": true,
  "handoff_evidence": "recording and measured state reference; verified by recording owner",
  "release_frame": 150,
  "clear_frame": 180,
  "release_evidence": "recording reference and verified frame",
  "clearance_evidence": "recording reference and verified frame"
}
```

The numbers above illustrate the schema, not recorded evidence. References are attestations; the portable validator does not inspect videos or certify physical outcomes.

The user subsequently confirmed that all 27 current episodes demonstrate inspection-to-bin placement and requested immediate SmolVLA training. This supports `operator_confirmed_demonstration_scope`, with sampled visual QC and explicit `individual_success_verified: false`. Each unreviewed episode remains an `operator_confirmed_placement_demonstration`; it is not relabeled `complete_success`. Do not invent release frames, bin-clearance evidence, or ACT joint-state parity. Exact handoff alignment remains an Intel deployment check.

Freeze a derivative dataset, manifest, and statistics together after recording closes. Preserve original recordings. If captions differ, rewrite the derivative task table with the exact instructions and retain the original caption/source mapping. Recompute its complete file inventory. Do not relabel an unverified destination or successful outcome.

Use `prepare_placement.py` on the closed derivative copy to create runnable manifests/configuration. It preserves original captions in `caption-provenance.json`, canonicalizes frame and episode task metadata, records source/environment identities, and computes training-only state/action statistics. The pinned native SmolVLA visual normalization is IDENTITY, so image bounds are fixed preprocessing values, not fitted pixel statistics. The two templates intentionally lack evidence and cannot launch training.

Use disjoint whole episodes: at least three training, one validation, and one final evaluation episode per destination. Five demonstrations per route therefore yield six training, two validation, and two final episodes. Fit statistics on training episodes only. Same-session pilots use `episode_pilot` and cannot establish independent generalization. Independent session/specimen groups may use `grouped_evaluation`.

For the 27-episode recording, the deterministic split uses 17 training, five validation, and five final episodes. BLUE contributes 9/3/3; PINK contributes 8/2/2. All episodes remain in exactly one split.

## Native training and evaluation

Two approaches were considered: build another trainer or extend the pinned native Studio runner. Reuse preserves its verified checkpoint loading, CUDA comparison, normalizers, GPU lock, and resumption checks. A strict `step_three_placement` scope excludes legacy anomaly-score prompts while retaining historical probe behavior.

The supported native path carries LeRobot `task` through the Studio adapter to the SmolVLA tokenizer. `selected_instruction` validates every loaded task against its frozen destination. Training records token lengths and rejects truncation. A fixed-input, fixed-noise comparison with a changed instruction must produce different predictions. This proves functional language conditioning; it does not prove correct placement.

Run preparation checks without fitting:

```sh
python3 scripts/h100/placement_contract.py --manifest /absolute/placement-manifest.json --config /absolute/placement-config.json
python3 scripts/h100/training_smolvla.py --config /workspace/second-look-h100/config/placement.json --mode check
```

Run model fitting only through the established Intel-to-H100 connection and transfer tooling. First inspect active GPU jobs and coordinate with their owner. The runner additionally refuses an occupied shared lock or active GPU compute process and requires a single H100. Keep inputs, source, caches, and outputs under `/workspace/second-look-h100`. Never train on Intel or Mac.

After staging and qualification, use the existing H100 environment:

```sh
python scripts/h100/training_smolvla.py --config /workspace/second-look-h100/config/placement.json --mode smoke
```

Use a distinct output directory for the full `--mode train` run. Resume only the same run with `--resume` and unchanged source/data/config identities. Checkpoints include optimizer recovery state. `run.json`, `packages.txt`, source hashes, normalizers, config, and base identities make the run reproducible.

Select a checkpoint using combined validation loss. Full placement training then records native offline validation loss separately for BLUE and PINK, followed by each final holdout. `placement-route-evaluation.json` and `result.json` contain the episode IDs and metrics. Final examples are used only after selection; do not tune against them. Smoke mode leaves them untouched. These are offline imitation metrics, not autonomous physical success rates.

## Intel deployment gate

Return the selected checkpoint, preprocessing configuration, train-only statistics, camera/joint mapping, source/package identities, and separate route results using the established transfer method. The existing export workflow must create the Intel artifact and verify native/export parity. Check both exact instructions with identical held-out cameras and measured states.

Measure cold load separately from warmed policy inference. Record actual Intel device, precision, per-route latency distribution, finite action dimensions, and parity tolerance. Native H100 reload parity does not prove Intel compatibility. Keep the deployment disarmed until Intel inference and the existing controller's supervised-run conditions pass. Record real placement successes, failures, interventions, and cycle times separately for both routes.

## Portable verification

```sh
python3 -m unittest discover -s scripts/h100 -p 'test_training_smolvla.py'
python3 -m unittest discover -s scripts/h100 -p 'test_placement_contract.py'
python -m unittest discover -s scripts/h100 -p 'test_prepare_placement.py'
```

These use synthetic fixtures. They check unsafe routing, missing data/evidence, leakage, normalization exclusions, and training boundaries without importing model packages or connecting hardware. The builder test needs Arrow, pandas, and NumPy from the existing Intel preprocessing environment. Native training and per-route evaluation require the prepared H100 environment and qualified data; portable success does not substitute for those checks.
