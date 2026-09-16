<!-- take-notes-session: "01a0a746-6411-7492-b884-db143a7b3bdd" -->
# Remote GPU training prompts

<!-- take-notes-payload-sha256: 42f74bbdaf0cdc9f519e9061d6fbddac9a348c99662c7e26536b8ecf344d4607 -->
Coverage: This note preserves the three full training prompts and the later remote-GPU addendum visible in this conversation. Earlier history is available only in part and through a summary.

## Use these prompts later

Append the remote training requirement below to each component prompt. The prompts are copied verbatim. Dataset counts describe the original prompt context; each prompt requires a fresh inspection before training.

### ACT: move blocks to inspection

```text
Train the ACT policy for step one: pick up one LEGO block from the starting area and place it in front of the close-up inspection camera.

Read AGENTS.md, PROJECT_SPEC.md, and AGENT_STATE.md. Continue from the current state. Preserve existing work and any active training.

Inspect dataset 11f099a2-1f91-432a-b582-7f031c035ae3, “Move blocks to inspection area.” It previously contained six episodes, 4,930 frames, and three camera streams. Verify current contents and successful demonstrations. Freeze a consistent snapshot before training; do not train against an active recording directory.

Use our native Studio ACT training and export path on the H100. Confirm that the configuration matches all three cameras, robot state, actions, and calibration. Both good and defective blocks have the same destination in this stage.

Split by whole episodes. With this small dataset, label results as a pilot. Start or resume training, save reproducible checkpoints, evaluate held-out action prediction, and verify export and Intel inference compatibility. Do not treat offline loss as proof of successful robot motion.

Own preparation, training, debugging, and reporting. Report dataset version, configuration, metrics, artifacts, and remaining physical validation. Coordinate hardware trials through the existing robot-control owner; do not start another robot-control process.
```

### Anomalib: inspect for defects

```text
Train and validate the Anomalib inspection component using close-up camera images.

Read AGENTS.md, PROJECT_SPEC.md, and AGENT_STATE.md. Inspect existing CV datasets, models, and running work before starting. Preserve useful artifacts and avoid duplicate jobs.

The output contract is:
- GOOD → blue bin
- DEFECTIVE → red bin
- UNKNOWN or invalid inspection → pause without bin placement

Use the actual agreed defect definition. Do not infer ground truth from block color or destination. If the defect definition is unavailable, ask only for that authoritative physical fact.

Audit existing labeled images and specimen identities. Use normal examples for an appropriate anomaly-detection baseline. Separate training, threshold calibration, and evaluation by physical specimen where possible. Do not place adjacent frames of the same specimen across splits and report them as independent evidence.

Select and train an Anomalib model suited to the available data and Intel deployment. Calibrate the decision threshold using labeled validation data. Handle missing blocks, occlusion, stale frames, and uncertain results explicitly.

Evaluate missed defects and false rejects. Report dataset limits honestly, including any lack of unseen-specimen evaluation. Export the model, verify prediction consistency, and benchmark actual Intel inference.

Own all digital work. Request only prepared physical captures if essential examples are missing. Deliver reproducible artifacts and integration instructions without taking robot-control ownership.
```

### SmolVLA: move inspected blocks into bins

```text
Fine-tune SmolVLA for step three: move a block from the inspection position into the bin selected by the inspection result.

Read AGENTS.md, PROJECT_SPEC.md, and AGENT_STATE.md. Preserve existing work and inspect the current SmolVLA implementation and datasets.

The confirmed mapping is:
- GOOD → BLUE bin
- DEFECTIVE → RED bin
- UNKNOWN → no placement action

Use these consistent task instructions:
1. “Pick up the block from the inspection area and place it in the blue bin.”
2. “Pick up the block from the inspection area and place it in the red bin.”

Condition the policy on the selected instruction, camera observations, and measured robot state. Anomalib supplies the defect decision. SmolVLA executes the selected placement task.

First inspect whether successful demonstrations exist for both routes. Step-one ACT demonstrations alone are insufficient. If placement data is missing, prepare the recording setup and request a short physical session with five pilot demonstrations per route. Start each demonstration at the actual ACT handoff pose and finish after releasing the block and clearing the bin.

Verify that the supported SmolVLA training path uses language conditioning. Fine-tune a pretrained checkpoint on both routes using a frozen dataset. Split by whole episodes and keep evaluation examples for each destination. Coordinate H100 usage with existing training; do not interrupt another job.

Save reproducible checkpoints and evaluate both routes separately. Verify Intel inference compatibility and latency before supervised motion. Distinguish offline evaluation from physical placement success.

Own preparation, training, debugging, and integration. Use the existing robot-control owner for physical trials. Report exact missing data if training cannot yet begin.
```

### Remote training requirement: append to all three

```text
REMOTE TRAINING REQUIREMENT

Run all model training and fine-tuning on the provided remote H100 node. Do not train on the MacBook or Intel robot PC. Use the Intel PC for deployed inference and verification.

Read FIRST_RUN_AGENT_GUIDE.md, especially Section 7, docs/REMOTE_ENVIRONMENT.md, and existing H100 training handoffs before connecting. Continue from the current setup.

The organizer’s GPU connection instructions are in ~/gpu_connect.txt on the Intel PC. The connection goes from the Intel PC to the H100. Use the supplied private key in place. Never copy it to the Mac or repository, print credentials, or enable agent forwarding.

Reuse the existing verified connection and transfer tooling. The basic Runpod SSH gateway does not support SCP/SFTP, so inspect the established transfer method before using it.

Keep all GPU work under /workspace/<team-project>. Inspect active jobs and coordinate GPU use before launching. Preserve other sessions and training runs. Never stop, restart, or shut down the provided node.

Verify the training process actually uses the H100. Return checkpoints, preprocessing configuration, evaluation results, and deployment artifacts to the Intel deployment workflow.

If remote access fails, diagnose it using the repository instructions. Report the exact blocker instead of silently falling back to local training.
```

## Repository references

- [H100 access guide](/Users/ohong/dev/intel-robotics/FIRST_RUN_AGENT_GUIDE.md:102)
- [Remote environment reference](/Users/ohong/dev/intel-robotics/docs/REMOTE_ENVIRONMENT.md)
