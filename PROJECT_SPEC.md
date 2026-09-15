# Second Look — project specification

**Budget:** the original six-hour build window, subject to the actual event deadline. Reuse recorded start/deadline information; this handoff does not reset the clock.  
**Engineering owner:** Codex, using the user's selected GPT-6 Astra/highest-reasoning configuration.  
**Operator role:** physical actions, otherwise inaccessible observations, and unavoidable access approvals.  
**Success:** a reproducible live system and honest evidence, not merely generated code or a polished mock.

## 1. Official challenge requirements

**Source:** [original brief](references/challenge-brief.pdf), pages 1–3. These are competition requirements, summarized without adding a particular sorting scenario.

Build one integrated system using the **SO-101 robotic arm, Intel Physical AI Studio, Anomalib, OpenVINO, and Intel Core Ultra Series 3**. Camera observations must lead to defect detection, VLA-based task reasoning, and an autonomous physical response. Optimize supported AI stages for the Intel processor using CPU/GPU/NPU resources where appropriate. The live demonstration must explain the architecture and optimization choices.

Page 3 describes the VLA layer as interpreting the task instruction and perception results, determining the response, and producing **action intent required by the robot controller**. It separately describes converting the selected action into executable SO-101 behavior.

**Interpretation, not an extra rule:** this language does not explicitly require training a fresh end-to-end motor policy. Select a genuinely supported Studio/VLA/controller architecture. A plain if/else dispatcher, an unrelated policy demo, or simply calling a generic language model a VLA does not establish compliance. Resolve a material architectural ambiguity from event-specific material or an organizer before relying on it for the judged demo; keep useful digital work moving meanwhile.

The exact defect definition, objects, operating conditions, and task constraints are deferred to Challenge Day. They are still absent from the materials supplied for this pack. Populate `docs/CHALLENGE_DAY.md` from actual sources. Do not treat object sorting, fixed presentation, fixtures, or reinspection as automatically allowed.

### Official judging weights

Source: brief page 4. The final column is our proposed evidence, not additional organizer wording.

| Official category | Points | Evidence to produce |
|---|---:|---|
| End-to-End Physical AI Solution | 25 | Continuous observation-to-autonomous-action trial using the integrated system. |
| Defect Detection with Anomalib | 20 | Actual normal/defective observations, detector outputs, and their downstream use. |
| VLA & Physical AI Studio Integration | 20 | Actual instruction/perception-to-action-intent or policy connection in Studio. |
| OpenVINO & Intel Core Ultra Series 3 Optimization | 20 | Real supported-workload deployment and measured performance/quality comparisons. |
| Robotic Execution & Reliability | 10 | Repeatability, manipulation outcomes, intervention counts, and appropriate recovery. |
| Innovation & Technical Demonstration | 5 | One useful differentiator, clear architecture, and a coherent demonstration. |
| **Total** | **100** | |

**Project priority:** the first four categories account for 85 points. Build an integrated baseline before investing in presentation extras.

## 2. Proposed experience: Second Look

Our working concept is an inspector that observes an object, detects defects with Anomalib, combines the result with the task instruction through the real Studio/VLA workflow, performs the required response, and checks the physical outcome.

Outcome verification is our implementation choice to make the loop observable. Use sensor evidence appropriate to the task; an empty pickup area alone is not proof of correct placement. Preserve association between the inspected object and the object being manipulated.

After the baseline is reliable, add at most one differentiator: bounded reinspection for an ambiguous observation, with a new physical viewpoint only if that motion is already validated and permitted. Unresolved ambiguity produces the approved safe review/stop behavior and is counted separately from autonomous task completion.

The operator view should show the real camera, Anomalib output/localization, instruction, actual VLA/policy result or action intent, controller status, outcome evidence, and timing. Use clear live/replay/mock labels. Do not invent internal model reasoning or display anomaly scores as calibrated probabilities.

## 3. Implementation freedom and practical choices

The agent chooses architecture, language, model, UI framework, IPC, test tools, and packaging from installed APIs and working examples. Reuse existing project code. Avoid elaborate new orchestration or broad framework migrations.

Find a credible VLA/controller route early. Check the actual camera keys, state inputs, joint ordering/units, action representation, normalizers, gripper conventions, and checkpoint metadata. Do not assume a base checkpoint is task-ready. Reuse appropriate artifacts, then own adaptation/training when needed; use the supplied H100 only when that shortens the critical path. See `docs/SOURCES.md` for the Studio implementation and model documentation.

Use a suitable installed Anomalib baseline. PatchCore is a candidate, not a mandate. Train/fit, select thresholds on validation data, and evaluate on separate observations/specimens as available. Do not split neighboring frames across train/test and claim independent generalization. Treat labels inferred only from model predictions as unverified; obtain irreducible physical ground truth in a small prepared request. Keep evaluation labels out of the inference/action path.

Verify an OpenVINO export and inference route early with the actual model, preprocessing, and postprocessing. Preserve a working supported fallback while investigating accelerated execution. Choose devices based on observed support and measurements, not a requirement to force every stage onto every device. Validate output quality after conversion or precision changes. NPU compatibility and benchmark configuration require version-matched documentation, not assumptions.

The complete deployed camera-to-controller loop and local fault supervision execute on the Intel PC. Codex is the development agent, not a live cloud control dependency. Mac development and H100 training do not substitute for Intel deployment evidence.

Anomaly localization is not itself a calibrated grasp target. Obtain valid object/workspace geometry through the chosen supported control method, installed configuration, and necessary physical setup. Do not invent physical coordinates or certify collision safety from a mock.

## 4. Safety and operator contract

Keep motion disarmed for bootstrap and ordinary software tests. A read/import/connect/calibration utility can have physical side effects; inspect relevant behavior before executing it.

Before supervised trials, establish the actual workspace, secured equipment, supported motion limits, fresh observations/state, one command owner, and a reachable approved stop procedure. One explicit bounded run enable can cover a test sequence; do not ask permission for every joint command. The operator stays present and clear. Pause when supervision or authorization ends, or the situation changes.

The controller must reject stale/malformed/inappropriate actions, prevent overlapping commands, bound recovery attempts, and use the approved local stop response on faults or loss of required supervision. Design the control session so it does not depend on an SSH terminal remaining open. Do not assume disconnect, automatic homing, gripper release, or torque disable is inherently safe. A software stop is not a certified emergency-stop system.

Prepare unavoidable physical requests before interrupting the operator:

> **NEED HANDS — [duration]**  
> **Do:** one specific physical action after a safe pause.  
> **Done when:** observable completion or a short confirmation.  
> **Next:** the capture, training, test, or repair the agent will run automatically.

Use cameras/tools before requesting descriptions. Batch compatible setup/capture actions. Essential examples or demonstrations may be needed during the build; do not postpone them merely to make the final handoff appear one-shot.

## 5. Evidence and acceptance

Use portable unit tests plus recorded-observation replay and a mock controller to cover empty/stale input, ambiguous detection, invalid action, unavailable hardware, timeout, failed outcome, and exhausted retries. Include robot state in replay when the selected policy needs it. Reuse a supplied physics scene only for a concrete useful check; do not build a new digital twin for presentation or equate simulation with physical validation.

Record each real trial's observation, task instruction, detector output, actual policy/action result, relevant robot state, controller result, physical outcome evidence, timing, deployed revision/configuration, and intervention. Retain failures. Report sample counts and limitations; do not cherry-pick a successful streak.

Compare actual workloads under consistent inputs and conditions. Separate cold start/compilation from warmed inference; record model-stage latency, observation-to-command delay, and complete physical cycle time separately. Report actual execution devices, tested precision/configuration, defect errors, task success, and interventions. Device enumeration and a toy-model test prove neither application acceleration nor robotic success.

**Definition of done:** the intended task runs from a clean launch on Intel, required components genuinely participate, models/configuration/artifacts are reproducible, meaningful failure paths were checked, real results are recorded, the running operator view was inspected, and concise demo/submission materials are prepared. The agent repairs failures rather than assigning them to the operator.

Use status labels: `HARDWARE VERIFIED`, `DIGITAL/REPLAY VERIFIED`, `BLOCKED`, `NOT TESTED`. Physical verification still pending is not full completion.

## 6. Time and handoff

Use remaining time, not an automatically renewed six hours. Recover the deadline from repo/event material. If absent, mark a provisional work budget and do not present it as the organizer's deadline.

With a full six hours available, aim for discovery in the first 15 minutes, a real integrated physical cycle within two hours when prerequisites permit, reliability next, then measured optimization. Reserve the final hour for fixes, clean-launch verification, evidence, and rehearsal; freeze nonessential features. Compress these phases for the actual remaining time. Do not spend the first two hours perfecting SSH, UI, or simulations.

Maintain only compact execution state and necessary documentation. Use focused subagents if helpful without competing hardware access or resource-heavy jobs. Produce a runnable application, launch/check/deploy interfaces, saved artifacts or reproducible manifests, tests/replay fixtures, results, and brief architecture/demo notes. Prepare publication/submission; obtain any required final account or external-publication approval.

The main handoff is a running, safely idle system with an actual application address, evidence-backed status, and one prepared physical verification session—not software homework. Continue useful digital work while physically blocked, within the active tool session; preserve state when execution must pause.
