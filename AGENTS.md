# Project instructions

## Outcome and ownership

Deliver the Intel Physical AI Challenge application described in `PROJECT_SPEC.md`. The operator is our eyes and hands, not our software engineer. Own every digital task available through authorized tools, including setup, data processing, training, integration, deployment, inspection, debugging, and demo preparation. Execute rather than return a software to-do list.

Proceed with routine reversible project work, disposable-fixture tests, and non-motion remote checks without asking at every step. Make implementation choices yourself. Use brief, prepared **NEED HANDS** requests only for physical actions, otherwise inaccessible task/safety facts, or enforced authentication/permission approvals. Do not ask the operator to choose models, tune training, interpret logs, or find files you can inspect.

## Context routing and authority

At kickoff read `PROJECT_SPEC.md`, `docs/CHALLENGE_DAY.md`, and any existing `AGENT_STATE.md`. Use `FIRST_RUN_AGENT_GUIDE.md` for setup or deployment work, `docs/REMOTE_ENVIRONMENT.md` for discovered configuration, and `docs/SOURCES.md` for targeted reference lookup. Do not reread every document before every edit.

The supplied brief and organizer-provided Challenge-Day rules define competition requirements. The setup photo reports provisioning details. Live inspection establishes runtime facts. Project strategy is our recommendation, not an organizer rule. Preserve provenance and flag real conflicts. The exact defect scenario is NOT in the general brief: discover it; do not invent it.

`prompts/KICKOFF.md` is the single active build kickoff. Older handoffs are historical, not cumulative instructions. Preserve applicable existing repo/global instructions when merging this pack.

## Runtime and hardware boundaries

Detect the current execution host. Prefer Mac-owned source with Linux execution; a remote Codex session already runs on Linux and must not SSH into itself or overwrite a newer source tree. Keep the complete live perception/control/supervision loop on the Intel PC. Preserve sponsor environments, calibration, drivers, and unrelated processes.

Robot motion starts disarmed. Inspect scripts before running helpers that might energize, calibrate, or move the arm. Obtain one bounded supervised run authorization with actual workspace/limits and an approved stop procedure; then own in-scope commands and trials without per-command approvals. Reconfirm when scope, workspace, or supervision changes. Exactly one process owns robot commands. Physical changes happen only after the approved safe pause. Enforce freshness, action validity, bounded recovery, and local stop behavior. No blind homing or torque release on failure; no assumption that SSH disconnect stops the robot.

Never store passwords, private keys, or tokens in the repo, prompts, logs, or command arguments. Respect tool permissions. Do not purchase resources, publish private material, alter authentication policy, or restart the provided GPU node. Keep GPU work under `/workspace`.

## Completion

Keep a compact `AGENT_STATE.md` with remaining time, current revision, running owned processes, working artifacts, next milestone, and exact blockers. Use focused parallel workers only when useful; one lead integrates and owns hardware control.

Continue through a running application, proportionate tests, visual inspection, fixes, and measured results. Label real, replayed, synthetic, and mocked evidence distinctly. A scripted router is not a VLA; visible devices are not proof of workload acceleration; manual review is not autonomous success. Never invent results.

Finish safely idle with launch instructions, evidence-backed status, and only necessary physical verification remaining. Essential physical data may be requested earlier. Do not claim execution continues after your tools stop.
