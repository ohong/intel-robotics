# Second Look — Intel Physical AI Challenge

**Deployed inspection pilot:** [Second Look](http://127.0.0.1:8088) uses Studio's low camera 1, a LEGO PatchCore detector fitted on H100, and OpenVINO GPU FP32 on the Intel PC. Second Look attaches to Studio shared frames. At 17:59:59 PDT, the camera was LIVE but the detector was INVALID; no inference ran. The last real overlay inspection was 17:43 PDT.

Source `2aa5149c57507428b28341312c1c9fd3aaaeecf2` produced release `a9ed13af508b6165`. All 121 Intel tests passed with no skips. The active source worktree is `/Users/ohong/dev/intel-robotics-integration`; the original checkout and its artifacts remain preserved.

**Quality and completion:** validation contains 14 repeated views from three specimens in one session. The deployed decision rule misses one defective view and returns UNKNOWN for four views. There is no unseen final test. The live policy remains BLOCKED, the controller remains DISARMED, and physical trials remain zero. This is an inspection pilot; the autonomous challenge remains incomplete.

See [build evidence](docs/BUILD_EVIDENCE.md), [exact pilot launch](docs/RUNBOOK.md), [demonstration](docs/DEMO.md), and [execution state](AGENT_STATE.md). The [H100 record](docs/H100_READINESS.md) includes fit and conversion receipts. Earlier smoke-model performance numbers apply only to their recorded models.

The original five episodes pass structural/video and native ACT loader checks. Their detector-conditioned training remains **BLOCKED**: all 635 predictions are ANOMALOUS, and none is linked to the picked object. A separate, later placement dataset has 27 inspection-to-bin episodes. Its frozen split, canonical instructions, and SmolVLA training path are prepared, but no step-three fine-tune or autonomous placement result is established. See the [placement status](docs/PLACEMENT_SMOLVLA_STATUS.md). The older original-caption H100 probe remains `REAL_DATA_PIPELINE_PROBE_NOT_DEPLOYABLE`.

The detector checkpoint archive is complete and verified. See [RUNBOOK.md](docs/RUNBOOK.md) for its receipt; the five episodes and VLA weights are excluded.

Recordings, detector annotations, and QC evidence are published separately at [huggingface.co/datasets/ohong/intel-robotics-dataset](https://huggingface.co/datasets/ohong/intel-robotics-dataset) (public). Pull with `hf download ohong/intel-robotics-dataset --repo-type dataset --local-dir artifacts/`.

## Continue the build

Resume from [AGENT_STATE.md](AGENT_STATE.md) and [prompts/KICKOFF.md](prompts/KICKOFF.md). The original six-hour start and cutoff are unknown; the clock has not restarted. Public judging is September 16.

The Demonstrations task owns robot control and the pending physical request. Following the camera 3 disconnection and stop request, it verified recording already STOPPED at 17:56:16 PDT, with no serial owners. Physical pose and torque remain unknown. Second Look sends no robot commands. Do not reconnect, move objects, or start a robot session before the approved safe pause and bounded authorization.

Codex owns the remaining digital work. The operator supplies physical setup, authoritative defect criteria, unseen specimens, demonstrations, and supervision. See the prepared session in [DEMO.md](docs/DEMO.md).

## Canonical files

| File | Responsibility |
|---|---|
| [AGENTS.md](AGENTS.md) | Persistent ownership rules, decision boundaries, and document routing. |
| [PROJECT_SPEC.md](PROJECT_SPEC.md) | Official challenge requirements, project strategy, and acceptance evidence. |
| [FIRST_RUN_AGENT_GUIDE.md](FIRST_RUN_AGENT_GUIDE.md) | Latest SSH facts, runtime discovery, context collection, and remote development workflow. |
| [prompts/KICKOFF.md](prompts/KICKOFF.md) | The only active end-to-end kickoff. |
| [docs/CHALLENGE_DAY.md](docs/CHALLENGE_DAY.md) | Task-specific rules and deadline; the agent discovers and records them. |
| [docs/REMOTE_ENVIRONMENT.md](docs/REMOTE_ENVIRONMENT.md) | Agent-maintained facts about actual hosts, interpreters, artifacts, and access tests. |
| [AGENT_STATE.md](AGENT_STATE.md) | Compact resumable execution state, not a second project specification. |
| [docs/SOURCES.md](docs/SOURCES.md) | Primary references and when to consult them. |
| [references/challenge-brief.pdf](references/challenge-brief.pdf) | Unmodified original four-page competition brief. |
| [references/SETUP_NOTES.md](references/SETUP_NOTES.md) and [photo](references/setup-notes.png) | Event provisioning notes and source image. |
| [.gitignore](.gitignore) | Default exclusions for credentials, environments, raw captures, and large model artifacts. |

## Retire overlapping handoffs

Do not load earlier `CODEX_GOD_PROMPT.md`, `CODEX_REMOTE_SETUP_PROMPT.md`, `MAC_REMOTE_WORKFLOW.md`, older `KICKOFF.md`, and old six-hour runbooks alongside this pack. These versions consolidate them. Preserve history outside active instruction discovery; archive old prompts rather than deleting user work. If this pack lands in an established repo, retain newer observed facts and merge its guidance rather than replacing facts with templates.

Fresh public-key SSH to `intel-robot` (`ird-demo@10.36.254.246:22`, `NUC16GDKX76`) passed during this build. See the access record for the verified configuration. No account password is included here.

## What the agent creates during implementation

Use the repo's existing conventions. Expected outputs include application source, meaningful tests, a compact replay/fault harness, launch/check/deploy entrypoints, trained/exported artifacts or a complete artifact manifest, real trial/benchmark evidence, and concise architecture/demo notes. Do not scaffold a large empty framework just to satisfy a directory diagram.

Keep this repository private unless publication is authorized; exclude credentials regardless of repository visibility.
