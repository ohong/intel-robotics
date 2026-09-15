# Second Look — Intel Physical AI Challenge

**Start here.** This is a repo-ready context and kickoff pack, not an implemented robot application. It consolidates the project conversation through September 15, 2026. No hardware tests were performed in producing these files.

The goal is one autonomous Codex build effort that produces a working SO-101 inspection-and-response system, optimized on the supplied Intel PC. Codex owns digital engineering; the operator supplies physical actions and otherwise inaccessible observations or access approvals.

## Start the build

Merge this pack's contents into the actual project root without overwriting existing application code, verified runtime records, or unrelated instructions. Use the kickoff in [prompts/KICKOFF.md](prompts/KICKOFF.md). Choose GPT-6 Astra and the highest available reasoning setting in your Codex session.

Prefer a **local Mac project with SSH execution on the Intel PC** for the user's requested Mac-owned source workflow. A Codex remote project instead runs its shell and uses files on the remote host. Detect and record the actual topology; do not create two competing source trees. See [FIRST_RUN_AGENT_GUIDE.md](FIRST_RUN_AGENT_GUIDE.md).

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

The old general first-run guide treated SSH as unknown. The user has since successfully connected to `ird-demo@10.36.254.246:22` and observed `NUC16GDKX76`, using password authentication. Key-based access and remote Codex readiness remain unverified until tested. No account password is included here.

## What the agent creates during implementation

Use the repo's existing conventions. Expected outputs include application source, meaningful tests, a compact replay/fault harness, launch/check/deploy entrypoints, trained/exported artifacts or a complete artifact manifest, real trial/benchmark evidence, and concise architecture/demo notes. Do not scaffold a large empty framework just to satisfy a directory diagram.

Keep this repository private unless publication is authorized; exclude credentials regardless of repository visibility.
