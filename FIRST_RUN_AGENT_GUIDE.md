# First-run guide — Mac development, Intel robot runtime

**Updated:** September 15, 2026, incorporating the user's successful SSH test.  
**Scope:** setup and deployment, not a second project kickoff. Continue the build after setup.  
**Provenance:** user terminal output and confirmation; the event photo; official references in `docs/SOURCES.md`. This pack's author has not connected to the machines.

## 1. Known connection facts

| Item | Latest evidence / status |
|---|---|
| Linux account | `ird-demo`, from terminal output. |
| Machine hostname | `NUC16GDKX76`, returned in the user's successful Mac-to-PC SSH test. |
| Latest successful address | `10.36.254.246`, Wi-Fi interface `wlo1`; revalidate if connectivity changes. |
| SSH port/server | `22`; service active and listening on IPv4/IPv6 according to supplied output. |
| Earlier address | `10.38.249.230`; do not use without fresh confirmation. The reason for the discrepancy is unknown. |
| Password SSH | User reports success. The password is intentionally excluded from this repository. |
| Desired SSH alias | `intel-robot`; configuration and key enrollment are not yet verified in this pack. |
| Desired desktop name | `Intel Robot PC`. |
| Unattended key authentication | NOT VERIFIED. Test from the agent's actual execution context. |
| Remote Codex installation/authentication | NOT VERIFIED. Relevant to a Codex desktop remote project. |
| Remote project directory | Discover existing directory; `~/second-look` is only a proposed fallback. |

Do not reinstall/restart SSH or repeat server bootstrap on a connection that already works.

## 2. Choose the correct execution topology

Inspect the current host, working directory, Git root/status, and existing access before changing anything.

**Preferred user workflow:** Codex runs in the local Mac project, owns source edits there, and uses SSH to deploy and test on Linux. Mac handles portable tests/UI/replay; Linux handles installed robotics libraries, cameras, real motion, Intel inference, and physical evidence. Keep the full live controller/supervision loop on Linux, not individual joint commands streamed over SSH.

**Already in a remote Codex project:** files and shell commands are on the remote host, even though the user is looking at the Mac app. Reuse that checkout as the active source and work directly there; do not SSH into the same PC or pretend Mac files are synchronized. Record this topology and preserve the Mac copy until an explicit one-way source handoff can be made without losing work. Never create two active writers to the same deployment. OpenAI documents this distinction in [remote connections](https://learn.chatgpt.com/docs/remote-connections).

Use the H100 only for useful authorized training. It is not the Intel deployment target.

## 3. Finish only the missing SSH setup

On a local Mac session, inspect existing SSH configuration/agent identities and reuse appropriate working access. If needed, create a dedicated Ed25519 identity without overwriting files; a proposed path is `~/.ssh/id_ed25519_intel_robot`. Prefer passphrase protection with supported local agent/Keychain loading. Keep the private key on the Mac and outside the repo.

Install only its public key in the Linux user's `~/.ssh/authorized_keys`, preserving existing entries and permissions. Use existing authorized access first. If a password is unavoidable, prepare an interactive local Terminal operation and have the operator enter it there. Do not solicit it in chat, hardcode it, or inject it through command arguments, environment variables, or password-automation utilities.

Preserve verified host trust. Investigate a changed fingerprint, and obtain physical-console confirmation if needed; do not delete trust records or use `StrictHostKeyChecking=no`. A matching machine hostname alone is not cryptographic host verification. Do not change server authentication policy, open root login, disable firewalls, or modify sudoers.

Create/merge a concrete `intel-robot` host alias using the verified account/address/port and actual identity path. Suggested settings are `IdentitiesOnly yes`, `ServerAliveInterval 30`, `ServerAliveCountMax 3`, `ConnectTimeout 10`, and `ForwardAgent no`. Preserve unrelated entries and inspect effective configuration with `ssh -G intel-robot`; handle `Host`/`Match`/`Include` ordering rather than appending conflicting entries blindly. See the [OpenSSH manual](https://man.openbsd.org/ssh) and [Ubuntu guide](https://documentation.ubuntu.com/server/how-to/security/openssh-server/).

Prove fresh noninteractive public-key access from the agent shell, not an existing multiplexed password session:

```bash
ssh -o BatchMode=yes \
    -o ControlPath=none \
    -o PreferredAuthentications=publickey \
    -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no \
    intel-robot 'whoami; hostname'
```

Expected output is `ird-demo` and `NUC16GDKX76`. This command is a test to run, not evidence that it has passed.

If user Terminal works but the agent shell fails, distinguish sandbox/network permission, key access, agent/passphrase loading, routing, host trust, and server authentication. Request only the missing approval. Do not scan the venue network or disable controls. If the address has changed, get a fresh address through a permitted route and retain host-key verification.

For a **desktop remote project**, verify the remote `codex` command, authentication, and login-shell PATH using current official documentation. Reuse the correct user's existing authentication; do not expose/copy credential files or log out another participant. If installation is necessary, use the official CLI and a user-owned location, without changing robotics environments. Prepare a supported login flow and ask only for unavoidable sign-in. Do not let remote-app setup block a local Codex session that already has working SSH command access.

## 4. Discover the actual installed stack without motion

Event-specific leads from [SETUP_NOTES.md](references/SETUP_NOTES.md):

- Conda environments: `hack_lerobot`, `hack_physical_ai`.
- Follower/leader calibration names: `hack_follower`, `hack_leader`.
- Camera discovery command: `lerobot-find-cameras`.
- GPU connection instructions: `~/gpu_connect.txt`; one H100; save work under `/workspace`; never stop/restart/shut down the nodes.

The generic Intel resource page advertises a preinstalled stack and examples using `intel_dev_env`, OpenVINO 2026.3, and Anomalib 2.6.0. These are documentation details, NOT live inventory for this PC. The sheet's environment names are better initial leads. Studio can have a separate runtime. Discover actual interpreters/package versions and preserve them rather than reinstalling the generic stack. Inspect `verify_stack.py` before use; its generic environment assertion or toy-device checks may not represent this event's deployment. [Intel reference](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html).

Find existing project/starter directories, editable installs and source commits, local changes, launch scripts, task-specific material, datasets/checkpoints, calibration files, camera settings, serial mappings, device availability, and relevant running services. Scope searches to likely project/cache locations; do not read unrelated personal files or dump credentials. Review scripts for physical side effects before executing them, including robot connection helpers.

## 5. Collect a small, useful development bundle

The agent discovers and copies these; the human should not hunt for files.

| Collect when relevant | Include |
|---|---|
| Event/task material | Actual defect rules, object definitions, starter README, working examples, and launch instructions. |
| Source/dependencies | Needed Studio/LeRobot integration code, local modifications, revision IDs, manifests/lockfiles. |
| Robot configuration | Actual calibration artifacts, serial mapping, camera configuration, and available workspace/limit definitions. Copies are references, not permission to overwrite live calibration. |
| Real replay/data sample | A compact set of relevant normal/defective observations and demonstrations, with timestamps, state/action metadata, and known-label provenance. |
| Model artifacts | Model/checkpoint identity, configuration, normalization, preprocessing/postprocessing, tokenizer where used, and weights only when needed locally. |
| Environment inventory | Actual Python paths, versions, safe package inventory, and relevant observed device/test results. |

Record source paths, revisions/hashes, and purpose in a manifest. Preserve complete selected artifact bundles; avoid dangling cache symlinks. Keep large weights/datasets on the appropriate machine with a verified transfer/loading path and manifest rather than duplicating entire caches.

Exclude private keys, tokens, `.env` secrets, SSH/Codex credentials, entire Conda environments, `.venv`, `node_modules`, drivers, full model caches, and unrelated home-directory files. Create a lightweight Mac environment only for portable tasks; do not clone Linux binary environments onto macOS. Put captures, local machine configuration, and large artifacts in ignored paths. Review sanitized files before versioning.

## 6. Make the development loop work, then build

Use a dedicated team directory without editing sponsor installations. Build simple project-appropriate commands for allowlisted source deployment, non-motion remote checks, owned-service start/status/stop, results retrieval, and optional localhost-only browser tunneling. Test a small transfer round trip. Use rsync-over-SSH when available; avoid destructive `--delete`, whole-home transfers, and bidirectional folder synchronization. [rsync reference](https://download.samba.org/pub/rsync/rsync.1).

Record the authoritative source root and deployed revision. Do not deploy/hot-reload during an armed trial. Do not kill unrelated processes. Use discovered interpreter paths or a verified Conda invocation wrapper for noninteractive remote commands; never assume SSH activates the environment seen in an interactive terminal.

Inspect the real UI/API/WebSocket ports before creating tunnels. The generic Studio examples mention 3000 and 7860, but these are not confirmed endpoints. Bind forwarding to loopback and verify a real browser/API interaction, not just an HTML response. Do not expose the Codex app server to the venue network.

Stop treating setup as the main task once access, a transfer test, correct-runtime smoke checks, and the useful development loop work. Continue implementation. Hardware motion remains gated by `PROJECT_SPEC.md`, not by SSH access.

## 7. Optional H100 access

The sheet's `ssh.runpod.io` command connects **PC → H100**, not Mac → Intel PC. Inspect `~/gpu_connect.txt` as data and use the supplied key in place on the PC. Never copy that private key to the Mac or into the repo; do not enable agent forwarding as a shortcut.

Runpod's basic gateway does not support SCP/SFTP. Verify the actual transport and file-transfer method; do not assume rsync, arbitrary remote-command automation, or port forwarding works through every endpoint. Reuse an existing organizer-approved direct SSH endpoint or supported transfer path when available. Do not reconfigure/restart a node to obtain one. [Runpod documentation](https://docs.runpod.io/pods/configuration/use-ssh).

Save all GPU work under `/workspace/<team-project>`, preserve others' jobs, and return complete deployment artifacts to Intel. Training infrastructure is optional unless the chosen solution needs it; do not let its setup displace a viable existing baseline.

## 8. Record and verify

Update `docs/REMOTE_ENVIRONMENT.md` with actual paths, versioned artifacts, interpreter mapping, tested commands, and explicit verification status. Include fresh SSH authentication, transfer, interpreter/import checks, relevant local replay checks, and UI/API tunneling where applicable. Camera availability or calibration presence alone is not a verified physical trial.

Document a later teardown that removes only the team's added SSH authorization and temporary personal Codex login, with access approval. Do not execute teardown during the build. Leave no unattended live motion or robot safety dependency on an SSH connection.
