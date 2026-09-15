# Sources and targeted reading

**Reference links checked September 15, 2026.** These are technical references, not proof of what is installed or what physically works on the event PC. Match API documentation to the installed version/commit before copying commands. Do not read the whole internet or vendor entire documentation sites before starting.

## Source authority

| Source | Location | Use |
|---|---|---|
| Original competition brief | [challenge-brief.pdf](../references/challenge-brief.pdf) | Pages 1–2: required stack/workflow; page 3: action intent and deliverables; page 4: rubric. Original uploaded name was `Physical AI Challenge Technical Brief.pdf`. |
| Organizer provisioning sheet | [photo](../references/setup-notes.png), [transcription](../references/SETUP_NOTES.md) | Environment/calibration names, camera-discovery command, GPU constraints. |
| Actual Challenge-Day material | [discovery record](CHALLENGE_DAY.md) | Exact scenario and rules, still missing at packaging. The agent must locate authoritative material. |
| User's ownership/workflow choices | [AGENTS.md](../AGENTS.md), [first-run guide](../FIRST_RUN_AGENT_GUIDE.md) | Codex owns digital work; user provides hands; prefer Mac-owned source and Intel runtime. |
| User's latest terminal evidence | [initial runtime record](REMOTE_ENVIRONMENT.md) | Password SSH was reported working; distinguish from fresh key-based verification. |
| Agent inspection/results | Runtime record, artifact manifests, trial evidence | Actual installed configuration and observed results; do not overwrite with generic documentation defaults. |

## Official technical references

| Reference | Link | Consult when |
|---|---|---|
| Intel Hack-a-thon Resources | [Documentation](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html) | Locating supplied verification/launch tools and troubleshooting. The generic installation differs from the sheet's environment names. Do not rerun install/reboot recipes on a working machine. |
| Physical AI Studio | [Repository](https://github.com/open-edge-platform/physical-ai-studio) | Inspecting real VLA/policy, training, export, runtime, and application interfaces. Prefer the installed checkout and its matching revision. |
| LeRobot SO-101 | [Documentation](https://huggingface.co/docs/lerobot/en/so101) | Understanding robot configuration/calibration/control. Do not run setup/calibration/teleoperation commands before reviewing physical side effects. |
| Anomalib PatchCore | [Reference](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html) | Evaluating a normal-data anomaly-detection baseline. Candidate only, not a mandatory model. |
| SmolVLA base model | [Model card](https://huggingface.co/lerobot/smolvla_base) | Only if considering this policy. Its base-model status is not proof of readiness for our task or embodiment. |
| OpenVINO NPU device | [Documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) | Checking model/operator/shape support and device requirements before claiming NPU execution. |
| OpenVINO benchmark tool | [Documentation](https://docs.openvino.ai/nightly/get-started/learn-openvino/openvino-samples/benchmark-tool.html) | Choosing and recording measurement settings. This link tracks nightly; use version-matched help for the installed tool. Model benchmarks are separate from robot-cycle timings. |
| OpenAI remote connections | [Documentation](https://learn.chatgpt.com/docs/remote-connections) | SSH aliases, desktop remote execution topology, and remote Codex login-shell readiness. |
| OpenAI AGENTS.md | [Documentation](https://developers.openai.com/codex/agent-configuration/agents-md) | Persistent instruction discovery and scoping; keep root guidance compact. |
| OpenAI Astra prompting | [Guidance](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra) | Designing focused context, clear boundaries, and an outcome that includes completion rather than a first draft. Not a robot-safety or capability guarantee. |
| Ubuntu OpenSSH server | [Documentation](https://documentation.ubuntu.com/server/how-to/security/openssh-server/) | Public-key enrollment and account/server configuration. Our server is already reported working. |
| OpenSSH client | [Manual](https://man.openbsd.org/ssh) | Host trust, identity selection, batch testing, and port forwarding. Check local installed options too. |
| rsync | [Manual](https://download.samba.org/pub/rsync/rsync.1) | Controlled one-way transfers and dry runs. No destructive whole-directory synchronization. |
| Runpod SSH | [Documentation](https://docs.runpod.io/pods/configuration/use-ssh) | Distinguishing basic gateway access from full SSH/file-transfer capabilities on the provided H100. |

The **Second Look** concept, choice to prioritize integration, suggested test evidence, source-ownership workflow, and bounded operator interaction are project recommendations. They are not extra challenge rules. No external page certifies this project's accuracy, safety, runtime readiness, or competition result.
