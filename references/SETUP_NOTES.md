# Organizer setup sheet — readable transcription

**Source:** user-supplied photograph [setup-notes.png](setup-notes.png), titled “Things are ready.” Original spelling/commands are retained where legible; list formatting is normalized. The photograph remains the source for comparison.

1. `conda activate hack_lerobot`
2. `conda activate hack_physical_ai`
3. SO-101 calibration files:
   - Follower: `hack_follower`
   - Leader: `hack_leader`
4. Check the camera indices: `lerobot-find-cameras`
5. Connecting to GPU Instance:
   - Command: `ssh <username>@ssh.runpod.io -i ~/.ssh/id_ed25519`
   - Command can be found under `~/gpu_connect.txt`
   - 1-H100 GPU instance
   - Do not stop/restart/shutdown the nodes
   - Always use `/workspace` to save your work

Printed reference:
<https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html>

## Interpretation, not additional printed instructions

The GPU command targets the provided H100, not the Intel workstation. Its identity path is relative to the machine executing it; keep the provided PC identity there. The sheet gives calibration names, not complete paths or proof that calibration is valid for the current setup. It does not specify the actual defect task, allowed manipulations, PC IP/account, or deadline. The later connection details are recorded separately in `../FIRST_RUN_AGENT_GUIDE.md`.
