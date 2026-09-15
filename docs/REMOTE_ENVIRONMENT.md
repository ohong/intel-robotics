# Remote environment — observed facts and verification

**Agent-maintained.** Preserve newer verified content when merging this template. A user-reported success is not an agent-run test. Never store passwords, private keys, tokens, or credential-file contents here.

## Initial evidence

| Field | Value | Evidence |
|---|---|---|
| Linux user | `ird-demo` | User terminal output. |
| Linux hostname | `NUC16GDKX76` | User reports successful SSH from Mac returning this hostname. |
| Last successful address/port | `10.36.254.246:22` | User-provided Wi-Fi/server output and follow-up. |
| Password SSH | USER-REPORTED SUCCESS | Not newly tested by this pack. |
| `intel-robot` alias / key auth | UNVERIFIED | Must test fresh from agent shell. |
| Conda environment names | `hack_lerobot`, `hack_physical_ai` | Organizer setup sheet, not inspected paths. |
| Calibration names | `hack_follower`, `hack_leader` | Organizer setup sheet, not motion verification. |
| H100 connection | `~/gpu_connect.txt` on PC | Sheet; access untested. |

## Discover and record

- Current agent host and OS; authoritative source root; remote deployment root; active/deployed revision.
- Actual SSH alias configuration, local identity **path**, public-key fingerprint, and trusted server fingerprint provenance.
- Remote Codex executable/version/login-shell PATH and authentication status, only as relevant.
- Conda executable and per-component interpreter paths; package versions and source revisions.
- Task material, calibration, camera/serial configuration, model/dataset manifests and observed device support.
- Owned service process IDs, start/stop commands, working UI/API endpoints and loopback tunnels.
- GPU work directory and verified transfer method; no credentials.

## Verification ledger

| Time | Check | Exact nonsecret command / evidence path | Result / limitation |
|---|---|---|---|
| Not run | Fresh agent public-key SSH | | UNVERIFIED |
| Not run | File-transfer round trip | | UNVERIFIED |
| Not run | Correct-interpreter import/smoke checks | | UNVERIFIED |
| Not run | Actual model export/inference on Intel | | UNVERIFIED |
| Not run | Live UI/API inspection | | UNVERIFIED |
| Not run | Physical trial | | NOT AUTHORIZED OR TESTED BY THIS PACK |

## Teardown notes

Document removal of only the team's newly enrolled key and temporary personal Codex login after the event. Do not perform teardown now or remove the organizer's credentials/calibration.
