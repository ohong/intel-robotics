# Installed Studio: supervised SO-101 teleoperation

## Camera One status fix — September 15, 15:56 PDT

User requested correction of the false Offline badge while retaining the matching feed.
Saved USB color index 4 and discovered index 0 share serial `344223022426`, USB bus, and sensor identity.
The UI previously compared every fingerprint field, so it marked the working camera Offline.

Deployed a presence-only comparator in `application/ui/src/features/cameras/hardware-presence.ts`,
with focused tests, and changed `application/ui/src/routes/cameras/layout.tsx` to use it.
USB aliases require matching driver, nonempty serial and bus, equal sensor, and every other field except index.
Exact matching remains for other cases. Camera settings and stream fingerprint keys remain unchanged.
Changing the saved index or globally changing fingerprint keys was rejected because either could alter stream selection.

Validation in isolated `/home/ird-demo/studio-ui-check`: 14 unit tests passed; TypeScript, focused ESLint,
and focused Prettier checks passed. The installed dev server rebuilt the camera-list boundary in 0.28 seconds.
Browser inspection confirmed **camera 1 (low): Online**, with the actual color scene at 1920×1080, 30 fps.
The saved configuration still uses `/dev/video4`. Both robot owner PIDs 79777/79778 and Studio service PIDs stayed unchanged.
No backend restart, robot reconnect, torque change, or calibration change occurred.

Local reproducible patch files: `artifacts/studio-camera/ui-fix/src/`.
Original remote layout backup: `/home/ird-demo/studio-ui-check/backup/layout.tsx`.
This is an uncommitted local fix in the installed Studio checkout, not an upstream release.

## Camera follow-up — September 15, approximately 15:48 PDT

The operator subsequently added cameras and reported successful teleoperation. This task did not exercise robot control.

Camera `aac8a131-088f-4bbf-8146-49de68581bff` initially requested RealSense RGB and depth at 1920×1080, 30 fps.
Live device profiles support 1080p color but not 1080p depth. The installed driver enables both with identical dimensions,
even for color output. Startup failed with `Couldn't resolve requests`. Studio kept sending its zero-filled image buffer,
which explains a black image without a visible error.

Changed only that camera's resolution to 1280×720 at 30 fps through the guarded camera API.
The operator concurrently changed its name to `camera 2 (high)` and SDK serial to `243622060187`;
those choices were preserved. The subsequent error was `VIDIOC_S_FMT ... errno=16 ... Device or resource busy`.

Live SDK `physical_port` establishes these different identifier namespaces:

| RealSense SDK serial | RGB node | udev serial |
|---|---|---|
| `243722062921` | `/dev/video4` | `344223022426` |
| `243622060187` | `/dev/video10` | `344223022299` |

Second Look's task owner paused its camera/inference service and released `/dev/video10`.
At 15:48:48 PDT Studio publisher PID 76982 connected SDK `243622060187` at 1280×720, 30 fps.
A separate camera-only browser page visibly showed the actual table/laptop scene; the repaired feed is VERIFIED.
The preview is available at <http://localhost:3000/projects/e5295086-77fd-46d7-958d-c605cf9ef8d7/cameras/aac8a131-088f-4bbf-8146-49de68581bff>.
The owner was told that `/dev/video4` is also configured in Studio and should not be claimed by Second Look.
Second Look needs a coordinated shared-feed route or a free camera before resuming capture.
Other camera entries `camera 1 (low)` and `high camera` remain unchanged.
No Studio restart, robot session restart, torque command, calibration write, or robot command was issued.
Before/after configuration evidence is in `artifacts/studio-camera/`.

Verified September 15, 2026, approximately 15:38 PDT from the Mac through `intel-robot`.

## Status

**DIGITAL VERIFIED; PHYSICAL TELEOP NOT TESTED.** Studio runs on NUC16GDKX76.
No robot session, torque operation, motor setup, recording, policy, or movement command was issued in this task.
Neither serial port had a process owner at the final inspection. This does not prove torque is off.
Earlier project state reports follower torque already enabled; this task did not query or change it.

- UI: <http://localhost:3000>, on the Intel PC or this Mac through the established SSH tunnel.
- Backend: `127.0.0.1:7860`; the UI proxies `/api` and WebSockets to it.
- Safe entry: <http://localhost:3000/projects/e5295086-77fd-46d7-958d-c605cf9ef8d7/environments>.
- Installed repository: `/home/ird-demo/physical-ai-studio`, revision `c4ff730fb49f84e5102d01088d52cfff1ba62854`; application VERSION `0.1.0`.
- Existing project and environment are both named `test`.

## Exact controls — only after supervised run authorization

Opening an environment is already a hardware action in this version.
Its preview automatically opens the robot runtime WebSocket and connects both arms.
The installed SO101 driver temporarily disables torque, configures servo registers, then enables follower torque.
The initial runtime mode holds the follower position and sends position targets.
Do not open the environment detail or editor before physical readiness and run authorization are established.

1. In project **test**, select **Robots → Environments**.
2. After physical readiness and authorization, open environment **test**.
3. Wait for the **Follower** panel to connect.
4. Turn the **Teleoperate** switch on to use the leader as the follower's command source.
5. To stop leader following, turn the same **Teleoperate** switch off.

Off changes the command source to `hold`, capturing the current follower position.
It keeps the arm powered and continues position commands. It is not a torque-off or emergency-stop control.
**Restart session** reconnects hardware; it is not the stop control.
**Record dataset** is unnecessary for plain teleoperation.

After switching Teleoperate off, leaving the preview detaches that browser subscriber.
Source code switches to hold after the last subscriber disappears and exits after the default 45-second idle interval.
Other clients may keep the session alive. Browser closure or SSH loss must not serve as the primary stop procedure.
The SO101 disconnect default also preserves follower torque and holds position.
These behaviors were inspected in installed source, not exercised on the hardware.

## Preserved configuration

| Role | Saved port | USB serial | Supplied calibration |
|---|---|---|---|
| SO101 Follower | `/dev/ttyACM1` | `5B79018445` | `hack_follower` |
| SO101 Leader | `/dev/ttyACM0` | `5B79018575` | `hack_leader` |

USB by-id symlinks agree with these saved assignments. Physical arm identity remains to be confirmed.
The environment pairs this follower with this leader. Both embedded Studio calibrations exactly equal the supplied JSON files:

- `~/.cache/huggingface/lerobot/calibration/robots/so_follower/hack_follower.json`
  SHA256 `d3aeabe2a145511792e852143ade7d503116792bea3013cee50ebeb94f6245ba`.
- `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/hack_leader.json`
  SHA256 `40b621aeb80c7a986c11cd2c3dc8e2118d4a65ac22ba031d0582f9fd4073eddc`.

There are no saved Studio cameras. The OS lists RealSense video nodes 0–11 and USB2.0_CAM1 nodes 12–13.
These are device nodes, not fourteen separate cameras. Existing Second Look PID 62868 owns `/dev/video10`.
No camera was opened or reconfigured by this task. Camera-free leader–follower teleoperation is supported by this environment.

Studio uses `application/backend/.venv` (Python 3.13), separate from Conda.
`hack_lerobot` exists. `hack_physical_ai` remains absent from the previously inspected installation and current Conda registry.
Neither environment was installed, changed, renamed, or upgraded.

## Running processes and launch configuration

Independent systemd user services survive the launching SSH command. They are transient, not configured for reboot startup.

- `physical-ai-studio-backend.service`: PID 64769 (`uv`), API PID 64785.
  Working directory `/home/ird-demo/physical-ai-studio/application/backend`.
  Command `./run.sh`, with `~/.local/bin` added to PATH to find the installed `uv`.
  The inspected wrapper uses `uv run --no-sync physicalai-studio serve`; no dependency sync occurred.
- `physical-ai-studio-ui.service`: npm PID 64849, server PID 65234.
  Working directory `/home/ird-demo/physical-ai-studio/application/ui`.
  Command `source ~/.nvm/nvm.sh && nvm use && npm run start -- --host 127.0.0.1 --port 3000`.
  Installed Node v24.2.0 and npm v11.19.1 were reused.
- Mac SSH tunnel PID 11092 forwards loopback ports 3000 and 7860.
- Existing `second-look-app.service`, port 8088, was left running and untouched.

Inspection commands:

```sh
ssh intel-robot 'systemctl --user status physical-ai-studio-backend physical-ai-studio-ui'
ssh intel-robot 'journalctl --user -u physical-ai-studio-backend -u physical-ai-studio-ui -n 60 --no-pager'
```

If the Mac tunnel later closes, recreate it only after confirming those local ports are free:

```sh
ssh -fN -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:3000:127.0.0.1:3000 \
  -L 127.0.0.1:7860:127.0.0.1:7860 intel-robot
```

## Evidence and remaining gate

- Both Studio services are active; backend and UI listen on loopback 7860 and 3000.
- Direct backend and UI-proxied `GET /api/projects` returned HTTP 200 with identical project data.
- Environment API returned the actual follower–leader pairing and empty camera list.
- Browser rendered the saved project and environment list. Backend logs confirm browser GET requests returned 200.
- Browser jobs WebSocket also connected successfully; no robot runtime WebSocket was opened by this task.
- Database was already at its migration head. Installed repository remained unchanged.
- UI control and backend action-source file hashes match the inspected local source copies.
- Nonfatal upstream model-docstring/deprecation messages appeared at startup; API startup succeeded.

Before opening the environment, the operator must confirm secured arms, a clear bounded workspace,
the physical leader/follower identities, supervision, and a reachable approved stop procedure.
Support the arm safely before session connection, which can briefly release torque.
No supervised-run authorization has been requested or granted in this task.

Source references: `application/ui/src/routes/environments/show.tsx`,
`application/ui/src/features/robots/environment-form/cells/robot-cell.tsx`,
`application/ui/src/features/robots/use-joint-state.ts`,
`application/backend/src/runtime/action_source.py`,
`application/backend/src/runtime/hosts/session_worker.py`, and installed
`physicalai/robot/so101/so101.py`.
