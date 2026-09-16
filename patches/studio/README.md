# Installed Studio UI fixes

These patches preserve authored changes deployed to Intel Physical AI Studio revision
`c4ff730fb49f84e5102d01088d52cfff1ba62854` (application `0.1.0`). Patch paths are relative
to its `application/ui` directory. They contain source and tests, without runtime data.

## Changes

- `camera-hardware-presence.patch`: fixes false Offline badges when USB discovery
  reports a different stream-node index for the same camera. Alias matching requires
  identical driver, nonempty serial and bus, matching sensor, and matching remaining
  fields. Stream-selection fingerprints remain unchanged. Includes 14 unit cases.
- `recording-follow-controls.patch`: adds explicit Pause/Resume following, reports
  torque-on hold, and blocks episode Start through button, form, and hotkey while
  held. Accept remains available for an active episode. Recording open/reconnect
  requests hold; inference loading remains unchanged. No motion shortcut is added.
- `test-recording-controls.cjs`: eight hardware-free checks using mocked UI/runtime
  callbacks and the real parsed provider `onOpen` callback.

## Apply and check

First use an offline checkout of the pinned revision. Review existing local changes.
Replace both absolute paths below with the actual repository paths:

```sh
cd /absolute/path/to/physical-ai-studio/application/ui
patch --dry-run -p1 -i /absolute/path/to/intel-robotics/patches/studio/camera-hardware-presence.patch
patch --dry-run -p1 -i /absolute/path/to/intel-robotics/patches/studio/recording-follow-controls.patch
patch -p1 -i /absolute/path/to/intel-robotics/patches/studio/camera-hardware-presence.patch
patch -p1 -i /absolute/path/to/intel-robotics/patches/studio/recording-follow-controls.patch
npm run type-check
npm run test:unit -- src/features/cameras/hardware-presence.test.ts
node /absolute/path/to/intel-robotics/patches/studio/test-recording-controls.cjs "$PWD/node_modules/typescript/lib/typescript.js" "$PWD"
```

Use the installed dependencies; these steps do not require dependency installation.
Do not apply a live UI change during an active recording or control session.

## Evidence and limits

Packaging check: both patches applied cleanly to a disposable source fixture and
reproduced all five authored source/test files byte-for-byte. All eight recorder
callback tests passed again against that patched fixture.

Camera patch: 14 unit cases, TypeScript, focused ESLint, and focused Prettier passed
in isolated Intel `studio-ui-check`. Browser inspection confirmed the repaired
Online badge alongside the actual camera feed, with its saved stream index preserved.

Recording patch: eight mocked callback tests, TypeScript, focused ESLint, and
focused Prettier passed. The lead deployed it after confirming no active serial
owners. Runtime-page rendered inspection and physical control testing remain pending.
No service restart or hardware command was part of that deployment.

The recorder tests transpile actual source and execute its callbacks with mocks.
They do not establish browser rendering, robot behavior, or physical safety.
Opening a runtime can initialize servos and enable torque. **Hold remains powered
position control; it is not disarmed or an emergency stop.**

For current deployment and fault status, see [dataset status](../../docs/DATASET_STATUS.md).
