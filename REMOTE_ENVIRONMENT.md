# Intel Robot PC remote environment

Verified 2026-09-15 from the Mac agent execution environment.
SSH development access works. Desktop connection itself is not tested.

## Connection values

| Item | Value |
|---|---|
| Desktop display name | Intel Robot PC |
| SSH alias | intel-robot |
| Address / port | 10.36.254.246 / 22 |
| Verified user | ird-demo |
| Verified hostname | NUC16GDKX76 |
| Remote project directory | /home/ird-demo/second-look |
| Mac SSH executable | /usr/bin/ssh, OpenSSH 10.3p1 |
| Mac private identity path | /Users/ohong/.ssh/id_ed25519_intel_robot |
| Public-key fingerprint | SHA256:EBNqt/AS9mZw/N+XHkDkZaQDzAMogqee7BJFqhJF/uE |
| Trusted ED25519 host fingerprint | SHA256:jaF8Nw6LEme5OsWYwIovxp8U9p0luNjNq+QO98lCkeY |

## Changes

- Preserved the OrbStack Include and added one concrete Host intel-robot block.
- Config backup: `/Users/ohong/.ssh/config.backup-intel-robot-20260915-144137`.
- Config uses the dedicated identity, IdentitiesOnly yes, ServerAliveInterval 30,
  ServerAliveCountMax 3, ConnectTimeout 10, ForwardAgent no, AddKeysToAgent yes,
  UseKeychain yes, and StrictHostKeyChecking yes.
- Local Terminal bootstrap enrolled the dedicated public key. Its identity is loaded
  in the Mac SSH agent. Private key and config have mode 600; remote .ssh is 700 and
  authorized_keys is 600. No password or token is stored in this document.
- Created `/home/ird-demo/second-look` as the fallback project folder.
- Added `/home/ird-demo/.local/bin/codex` symlink to existing
  `/usr/lib/chatgpt/resources/codex`. Existing .profile already includes ~/.local/bin.
  No download, shell-profile edit, system-tool replacement, or Conda change was needed.

## Codex readiness

Remote user's actual login shell is `/bin/bash`.
Verified through `"$SHELL" -lc` over a fresh SSH connection:

- Command path: `/home/ird-demo/.local/bin/codex`.
- Version: `codex-cli 0.154.0-alpha.6.2`.
- Authentication status: `Logged in using ChatGPT`.

The existing login was left untouched. Account ownership and a live model request
were not verified. No Mac authentication was copied and no new login was created.

[Official OpenAI documentation](https://learn.chatgpt.com/docs/remote-connections)
requires a concrete SSH alias, a remote login-shell Codex executable, and remote
Codex authentication. In the desktop app, open **Settings → Connections**, enable
**intel-robot**, use **Intel Robot PC** as its display name, and select
**/home/ird-demo/second-look**. The tools exposed no SSH-host configuration control;
Computer Use cannot operate Codex. No remote saved project appeared in the final
project listing, so desktop connection completion remains unverified.

## Python environments

| Environment | Interpreter | Result |
|---|---|---|
| hack_lerobot | /home/ird-demo/miniforge3/envs/hack_lerobot/bin/python | Noninteractive invocation passed; Python 3.12.14 |
| intel_dev_env | /home/ird-demo/miniforge3/envs/intel_dev_env/bin/python | Noninteractive invocation passed; Python 3.11.16 |
| hack_physical_ai | Not found | Absent from Conda registry, miniforge envs, and bounded /home and /opt directory search |

Do not assume intel_dev_env replaces hack_physical_ai. No environments were installed
or changed. Hardware packages and robot-control code were not invoked.

## Tested commands and results

Fresh unattended key-only connection (no reused control socket):

```sh
ssh -o BatchMode=yes -o ControlPath=none \
  -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no intel-robot 'whoami; hostname'
```

Passed with exactly `ird-demo` and `NUC16GDKX76`.
`ssh -G intel-robot` verified effective connection settings.

```sh
ssh intel-robot
ssh -o BatchMode=yes intel-robot '"$SHELL" -lc "command -v codex; codex --version; codex login status"'
ssh -o BatchMode=yes intel-robot '/home/ird-demo/miniforge3/envs/hack_lerobot/bin/python -c "import sys; print(sys.executable); print(sys.version)"'
```

Both hosts have `/usr/bin/rsync`. A unique temporary file was uploaded into a unique
`.ssh-check.*` directory under second-look, downloaded, and compared
byte-for-byte. SHA256: `8afb625f6f7f21290f592f9e1b60147f032b94306e2f990456279a9b3202d4db`.
All test files and temporary directories were removed. This proves project read/write
access and a bidirectional transfer round trip, not automatic synchronization.

Transfer command forms used (substitute explicit source/destination files):

```sh
rsync -a -e 'ssh -o BatchMode=yes -o ControlPath=none' LOCAL_FILE intel-robot:/home/ird-demo/second-look/REMOTE_FILE
rsync -a -e 'ssh -o BatchMode=yes -o ControlPath=none' intel-robot:/home/ird-demo/second-look/REMOTE_FILE LOCAL_FILE
```

`ss -ltnp` showed SSH, DNS and local printing listeners only. No project web UI was
running. No tunnel was needed or tested, and no service or robot process was started.

## DHCP address changes

1. Confirm the new address on the PC.
2. Back up SSH config and change only this alias's HostName.
3. Preserve known_hosts records. Verify the new address's host fingerprint against
   the trusted value above before adding trust. Investigate any mismatch.
4. Run `ssh -G intel-robot` and repeat the fresh key-only identity check.

Never bypass host checks or reuse the stale address from earlier attempts.

## Event teardown — documented only

1. Read the Mac dedicated .pub file and verify the fingerprint above.
2. Back up the remote authorized_keys file with mode 600.
3. Remove only entries matching that public key's type and base64 value, preserving
   every other entry and any options/comments. Check the result before replacing
   authorized_keys; keep its mode 600.
4. This setup created no Codex authentication. Do not log out the pre-existing
   account. If a later task creates temporary personal authentication, record its
   account and CODEX_HOME, then use `codex logout` only for that specific login.
5. Remove only the symlink `/home/ird-demo/.local/bin/codex` if still pointing to
   `/usr/lib/chatgpt/resources/codex` and no longer needed.
6. Remove the dedicated agent identity with `ssh-add -d ~/.ssh/id_ed25519_intel_robot`.
   Remove only its corresponding Keychain item if saved, then the dedicated key pair
   and Host intel-robot block if no longer needed. Preserve unrelated SSH entries.

Keep project files unless separately authorized to delete them. No robot movement,
calibration, firmware, drivers, SSH service, firewall, sudoers, GPU, or Runpod
credentials were changed. The Mac repository was not migrated or synchronized.
