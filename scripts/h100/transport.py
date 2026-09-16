#!/usr/bin/env python3
"""Run H100 commands and move files through Intel without exporting credentials.

The Mac sends this worker to Intel over the existing ``intel-robot`` SSH alias.
Only Intel reads gpu_connect.txt. Transfer commands accept individual files;
package a directory as an archive before transferring it.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


INTEL_ALIAS = "intel-robot"
CONNECTION_FILE = "/home/ird-demo/gpu_connect.txt"
GPU_ROOT = "/workspace/second-look-h100"
EXPECTED_HOSTNAME = "55a9e1e03378"
SSH_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "ForwardAgent=no",
    "-o", "ConnectTimeout=20",
    "-o", "LogLevel=ERROR",
]


class TransportError(Exception):
    """An error safe to show without connection details."""


def encode(value: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def gpu_path(value: str) -> str:
    """Reject traversal and SCP metacharacters before touching a remote path."""
    path = PurePosixPath(value)
    root = PurePosixPath(GPU_ROOT)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path == root
        or root not in path.parents
        or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value)
    ):
        raise TransportError(f"GPU file paths must be below {GPU_ROOT}, without traversal or shell characters.")
    return str(path)


def parse_connection(contents: str) -> tuple[str, str, str]:
    """Accept only one ssh destination, one port, and one identity-file path."""
    try:
        tokens = shlex.split(contents, comments=True)
    except ValueError:
        raise TransportError("Intel connection file has invalid quoting.") from None
    if not tokens or tokens.pop(0) != "ssh":
        raise TransportError("Intel connection file must contain one SSH command.")
    destination = port = identity = None
    while tokens:
        token = tokens.pop(0)
        if token in {"-p", "-i"}:
            if not tokens:
                raise TransportError("Intel connection file is missing an option value.")
            value = tokens.pop(0)
            if token == "-p" and port is None:
                port = value
            elif token == "-i" and identity is None:
                identity = value
            else:
                raise TransportError("Intel connection file has duplicate options.")
        elif destination is None and re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+", token):
            destination = token
        else:
            raise TransportError("Intel connection file contains unsupported arguments.")
    if not destination or not port or not identity:
        raise TransportError("Intel connection file must specify destination, port, and identity file.")
    if not port.isascii() or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise TransportError("Intel connection file has an invalid port.")
    key = Path(identity).expanduser()
    if not key.is_absolute() or not key.is_file():
        raise TransportError("Intel identity file is unavailable or is not an absolute path.")
    return destination, port, str(key)


# This helper executes on H100. Path checks use real paths, so a symlink cannot
# redirect a transfer outside the dedicated directory. It contains no endpoint
# details or credentials, and reports only small JSON replies.
GPU_HELPER = r'''
import base64, json, os, shutil, stat, sys, tempfile
from pathlib import Path

request = json.loads(base64.urlsafe_b64decode(sys.argv[1]))
root = Path("/workspace/second-look-h100")

def checked(value):
    p = Path(value)
    if not p.is_absolute() or ".." in p.parts or p == root or root not in p.parents:
        raise ValueError("outside root")
    if root.resolve() != root:
        raise ValueError("redirected root")
    if not p.resolve().is_relative_to(root):
        raise ValueError("redirected path")
    return p

try:
    action = request["action"]
    if action == "root":
        if root.resolve() != root:
            raise ValueError("redirected root")
        root.mkdir(parents=True, exist_ok=True)
        result = {}
    elif action == "prepare_push":
        target = checked(request["path"])
        if os.path.lexists(target) and not request["overwrite"]:
            raise FileExistsError()
        target.parent.mkdir(parents=True, exist_ok=True)
        checked(str(target))
        fd, stage = tempfile.mkstemp(prefix=".h100-transfer-", dir=target.parent)
        os.close(fd)
        result = {"stage": stage}
    elif action == "check_pull":
        source = checked(request["path"])
        if not source.is_file():
            raise ValueError("not a regular file")
        result = {}
    elif action == "finish_push":
        target = checked(request["path"])
        stage = checked(request["stage"])
        if not stage.name.startswith(".h100-transfer-") or stage.parent != target.parent:
            raise ValueError("invalid staging file")
        if request["overwrite"]:
            os.replace(stage, target)
        else:
            # O_EXCL protects existing destinations even if they appear after
            # prepare_push. Copying also works on volumes without hard links.
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as out, stage.open("rb") as src:
                    shutil.copyfileobj(src, out)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            stage.unlink()
        result = {}
    elif action == "cleanup":
        stage = checked(request["stage"])
        if not stage.name.startswith(".h100-transfer-"):
            raise ValueError("invalid staging file")
        stage.unlink(missing_ok=True)
        result = {}
    else:
        raise ValueError("unknown action")
    print(json.dumps({"ok": True, **result}))
except FileExistsError:
    print(json.dumps({"ok": False, "error": "Destination exists; pass --overwrite only when replacement is intended."}))
    sys.exit(1)
except Exception:
    print(json.dumps({"ok": False, "error": "GPU path or file operation failed; no connection details were logged."}))
    sys.exit(1)
'''


class IntelWorker:
    def __init__(self, timeout: int):
        try:
            contents = Path(CONNECTION_FILE).read_text()
        except OSError:
            raise TransportError("Cannot read the connection file on Intel.") from None
        self.destination, self.port, self.identity = parse_connection(contents)
        self.timeout = timeout

    def ssh(self, command: str, *, capture: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", *SSH_OPTIONS, "-p", self.port, "-i", self.identity, self.destination, command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.DEVNULL,
            timeout=self.timeout,
            check=False,
        )

    def verify_identity(self) -> None:
        result = self.ssh("id -un; hostname")
        if result.returncode or result.stdout.decode(errors="replace").splitlines() != ["root", EXPECTED_HOSTNAME]:
            raise TransportError("H100 identity check failed; expected host and root user were not verified.")

    def helper(self, action: str, **kwargs) -> dict:
        result = self.ssh(shlex.join(["python3", "-c", GPU_HELPER, encode({"action": action, **kwargs})]))
        try:
            reply = json.loads(result.stdout)
        except (ValueError, UnicodeDecodeError):
            raise TransportError("H100 file operation failed before a valid reply.") from None
        if result.returncode or not reply.get("ok"):
            # Accept only our fixed error strings, never arbitrary remote output.
            if reply.get("error", "").startswith("Destination exists;"):
                raise TransportError("Destination exists; pass --overwrite only when replacement is intended.")
            raise TransportError("H100 file operation failed; inspect the requested path and permissions.")
        return reply

    def scp(self, source: str, destination: str) -> None:
        result = subprocess.run(
            ["scp", "-q", *SSH_OPTIONS, "-P", self.port, "-i", self.identity, "--", source, destination],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=self.timeout, check=False,
        )
        if result.returncode:
            raise TransportError("SCP failed; endpoint and authentication details were suppressed.")

    def push(self, source: str, destination: str, overwrite: bool) -> None:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise TransportError("Intel source must be an existing regular file.")
        destination = gpu_path(destination)
        stage = self.helper("prepare_push", path=destination, overwrite=overwrite)["stage"]
        try:
            self.scp(str(source_path), f"{self.destination}:{stage}")
            self.helper("finish_push", path=destination, stage=stage, overwrite=overwrite)
        finally:
            try:
                self.helper("cleanup", stage=stage)
            except (TransportError, subprocess.TimeoutExpired):
                print("Warning: a temporary H100 transfer file may remain.", file=sys.stderr)

    def pull(self, source: str, destination: str, overwrite: bool) -> None:
        source = gpu_path(source)
        target = Path(destination).expanduser().absolute()
        if os.path.lexists(target) and not overwrite:
            raise TransportError("Intel destination exists; pass --overwrite only when replacement is intended.")
        self.helper("check_pull", path=source)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, stage_name = tempfile.mkstemp(prefix=".h100-transfer-", dir=target.parent)
        os.close(fd)
        stage = Path(stage_name)
        try:
            self.scp(f"{self.destination}:{source}", str(stage))
            if overwrite:
                os.replace(stage, target)
            else:
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with os.fdopen(fd, "wb") as out, stage.open("rb") as src:
                        shutil.copyfileobj(src, out)
                except BaseException:
                    target.unlink(missing_ok=True)
                    raise
        finally:
            stage.unlink(missing_ok=True)


def run_worker(request: dict) -> int:
    worker = IntelWorker(request["timeout"])
    worker.verify_identity()
    action = request["action"]
    if action == "exec":
        worker.helper("root")
        command = "cd " + shlex.quote(GPU_ROOT) + " && " + shlex.join(["sh", "-c", request["command"]]) + " 2>&1"
        result = worker.ssh(command, capture=False)
        if result.returncode:
            print(f"H100 command failed (exit {result.returncode}); SSH stderr is suppressed.", file=sys.stderr)
        return result.returncode
    if action == "push":
        worker.push(request["source"], request["destination"], request["overwrite"])
    elif action == "pull":
        worker.pull(request["source"], request["destination"], request["overwrite"])
    else:
        raise TransportError("Unknown action.")
    print(f"{action} complete (Intel <-> H100).")
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--intel-worker":
        return run_worker(json.loads(base64.urlsafe_b64decode(sys.argv[2])))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for each remote operation (default: 600).")
    sub = parser.add_subparsers(dest="action", required=True)
    execute = sub.add_parser("exec", help="Run a shell command on H100, starting in the dedicated workspace.")
    execute.add_argument("command", help="One shell command string. Quote it for your local shell.")
    for name, help_text in [("push", "Copy one Intel file to H100."), ("pull", "Copy one H100 file to Intel.")]:
        transfer = sub.add_parser(name, help=help_text)
        transfer.add_argument("source")
        transfer.add_argument("destination")
        transfer.add_argument("--overwrite", action="store_true", help="Explicitly permit replacing the destination file.")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    request = vars(args)
    if args.action in {"push", "pull"}:
        gpu_path(args.destination if args.action == "push" else args.source)
    command = shlex.join(["python3", "-", "--intel-worker", encode(request)])
    result = subprocess.run(
        ["ssh", *SSH_OPTIONS, INTEL_ALIAS, command],
        input=Path(__file__).read_bytes(), stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode:
        print("Transport failed; Intel SSH diagnostics were suppressed to protect connection details.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TransportError as exc:
        print(f"Transport error: {exc}", file=sys.stdout if "--intel-worker" in sys.argv else sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Transport timed out; a remote command may still be running.", file=sys.stdout if "--intel-worker" in sys.argv else sys.stderr)
        sys.exit(124)
    except KeyboardInterrupt:
        print("Transport interrupted; a remote command may still be running.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        print("Transport failed; connection details were suppressed.", file=sys.stdout if "--intel-worker" in sys.argv else sys.stderr)
        sys.exit(1)
