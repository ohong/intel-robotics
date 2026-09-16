#!/usr/bin/env python3
"""Continuously mirror verified Intel CV captures to the Mac over rsync."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import uuid


DEFAULT_REMOTE_HOST = "intel-robot"
DEFAULT_REMOTE_PATH = "/home/ird-demo/second-look/artifacts/cv/lego-capture-20260915"
DEFAULT_LOCAL_PATH = "/Users/ohong/dev/intel-robotics/artifacts/cv/lego-capture-20260915"
SUBPROCESS_TIMEOUT = 20.0
STATUS_NAME = "sync-status.json"
JOURNAL_NAME = "captures.jsonl"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]*$")
_REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_EXCLUDED_PARTS = {"tmp", "thumbnail", "thumbnails", "preview", "previews"}
_EXCLUDED_STEMS = {"contact-sheet", "contact_sheet", "thumbnail", "preview"}

LOG = logging.getLogger("cv_capture_mirror")


class MirrorError(RuntimeError):
    """An expected mirror operation failure that can be retried."""


def validate_remote_host(value: str) -> str:
    """Allow SSH aliases and simple host names without shell metacharacters."""
    if not value or not _HOST_RE.fullmatch(value) or value.startswith("-"):
        raise ValueError("remote host contains unsupported characters")
    return value


def validate_remote_path(value: str) -> str:
    """Allow an absolute POSIX path without traversal or remote-shell syntax."""
    if not value or not _REMOTE_PATH_RE.fullmatch(value):
        raise ValueError("remote path must be an absolute safe POSIX path")
    normalized = value.rstrip("/") or "/"
    path = PurePosixPath(normalized)
    if not path.is_absolute() or ".." in path.parts or any(part == "" for part in path.parts):
        raise ValueError("remote path must not contain traversal or empty components")
    if normalized == "/":
        raise ValueError("remote path must identify a project directory")
    return path.as_posix()


def validate_local_path(value: str) -> Path:
    """Resolve a local destination and reject a pre-existing symlink."""
    if not value or "\x00" in value:
        raise ValueError("local path is empty or contains NUL")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = Path(os.path.abspath(os.fspath(path)))
    if path == Path("/"):
        raise ValueError("refusing the filesystem root as a capture destination")
    if path.exists() and path.is_symlink():
        raise ValueError("refusing a symlink capture destination")
    return path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one bounded subprocess without invoking a local shell or exposing output."""
    try:
        return subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, timeout=SUBPROCESS_TIMEOUT,
                              check=False, shell=False)
    except FileNotFoundError as exc:
        raise MirrorError(f"required command is unavailable: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MirrorError(f"command timed out after {SUBPROCESS_TIMEOUT:.0f}s: {command[0]}") from exc
    except OSError as exc:
        raise MirrorError(f"could not start {command[0]}: {type(exc).__name__}") from exc


def _check_result(result: subprocess.CompletedProcess[str], command_name: str) -> None:
    if result.returncode:
        # Keep remote command output local and out of status files. Do not print it because
        # SSH diagnostics can include paths or environment details that should stay local.
        raise MirrorError(f"{command_name} failed with exit code {result.returncode}")


def _is_excluded_png(relative: PurePosixPath) -> bool:
    parts = tuple(part.lower() for part in relative.parts)
    if any(part.startswith(".") or part in _EXCLUDED_PARTS for part in parts):
        return True
    stem = relative.stem.lower()
    if stem in _EXCLUDED_STEMS or "preview" in stem or "thumb" in stem:
        return True
    return False


def _safe_relative_png(root: Path, value: object) -> tuple[Path, PurePosixPath] | None:
    """Turn a journal path into a safe local path below root."""
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return None
    try:
        relative = PurePosixPath(value)
    except (TypeError, ValueError):
        return None
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return None
    if relative.suffix.lower() != ".png" or _is_excluded_png(relative):
        return None
    candidate = root.joinpath(*relative.parts)
    try:
        resolved_root = root.resolve()
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if candidate.is_symlink():
        return None
    return candidate, relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _usable_png(path: Path) -> bool:
    """Reject missing, empty, and obviously truncated/non-PNG files."""
    try:
        if not path.is_file() or path.stat().st_size <= len(PNG_SIGNATURE):
            return False
        with path.open("rb") as stream:
            return stream.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


def _journal_rows(root: Path) -> list[dict]:
    """Read only newline-terminated JSONL rows, skipping a possibly truncated tail."""
    journal = root / JOURNAL_NAME
    try:
        data = journal.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise MirrorError(f"cannot read local capture journal: {type(exc).__name__}") from exc

    rows: list[dict] = []
    for raw_line in data.splitlines(keepends=True):
        if not raw_line.endswith(b"\n"):
            continue
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            LOG.warning("ignored malformed complete capture journal row")
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _row_path_and_hash(row: dict) -> tuple[object, object]:
    """Read the current worker schema and a few harmless compatibility aliases."""
    path = row.get("path")
    if path is None:
        for key in ("file", "filename", "image_path", "image", "png"):
            if row.get(key) is not None:
                path = row[key]
                break
    expected = row.get("sha256")
    if expected is None:
        for key in ("sha256", "hash", "digest"):
            if row.get(key) is not None:
                expected = row[key]
                break
    return path, expected


def verified_png_manifest(root: Path) -> dict[str, str]:
    """Return unique journal-referenced PNGs whose local bytes pass expected hashes."""
    rows = _journal_rows(root)
    entries: dict[PurePosixPath, str | None] = {}
    conflicts: set[PurePosixPath] = set()
    for row in rows:
        value, expected = _row_path_and_hash(row)
        safe = _safe_relative_png(root, value)
        if safe is None:
            continue
        _candidate, relative = safe
        normalized_expected: str | None
        if expected is None:
            normalized_expected = None
        elif isinstance(expected, str) and _SHA256_RE.fullmatch(expected):
            normalized_expected = expected.lower()
        else:
            conflicts.add(relative)
            continue
        if relative in entries and entries[relative] != normalized_expected:
            conflicts.add(relative)
        else:
            entries.setdefault(relative, normalized_expected)

    manifest: dict[str, str] = {}
    for relative in sorted(entries, key=lambda item: item.as_posix()):
        if relative in conflicts:
            LOG.warning("ignored capture with conflicting journal hashes: %s", relative.name)
            continue
        safe = _safe_relative_png(root, relative.as_posix())
        if safe is None:
            continue
        candidate, _ = safe
        if not _usable_png(candidate):
            continue
        actual = _sha256(candidate)
        expected = entries[relative]
        if expected is not None and actual != expected:
            LOG.warning("ignored capture whose copied bytes failed journal hash: %s", relative.name)
            continue
        manifest[relative.as_posix()] = actual
    return manifest


def _write_atomic_json(path: Path, value: dict) -> None:
    """Write a local status file through a same-directory fsync and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise MirrorError("refusing a symlink sync-status.json")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(os.fspath(temporary), flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(os.fspath(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The file replacement is atomic even on filesystems that do not fsync directories.
            pass
    except OSError as exc:
        raise MirrorError(f"cannot write local sync status: {type(exc).__name__}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            LOG.warning("could not remove temporary local sync status")


def _status_payload(manifest: dict[str, str]) -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "copied_count": len(manifest),
        "last_sync_utc": now,
        "hash_manifest": manifest,
    }


def _remote_temp_path(remote_path: str) -> str:
    return f"{remote_path}/.sync-status.{os.getpid()}.{uuid.uuid4().hex}.tmp"


def mirror_once(remote_host: str, remote_path: str, local_path: Path) -> dict:
    """Copy remote captures, publish verified local status, then atomically ack remotely."""
    local_path.mkdir(parents=True, exist_ok=True)
    source = f"{remote_host}:{remote_path}/"
    destination = f"{local_path.as_posix()}/"
    result = _run([
        "rsync", "-a", "--exclude=.*", "--exclude=tmp", "--exclude=sync-status.json",
        source, destination,
    ])
    _check_result(result, "capture rsync")

    manifest = verified_png_manifest(local_path)
    payload = _status_payload(manifest)
    status_path = local_path / STATUS_NAME
    _write_atomic_json(status_path, payload)

    remote_temp = _remote_temp_path(remote_path)
    upload = _run(["rsync", "-a", os.fspath(status_path), f"{remote_host}:{remote_temp}"])
    _check_result(upload, "status upload")
    remote_status = f"{remote_path}/{STATUS_NAME}"
    move_command = "mv -- {} {}".format(shlex.quote(remote_temp), shlex.quote(remote_status))
    moved = _run(["ssh", "-o", "BatchMode=yes", remote_host, move_command])
    _check_result(moved, "status acknowledgement")
    return payload


def _install_signal_handlers(stop: "__import__('threading').Event") -> None:
    def request_stop(signum: int, _frame: object) -> None:
        LOG.info("received signal %s; stopping after the current operation", signum)
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def run(args: argparse.Namespace) -> int:
    import threading

    remote_host = validate_remote_host(args.remote_host)
    remote_path = validate_remote_path(args.remote_path)
    local_path = validate_local_path(args.local_path)
    if not isinstance(args.interval, (int, float)) or not 0 < args.interval <= 3600:
        raise ValueError("interval must be greater than zero and at most 3600 seconds")

    stop = threading.Event()
    _install_signal_handlers(stop)
    retry_delay = min(float(args.interval), 10.0)
    while not stop.is_set():
        try:
            payload = mirror_once(remote_host, remote_path, local_path)
            LOG.info("mirrored and acknowledged %d verified PNG capture(s)", payload["copied_count"])
            if args.once:
                return 0
            retry_delay = min(float(args.interval), 10.0)
            if stop.wait(float(args.interval)):
                break
        except (MirrorError, OSError, ValueError) as exc:
            LOG.error("capture mirror attempt failed: %s", exc)
            if args.once:
                return 2
            if stop.wait(retry_delay):
                break
            retry_delay = min(10.0, max(0.1, retry_delay * 2.0))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-path", default=DEFAULT_REMOTE_PATH)
    parser.add_argument("--local-path", default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="mirror one time and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(build_parser().parse_args(argv))
    except (MirrorError, OSError, ValueError) as exc:
        LOG.error("capture mirror stopped: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
