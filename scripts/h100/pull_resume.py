#!/usr/bin/env python3
"""Resume one H100 file onto Intel, then verify SHA-256 before publishing it."""

import argparse
import inspect
from pathlib import Path
import re
import shlex
import subprocess
import sys

import transport


def run_resume(request):
    # This function is sent to Intel together with transport.py. Credentials
    # remain on Intel; failed transfers retain the stable .part file.
    import fcntl
    import hashlib
    import time

    target = Path(request["destination"]).expanduser().absolute()
    part = Path(str(target) + ".part")
    receipt = Path(str(target) + ".transfer.json")
    lock_path = Path(str(part) + ".lock")
    metadata = {"source": request["source"], "destination": str(target),
                "part": str(part), "expected_sha256": request["sha256"],
                "started_at_unix": time.time(), "status": "starting"}

    def save(status, **values):
        metadata.update(status=status, updated_at_unix=time.time(), **values)
        fd, name = tempfile.mkstemp(prefix=".transfer-receipt-", dir=target.parent)
        with os.fdopen(fd, "w") as output:
            json.dump(metadata, output, indent=2)
            output.write("\n")
        os.replace(name, receipt)

    try:
        if os.path.lexists(target):
            raise TransportError("Destination exists; resumable pull never overwrites it.")
        if part.is_symlink() or (part.exists() and not part.is_file()):
            raise TransportError("Partial file must be a regular file, not a symbolic link.")
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock_fd, "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise TransportError("Another resumable pull owns this destination.") from None
            worker = IntelWorker(request["timeout"])
            worker.verify_identity()
            source = gpu_path(request["source"])
            worker.helper("check_pull", path=source)
            if receipt.exists():
                previous = json.loads(receipt.read_text())
                if previous.get("source") != source or previous.get("expected_sha256") != request["sha256"]:
                    raise TransportError("Existing receipt belongs to a different source or expected hash.")
            save("transferring", initial_bytes=part.stat().st_size if part.exists() else 0)
            print("Resuming into Intel .part file; progress follows. Receipt: " + str(receipt), flush=True)
            ssh = ["ssh", *SSH_OPTIONS, "-p", worker.port, "-i", worker.identity]
            result = subprocess.run(
                ["rsync", "--partial", "--append-verify", "--info=progress2",
                 "-e", shlex.join(ssh), "--", worker.destination + ":" + source, str(part)],
                stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=request["timeout"], check=False,
            )
            if result.returncode:
                save("interrupted", bytes=part.stat().st_size if part.exists() else 0)
                raise TransportError("Rsync failed; partial bytes remain available for another run.")
            actual = hashlib.sha256()
            with part.open("rb") as content:
                for block in iter(lambda: content.read(1024 * 1024), b""):
                    actual.update(block)
            digest = actual.hexdigest()
            if digest != request["sha256"]:
                save("hash_mismatch", bytes=part.stat().st_size, actual_sha256=digest)
                raise TransportError("Whole-file SHA-256 mismatch; partial file retained, destination unpublished.")
            # A hard link publishes atomically without replacing an existing
            # file. Both paths share the same Intel directory and filesystem.
            os.link(part, target)
            part.unlink()
            save("verified", bytes=target.stat().st_size, actual_sha256=digest)
            print("Verified complete: " + str(target) + " SHA-256 " + digest, flush=True)
            return 0
    except subprocess.TimeoutExpired:
        save("interrupted", bytes=part.stat().st_size if part.exists() else 0)
        print("Resumable pull timed out; partial bytes remain available for another run.", flush=True)
        return 124
    except TransportError as error:
        print("Resumable pull: " + str(error), flush=True)
        return 1
    except Exception:
        print("Resumable pull failed; partial files retained and connection details suppressed.", flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="H100 file below /workspace/second-look-h100.")
    parser.add_argument("destination", help="Final Intel path; its .part file is reused.")
    parser.add_argument("--sha256", required=True, help="Expected whole-file SHA-256.")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per remote operation (default: 3600).")
    request = vars(parser.parse_args())
    if not re.fullmatch(r"[0-9a-fA-F]{64}", request["sha256"]):
        parser.error("--sha256 must contain exactly 64 hexadecimal characters")
    request["sha256"] = request["sha256"].lower()
    if request["timeout"] < 1:
        parser.error("--timeout must be positive")
    transport.gpu_path(request["source"])
    if not request["destination"].startswith(("/", "~/")):
        parser.error("destination must be an absolute Intel path or start with ~/")
    source = Path(transport.__file__).read_text().replace('if __name__ == "__main__":', "if False:")
    source += "\n" + inspect.getsource(run_resume)
    source += "\nsys.exit(run_resume(json.loads(base64.urlsafe_b64decode(sys.argv[1]))))\n"
    command = shlex.join(["python3", "-", transport.encode(request)])
    result = subprocess.run(["ssh", *transport.SSH_OPTIONS, transport.INTEL_ALIAS, command],
                            input=source.encode(), stderr=subprocess.DEVNULL, check=False)
    if result.returncode:
        print("Resumable transfer incomplete; SSH diagnostics suppressed.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except transport.TransportError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted; inspect the Intel receipt before resuming.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        print("Resumable pull failed; connection details suppressed.", file=sys.stderr)
        sys.exit(1)
