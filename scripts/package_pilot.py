#!/usr/bin/env python3
"""Freeze and verify the recorded detector pilot; never access a remote host."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import tempfile

SOURCE = Path(__file__).resolve().parents[1]
RELEASE = "a9ed13af508b6165"
LIVE_REVISION = "2aa5149c57507428b28341312c1c9fd3aaaeecf2"
ARTIFACT_ID = "82fdd2d1db11d43aeb8376ee4b61cf4f85bd28f0102c4ed48d2ec41ac5d8a6c0"
METADATA_SHA = "c29cfb2aa9e6446e4f78cc8e4b7a9cb7cc745e293e171e400cb326465f8671ec"
MODEL = "models/h100-patchcore-v3"
CAPTURE = "cv/lego-capture-20260915"
PILOT = "cv/lego-pilot-v1"
MODEL_FILES = "metadata.json model.pt model.xml model.bin run.json events.jsonl".split()
PILOT_FILES = """manifest.json exclusions.json calibration-v2.json parity-h100-v3.json
capture-snapshot.json foreground-crops.jpg calibration.json parity-v2.json
missed-review/records.json missed-review/inspection.md missed-review/misses.jpg""".split()
VERIFICATION = """final-pilot-launch.json final-pilot-verification.json
intel-checks-a9ed13af508b6165.txt pilot-model-install.json pilot-model-metadata.json
pilot-benchmark.json dataset-final-read.json pilot-browser-verification.json invalid-browser-verification.json
overlay-retention-regression.json missing-camera-result.json""".split()
POLICY = """smolvla-same-image-inference.json smolvla-same-image-anomaly.json
smolvla-same-image-anomaly-map.npy smolvla-engineering-inference.json
smolvla-downloads.json architecture-check.json""".split()
README = """Second Look detector pilot snapshot

Use source/ as the extracted project root. Read source/docs/RUNBOOK.md and
source/docs/BUILD_EVIDENCE.md before launch. From source/, portable checks are:
  python3 scripts/run_checks.py

This package preserves the tested runtime source, immutable detector model,
30 operator-labeled captures, and selected historical evidence. Model files are
under source/artifacts/models/h100-patchcore-v3; do not refit or recalibrate them
to reproduce the recorded decisions. Existing Intel deployment uses Python 3.11
in intel_dev_env and Studio's existing backend interpreter for camera access.
No new-machine environment setup or hardware verification is claimed.

source/source-manifest.json identifies the live runtime revision and 40 hashes.
bundle-manifest.json identifies the later source Git revision and inventories
all payload files (the manifest excludes itself). The adjacent pilot-bundle.json
receipt hashes the entire archive after every member has been read and checked.
Historical JSON paths remain unchanged; map captured image basenames to the
bundled capture directory. Any optional replay image mapping is in the manifest.

This is a detector-only snapshot, not live VLA/control integration. Five real
Studio episodes were reported; they are not bundled or validated here. Physical
pose, torque, safety, and autonomous success remain UNKNOWN. Recorded SmolVLA
inference uses synthetic state/normalizers. VLA weights are not included;
source/artifacts/policy/smolvla-downloads.json records pinned downloads and hashes
when available. Runtime dependencies and sponsor environments are not bundled.
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(SOURCE), *args])


def digest(path: Path) -> dict:
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": checksum, "bytes": path.stat().st_size}


def safe_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    require(bool(name) and not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe relative path: {name}")
    path = root
    for part in relative.parts:
        path /= part
        require(not path.is_symlink(), f"Symlink is not allowed: {path}")
    require(stat.S_ISREG(path.stat().st_mode), f"Not a regular file: {path}")
    return path


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def select_inputs(artifacts: Path, manifest_path: Path, source_root: Path = SOURCE):
    manifest = json.loads(manifest_path.read_text())
    require(manifest["release"] == RELEASE and manifest["git_revision"] == LIVE_REVISION
            and manifest["git_dirty"] is False and len(manifest["files"]) == 40,
            "Expected the clean, 40-file live release manifest")
    entries, modes, skipped = {}, {}, []
    for row in git("ls-files", "--stage", "-z").split(b"\0"):
        if not row:
            continue
        header, raw_name = row.split(b"\t", 1)
        mode, _, stage = header.split()
        name = raw_name.decode()
        require(mode in (b"100644", b"100755") and stage == b"0",
                f"Unsupported Git entry: {name}")
        entries[f"source/{name}"] = safe_file(source_root, name)
        modes[f"source/{name}"] = 0o755 if mode == b"100755" else 0o644
    for name, checksum in manifest["files"].items():
        require(f"source/{name}" in entries, f"Runtime file is untracked: {name}")
        require(digest(safe_file(source_root, name))["sha256"] == checksum,
                f"Live source hash mismatch: {name}")
    entries["source/source-manifest.json"] = manifest_path

    def artifact(name: str, optional: bool = False) -> None:
        if optional and not (artifacts / name).exists():
            skipped.append(name)
            return
        entries[f"source/artifacts/{name}"] = safe_file(artifacts, name)

    for name in MODEL_FILES:
        artifact(f"{MODEL}/{name}")
    metadata = json.loads((artifacts / MODEL / "metadata.json").read_text())
    require(digest(artifacts / MODEL / "metadata.json")["sha256"] == METADATA_SHA
            and metadata["artifact_id"] == ARTIFACT_ID, "Model metadata identity mismatch")
    require(set(metadata["files"]) == {"model.pt", "model.xml", "model.bin"},
            "Unexpected model file inventory")
    for name, checksum in metadata["files"].items():
        require(digest(safe_file(artifacts, f"{MODEL}/{name}"))["sha256"] == checksum,
                f"Model hash mismatch: {name}")
    for name in ("collection.json", "captures.jsonl", "specimens.jsonl"):
        artifact(f"{CAPTURE}/{name}")
    records = [json.loads(line) for line in (artifacts / CAPTURE / "captures.jsonl").read_text().splitlines() if line]
    names = [record["path"] for record in records]
    require(len(records) == len(set(names)) == 30, "Expected 30 unique capture records")
    require(set(names) == {p.name for p in (artifacts / CAPTURE).glob("*.png")},
            "Capture PNG inventory differs from journal")
    for record in records:
        name = f"{CAPTURE}/{record['path']}"
        artifact(name)
        require(digest(entries[f"source/artifacts/{name}"])["sha256"] == record["sha256"],
                f"Capture hash mismatch: {name}")
    for name in PILOT_FILES:
        artifact(f"{PILOT}/{name}")
    for directory, names in (("verification", VERIFICATION), ("policy", POLICY)):
        for name in names:
            artifact(f"{directory}/{name}", optional=True)
    dependencies = ["Runtime environments are not bundled.",
                    "Five reported Studio episodes are not bundled or validated.",
                    "VLA weights are excluded; see policy/smolvla-downloads.json when bundled."]
    mappings = []
    replay = artifacts / "policy/smolvla-same-image-inference.json"
    if replay.exists():
        image = json.loads(replay.read_text())["image"]
        prefix = "/home/ird-demo/second-look/artifacts/"
        recorded = image["path"]
        relative = recorded[len(prefix):] if recorded.startswith(prefix) else None
        local_relative = relative
        # This earlier local transfer predates the recorded remote directory name.
        if relative == "cv/scout-20260915/camera1.png" and not (artifacts / relative).exists():
            local_relative = "cv/scout/camera1.png"
        if local_relative and (artifacts / local_relative).exists():
            replay_image = safe_file(artifacts, local_relative)
            require(digest(replay_image)["sha256"] == image["sha256"],
                    "Replay image hash mismatch")
            entries[f"source/artifacts/{relative}"] = replay_image
            mappings.append({"recorded_path": recorded, "bundle_path": f"source/artifacts/{relative}"})
        else:
            dependencies.append(f"Replay image not bundled: {recorded} SHA256 {image['sha256']}")
    return entries, modes, skipped, dependencies, mappings


def verify_archive(path: Path, root: str, inventory: dict) -> None:
    seen = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            require(member.isfile() and member.name.startswith(root + "/"),
                    f"Unsafe archive member: {member.name}")
            name = member.name[len(root) + 1:]
            require(name in inventory and name not in seen, f"Unexpected archive member: {name}")
            with archive.extractfile(member) as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            require(checksum == inventory[name]["sha256"] and member.size == inventory[name]["bytes"],
                    f"Archive verification failed: {name}")
            seen.add(name)
    require(seen == set(inventory), "Archive is missing expected members")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs; allow dirty docs; write nothing")
    args = parser.parse_args()
    artifacts = args.artifact_root.resolve(strict=True)
    manifest_path = safe_file(args.manifest.absolute().parent, args.manifest.name)
    output = args.output.absolute()
    receipt_path = output.with_name("pilot-bundle.json")
    require(not output.exists() and not output.is_symlink(), f"Output already exists: {output}")
    require(not receipt_path.exists() and not receipt_path.is_symlink(), f"Receipt already exists: {receipt_path}")
    revision = git("rev-parse", "HEAD").decode().strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    require(args.dry_run or not dirty, "Commit tracked source changes before building")
    entries, modes, skipped, dependencies, mappings = select_inputs(artifacts, manifest_path)
    require(args.dry_run or "source/scripts/package_pilot.py" in entries, "Commit the packager before building")
    inventory = {name: digest(path) for name, path in sorted(entries.items())}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "input_files": len(entries), "input_bytes": sum(v["bytes"] for v in inventory.values()),
                          "source_revision": revision, "tracked_source_dirty": dirty, "live_release": RELEASE,
                          "skipped_optional_files": skipped, "unbundled_dependencies": dependencies}, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pilot-build-", dir=output.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        for name, source in entries.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            require(digest(target) == inventory[name] == digest(source), f"Input changed while freezing: {name}")
        require(git("rev-parse", "HEAD").decode().strip() == revision
                and not git("status", "--porcelain", "--untracked-files=no"), "Source changed while freezing")
        frozen, _, _, _, _ = select_inputs(
            staging / "source/artifacts", staging / "source/source-manifest.json", staging / "source")
        require(set(frozen) == set(entries), "Frozen file selection changed")
        # All payload reads below use the frozen snapshot, never live evidence files.
        (staging / "README.txt").write_text(README)
        inventory["README.txt"] = digest(staging / "README.txt")
        write_json(staging / "bundle-manifest.json", {
            "schema_version": 1, "source_git_revision": revision, "live_source_revision": LIVE_REVISION,
            "release": RELEASE, "artifact_id": ARTIFACT_ID, "files": inventory,
            "inventory_scope": "All payload files; excludes bundle-manifest.json itself",
            "quality_limits": ["Detector-only snapshot; not live VLA/control integration.",
                               "Five real episodes reported, not bundled or validated.",
                               "Physical pose, torque, safety, and autonomous success UNKNOWN.",
                               "One missed defective view, four abstentions; no unseen final test."],
            "skipped_optional_files": skipped, "unbundled_dependencies": dependencies,
            "replay_path_mappings": mappings,
        })
        inventory["bundle-manifest.json"] = digest(staging / "bundle-manifest.json")
        archive_path = Path(temporary) / "bundle.tar.gz"
        root = f"second-look-pilot-{RELEASE}"
        with archive_path.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(inventory):
                    info = tarfile.TarInfo(f"{root}/{name}")
                    info.size, info.mode = inventory[name]["bytes"], modes.get(name, 0o644)
                    info.uid = info.gid = info.mtime = 0
                    with (staging / name).open("rb") as stream:
                        archive.addfile(info, stream)
        verify_archive(archive_path, root, inventory)
        receipt = {"archive": output.name, **digest(archive_path), "members": len(inventory),
                   "live_release": RELEASE, "source_revision": revision, "live_source_revision": LIVE_REVISION,
                   "artifact_id": ARTIFACT_ID, "verification_passed": True}
        # Hard links publish complete files atomically and refuse existing destinations.
        write_json(Path(temporary) / "receipt.json", receipt)
        os.link(archive_path, output)
        os.link(Path(temporary) / "receipt.json", receipt_path)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"package_pilot: {error}") from error
