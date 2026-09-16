#!/usr/bin/env python3
"""Run portable, non-motion checks for the Second Look checkout.

The checks compile Python into a temporary directory, run the repository's
unittest suite, and exercise the application's argument parser with ``--help``.
They do not open a camera, connect to a robot, start a server, or use SSH.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "scripts" / "run_app.py"


@dataclass(frozen=True)
class Result:
    name: str
    status: str
    detail: str = ""


def _python_files() -> list[Path]:
    paths: list[Path] = []
    for base in (ROOT / "secondlook", ROOT / "scripts", ROOT / "tests"):
        if not base.is_dir():
            continue
        paths.extend(
            path
            for path in base.rglob("*.py")
            if "__pycache__" not in path.parts and path.is_file() and not path.is_symlink()
        )
    return sorted(paths)


def check_syntax() -> Result:
    files = _python_files()
    if not files:
        return Result("syntax", "SKIP", "no Python files found")
    with tempfile.TemporaryDirectory(prefix="secondlook-compile-") as directory:
        destination = Path(directory)
        try:
            for path in files:
                relative = path.relative_to(ROOT)
                output = destination / relative.with_suffix(".pyc")
                output.parent.mkdir(parents=True, exist_ok=True)
                py_compile.compile(str(path), cfile=str(output), doraise=True)
        except py_compile.PyCompileError as error:
            return Result("syntax", "FAIL", str(error).strip())
    return Result("syntax", "PASS", f"compiled {len(files)} Python files")


def check_unittest(timeout: float) -> Result:
    tests = ROOT / "tests"
    if not tests.is_dir() or not any(tests.glob("test*.py")):
        return Result("unittest", "SKIP", "no test modules found")
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_path if existing_path else "")
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Result("unittest", "FAIL", f"timed out after {timeout:g}s")
    output = (completed.stdout or "").strip().splitlines()
    tail = " | ".join(output[-4:])
    if completed.returncode:
        return Result("unittest", "FAIL", tail or f"exit {completed.returncode}")
    return Result("unittest", "PASS", tail or "unittest discovery passed")


def check_app_help(timeout: float) -> Result:
    if not APP.is_file():
        return Result("run_app --help", "SKIP", "scripts/run_app.py is not present")
    try:
        completed = subprocess.run(
            [sys.executable, str(APP), "--help"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Result("run_app --help", "FAIL", f"timed out after {timeout:g}s")
    if completed.returncode:
        output = (completed.stdout or "").strip().splitlines()
        return Result("run_app --help", "FAIL", " | ".join(output[-4:]) or f"exit {completed.returncode}")
    return Result("run_app --help", "PASS", "argument parser exited without starting the server")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="skip unittest discovery")
    parser.add_argument("--skip-app-help", action="store_true", help="skip the run_app.py --help check")
    parser.add_argument("--timeout", type=float, default=60.0, help="timeout per subprocess check in seconds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2
    results = [check_syntax()]
    if args.skip_tests:
        results.append(Result("unittest", "SKIP", "disabled by --skip-tests"))
    else:
        results.append(check_unittest(args.timeout))
    if args.skip_app_help:
        results.append(Result("run_app --help", "SKIP", "disabled by --skip-app-help"))
    else:
        results.append(check_app_help(args.timeout))
    for result in results:
        suffix = f" — {result.detail}" if result.detail else ""
        print(f"[{result.status}] {result.name}{suffix}")
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
