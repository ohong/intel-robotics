#!/usr/bin/env python3
"""Reproducible, non-motion deployment helpers for the Intel robot PC."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import uuid
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
ALIAS = "intel-robot"
REMOTE_ROOT = "/home/ird-demo/second-look"
INTEL_PYTHON = "/home/ird-demo/miniforge3/envs/intel_dev_env/bin/python"
STUDIO_PYTHON = "/home/ird-demo/physical-ai-studio/application/backend/.venv/bin/python"
UNIT = "second-look-app.service"
DEFAULT_PORT = 8088
STUDIO_CAMERA_SERVICE = "physicalai/camera/RealSenseCamera/243622060187/frame"
DEFAULT_ANOMALY_ARTIFACT = f"{REMOTE_ROOT}/artifacts/models/scene-smoke"
TASK_INSTRUCTION = "config/task-instruction.txt"
ALLOWLIST = ("secondlook", "scripts/run_app.py", "scripts/studio_camera_worker.py", "scripts/studio_robot_worker.py", "scripts/anomaly_cli.py", "scripts/cv_capture.py", "scripts/cv_capture_server.py", "scripts/cv_capture_mirror.py", "scripts/cv_manifest.py", "scripts/policy_worker.py", "scripts/recording/native_policy_guard.py", "scripts/recording/native_policy_provenance.py", "scripts/remote.py", "scripts/run_checks.py", "tests", "config", "pyproject.toml")
EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "artifacts", "captures", "weights", "models", "checkpoints", "data"}
EXCLUDED_SUFFIXES = {".pyc", ".pem", ".key", ".p12", ".pfx", ".pt", ".pth", ".onnx", ".safetensors", ".bin", ".ckpt", ".npz", ".npy", ".mp4", ".mov", ".webm", ".jpg", ".jpeg", ".png", ".gif"}
SECRET = (re.compile(b"-----" + b"BEGIN .*" + b"PRIVATE KEY" + b"-----"), re.compile(rb"\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{20,}"))
GUARD = "runtime/armed.json"
EVIDENCE_ROOTS = ("artifacts/runtime", "artifacts/evidence", "artifacts/benchmarks", "artifacts/trials", "evidence")
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ControlPath=none", "-o", "ConnectTimeout=10", "-o", "PreferredAuthentications=publickey", "-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no")
class ToolError(RuntimeError):
    pass
def run(argv: Sequence[str], *, input_data: bytes | None = None, timeout: float | None = 60) -> subprocess.CompletedProcess[bytes]:
    try:
        p = subprocess.run(list(argv), input=input_data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"command unavailable or timed out: {argv[0]}") from exc
    if p.returncode:
        detail = p.stderr.decode(errors="replace").strip().splitlines()
        raise ToolError(detail[-1] if detail else f"{argv[0]} exited {p.returncode}")
    return p
def ssh(command: str, *, input_data: bytes | None = None, timeout: float | None = 60) -> subprocess.CompletedProcess[bytes]:
    return run(("ssh", *SSH_OPTS, ALIAS, command), input_data=input_data, timeout=timeout)
def remote_python(source: str, args: Sequence[str] = (), *, input_data: bytes | None = None, timeout: float | None = 60) -> subprocess.CompletedProcess[bytes]:
    command = "python3 -c %s %s" % (shlex.quote(source), " ".join(shlex.quote(a) for a in args))
    return ssh(command, input_data=input_data, timeout=timeout)
def manifest() -> dict[str, Any]:
    files: dict[str, str] = {}
    for name in ALLOWLIST:
        base = ROOT / name
        if not base.exists() or base.is_symlink():
            continue
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in candidates:
            if path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            pure = PurePosixPath(rel)
            if ".." in pure.parts or any(p in EXCLUDED_DIRS for p in pure.parts) or path.suffix.lower() in EXCLUDED_SUFFIXES or path.name == ".env" or path.name.startswith(".env."):
                continue
            data = path.read_bytes()
            if len(data) > 5 * 1024 * 1024:
                continue
            if any(pattern.search(data) for pattern in SECRET):
                raise ToolError(f"possible credential signature in selected source: {rel}")
            if "\n" in rel or "\r" in rel:
                raise ToolError(f"unsafe source path: {rel}")
            files[rel] = hashlib.sha256(data).hexdigest()
    try:
        revision = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(("git", "status", "--porcelain", "--untracked-files=all"), cwd=ROOT, stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    identity = {"git_revision": revision, "git_dirty": dirty, "files": dict(sorted(files.items()))}
    release = hashlib.sha256((json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()[:16]
    return {"schema_version": 1, "release": release, "git_revision": revision, "git_dirty": dirty, "files": dict(sorted(files.items()))}
def guard(data: dict[str, Any], label: str) -> None:
    if not isinstance(data, dict) or type(data.get("armed")) is not bool:
        raise ToolError(f"{label} {GUARD} is missing or does not explicitly set armed=false")
    if data["armed"]:
        raise ToolError(f"refusing while {label} {GUARD} says armed=true")
STATUS = r'''import json,os,pathlib,re,shlex,shutil,subprocess,sys
root=pathlib.Path(sys.argv[1]); unit=sys.argv[2]; interpreter=sys.argv[3]; studio_python=sys.argv[4]; studio_service=sys.argv[5]; expected_port=sys.argv[6]; expected_artifact=sys.argv[7]
g=root/"runtime/armed.json"
try:
 d=json.loads(g.read_text()) if g.is_file() and not g.is_symlink() else None
 guard={"state":"disarmed" if isinstance(d,dict) and type(d.get("armed")) is bool and not d["armed"] else "armed" if isinstance(d,dict) and d.get("armed") is True else "unknown","present":g.is_file() and not g.is_symlink()}
except Exception as e: guard={"state":"unknown","present":True,"error":str(e)}
cur=root/"current"; current={"exists":os.path.lexists(cur)}
if cur.is_symlink():
 target=cur.resolve(strict=False); current.update(type="symlink",target=os.readlink(cur),resolved=str(target))
 meta=target/"source-manifest.json"
 try: m=json.loads(meta.read_text()); current["owned"]=target.parent==root/"releases" and bool(re.fullmatch(r"[0-9a-f]{16}",target.name)) and m.get("schema_version")==1 and m.get("release")==target.name
 except Exception: current["owned"]=False
elif os.path.lexists(cur): current.update(type="other",owned=False)
else: current.update(type="absent",owned=False)
unit_info={"name":unit,"available":bool(shutil.which("systemctl"))}
if unit_info["available"]:
 p=subprocess.run(["systemctl","--user","show",unit,"--property=LoadState,ActiveState,MainPID,ExecStart,WorkingDirectory","--no-pager"],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
 unit_info.update({k:v for k,v in (line.split("=",1) for line in p.stdout.splitlines() if "=" in line)}); unit_info["returncode"]=p.returncode
 work=unit_info.get("WorkingDirectory",""); exe=unit_info.get("ExecStart","")
 path_match=re.search(r"path=([^ ;]+) ;",exe); argv_match=re.search(r"argv\[\]=(.*?)(?: ;|$)",exe)
 argv=shlex.split(argv_match.group(1)) if argv_match else []
 try: resolved=pathlib.Path(work).resolve(strict=False)
 except Exception: resolved=pathlib.Path("/")
 work_ok=resolved==root/"current" or root/"releases" in resolved.parents
 for key, flag in (("revision","--revision"),("camera_service","--studio-camera-service"),("studio_python","--studio-python"),("instruction_file","--instruction-file"),("port","--port"),("anomaly_artifact","--anomaly-artifact"),("robot_session","--studio-robot-session"),("replay_dataset","--replay-dataset")):
  try: unit_info[key]=argv[argv.index(flag)+1]
  except (ValueError,IndexError): unit_info[key]=None
 unit_info["owned"]=bool(unit_info.get("LoadState")=="loaded" and path_match and path_match.group(1)==interpreter and argv and argv[0]==interpreter and "scripts/run_app.py" in argv and work_ok)
print(json.dumps({"host":os.uname().nodename,"root":str(root),"root_exists":root.is_dir(),"guard":guard,"current":current,"interpreter":interpreter,"studio_interpreter":studio_python,"studio_camera_service":studio_service,"expected_port":expected_port,"expected_anomaly_artifact":expected_artifact,"unit":unit_info},sort_keys=True))'''


def status(studio_service: str = STUDIO_CAMERA_SERVICE, expected_port: int = DEFAULT_PORT,
           expected_artifact: str = DEFAULT_ANOMALY_ARTIFACT) -> dict[str, Any]:
    out = remote_python(STATUS, (REMOTE_ROOT, UNIT, INTEL_PYTHON, STUDIO_PYTHON,
                                 studio_service, str(expected_port), expected_artifact), timeout=30).stdout
    try:
        return json.loads(out.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolError("remote status returned invalid JSON") from exc
def output(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))
PREPARE = r'''import json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]);stage=pathlib.Path(sys.argv[2]);release=pathlib.Path(sys.argv[3]);data=sys.stdin.buffer.read()
if not root.is_dir(): raise SystemExit("remote root is missing")
(root/"releases").mkdir(exist_ok=True)
if os.path.lexists(release):
 meta=release/"source-manifest.json"
 try: old=json.loads(meta.read_text());new=json.loads(data)
 except Exception: raise SystemExit("release ownership metadata is invalid")
 if not meta.is_file() or any(old.get(k)!=new.get(k) for k in ("schema_version","release","files")): raise SystemExit("release exists with different ownership metadata")
 print("reuse")
else:
 if os.path.lexists(stage): raise SystemExit("staging path already exists")
 stage.mkdir();(stage/"source-manifest.json").open("xb").write(data);print("stage")'''
VERIFY = r'''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);m=json.loads((root/"source-manifest.json").read_text())
for name,digest in m["files"].items():
 p=root/pathlib.Path(*pathlib.PurePosixPath(name).parts)
 if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise SystemExit("checksum mismatch: "+name)
print("verified="+str(len(m["files"])))'''
FINALIZE = r'''import os,pathlib,sys
s=pathlib.Path(sys.argv[1]);r=pathlib.Path(sys.argv[2])
if not s.is_dir() or os.path.lexists(r): raise SystemExit("unsafe release finalize")
os.rename(s,r);print(str(r))'''
ACTIVATE = r'''import os,pathlib,sys
root=pathlib.Path(sys.argv[1]);release=pathlib.Path(sys.argv[2]).resolve();expected=sys.argv[3];cur=root/"current"
if not release.is_dir() or release.parent!=root/"releases" or not (release/"source-manifest.json").is_file(): raise SystemExit("release is not owned")
if os.path.lexists(cur):
 if not cur.is_symlink() or os.readlink(cur)!=expected: raise SystemExit("current changed; refusing pointer update")
elif expected!="<absent>": raise SystemExit("current changed; refusing pointer update")
tmp=root/(".current."+release.name+".tmp")
if os.path.lexists(tmp): raise SystemExit("temporary current pointer exists")
os.symlink(os.path.relpath(release,root),tmp);os.replace(tmp,cur);print(os.readlink(cur))'''


def deploy(args: argparse.Namespace) -> int:
    m = manifest(); release = f"{REMOTE_ROOT}/releases/{m['release']}"
    if args.dry_run:
        output({"action":"deploy","release":m["release"],"remote_release":release,"activate":not args.stage_only,"manifest":m}); return 0
    before = status(); guard_remote = before.get("guard", {})
    if not args.stage_only:
        local_guard = ROOT / GUARD
        if not local_guard.is_file() or local_guard.is_symlink(): raise ToolError(f"create an explicit local {GUARD} with armed=false before deploy")
        guard(json.loads(local_guard.read_text()), "local")
        if guard_remote.get("state") != "disarmed": raise ToolError("remote armed state is unknown or armed; refusing deploy")
    if not before.get("root_exists"): raise ToolError("remote project root is missing")
    current = before.get("current", {}); expected = current.get("target") if current.get("type")=="symlink" else "<absent>"
    if current.get("type")=="other" or (current.get("exists") and not current.get("owned")): raise ToolError("remote current is not an owned release symlink")
    stage = f"{REMOTE_ROOT}/releases/.staging-{m['release']}-{uuid.uuid4().hex[:8]}"; data=(json.dumps(m,sort_keys=True,indent=2)+"\n").encode()
    prep = remote_python(PREPARE,(REMOTE_ROOT,stage,release),input_data=data).stdout.decode().strip()
    if prep == "stage":
        names=("\n".join(m["files"]) + "\n").encode()
        rsync=("rsync","-a","--files-from=-","--exclude=.env*","--exclude=artifacts/***","--exclude=captures/***","--exclude=weights/***","-e","ssh " + " ".join(shlex.quote(o) for o in SSH_OPTS),f"{ROOT.as_posix()}/",f"{ALIAS}:{stage}/")
        run(rsync,input_data=names,timeout=180)
    if prep in ("stage", "reuse"):
        remote_python(VERIFY,(stage if prep=="stage" else release,),timeout=180)
    if prep == "stage": remote_python(FINALIZE,(stage,release))
    elif prep != "reuse": raise ToolError(f"unexpected remote prepare result: {prep}")
    activated = False
    if not args.stage_only:
        latest=status(); guard_remote=latest.get("guard",{})
        if guard_remote.get("state")!="disarmed": raise ToolError("remote armed state changed; refusing current update")
        now=latest.get("current",{}); now_expected=now.get("target") if now.get("type")=="symlink" else "<absent>"
        if now.get("type")=="other" or now_expected!=expected: raise ToolError("remote current changed; refusing pointer update")
        remote_python(ACTIVATE,(REMOTE_ROOT,release,now_expected)); activated=True
    output({"release":m["release"],"staged":prep=="stage","activated":activated,"git_revision":m["git_revision"],"git_dirty":m["git_dirty"]}); return 0


CHECK = r'''import pathlib,subprocess,sys,json
root=pathlib.Path(sys.argv[1]);p=root/"current/scripts/run_app.py";r={"app":str(p),"exists":p.is_file()}
if p.is_file():
 x=subprocess.run([sys.argv[2],str(p),"--help"],cwd=root/"current",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20,check=False);r.update(returncode=x.returncode,output=x.stdout.decode(errors="replace")[-1000:])
print(json.dumps(r));raise SystemExit(0 if r.get("returncode")==0 else 2)'''


def check() -> int:
    s=status()
    if not s.get("current",{}).get("owned"): raise ToolError("remote check requires an owned current release")
    output({"status":s,"app_help":json.loads(remote_python(CHECK,(REMOTE_ROOT,INTEL_PYTHON),timeout=30).stdout.decode()),"full_launch_verified":False}); return 0


def evidence_path(value: str) -> str:
    p=PurePosixPath(value.rstrip("/"))
    if p.is_absolute() or ".." in p.parts or not any(p.as_posix()==r or p.as_posix().startswith(r+"/") for r in EVIDENCE_ROOTS): raise ToolError("pull path is outside the evidence allowlist")
    return p.as_posix()


def pull(args: argparse.Namespace) -> int:
    path=evidence_path(args.path); dest=Path(args.destination).expanduser()
    if dest.is_symlink(): raise ToolError("refusing a local symlink destination")
    dest.mkdir(parents=True,exist_ok=True)
    rsync=("rsync","-a","--ignore-existing","--exclude=.env*","--exclude=captures/***","--exclude=weights/***","--exclude=models/***","-e","ssh " + " ".join(shlex.quote(o) for o in SSH_OPTS),f"{ALIAS}:{REMOTE_ROOT}/{path}/",f"{dest.as_posix()}/")
    if args.dry_run: print(" ".join(shlex.quote(x) for x in rsync)); return 0
    run(rsync,timeout=180); output({"pulled":path,"destination":str(dest),"overwrites":False}); return 0


def anomaly_artifact_path(value: str) -> str:
    base = PurePosixPath(f"{REMOTE_ROOT}/artifacts/models")
    path = PurePosixPath(value)
    if (not path.is_absolute() or path == base or base not in path.parents
            or ".." in path.parts or "\n" in value or "\r" in value):
        raise argparse.ArgumentTypeError(
            f"anomaly artifact must be an absolute path below {base} without traversal"
        )
    return path.as_posix()


def service(action: str, port: int, studio_service: str = STUDIO_CAMERA_SERVICE,
            anomaly_artifact: str = DEFAULT_ANOMALY_ARTIFACT, robot_session: str | None = None,
            replay_dataset: str | None = None) -> int:
    try:
        anomaly_artifact = anomaly_artifact_path(anomaly_artifact)
    except argparse.ArgumentTypeError as exc:
        raise ToolError(str(exc)) from exc
    s=status(studio_service, port, anomaly_artifact); unit=s.get("unit",{})
    if action=="stop" and unit.get("LoadState")=="not-found": print(f"{UNIT} is already stopped"); return 0
    if action=="start":
        if s.get("guard",{}).get("state")!="disarmed": raise ToolError("service start requires runtime/armed.json with armed=false")
        if not s.get("current",{}).get("owned"): raise ToolError("service start requires an owned current release")
        release=Path(s["current"]["resolved"]).name
        if unit.get("ActiveState")=="active":
            if not unit.get("owned"): raise ToolError("refusing to manage a systemd unit that fails Second Look ownership checks")
            mismatches=[]
            if unit.get("revision")!=release: mismatches.append(f"revision {unit.get('revision')} != {release}")
            if unit.get("camera_service")!=studio_service: mismatches.append(f"camera service {unit.get('camera_service')} != {studio_service}")
            if unit.get("studio_python")!=STUDIO_PYTHON: mismatches.append(f"Studio Python {unit.get('studio_python')} != {STUDIO_PYTHON}")
            if unit.get("instruction_file")!=TASK_INSTRUCTION: mismatches.append(f"instruction file {unit.get('instruction_file')} != {TASK_INSTRUCTION}")
            if unit.get("port")!=str(port): mismatches.append(f"port {unit.get('port')} != {port}")
            if unit.get("anomaly_artifact")!=anomaly_artifact: mismatches.append(f"anomaly artifact {unit.get('anomaly_artifact')} != {anomaly_artifact}")
            if unit.get("robot_session")!=robot_session: mismatches.append(f"Studio robot session {unit.get('robot_session')} != {robot_session}")
            if unit.get("replay_dataset")!=replay_dataset: mismatches.append(f"replay dataset {unit.get('replay_dataset')} != {replay_dataset}")
            if mismatches: raise ToolError("active service arguments do not match current start request: " + "; ".join(mismatches))
            print(f"{UNIT} is already running"); return 0
        if unit.get("LoadState") not in ("not-found", None):
            if not unit.get("owned"): raise ToolError("refusing to manage a systemd unit that fails Second Look ownership checks")
            ssh(f"systemctl --user stop {shlex.quote(UNIT)}",timeout=30)
            if status().get("unit",{}).get("LoadState")!="not-found": raise ToolError(f"{UNIT} remains loaded after stop; refusing replacement")
        command=("systemd-run","--user",f"--unit={UNIT}","--collect",f"--property=WorkingDirectory={REMOTE_ROOT}/current","--property=EnvironmentFile=-/home/ird-demo/.config/secondlook/fal.env","--property=EnvironmentFile=-/home/ird-demo/.config/secondlook/openai.env","--",INTEL_PYTHON,"scripts/run_app.py","--studio-camera-service",studio_service,"--studio-python",STUDIO_PYTHON,"--instruction-file",TASK_INSTRUCTION,"--port",str(port),"--anomaly-artifact",anomaly_artifact,"--backend","openvino","--device","GPU","--precision","f32","--revision",release,"--inference-interval",".5","--evidence",f"{REMOTE_ROOT}/artifacts/runtime/observations.jsonl","--lock-file",f"{REMOTE_ROOT}/runtime/app.lock")
        if robot_session: command+=("--studio-robot-session",robot_session)
        if replay_dataset: command+=("--replay-dataset",replay_dataset)
        ssh(" ".join(shlex.quote(x) for x in command),timeout=30)
        print(f"{UNIT} launch requested for release {release}"); return 0
    if not unit.get("owned"): raise ToolError("refusing to manage a systemd unit that fails Second Look ownership checks")
    ssh(f"systemctl --user {action} {shlex.quote(UNIT)}",timeout=30)
    print(f"{action} requested for {UNIT} (port={port})"); return 0


def tunnel(args: argparse.Namespace) -> int:
    command=("ssh",*SSH_OPTS,"-o","ExitOnForwardFailure=yes","-N","-L",f"127.0.0.1:{args.local_port}:127.0.0.1:{args.remote_port}",ALIAS)
    if args.print_only: print(" ".join(shlex.quote(x) for x in command)); return 0
    run(command,timeout=None); return 0


def port(value: str) -> int:
    try: n=int(value)
    except ValueError as exc: raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1<=n<=65535: raise argparse.ArgumentTypeError("port must be 1..65535")
    return n


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    x=sub.add_parser("manifest");x.set_defaults(fn=lambda a: output(manifest()) or 0)
    x=sub.add_parser("deploy");x.add_argument("--stage-only",action="store_true");x.add_argument("--dry-run",action="store_true");x.set_defaults(fn=deploy)
    x=sub.add_parser("status");x.set_defaults(fn=lambda a: output(status()) or 0)
    x=sub.add_parser("check");x.set_defaults(fn=lambda a: check())
    x=sub.add_parser("start");x.add_argument("--port",type=port,default=DEFAULT_PORT);x.add_argument("--studio-camera-service",default=STUDIO_CAMERA_SERVICE);x.add_argument("--anomaly-artifact",type=anomaly_artifact_path,default=DEFAULT_ANOMALY_ARTIFACT);x.add_argument("--studio-robot-session",help="read-only joint telemetry for mission control, rt-<follower id>");x.add_argument("--replay-dataset",help="remote LeRobot v3 dataset path shown as REPLAY");x.set_defaults(fn=lambda a: service("start",a.port,a.studio_camera_service,a.anomaly_artifact,a.studio_robot_session,a.replay_dataset))
    x=sub.add_parser("stop");x.add_argument("--port",type=port,default=DEFAULT_PORT);x.add_argument("--studio-camera-service",default=STUDIO_CAMERA_SERVICE);x.set_defaults(fn=lambda a: service("stop",a.port,a.studio_camera_service))
    x=sub.add_parser("pull");x.add_argument("--path",default="artifacts/runtime");x.add_argument("--destination",default="artifacts/runtime");x.add_argument("--dry-run",action="store_true");x.set_defaults(fn=pull)
    x=sub.add_parser("tunnel");x.add_argument("--local-port",type=port,default=DEFAULT_PORT);x.add_argument("--remote-port",type=port,default=DEFAULT_PORT);x.add_argument("--print-only",action="store_true");x.set_defaults(fn=lambda a:tunnel(a))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try: return int(args.fn(args))
    except ToolError as exc: print(f"error: {exc}",file=sys.stderr); return 2
    except KeyboardInterrupt: return 130


if __name__ == "__main__": raise SystemExit(main())
