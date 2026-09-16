#!/usr/bin/env bash
# Build a separate Studio environment. Never install into the sponsor environment.
set -euo pipefail
TASK_ROOT=${TASK_ROOT:-/workspace/second-look-h100}
[[ "$TASK_ROOT" == /workspace/* && "$TASK_ROOT" != *..* ]] || { echo 'TASK_ROOT must be below /workspace' >&2; exit 2; }
mkdir -p "$TASK_ROOT/cache" "$TASK_ROOT/tmp"
TASK_ROOT=$(realpath "$TASK_ROOT")
[[ "$TASK_ROOT" == /workspace/* ]] || exit 2
export TMPDIR="$TASK_ROOT/tmp" UV_CACHE_DIR="$TASK_ROOT/cache/uv" PIP_CACHE_DIR="$TASK_ROOT/cache/pip"
export CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"
export HF_HOME="$TASK_ROOT/cache/huggingface" XDG_CACHE_HOME="$TASK_ROOT/cache/xdg" TORCH_HOME="$TASK_ROOT/cache/torch"
STUDIO_REVISION=c4ff730fb49f84e5102d01088d52cfff1ba62854
RUNTIME_REVISION=8e4021703ef43387a835c6647b993cecc069ca85
SOURCE="$TASK_ROOT/code/physical-ai-studio"
ENVIRONMENT="$TASK_ROOT/training-env"
command -v uv >/dev/null
if [[ ! -d "$SOURCE" ]]; then
  mkdir -p "$TASK_ROOT/code"
  git clone --filter=blob:none --no-checkout https://github.com/open-edge-platform/physical-ai-studio.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$STUDIO_REVISION"
fi
[[ $(git -C "$SOURCE" rev-parse HEAD) == "$STUDIO_REVISION" ]] || { echo 'Existing Studio checkout has wrong revision' >&2; exit 2; }
[[ -z $(git -C "$SOURCE" status --porcelain --untracked-files=no) ]] || { echo 'Existing Studio checkout modified' >&2; exit 2; }
# Inherit the provided CUDA Torch wheels. Install all additions only in this venv.
if [[ ! -d "$ENVIRONMENT" ]]; then
  python3 -m venv --system-site-packages "$ENVIRONMENT"
fi
[[ -f "$ENVIRONMENT/pyvenv.cfg" && -x "$ENVIRONMENT/bin/python" ]] || { echo "Invalid existing environment" >&2; exit 2; }
"$ENVIRONMENT/bin/python" - <<'PY' > "$TASK_ROOT/torch-constraints.txt"
from importlib.metadata import version
for name in ('torch', 'torchvision'):
    print(f'{name}=={version(name)}')
PY
uv --version > "$TASK_ROOT/uv-version.txt"
# uv selects the official PyTorch cu128 index only for the Torch ecosystem.
uv pip install --python "$ENVIRONMENT/bin/python" --torch-backend cu128 \
  --constraint "$TASK_ROOT/torch-constraints.txt" \
  "physicalai @ git+https://github.com/openvinotoolkit/physicalai.git@$RUNTIME_REVISION" \
  --editable "$SOURCE/library[smolvla]"
uv pip freeze --python "$ENVIRONMENT/bin/python" > "$TASK_ROOT/training-requirements.freeze.txt"
"$ENVIRONMENT/bin/python" - <<'PY' > "$TASK_ROOT/training-imports.json"
import inspect, json, torch
from importlib.metadata import version
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.policies.smolvla.policy import SmolVLA
from physicalai.train import Trainer
print(json.dumps({
    'status': 'IMPORTS_VERIFIED', 'training_executed': False,
    'torch': torch.__version__, 'cuda_build': torch.version.cuda,
    'packages': {n: version(n) for n in ('physicalai', 'physicalai-train', 'lerobot', 'lightning', 'transformers')},
    'source': inspect.getfile(SmolVLA),
    'trainer_fit': str(inspect.signature(Trainer.fit)),
    'smolvla': str(inspect.signature(SmolVLA)),
    'datamodule': str(inspect.signature(LeRobotDataModule)),
}, indent=2))
PY
printf 'Environment: %s\nImport report: %s\n' "$ENVIRONMENT" "$TASK_ROOT/training-imports.json"
