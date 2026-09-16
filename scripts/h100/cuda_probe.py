#!/usr/bin/env python3
"""Bounded synthetic CUDA/serialization check. Does not exercise the robot trainer."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/workspace/second-look-h100')
for name in ('checks', 'cache', 'logs', 'checkpoints', 'tmp'):
    (ROOT / name).mkdir(parents=True, exist_ok=True)
for key, value in {'XDG_CACHE_HOME': ROOT/'cache', 'TORCH_HOME': ROOT/'cache/torch',
                   'HF_HOME': ROOT/'cache/huggingface', 'TMPDIR': ROOT/'tmp',
                   'CUDA_CACHE_PATH': ROOT/'cache/cuda'}.items():
    os.environ[key] = str(value)
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
if apps:
    raise SystemExit('BLOCKED: an existing GPU process is present; coordinate before probing')
import torch
if not torch.cuda.is_available():
    raise SystemExit('BLOCKED: torch cannot initialize CUDA')
torch.manual_seed(17)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
start = time.monotonic()
a, b = torch.randn(256, 256), torch.randn(256, 256)
gpu = (a.cuda() @ b.cuda()).cpu()
reference = a @ b
torch.testing.assert_close(gpu, reference, rtol=1e-4, atol=1e-4)
# Small optimizer state is saved and reloaded; this is explicitly synthetic.
model = torch.nn.Linear(8, 2).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
x, y = torch.randn(16, 8, device='cuda'), torch.randn(16, 2, device='cuda')
losses = []
for step in range(3):
    optimizer.zero_grad()
    loss = (model(x)-y).square().mean()
    loss.backward(); optimizer.step()
    losses.append(loss.item())
run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
checkpoint = ROOT/'checkpoints'/('cuda-fixture-'+run_id+'.pt')
with checkpoint.open('xb') as stream:
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': 3, 'seed': 17}, stream)
loaded = torch.load(checkpoint, map_location='cuda', weights_only=True)
restored = torch.nn.Linear(8, 2).cuda()
restored.load_state_dict(loaded['model'])
restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
restored_optimizer.load_state_dict(loaded['optimizer'])
torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
restored_optimizer.zero_grad(); resumed_loss = (restored(x)-y).square().mean()
resumed_loss.backward(); restored_optimizer.step()
torch.cuda.synchronize()
result = {'status': 'PASS', 'fixture': 'synthetic Linear(8,2), not Studio/SmolVLA or real task data',
          'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
          'seed': 17, 'optimizer_steps': 3, 'resume_step_executed': True,
          'matrix_max_abs_error': (gpu-reference).abs().max().item(), 'losses': losses,
          'elapsed_seconds': time.monotonic()-start, 'checkpoint': str(checkpoint),
          'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
report = ROOT/'logs'/('cuda-fixture-'+run_id+'.json')
report.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
