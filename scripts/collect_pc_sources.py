#!/usr/bin/env python3
"""Collect a verified, tracked-only reference snapshot without changing the PC."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'pc-context/sources/physical-ai-studio'
MANIFEST = ROOT / 'pc-context/manifests/physical-ai-studio.json'
REMOTE = '/home/ird-demo/physical-ai-studio'
EXPECTED = 'c4ff730fb49f84e5102d01088d52cfff1ba62854'

INVENTORY = r'''
import hashlib,json,os,re,subprocess
from pathlib import Path
root=Path('/home/ird-demo/physical-ai-studio')
os.chdir(root)
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if revision != 'c4ff730fb49f84e5102d01088d52cfff1ba62854':
    raise RuntimeError('Remote revision changed')
if subprocess.check_output(['git','status','--porcelain','--untracked-files=no']):
    raise RuntimeError('Remote tracked source is modified')
paths=subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
files=[]; exclusions=[]
secret_patterns=[
    rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    rb'\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{20,}',
    rb'\bAKIA[A-Z0-9]{16}\b',
]
excluded_parts={'.git','.agents','.claude','skills','node_modules','.venv','venv','__pycache__','.pytest_cache','.mypy_cache','dist','build'}
# These exact tracked templates were reviewed: credentials are blank and other values are defaults.
reviewed_templates={
    'application/backend/.env.example':'e8f71ebbd8fd4498a19ea016e55c6dd7e5ba332c7eff73f0529d8c17dbadca59',
    'application/docker/.env.example':'843e8a4b1c136ab77bb6a4ccd689a3a6ef89a685f177bfc5993cd8a2d112aa5b',
    'application/docker/.env.trainer.example':'c47477bff92cfff7f1cc22f77c051bd81f60f290cef89357b83464d15cc99324',
}
for name in filter(None,paths):
    p=Path(name); reason=None
    if p.is_absolute() or '..' in p.parts: reason='unsafe path'
    elif any(x in excluded_parts for x in p.parts) or p.name in {'AGENTS.md','CLAUDE.md'} or 'instructions' in p.name.lower(): reason='agent instructions or excluded generated directory'
    elif (p.name == '.env' or p.name.startswith('.env.')) and name not in reviewed_templates: reason='credential or environment file'
    elif p.suffix.lower() in {'.pem','.key','.p12','.pfx'} or p.name in {'credentials.json','id_rsa','id_ed25519'}: reason='credential file'
    elif p.is_symlink(): reason='symlink omitted conservatively'
    elif not p.is_file(): reason='not a regular file'
    elif p.stat().st_size > 5*1024*1024: reason='file larger than 5 MiB'
    elif p.suffix.lower() in {'.pyc','.pyo','.so','.dylib','.dll','.whl','.pt','.pth','.onnx','.safetensors','.bin','.zip','.gz','.mp4','.mov','.webm'}: reason='binary build, model, archive, or video'
    if reason:
        exclusions.append({'original_path':name,'reason':reason}); continue
    data=p.read_bytes()
    if name in reviewed_templates and hashlib.sha256(data).hexdigest()!=reviewed_templates[name]:
        raise RuntimeError('Reviewed environment template changed: '+name)
    if any(re.search(pattern,data) for pattern in secret_patterns):
        exclusions.append({'original_path':name,'reason':'possible embedded credential signature'});continue
    files.append({'original_path':name,'local_path':'pc-context/sources/physical-ai-studio/'+name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps({'source_host':'intel-robot','source_root':str(root),'revision':revision,'tracked_count':len(list(filter(None,paths))),'files':files,'exclusions':exclusions}))
'''

ARCHIVE = r'''
import json,sys,tarfile
from pathlib import Path
root=Path('/home/ird-demo/physical-ai-studio')
names=json.load(sys.stdin)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
    for name in names:
        p=root/name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
            raise RuntimeError('Unsafe source changed during collection')
        tar.add(p,arcname=name,recursive=False)
'''

def ssh(script, input_data=None):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','intel-robot',
                           'python3 -c '+shlex.quote(script)], input=input_data,
                          stdout=subprocess.PIPE,check=True).stdout

def main():
    inventory=json.loads(ssh(INVENTORY))
    if MANIFEST.exists():
        previous=json.loads(MANIFEST.read_text())
        if any(previous.get(key)!=inventory[key] for key in ['source_host','source_root','revision']):
            raise SystemExit('Refusing to replace a different snapshot manifest')
    expected={f['original_path']:f for f in inventory['files']}
    checked={}
    # Reuse only byte-identical files; never overwrite local edits or follow local symlinks.
    for name,record in expected.items():
        target=DEST/name
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise SystemExit('Refusing a local symlink: '+name)
        if target.exists():
            if not target.is_file():
                raise SystemExit('Local path is not a file: '+name)
            data=target.read_bytes()
            if len(data)!=record['size'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
                raise SystemExit('Refusing to overwrite different local data: '+name)
            checked[name]=data
    reused=set(checked)
    payload=ssh(ARCHIVE,json.dumps([name for name in expected if name not in checked]).encode())
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        for member in tar:
            if member.name not in expected or not member.isfile() or member.name in checked:
                raise RuntimeError('Unexpected archive member')
            data=tar.extractfile(member).read()
            record=expected[member.name]
            if len(data)!=record['size'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
                raise RuntimeError('Source changed or transfer checksum mismatch: '+member.name)
            checked[member.name]=data
    if checked.keys()!=expected.keys():
        raise RuntimeError('Archive is incomplete')
    for name,data in checked.items():
        target=DEST/name
        if name not in reused:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=expected[name]['sha256']:
            raise RuntimeError('Local verification failed')
    inventory['verification']='All copied file sizes and SHA-256 hashes match the remote inventory and local files.'
    inventory['total_bytes']=sum(f['size'] for f in inventory['files'])
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    manifest_text=json.dumps(inventory,indent=2)+'\n'
    if not MANIFEST.exists() or MANIFEST.read_text()!=manifest_text:
        staging=MANIFEST.with_suffix('.json.tmp')
        with staging.open('x') as stream:
            stream.write(manifest_text)
        staging.replace(MANIFEST)
    print(json.dumps({'verified_files':len(checked),'new_files':len(checked)-len(reused),'excluded_files':len(inventory['exclusions']),'bytes':inventory['total_bytes'],'manifest':str(MANIFEST)}))

if __name__=='__main__':
    main()
