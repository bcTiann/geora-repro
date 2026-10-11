"""Validate a temporary model copy on login CPU before allocating one GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from safetensors import safe_open


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--destination', type=Path, required=True)
parser.add_argument('--expected', type=Path, required=True)
arguments = parser.parse_args()
started = time.perf_counter()
expected = json.loads(arguments.expected.read_text())
if arguments.source.resolve() != Path(expected['source']).resolve():
    raise RuntimeError('Model source differs from the previously validated pinned source')
arguments.destination.mkdir(parents=True, exist_ok=False)
files = {}
for name, metadata in expected['files'].items():
    source = arguments.source / name
    target = arguments.destination / name
    source_hash = digest(source)
    if source_hash != metadata['source_sha256']:
        raise RuntimeError('Original model file changed: ' + name)
    shutil.copyfile(source, target)
    if digest(target) != source_hash:
        raise RuntimeError('Copied file differs: ' + name)
    files[name] = dict(bytes=target.stat().st_size, source_sha256=source_hash, copied_sha256=source_hash)
    print('PASS: exact pinned file copy', name, flush=True)
with safe_open(str(arguments.destination / 'model.safetensors'), framework='pt', device='cpu') as checkpoint:
    count = len(checkpoint.keys())
    if count != 338:
        raise RuntimeError('Unexpected model tensor count')
record = dict(status='validated', source=str(arguments.source.resolve()),
              staged=str(arguments.destination.resolve()), model_revision=expected['model_revision'],
              files=files, source_tensor_count=count, all_files_exact=True,
              elapsed_seconds=time.perf_counter() - started)
(arguments.destination / 'staging_record.json').write_text(json.dumps(record, indent=2) + '\n')
print('CPU staging passed:', arguments.destination, flush=True)
