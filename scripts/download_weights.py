"""Download the immutable paper weight bundle and verify SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

def verify(path, spec):
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        return False
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest() == spec['sha256']


def assemble(output, manifest):
    for name, parts in manifest.get('assembly', {}).items():
        target = output / name
        if verify(target, manifest['files'][name]):
            continue
        temporary = target.with_suffix(target.suffix + '.tmp')
        try:
            with temporary.open('wb') as dst:
                for part in parts:
                    source = output / part
                    if not verify(source, manifest['downloads'][part]):
                        raise RuntimeError(f'Chunk verification failed: {part}')
                    with source.open('rb') as src:
                        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            if not verify(temporary, manifest['files'][name]):
                raise RuntimeError(f'Assembled weight verification failed: {name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('weights'))
    a = p.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'configs/weights.json').read_text())
    if not manifest['revision']:
        raise RuntimeError('Release has not yet been pinned to a Hub revision')
    required = manifest.get('downloads', manifest['files'])
    complete = {name for name, spec in manifest['files'].items() if verify(a.output / name, spec)}
    skipped_parts = {part for name in complete for part in manifest.get('assembly', {}).get(name, [])}
    for name, spec in required.items():
        if name in complete or name in skipped_parts:
            continue
        path = Path(hf_hub_download(manifest['repo_id'], name, revision=manifest['revision'], local_dir=a.output))
        if not verify(path, spec):
            raise RuntimeError(f'Weight verification failed: {name}')
        print(f'Verified {name}')
    assemble(a.output, manifest)
    for name, spec in manifest['files'].items():
        if not verify(a.output / name, spec):
            raise RuntimeError(f'Weight verification failed: {name}')
    print(f'Weights ready: {a.output}')

if __name__ == '__main__':
    main()
