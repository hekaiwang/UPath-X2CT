import hashlib
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('download_weights',Path(__file__).resolve().parents[1]/'scripts/download_weights.py')
download=importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

def checksum(value):
    return {'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}

def bundle(tmp_path):
    chunks=[b'first checkpoint bytes\x00\xff',b'second checkpoint bytes']
    for i,data in enumerate(chunks):(tmp_path/f'part{i}').write_bytes(data)
    return {'assembly':{'model.pt':['part0','part1']},'files':{'model.pt':checksum(b''.join(chunks))},'downloads':{f'part{i}':checksum(data) for i,data in enumerate(chunks)}}

def test_chunk_reassembly_is_byte_exact_and_idempotent(tmp_path):
    manifest=bundle(tmp_path)
    download.assemble(tmp_path,manifest)
    assert download.verify(tmp_path/'model.pt',manifest['files']['model.pt'])
    (tmp_path/'part0').unlink()
    download.assemble(tmp_path,manifest)
    assert download.verify(tmp_path/'model.pt',manifest['files']['model.pt'])

def test_corrupt_chunk_cannot_install_weights(tmp_path):
    manifest=bundle(tmp_path)
    (tmp_path/'part0').write_bytes(b'corruption')
    with pytest.raises(RuntimeError,match='Chunk verification'):
        download.assemble(tmp_path,manifest)
    assert not (tmp_path/'model.pt').exists()
    assert not (tmp_path/'model.pt.tmp').exists()

def test_wrong_chunk_order_cannot_install_weights(tmp_path):
    manifest=bundle(tmp_path)
    manifest['assembly']['model.pt'].reverse()
    with pytest.raises(RuntimeError,match='Assembled weight'):
        download.assemble(tmp_path,manifest)
    assert not (tmp_path/'model.pt').exists()
