"""Verify the frozen team split files were not replaced after scoring."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
manifest=json.loads((ROOT/'data/split_manifest.json').read_text(encoding='utf-8'))
for filename, metadata in manifest.items():
    path=ROOT/'data'/filename
    data=path.read_bytes()
    assert len(data)==metadata['bytes'],f'{filename}: size changed'
    assert hashlib.sha256(data).hexdigest()==metadata['sha256'],f'{filename}: SHA256 mismatch'
    print('PASS',filename,len(data),'bytes')
