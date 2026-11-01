"""Preserve literal executed source bytes, including Windows/WSL line endings."""
import hashlib
import json
import zipfile
from pathlib import Path

out=Path(__file__).resolve().parent
root=out.parents[3]
frozen=json.loads((out.parent/'implementation_manifest.json').read_text())['source_hashes']
paths=set(frozen)|{str(p.relative_to(root)) for p in (root/'r3_final').glob('*.py')}
files={}
with zipfile.ZipFile(out/'executed_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(paths):
        raw=(root/name).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if name in frozen:assert digest==frozen[name]
        files[name]=digest
        archive.writestr(name,raw)
manifest=dict(zip_sha256=hashlib.sha256((out/'executed_sources.zip').read_bytes()).hexdigest(),
              scientific_sources_match_seal=True,files=files)
(out/'executed_source_archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
