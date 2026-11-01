"""Archive the exact tested binaries, without importing the solver or CUDA."""
import gzip
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[4]
out=Path(__file__).resolve().parent
manifest={}
for label,path in (
    ('executed',root/'.r3-build/native/faco_opt.cpython-313-x86_64-linux-gnu.so'),
    ('original_reference',root.parent/'DyNACO-r3/.r3-build/native/faco_opt.cpython-313-x86_64-linux-gnu.so')):
    raw=path.read_bytes()
    target=out/f'{label}_faco_opt.so.gz'
    target.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
    manifest[label]=dict(source_path=str(path),uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                         compressed_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         uncompressed_bytes=len(raw),compressed_bytes=target.stat().st_size)
assert manifest['executed']['uncompressed_sha256']==json.loads((out.parent/'implementation_manifest.json').read_text())['native_binary_sha256']
(out/'native_archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
