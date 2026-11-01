"""Final artifact/provenance checks; never executes a solver or generates data."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/r3_final_end_to_end/v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    required=['FINAL_RESEARCH_REPORT.md','EXECUTIVE_DECISION.md','SCIENTIFIC_PROTOCOL.md',
              'scientific_protocol.sha256','implementation_manifest.json','environment_manifest.json',
              'dataset_manifest.json','execution_manifest.json','test_results.json','HANDOFF.md',
              'analysis/per_run_results.csv','analysis/per_instance_results.csv',
              'analysis/aggregate_results.json','analysis/statistical_tests.json','analysis/decision.json',
              'reproducibility/commands.md','reproducibility/source_hashes.json','reproducibility/execution_log.md']
    assert all((OUT/p).is_file() for p in required)
    manifest=json.loads((OUT/'execution_manifest.json').read_text())
    assert manifest['status']=='COMPLETE' and manifest['valid_runs']==192
    assert len(manifest['planned'])==len(manifest['runs'])==192
    assert {z['run_id'] for z in manifest['planned']}=={z['identity']['run_id'] for z in manifest['runs']}
    scientific=json.loads((OUT/'implementation_manifest.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in scientific['source_hashes'].items())
    assert sha(OUT/'SCIENTIFIC_PROTOCOL.md')==scientific['protocol_sha256']
    assert sha(OUT/'principal_coords.npy')==json.loads((OUT/'dataset_manifest.json').read_text())['sha256']
    native=json.loads((OUT/'reproducibility/native_archive_manifest.json').read_text())
    for label,row in native.items():
        packed=OUT/'reproducibility'/f'{label}_faco_opt.so.gz'
        assert sha(packed)==row['compressed_sha256']
        assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==row['uncompressed_sha256']
    source_all={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'r3_final').glob('*.py'))}
    ledger={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('final_delivery_audit.json','principal.lock')}
    result=dict(status='PASS',valid_runs=192,required_files_complete=True,frozen_scientific_sources_exact=True,
                dataset_and_protocol_unchanged=True,archived_native_binaries_exact=True,
                figures_png=len(list((OUT/'figures').glob('0*.png'))),figures_svg=len(list((OUT/'figures').glob('0*.svg'))),
                attempt_directories=len(list((OUT/'raw_results').glob('*/attempt_*'))),
                failed_attempts=len(list((OUT/'raw_results').glob('*/attempt_*/failure.json'))),
                archive_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
                all_final_python_source_hashes=source_all,artifact_sha256=ledger)
    assert result['figures_png']==result['figures_svg']==8
    (OUT/'reproducibility/final_delivery_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_sha256','all_final_python_source_hashes')},indent=2))


if __name__=='__main__':main()
