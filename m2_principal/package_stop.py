"""Package the failed admission, without retry, generation or solver launch."""
import json
from pathlib import Path
from srr_p2.core import ROOT,read,save,sha,JOURNAL_DTYPE
OUT=ROOT/'artifacts/m2_principal/v1'
def main():
    p=read(OUT/'preflight.json');assert p['status']=='BLOCKED'
    rh=p['registration_sha256'];claim=ROOT/'.srr-build'/('principal_'+rh+'.claim')
    assert not claim.exists() and not (OUT/'execution').exists()
    base=ROOT/'artifacts/srr_p2/v1/final_validation'
    sizes={}
    for arm in (0,1):
        sizes[str(arm)]={ext:(base/f'development_arm{arm}.{ext}').stat().st_size for ext in ('json','npz','bin')}
    # Supplementary sizing only: no revised admission decision and no retry.
    save(OUT/'storage_audit.json',dict(development_1000_iteration_file_bytes=sizes,journal_record_bytes=JOURNAL_DTYPE.itemsize,
        observed_max_case_bytes=max(sum(v.values()) for v in sizes.values()),
        pilot_scaled_192_cases_bytes=192*max(sum(v.values()) for v in sizes.values()),
        admission=p['disk'],explanation='Pilot-scaled size is much smaller than the theoretical wall-loop-ceiling bound. Admission used the latter with 2x headroom; this was a new conservative operational check, not a registered scientific parameter or an observed disk-full error. No alternate threshold was applied after failure.'))
    save(OUT/'decision.json',dict(decision='M2 RESOURCE-INCONCLUSIVE',stage='A: pre-launch resource admission',cause=p['error'],completed_runs=0,valid_runs=0,planned_runs=192,registration_sha256=rh,dataset_sha256=None,dataset_generated=False,launch_claim_created=False,execution_authorization_created=False,approval_basis_sha256=sha(OUT/'authorization_basis.txt'),scientific_GO_NO_GO_computed=False,resource_observations=p))
    save(OUT/'full_grid_audit.json',dict(status='NOT_RUN',reason='Preflight stopped before data generation; no official statistics on a partial grid',completed=0,valid=0,missing=192,equal_iteration_findings=None,equal_wall_findings=None))
    names=['FINAL_PROTOCOL.md','registration.json','registration.sha256','dispatch.json','FINAL_REVIEW_REPORT.md','independent_audit.json','final_validation/tests.json']
    save(OUT/'registration_reference.json',dict(expected_packaging_HEAD='466706b1163155561ecad37be8367cadba61a9ef',implementation_commit='07af7acf09e1685db3c61500579f65420148b3dc',files={n:sha(ROOT/'artifacts/srr_p2/v1'/n) for n in names}))
    sizing=read(OUT/'storage_audit.json');d=p['disk']
    text=f'''# Registered M2 study: pre-launch resource stop

**Official decision: M2 RESOURCE-INCONCLUSIVE.** Completed principal runs:0/192;
valid principal runs:0/192. No dataset was generated or sealed, no principal
launch claim was created, and no registered execution authorization was issued.
The actual user instruction authorizing the workflow is archived verbatim with
its session record and SHA256; no human identity/signature was invented.

## Research question and planned method

Does one additional unchanged K32 SRR pass improve end-to-end DyNACO under matched
computational budgets? The frozen design specified16 uniform Euclidean TSP1K
instances, PCG64 seed202612070201, solver seeds31415/27182/16180,100 ants,
one frozen checkpoint, original seed/guidance/candidate/pheromone semantics and
two arms. Planned modes were1000 iterations and4.0s instrumented search,
192 runs total, two CPU threads and sequential RTX3050 execution.
Registered limits were1GiB allocated CUDA,3GiB RSS,60s search,120s setup,
2700s aggregate. Co-primary screening required both modes to satisfy median
instance gain>=0.1%, positive mean-bootstrap lower bound and>=12/16 positive
instances;10000 paired instance resamples use PCG64 seed202612070202.

Registration SHA256: `{rh}`.
Protocol SHA256: `{p['protocol_sha256']}`.
Dataset SHA256: **NOT CREATED**. Authoritative protocol/source/binary/checkpoint
hashes passed. Frozen Python3.13.18, NumPy2.5.4, Torch2.7.1+cu126,
CUDA12.6, GCC13.3.0, exact RTX3050 identity and two CPU threads passed.
The GPU compute-process query reported no conflicting compute workload.

## Mandatory gate failure

The preflight stopped on conservative disk headroom. Available bytes:
{d['free']}; admission maximum-output estimate:{d['estimated_max_bytes']};
required twofold headroom:{d['required_headroom_bytes']}. The bound allocates
96 iteration-mode cases at1000 iterations plus96 wall cases at the frozen
100000-iteration loop ceiling, with8584 bytes per iteration for journals,
tour/cost states and metadata. Actual disk exhaustion did not occur.

The new admission check was deliberately conservative and was not itself a
frozen scientific threshold. For transparency, actual P2 development artifacts
have at most{sizing['observed_max_case_bytes']} bytes per1000-iteration case;
scaling that observed footprint to192 cases gives
{sizing['pilot_scaled_192_cases_bytes']} bytes before headroom and variable
wall-run iteration counts. Thus the blocked ceiling estimate must not be
described as expected consumption. No threshold was relaxed after the failed
mandatory gate, no alternative storage/dataset was selected, and no retry ran.

Preflight elapsed:{p['elapsed_seconds']:.3f}s; peak process RSS:
{p['peak_rss_bytes']} bytes. CUDA free memory:{p['cuda_free_bytes']} bytes of
{p['cuda_total_bytes']}. Available system RAM was
{p['system_memory']['MemAvailable']} bytes. These are admission observations,
not principal study resource use. Per-run search, setup, warmup, allocated/
reserved CUDA peaks and principal evaluation counters are NOT RUN/NOT MEASURED.
Checks after the failed disk gate, including the fresh watchdog smoke test,
were not executed. Prior P2 tests passed, but were not substituted for new tests.

## Results and statistical interpretation

Equal iterations: NOT RUN; no quality, runtime or evaluation comparison.
Equal wall time: NOT RUN; no eligible principal checkpoints or outcomes.
No official mean, median, uncertainty, anytime curve or scientific GO/NO-GO is
computed from this incomplete grid. A technical/resource failure is not zero
improvement and is not a scientific NO-GO. The existing M1v2 exploratory signal
is neither confirmed nor contradicted by this admission failure.

## Limitations and recommendation

The planned scientific inference would cover only TSP1K,16 independent instances,
three seeds, one checkpoint and the RTX3050 environment. Instrumented equal-time
results would include telemetry overhead (P1 approximately19.9% control and
32.2% treatment), without subtraction. The prospective co-primary screening
rule is not formally calibrated. No causal isolation of native duplicate
suppression or evidence for adaptive candidate expansion is provided.

Recommendation: prepare a separately reviewed future launch with sufficient
audited storage capacity, or prospectively justify a practical output admission
bound using observed iteration rates. Do not silently rerun this stopped study
or alter the frozen scientific budgets. No follow-up is executed here.

Historical decisions remain Stage A NO-GO, secondary INCONCLUSIVE, M0 VALIDATED,
M1v1 technical INCONCLUSIVE, M1v2 EXPLORATORY SIGNAL, M1.5 536/396/140,
P0/P1/P2 readiness. Frozen source and historical evidence remain unchanged.
'''
    (OUT/'M2_RESEARCH_REPORT.md').open('x').write(text)
    (OUT/'HANDOFF.md').open('x').write('M2 RESOURCE-INCONCLUSIVE at pre-launch disk admission. 0/192 runs; no dataset, claim or execution authorization. See M2_RESEARCH_REPORT.md, preflight.json, storage_audit.json and decision.json. Actual development-scaled storage is distinguished from the conservative loop-ceiling bound. Future work requires separate reviewed admission/launch; no retry or scientific NO-GO is implied. Historical scientific decisions are unchanged.\n')
    save(OUT/'input_manifest.json',dict(registration_sha256=rh,authorization_basis_sha256=sha(OUT/'authorization_basis.txt'),source_files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'m2_principal').glob('*.py')}))
    save(OUT/'output_manifest.json',dict(files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json'}))
if __name__=='__main__':main()
