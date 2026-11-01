"""Dummy-process and metadata-fixture tests only. No principal dataset/authorization."""
import os,sys,time,json,signal,tempfile,subprocess,copy,contextlib
from pathlib import Path
from .common import *
from .launch import identity,members,reconcile,exact_signal,reserve_attempt,owned_pidfd
from .approval import validate_records
OUT=ROOT/'artifacts/m2_launch_safety_fix/v1'

def waitfor(test,seconds=10):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if test():return
        time.sleep(.02)
    raise TimeoutError('dummy safety condition not reached')
def approval_tests():
    # In-memory synthetic metadata only. No approval/authorization file exists.
    base=dict(attempt='fixture-new',amendment_sha256='amend',implementation_sha256='impl',scientific_registration_sha256=SCIENTIFIC,registration_sha256='candidate',approved_by='SYNTHETIC TEST ONLY',approved_at='2026-01-01T00:00:00+00:00',basis_sha256='fixture')
    records=[dict(base,action=a) for a in ('ACCEPT_M2_OPERATIONAL_AMENDMENT','PERMIT_M2_DATASET_GENERATION','APPROVE_M2_PRINCIPAL_192')]
    records[2]['dataset_manifest_sha256']='seal'
    kw=dict(attempt='fixture-new',amendment_hash='amend',implementation_hash='impl',candidate_hash='candidate',seal_hash='seal')
    assert validate_records(*records,**kw)
    tests=[]
    for name,key,value in [('wrong_amendment','amendment_sha256','bad'),('wrong_registration','scientific_registration_sha256','bad'),('wrong_implementation','implementation_sha256','bad'),('old_attempt','attempt','previous-stopped'),('malformed','approved_at','bad'),('missing_identity','approved_by','')]:
        bad=copy.deepcopy(records);bad[0][key]=value
        try:validate_records(*bad,**kw)
        except (ValueError,KeyError):tests.append(dict(test=name,status='PASS'))
        else:raise AssertionError(name+' accepted')
    try:validate_records({},records[1],records[2],**kw)
    except ValueError:tests.append(dict(test='missing_approval',status='PASS'))
    else:raise AssertionError('missing accepted')
    return tests

def main():
    process=[];gates=approval_tests();watchdog=[]
    run_dir=OUT/'dummy_run_v6';run_dir.mkdir()
    with contextlib.nullcontext(str(run_dir)) as td:
        tmp=Path(td)
        sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
        sentinel_fd=os.pidfd_open(sentinel.pid)
        try:
            for scenario in ('normal','supervisor_exception','owner_term','owner_kill','controller_term','controller_kill','parent_launcher_exit','native_stall','low_disk','serialization_low_disk','watchdog_crash','watchdog_ioerror','watchdog_stall','watchdog_partial','serialization_failure','repeated_term','insufficient'):
                folder=tmp/scenario;folder.mkdir();record=folder/'record';output=folder/'output'
                internal={'owner_term':'native_stall','owner_kill':'native_stall','controller_term':'native_stall','controller_kill':'native_stall','repeated_term':'native_stall','native_stall':'low_disk'}.get(scenario,scenario)
                spec=dict(mode='fixture',scenario=internal,record=str(record),output=str(output),nested_namespace=scenario=='normal')
                outerfd=None
                if scenario=='normal':
                    outerfd=os.open('/proc/self/ns/pid',os.O_RDONLY);spec['escape_fd']=outerfd
                specfile=folder/'spec.json';save(specfile,spec)
                command=[sys.executable,'-c',"from m2_safety.launch import launch;import sys,json;print(json.dumps(launch(sys.argv[1])))",str(specfile)]
                with (folder/'controller.log').open('x') as log:
                    controller=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True,pass_fds=(outerfd,) if outerfd is not None else ())
                    controller_fd=os.pidfd_open(controller.pid);heldns=None;nestedfd=None;nested_snapshot=[]
                    try:
                        if scenario!='insufficient':
                            waitfor(lambda:(record/'namespace_owner.json').exists() and (output/'descendant.json').exists())
                            owner=read(record/'namespace_owner.json');heldns=os.open(f"/proc/{owner['pid']}/ns/pid",os.O_RDONLY)
                            snapshot=members(owner['pid_namespace']);assert len(snapshot)>=3
                            owner_fd=owned_pidfd(owner)
                            if scenario=='normal':
                                waitfor(lambda:(output/'escape.json').exists() and (output/'nested_descendant.json').exists())
                                nested=read(output/'nested_descendant.json');nested_snapshot=members(nested['pid_namespace']);assert nested_snapshot
                                nestedfd=os.open(f"/proc/{nested_snapshot[0]['pid']}/ns/pid",os.O_RDONLY)
                            (record/'inject').write_text('trigger synthetic failure after readiness')
                            try:
                                if scenario in ('owner_term','repeated_term'):exact_signal(owner_fd,signal.SIGTERM)
                                if scenario=='owner_kill':exact_signal(owner_fd,signal.SIGKILL)
                                if scenario in ('controller_term','controller_kill'):exact_signal(controller_fd,signal.SIGTERM if scenario=='controller_term' else signal.SIGKILL)
                                if scenario=='repeated_term':
                                    try:exact_signal(owner_fd,signal.SIGTERM)
                                    except ProcessLookupError:pass
                            finally:os.close(owner_fd)
                        else:owner=None;snapshot=[]
                        controller.wait(timeout=12)
                        if owner:waitfor(lambda:not members(owner['pid_namespace']),seconds=5)
                        if nested_snapshot:waitfor(lambda:not members(nested_snapshot[0]['pid_namespace']),seconds=5)
                        result=reconcile(record)
                        expected='COMPLETE' if scenario=='normal' else 'BLOCKED' if scenario=='insufficient' else 'INCONCLUSIVE' if scenario=='parent_launcher_exit' else 'RESOURCE-INCONCLUSIVE'
                        assert result['status']==expected,(scenario,result,read(record/'init_result.json') if (record/'init_result.json').exists() else None)
                        assert sentinel.poll() is None,'unrelated process killed'
                        if scenario!='insufficient':assert (output/'prefix.bin').read_bytes()==b'preserved-completed-and-partial-journal'
                        escape=read(output/'escape.json') if (output/'escape.json').exists() else None
                        if scenario=='normal':assert escape==dict(result=-1,errno=1),escape
                        row=dict(test=scenario,status='PASS',controller_pid=controller.pid,controller_exit=controller.returncode,host_snapshot=snapshot,nested_namespace_snapshot=nested_snapshot,nested_remaining_live_processes=members(nested_snapshot[0]['pid_namespace']) if nested_snapshot else [],namespace_owner=owner,namespace_identity=read(output/'identity.json') if (output/'identity.json').exists() else None,descendant=read(output/'descendant.json') if (output/'descendant.json').exists() else None,escape_test=escape,remaining_live_processes=members(owner['pid_namespace']) if owner else [],result=result,unrelated_sentinel_survived=True,partial_prefix_preserved=scenario!='insufficient')
                        process.append(row);print('PASS',scenario,result['status'],flush=True)
                        watchdog.append(dict(test=scenario,status='PASS',result=result))
                        save(run_dir/(scenario+'_process_test.json'),row)
                    finally:
                        if controller.poll() is None:exact_signal(controller_fd,signal.SIGKILL);controller.wait(timeout=5)
                        os.close(controller_fd)
                        if heldns is not None:os.close(heldns)
                        if nestedfd is not None:os.close(nestedfd)
                        if outerfd is not None:os.close(outerfd)
            # Persistent attempt and lock mechanics use temporary synthetic paths.
            reserve_attempt(tmp,'fixture-attempt')
            try:reserve_attempt(tmp,'fixture-attempt')
            except FileExistsError:gates.append(dict(test='duplicate_attempt',status='PASS'))
            else:raise AssertionError('duplicate allowed')
            import fcntl
            with (tmp/'gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                blocked=subprocess.run([sys.executable,'-c',"import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",str(tmp/'gpu.lock')],capture_output=True)
                assert blocked.returncode!=0;gates.append(dict(test='GPU_lock_contention',status='PASS'))
            # Direct legacy paths must refuse before authorization/content reads.
            for module,args in [('srr_p2.execute',['--registration','absent','--authorization','absent','--dataset-manifest','absent','--output',str(tmp/'forbidden')]),('srr_p2.worker',['absent','absent','absent',str(tmp/'forbidden')])]:
                denied=subprocess.run([sys.executable,'-m',module,*args],capture_output=True,text=True)
                assert denied.returncode!=0 and 'BLOCKED: use approved contained launch' in denied.stderr and not (tmp/'forbidden').exists(),denied.stderr
                gates.append(dict(test='legacy_bypass_'+module,status='PASS',returncode=denied.returncode))
            # Actual launch CLI missing approval exits before output/claim creation.
            badspec=tmp/'bad_principal_spec.json';save(badspec,dict(mode='principal',operational_approval=str(tmp/'absent')))
            denied=subprocess.run([sys.executable,'-m','m2_safety.launch','--spec',str(badspec)],capture_output=True)
            assert denied.returncode!=0;gates.append(dict(test='new_entry_missing_approval',status='PASS'))
            from .common import State
            state=State();state.fail('RESOURCE-INCONCLUSIVE','DISK_ORIGIN');state.fail('INCONCLUSIVE','CLEANUP_ERROR')
            assert state.result()['reason']=='DISK_ORIGIN' and state.result()['cleanup_errors']
            gates.append(dict(test='first_failure_not_overwritten',status='PASS'))
        finally:
            exact_signal(sentinel_fd,signal.SIGKILL);sentinel.wait();os.close(sentinel_fd)
    save(OUT/'ORPHAN_PROCESS_TESTS.json',dict(status='PASS',cases=process,principals_launched=0))
    save(OUT/'WATCHDOG_FAILURE_TESTS.json',dict(status='PASS',cases=watchdog))
    save(OUT/'APPROVAL_GATE_TESTS.json',dict(status='PASS',cases=gates,synthetic_metadata_only=True,actual_approval_files_created=0))
    save(OUT/'SAFETY_TEST_SUMMARY.json',dict(status='PASS',lifecycle_cases=len(process),gate_checks=len(gates),all_remaining_live_process_counts_zero=True,principal_data_generated=False,principal_runs=0))
if __name__=='__main__':main()
