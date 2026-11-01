"""One-time read-only admission audit; no principal generator or solver call."""
import os,sys,json,time,subprocess,shutil,fcntl,tempfile,resource
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from srr_p2.core import ROOT,sha,read,save,verify_registration,monitor,JOURNAL_DTYPE
OUT=ROOT/'artifacts/m2_principal/v1'
RH='5d2d8942f900d4d1a4ef89e069668c3d0290d17c628d058150ffe6998d5bca45'
def main():
    start=time.monotonic();record=dict(timestamp=datetime.now(timezone.utc).isoformat(),checks=[])
    def check(name,ok):
        record['checks'].append(dict(name=name,passed=bool(ok)))
        if not ok:raise RuntimeError('mandatory preflight gate: '+name)
    try:
        r=verify_registration(ROOT/'artifacts/srr_p2/v1/registration.json',RH)
        check('registration_and_all_source_hashes',True)
        record['registration_sha256']=RH
        record['protocol_sha256']=sha(ROOT/'artifacts/srr_p2/v1/FINAL_PROTOCOL.md')
        record['dispatch_sha256']=sha(ROOT/r['schedule'])
        check('192_unique_dispatch',len(read(ROOT/r['schedule']))==192 and len({i['id'] for i in read(ROOT/r['schedule'])})==192)
        for name in ('independent_audit.json','final_validation/tests.json'):
            check('P2_'+name,read(ROOT/'artifacts/srr_p2/v1'/name)['status']=='PASS')
        sys.path.insert(0,str(ROOT/r['binary_dir']));sys.path.insert(1,str(Path(r['diagnostic_binary']).parent))
        import torch,faco_opt,r3_diagnostic
        from r3.common import strict_cuda
        strict_cuda();env=r['environment']
        check('python_numpy_torch_cuda_versions',(sys.version,np.__version__,torch.__version__,torch.version.cuda)==(env['python'],env['numpy'],env['torch'],env['cuda']))
        compiler='/mnt/d/GitRepository/DyNACO/.r1-env/sysroot/usr/bin/x86_64-linux-gnu-g++-13'
        check('compiler',subprocess.check_output([compiler,'--version'],text=True)==env['compiler'])
        check('production_binary_loaded',sha(faco_opt.__file__)==r['files'][r['production_binary']])
        check('RTX3050_identity',torch.cuda.get_device_name()==env['gpu'])
        check('two_CPU_threads',torch.get_num_threads()==2)
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        record['gpu_processes']=gpu
        # This preflight's own CUDA context is permitted; no other compute PID.
        check('no_conflicting_GPU_process',all(line.split(',')[0].strip()==str(os.getpid()) for line in gpu.splitlines() if line.strip()))
        free,total=torch.cuda.mem_get_info();record['cuda_free_bytes']=free;record['cuda_total_bytes']=total
        check('CUDA_free_above_cap',free>1024**3)
        mem=dict((l.split(':')[0],int(l.split()[1])*1024) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','MemTotal:')))
        record['system_memory']=mem;check('available_RAM_above_RSS_cap',mem['MemAvailable']>3*1024**3)
        # Reserve even the registered 100000-iteration ceiling for every wall run,
        # plus uncompressed state/cost/journal, JSON and twofold headroom.
        per_iteration=JOURNAL_DTYPE.itemsize+4000+400+160
        estimate=96*(1000+100000)*per_iteration
        disk=shutil.disk_usage(OUT);record['disk']=dict(free=disk.free,estimated_max_bytes=estimate,required_headroom_bytes=2*estimate,per_iteration_bytes=per_iteration)
        check('conservative_disk_headroom',disk.free>2*estimate)
        check('no_launch_claim',not (ROOT/'.srr-build'/('principal_'+RH+'.claim')).exists())
        check('no_conflicting_output',not (OUT/'execution').exists())
        with (ROOT/'.srr-build/principal_gpu.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);check('GPU_lock_available',True)
        with tempfile.TemporaryDirectory(dir=OUT) as td:
            p=Path(td);(p/'write_check').write_text('write/read');check('write_permission',(p/'write_check').read_text()=='write/read')
            proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],start_new_session=True)
            result=monitor(proc,p/'events',time.monotonic()+.05)
            check('parent_watchdog_kills_blocked_child',result['status']=='RESOURCE-INCONCLUSIVE' and proc.poll() is not None)
        record['status']='PASS'
    except BaseException as e:
        import traceback
        record.update(status='BLOCKED',error=str(e),traceback=traceback.format_exc())
    record['elapsed_seconds']=time.monotonic()-start
    record['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    save(OUT/'preflight.json',record)
    print(json.dumps(record,indent=2))
    if record['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
