"""Default-deny guard for legacy entry points; sealed grant inside safety namespace."""
import os,json,fcntl
from pathlib import Path
from .common import sha,implementation
def require(role,registration,authorization,dataset_manifest,output):
    try:
        fd=int(os.environ['DYNACO_SAFETY_FD'])
        required=fcntl.F_SEAL_SEAL|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE
        if fcntl.fcntl(fd,fcntl.F_GET_SEALS)&required!=required:raise ValueError('unsealed capability')
        grant=json.loads(os.pread(fd,65536,0))
        if grant['mode']!='principal' or grant['implementation_sha256']!=implementation():raise ValueError('grant implementation')
        if sha(grant['spec_file'])!=grant['spec_sha256']:raise ValueError('launch specification drift')
        from .approval import validate_bundle
        validate_bundle(json.loads(Path(grant['spec_file']).read_text()))
        authorization=authorization or grant['authorization_file']
        if Path('/proc/1/ns/pid').stat().st_ino!=grant['pid_namespace'] or os.getpid()==1:raise ValueError('containment identity')
        if grant['registration_sha256']!=sha(registration) or grant['authorization_sha256']!=sha(authorization) or grant['dataset_manifest_sha256']!=sha(dataset_manifest) or grant['output']!=str(Path(output).resolve()):raise ValueError('launch capability mismatch')
        if Path('/proc/1/cmdline').read_bytes().find(b'm2_safety.init')<0:raise ValueError('missing safety owner')
        if role=='worker' and b'srr_p2.execute' not in Path(f'/proc/{os.getppid()}/cmdline').read_bytes():raise ValueError('worker must be child of admitted launcher')
        return fd
    except Exception as e:raise RuntimeError('BLOCKED: use approved contained launch; '+str(e)) from e
