"""Race-protected parent-death chain into util-linux PID namespace."""
import os,sys,ctypes,signal
def main():
    parent=int(sys.argv[1]);lib=ctypes.CDLL(None,use_errno=True)
    if lib.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    if os.getppid()!=parent:os._exit(125)
    os.execv('/usr/bin/unshare',['/usr/bin/unshare','--user','--map-root-user','--pid','--fork','--mount-proc','--kill-child=KILL',sys.executable,'-m','m2_safety.init',*sys.argv[2:]])
if __name__=='__main__':main()
