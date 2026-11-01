# Process containment audit

Linux-specific chain: host controller → race-protected bootstrap → pinned
/usr/bin/unshare → namespace init PID1 → sampler and P2 launcher → P2 worker.
The launcher/worker may use setsid; this does not escape the PID namespace.
All three failure boundaries were tested on WSL2 Linux5.15.167.4, util-linux2.39.3.

Kernel semantics supporting init-death containment are documented by
[Linux pid_namespaces(7)](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html).
Parent-death signal semantics and credential-reset caveats are documented by
[PR_SET_PDEATHSIG](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html).
The bootstrap sets SIGKILL and rechecks its parent before exec. The selected
util-linux --kill-child implementation arms the child and checks the original
parent via pidfd before continuing; see its
[v2.39.3 source](https://github.com/util-linux/util-linux/blob/v2.39.3/sys-utils/unshare.c).
The installed binary contains that pidfd-race-check branch; its hash is recorded.

Host pidfds target exact processes. Namespace-owner FD acquisition verifies
start ticks and namespace identity. Namespace FDs are held during tests to
avoid inode reuse while checking zero remaining members. Host snapshots include
PID, PPID, PGID, SID, start time and namespace inode; inner identities show the
different namespace PIDs. No unrelated process group is signaled. The sentinel
outside containment survives every test. An inherited outer-namespace FD in
the fixture cannot be used to join the ancestor namespace (EPERM). A nested
PID namespace also loses all live members when outer init exits.

The external controller waits for bootstrap exit, then verifies no live namespace
member before finalization. Native PID1 SIGKILL is handled by the kernel. A
controller SIGKILL cannot write an audit itself: the existing record stays
incomplete until reconcile verifies exit and assigns RESOURCE-INCONCLUSIVE.
No surviving process is treated as successfully cleaned up. Zombie entries
have no executing worker and are recorded separately from live process checks.

Tests show no living descendant after successful shutdown on this target
environment. They do not claim a real-time bound for an unresponsive kernel or
proof against a malicious host administrator. If namespace creation, pidfds or
parent-death setup fail, launch refuses; there is no process-snapshot fallback.
The five existing CUDA development tests also pass within this namespace.
