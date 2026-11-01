# Before implementation: S2F architecture and risks

Starting HEAD136a1ed99c1439c6f75dcca50b23e3d4303b6a1d is clean. S2 found
four blockers: optional admission/watchdog, monitor exceptions fail open,
orphan separate-session worker, and absent status/approval arbitration.

Unprivileged user/PID/mount namespaces are available on this WSL2 Linux5.15
host; delegated writable cgroup2 control is not available to this user. Select
a dedicated PID-namespace init as the safety owner. Linux terminates every
process in a PID namespace when its init exits, including separate sessions
and nested namespaces. This is stronger than process-tree snapshots. Test
actual SIGKILL paths before readiness; no try/finally-only containment claim.

An outer controller starts a bootstrap with race-protected parent-death SIGKILL;
bootstrap execs unshare --fork --pid --mount-proc --user --map-root-user
--kill-child=KILL. The init runs admission, disk/time/RSS monitoring and only
then starts the registered launcher/worker. Losing init kills the namespace;
losing controller must kill bootstrap and init. Use pidfds in host tests to
signal exact owned processes, recording starttime/PID/PPID/PGID/SID/ns inode.
No signal is sent to a guessed or numerically sorted process group.

Disk sampling runs separately from init's heartbeat loop. A sampler crash,
malformed response, missing heartbeat, filesystem error or low free space
must terminate containment. First failure is immutable; cleanup errors append
without replacing it. An independent external owner writes the final status
after namespace exit, before accepting any COMPLETE result. Retain raw P2
records without rewriting historical/frozen result JSON.

The actual supported entry path must bind amendment/source/attempt approval,
fresh admission on the output filesystem, GPU lock and persistent claim before
worker creation. Purely external wrapping cannot reject direct use of Python
P2 entry points owned by the same user. Minimal guards in execute.py/worker.py
are therefore necessary. These are registration-pinned bytes: preserve original
registration, produce exact diffs and a new registration CANDIDATE, and require
independent re-registration review. No native/runner/scientific edits planned.

All tests use tiny synthetic artifacts/dummy processes. No principal identity,
dataset, human approval or real execution authorization is created. Numerical
validation may replay saved local-search states with the unchanged native
binary; no principal sampling is permitted. Linux-specific containment,
CUDA compatibility and operational overhead must remain explicit review topics.
