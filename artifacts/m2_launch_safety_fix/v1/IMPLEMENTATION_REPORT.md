# M2-S2F implementation

**M2-S2F READY FOR INDEPENDENT SAFETY RE-REVIEW.** No launch is approved. No
principal coordinates, execution authorization or principal identities were
generated/launched. Original RESOURCE-INCONCLUSIVE,0/192 stays unchanged.

## Four blockers addressed

F1: `m2_safety.launch` is the only public principal entry. It validates externally
supplied amendment/generation/execution approvals, hashes and a new attempt,
checks actual free space and reserves a persistent safety attempt. A contained
init repeats validation, checks registered source/dataset integrity, requires
fresh admission and a healthy disk sampler before starting P2. The two legacy
entry points require a sealed inherited capability inside the owned namespace;
direct execution refuses before GPU lock/claim/data-content reads. Worker
refusal is outside its old failure-writer block, so unauthorized attempts do
not touch an existing output directory.

F2: The disk sampler is a separate process. A crash, malformed response, missing
heartbeat, filesystem change or I/O error fails closed. Init checks heartbeats
within2s while sampler sleeps0.25s. Unexpected init/controller death is covered
by kernel containment and parent-death signaling, not try/finally alone.

F3: A dedicated Linux PID namespace owns launcher, sampler and worker, including
separate sessions/nested namespaces. Losing init terminates all descendants.
Host tests use pidfds with ownership/start-time/namespace checks; no numeric
PGID sorting or guessed process-group signaling is used. The exact util-linux
binary is pinned, and early controller-death races are tested.

F4: `authoritative_status.json` is the single final interpretation after namespace
exit verification. Disk and existing P2 resource stops yield RESOURCE-INCONCLUSIVE;
correctness/hash failures yield INCONCLUSIVE. Missing approvals/admission refuse
before work as BLOCKED. Cleanup errors append and cannot replace the first
origin. Losing the controller without a final write requires read-only
reconciliation; lack of a complete status can never mean scientific NO-GO.

## Validation evidence

- Final main suite:17 lifecycle cases and13 gate checks PASS; every verified
  shutdown has zero live owned processes and an unaffected external sentinel.
- Covers normal, supervisor exception, init/controller TERM/KILL, early launcher
  exit, simulated native stall, disk low-floor, serialization-phase simulation,
  sampler crash/heartbeat expiry/I/O error, serializer failure, repeated signals,
  insufficient admission and interrupted attempts. Tiny prefixes are retained.
- Nested PID namespace descendants are killed; attempted join of ancestor PID
  namespace returns EPERM. Exact PID/PPID/PGID/SID/start ticks and exit statuses
  are in ORPHAN_PROCESS_TESTS.json and per-case records.
- PINNED_BOOTSTRAP_TESTS.json additionally covers exact-binary normal completion,
  early interrupted bootstrap and controller SIGKILL.
- SOURCE_INVARIANCE.json proves P2 AST matches the original after stripping only
  guards and the added FD inheritance. No solver/runner logic changed.
-32 saved middle/late local-search states, both arms and all three measurement
  modes reproduce exact route/checklist/reference counters via the existing
  P1 saved-state gate. No new ACO trajectory is run by this saved-state gate.
-12 existing gate/scanner tests PASS. Five original inference/state-capture
  regression tests PASS inside the selected namespace with CUDA on tiny
  deterministic development TSP100 inputs. They check normal/traced paths,
  guidance/RNG/source/best/pheromone invariance; they are not TSP1K principal
  collection. No model is trained or checkpoint changed.

The first two dummy harness attempts timed out waiting for descendant readiness
because a timed injection preceded slow child startup. Attempt1's temporary
namespace logs were automatically removed by TemporaryDirectory cleanup; its
outer traceback and normal-case row remain. Attempt2 retained all case files.
The harness now uses an explicit fixture readiness/injection barrier. Passing
validation3/4/5 and final validation6 remain separately preserved. No failed
principal case existed or was retried. No numerical tolerance changed.

## Registration and limits

Two pinned Python files changed: srr_p2/execute.py and worker.py. Original frozen
registration/protocol/binary/checkpoint remain immutable, but the old executable
registration deliberately no longer verifies this checkout. A NEW executable
registration candidate and exact source delta accompany this package; independent
re-registration review is mandatory. All scientific settings/decisions remain.

S1 admission17817796608 bytes remains a conditional engineering policy, not a
hard bound. Both old and new source checks are retained explicitly; the new
candidate names the original scientific registration and executable source
commit. No approval was auto-created. Valid authorized principal collection
is deliberately NOT exercised, because this task authorizes only safety work.

The capability/approval checks protect supported entry paths, not a hostile
operator who can replace source files or forge human provenance. Kernel/host
failure, uninterruptible kernel I/O or sudden filesystem loss can delay exit or
prevent a final write; unverified exit is never COMPLETE. Zombified, non-running
process entries are distinguished from live processes. Additional monitor CPU/
I/O overhead applies equally and is not subtracted from search time. A future
review must accept the exact candidate/amendment and recheck the real launch
environment/capacity. There is no scientific GO or Stage B authorization here.

Final IPC review found a possible partial-line sampler stall. Non-blocking, bounded message reads now preserve heartbeat deadlines; validation6 includes the partial-message stop and passes all17 lifecycle cases. The pre-fix unapproved candidate and metadata remain in pre_nonblocking_candidate/. The final candidate pins the follow-up source commit.
