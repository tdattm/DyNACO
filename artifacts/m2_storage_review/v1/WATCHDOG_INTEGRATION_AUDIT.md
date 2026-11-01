# Actual launch-path and failure audit

Existing path: separate m2_principal/preflight.py (still old166.46GB admission)
→ externally authorized dataset generation/seal → operator authorization file
→ srr_p2.execute → GPU flock → authorization hash check → persistent claim
→ output/launch manifest → worker separate process session → frozen parent
time/RSS monitor → final watchdog JSON. Worker verifies inputs/environment,
loads checkpoint, warms up, then runs192 cases with phase events, journals,
NPZ/JSON, validation and full-grid analysis.

Missing path: an enforced approved operational wrapper connecting the NEW
admission to that entry point and supervising its entire lifetime. The S1
watch function observes an already-started child; no launcher invokes it.
The model's admit function is exercised only in tests/reported resource review.
No dataset generator or execution authorization was run by S2 to test this.

| Condition | Implemented behavior/evidence | Review |
|---|---|---|
| Insufficient space before launch | admit returns bool; not called by actual launch | BLOCKED |
| Low free space during search | standalone watch kills known descendant groups; S2 prefix probe passes | Partial only; not wired |
| Low free space during serialization | same proposed poll function; no integrated serializer test | NOT DEMONSTRATED |
| External writer consumes disk | free-rate arithmetic exists; no competing-writer launch test | Conditional/unverified |
| ENOSPC/quota write error | text scan/OSError catch in standalone utility; frozen worker often says INCONCLUSIVE | No authoritative arbitration |
| Native/CUDA worker stalls | unchanged P2 time/RSS watchdog can kill live worker while parent lives | Existing guard, not disk wiring |
| Supervisor RuntimeError/death | synthetic RuntimeError left child alive and no stop file | REPRODUCED UNSAFE |
| Launcher exits while worker survives | watch returns CHILD_EXITED with orphan worker alive | REPRODUCED UNSAFE |
| Sampler blocks in filesystem I/O | synchronous sample has no independent heartbeat deadline | Failure protection missing |
| Partial NPZ/journal | no deletion by watch; S2 tiny prefix survives stop | Retain; never validate partial as complete |
| Report write fails | stderr fallback only; no guaranteed separate evidence sink | Missing integrated persistence |
| Final analysis after disk stop | no top-level status precedence contract | BLOCKED |

GPU lock, dataset-bound authorization, exclusive output directory and persistent
claim remain unchanged in hashed P2 source. This establishes preservation, not
new end-to-end enforcement of operational admission or three approval gates.
No old approval/result is overwritten; no claim is removed. Future repairs
must retain these guards, no retry/resume/replacement, and no official partial
scientific aggregation. A new entry point plus wrapper acceptance must prevent
the old direct entry path from silently bypassing the approved safety policy.

No registered source was modified to address any issue. If avoiding the bypass
requires modifying pinned execute/worker/core, a new registration is required.
