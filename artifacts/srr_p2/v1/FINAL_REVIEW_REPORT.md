# M2-P2 final technical review

**M2-P2 READY FOR EXPLICIT PRINCIPAL APPROVAL.** This is execution readiness,
not approval to generate data or collect principal outcomes. No principal
coordinates were generated, inspected or run. No reserved Stage A confirmation
data was accessed. No production file or P1/historical artifact was changed.

The dedicated branch is `codex/srr-registration-p2`, based exactly on reviewed
P1 `ac24f9323ca0ac99037f5790378e5fdd18fc3029`. The immutable implementation
commit is recorded in registration.json; a later packaging commit carries the
registration and audit evidence. P1's original worktree remains intact.

## Scientific registration

FINAL_PROTOCOL.md retains the 16-instance PCG64 specification, three paired
solver seeds, 100 ants, frozen checkpoint, K32 versus exactly two-pass K32,
1000 iterations and 4.0s modes, 192 deterministic dispatch identities, paired
instance bootstrap and co-primary decision rules. No principal quality was
observed or used to tune any design choice.

Equal-time results concern instrumented implementations. The P1 overhead means
were 19.9% and 32.2%; neither is subtracted or used to correct observations.
Compact P2 per-iteration crash journaling is applied equally, with its time on
the search clock. This extends persistence, not the solver's selection policy.
P1's 0.1% median / positive bootstrap lower bound / 12-of-16 conjunction remains
a prospective screening proposal with correlated criteria and small-n limits,
not a calibrated scientific guarantee.

Protocol freeze pins source, production binary, checkpoint, environment,
generator specification and dispatch before data generation. Dataset sealing
occurs only after explicit approval and before the first solver run. A future
external seal supplies its then-computed content hash. The registration has a
null dataset-content hash now by design. Execution authorization must bind the
registration and seal; operators must preserve the approval record externally.

## Execution and failure safety

P2 reuses the P1 runner body with phase callbacks, an append-only completed-state
journal and initial-state persistence. Source comparison tests strip exactly
these additions and require the remaining body to match P1. There is no
candidate, arithmetic, float32 gain, no-op, RNG, neural, NLS, or pheromone change.
The validated P1 production native binary is copied byte-for-byte, not rebuilt.

The orchestrator has required CLI arguments and an explicit authorization file;
there is no default launch or dataset generation side effect. It validates
source/binary/checkpoint hashes, environment, sealed coordinates, dispatch,
permutations, credited global-best float64 length and native counts. Every run
has an exclusive pre-run manifest and raw results. Only a complete valid grid
reaches the P1 instance aggregation. A per-registration persistent launch claim
and exclusive GPU lock prevent accidental duplicates and concurrent workloads
using this registered checkout. They do not control unrelated external GPU jobs.

A parent process watchdog remains independent of model/CUDA/native calls,
checks phase deadlines and process RSS, and terminates the whole worker group.
Tests scale deadlines down rather than waiting 120s/2700s. Model initialization,
warmup, setup, search and serialization are under these watchdogs. All completed
records and the current journal prefix survive failure. Unfinished in-flight
iteration work after a hard kill cannot be reconstructed exactly: that case
invalidates the study and is never replaced or counted as a valid case.

Future warmups explicitly save durations for each arm. Allocated CUDA and RSS
caps are checked alongside the parent RSS watchdog; reserved CUDA is reported
separately. Resource stops are RESOURCE-INCONCLUSIVE; correctness/hash/data/
environment errors are INCONCLUSIVE. No automatic retry or resume exists.

## Executed development checks

`final_validation/tests.json` and `final_tests.log` contain the passing checks.
Dispatch order/uniqueness, native seed derivation, last eligible timestamp,
zero-eligible rejection, actual saved P1 wall-tour float64 length, malformed
records, missing/duplicate grid records, seal/hash/shape/dtype/nonfinite
rejection, exclusive GPU lock, persistent duplicate claim, blocked startup,
blocked native search, aggregate and RSS failures all passed.

One 1000-iteration development run per arm with compact journals matched P1
tour arrays, all ant costs, source, global best and pheromone exactly. Their
journals match every saved completed timestamp/tour and final counters. Control
used 100000 calls / 25644615 evaluations; treatment used 200000 / 37994118.
The observed development search times were 4.429s and 2.930s, with peak RSS
at most 1375182848 bytes and CUDA allocated at most 128539136 bytes. These
single samples are not speed comparisons and do not revise budgets. Two earlier
development pairs were also retained; no full 192-run development matrix ran.

The intermediate journal test's source-comparison assertion failed after a
concurrent edit added initial-state persistence to the return value. Native
outcome comparisons had passed. Its log and outputs remain under
`journal_validation/` and `journal_tests.log`. The final frozen source was
retested successfully in `final_validation/`; no scientific tolerance changed.

`independent_audit.json` separately rereads archives/journals and registration,
rejects source/registration drift and unauthorized launch, and records P1 input
hashes. This is a separate technical check within the same agent session, not
an external research review. Full authorized principal orchestration remains
unexecuted, as required.

## Resource interpretation and handoff

P1's 192-run estimate was 750.1s, with doubled envelope 1459.2s against 2700s.
Using the slower single P2 development search sample for every iteration-mode
case would give about 425s for those 96 searches plus about 384s for 96 wall
searches, before setup/overshoot/output. This remains a feasibility projection
from observed development, not a guarantee across independent instances or
hardware contention. Hard caps remain binding; no budget changes, replacement
or partial scientific GO are permitted after launch.

Next action is independent protocol/registration review and explicit principal
approval. This delivery commits locally only. Stage A NO-GO, secondary
INCONCLUSIVE, M0 VALIDATED, M1v1 INCONCLUSIVE, M1v2 EXPLORATORY SIGNAL,
M1.5 536/396/140, P0 and P1 readiness decisions remain unchanged.
