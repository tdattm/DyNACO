# M2-S2 technical review

**M2-S2 BLOCKED: STORAGE SAFETY.** Storage arithmetic and frozen identity pass;
mandatory launch wiring and fail-closed supervision do not. Do not generate
principal data or authorize execution on this evidence.

This is a new review turn by Codex in the same conversation as the implementing
agent. It independently recomputes hashes/math and uses new synthetic probes;
it is not an external independent reviewer or human approval. S1's reported
22 PASS results are separated from this turn's reproduced observations.

## Identity and preservation

Before creating this review branch, checked-out S1 HEAD was exactly
01fa5ae8580540da53db5f70a6f35866b77f98a7, with a clean working tree. The complete
S1 delta from c40a034 consists of16 new files under storage_recovery/ and
artifacts/m2_storage_recovery/v1/; no prior tracked file was edited. Source,
reports, manifests and recorded tests were inspected.

Independent SHA256 checks passed for108 registered files,1216 historical input
files,12 entries in the S1 artifact manifest and both S1 source entries.
Registration5d2d8942f900d4d1a4ef89e069668c3d0290d17c628d058150ffe6998d5bca45
and protocol574cd6781e7a026951521e12bfdef28f5790752289c029f21b8db0f4e5e5dedd
remain exact. This includes native binaries/checkpoint. Original principal
attempt remains RESOURCE-INCONCLUSIVE,0/192, no dataset or claim. No principal
coordinate file, solver launch or authorization file was created/accessed here.

The S1 amendment's immutable review reference is SHA256
fc88ecb001f219bf1ee414df30a4405f158055b9e1417deb6674b8c417fe537c.
This identifies content only and is not acceptance or a signature.

## Blocking findings

**F1 — No mandatory integrated launch path (blocker).**
storage_recovery/supervisor.py:7 defines a boolean admission function, and:35
defines watch on an already-started child. Only storage_recovery/review.py's
synthetic tests call them. srr_p2/execute.py imports neither; its public entry
point still acquires the GPU lock, validates the dataset-bound authorization,
creates the persistent claim, starts the worker in a separate session and
uses the existing time/RSS monitor. That monitor has no disk check. No versioned
launch wrapper connects admission, actual output path and monitoring. Thus
the17.82GB alternative is not an operationally enforced replacement and the
conditional storage proposal cannot safely justify launch.

**F2 — Supervisor exception leaves child alive (blocker, reproduced).**
supervisor.py:59 catches OSError only. A synthetic RuntimeError from the sampler
propagated out of watch, left the child running and wrote no stop status.
The reviewer harness then terminated it. Unexpected monitor failure or process
death has no separate heartbeat/owner that guarantees worker termination.
The proposed monitoring premise must fail closed even when monitoring fails.

**F3 — Loss of launcher leaves separate worker alive (blocker, reproduced).**
supervisor.py:52 returns CHILD_EXITED once its immediate child exits. The new
probe made a launcher start a worker in a separate session, then exit. watch
returned CHILD_EXITED while that worker remained alive. This mirrors the P2
separate-session topology, without invoking P2 or CUDA. The reviewer cleaned
up the orphan. A one-time descendant snapshot cannot recover an already
reparented worker; sorting group IDs also does not establish tree depth despite
the comment at:29. A stable owned job/process boundary is missing.

**F4 — No authoritative status/approval adapter (blocker by source review).**
The external disk report is a supplement only. Frozen worker.py:54 does not
classify general ENOSPC/EDQUOT OSError as resource; the proposed supervisor can
recognize selected text after exit, but nothing merges that stop with the
frozen study_status/watchdog/analysis outputs into one official result. No
adapter prevents scientific interpretation after an external resource stop.
No launcher checks acceptance of the amendment hash before data generation,
nor is that acceptance bound to the new execution authorization. The unchanged
P2 entry point bypasses every new disk/operational-approval check.

## Storage assessment

Old166460928000-byte and proposed17817796608-byte calculations reproduce
exactly. Journal4024 bytes, NPZ layouts, compression observations and wall
attempt counts were reread from development evidence. Details are in
storage_recalculation.json and STORAGE_REVIEW.md. The new threshold is a
conditional engineering admission requirement, not a deterministic upper
bound or statistically calibrated probabilistic bound. Fourfold throughput,
10KiB/iteration,1MiB/case,64MiB overhead, factor2 delivery and4GiB floor are
explicit provisioning assumptions. They have not been demonstrated as safe
under a complete integrated launch or competing filesystem writers.

The S1 observed21,037,568,000 bytes leaves3,219,771,392 bytes over the proposal
(18.0705% of required capacity). That historical margin cannot override F1-F4.
S2 also records a new free-space observation; any future launch needs another
measurement on the actual output filesystem immediately before admission.
The threshold must not be reduced or substituted merely to obtain a pass.

## Scope and next action

No fix was implemented in S1, P2 or scientific sources. This review adds only
artifacts/m2_storage_review/v1/. The16-instance PCG64 specification, three seeds,
K32 one/two-pass arms,100 ants,1000 iterations/4s,192-run order, frozen neural
guidance/checkpoint/pheromone/RNG/no-op arithmetic, float64 evaluation and
co-primary rules remain byte-for-byte unchanged in registered sources/protocol.

An external-only operational repair may be possible without changing scientific
registration, but must get a new immutable amendment, integration tests and
review. If any of the108 pinned files must change, require a new registration
and review before execution. No self-approved exception is available.

Next step: separately authorize implementation and synthetic testing of an
enforced launch wrapper, owned worker lifetime and fail-closed status arbitration.
Then repeat operational review before the three approval gates below. This
review grants none of those approvals and performs no recovery retry.
