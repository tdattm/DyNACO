# Registration impact — independent re-registration review REQUIRED

Original scientific registration SHA256 remains
5d2d8942f900d4d1a4ef89e069668c3d0290d17c628d058150ffe6998d5bca45.
Original protocol SHA256 remains
574cd6781e7a026951521e12bfdef28f5790752289c029f21b8db0f4e5e5dedd.
Those files and every P0/P1/P2/S1/S2 research artifact are unchanged.

Two active pinned files necessarily change to reject legacy bypasses:
execute.py adds guard import/check and inherited capability FD; worker.py adds
guard import/checks before its failure-writer and before work. Exact old/new
hashes are in SOURCE_INVARIANCE.json and the complete diff in source_delta.patch.
Stripping just those guards/FD inheritance yields the original AST exactly.
All other106 originally pinned files retain original hashes, including native
binary/checkpoint, runner/analysis, dispatch, RNG, candidate and pheromone code.

A separate registration_candidate.json will pin these two new hashes, all
m2_safety sources, the exact unshare binary and the new operational amendment.
Its source commit is the immutable safety implementation commit; packaging
comes in a descendant commit to avoid self-referential hashes. Its scientific
identity refers to the original registration; it is not silently substituted.
The original registration is expected to REFUSE this guarded checkout because
two source hashes differ. Do not disable those checks.

All16 TSP1K/PCG64 settings, three seeds,100 ants,K32 one/two-pass semantics,
1000 iterations,4s search,192-run order, float64 outcomes and scientific criteria
remain unchanged.32 saved-state native cases and17 existing correctness tests
passed (12 gate/scanner plus5 CUDA inference/state-capture inside namespace).

The candidate is PENDING review, not a new approved registration. A human must
accept its executable identity and exact amendment/version before any dataset
permission or execution authorization. The current task supplies none. Any
subsequent code/hash change invalidates that acceptance and needs review again.

Historical checkouts remain unchanged for reproducibility and are not supported
new-launch implementations. This is supported-path enforcement in the approved
candidate, not a host-wide security policy against a source-owning operator
running another checkout. Future approval must identify this exact implementation;
no old stopped-launch authorization is accepted as its amendment approval.
