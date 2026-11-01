# Admission/watchdog lifecycle

1. Public command requires a spec; there is no default launch or generator.
2. Validate three approvals, their actual basis hashes, chronology, amendment,
   candidate registration, source-version and NEW attempt identity.
3. Check unused outputs and free bytes on their actual parent filesystem;
   reserve an immutable safety attempt claim. No principal claim is removed.
4. Establish namespace and parent-death chain. Contained init repeats approval,
   registration and sealed-input checks. An old registration cannot pass the
   new guarded source hashes.
5. Init requires17817796608 bytes and a first valid sample before creating
   the launcher; admission older than2s is rejected, free bytes are rechecked
   immediately before invocation. The sampler targets output itself once it
   exists, otherwise its parent; symlink substitution/device drift refuses.
6. Only then create a sealed, inherited launch capability. Original P2 GPU
   flock, bound authorization, exclusive outputs and persistent claim remain.
7. Sampler runs throughout initialization, warmup, search, journals, NPZ/JSON
   serialization and final P2 analysis. It has no dependency on native solver
   or CUDA callbacks. Low4GiB floor/growth reserve or I/O faults stop the whole
   containment. A missing/malformed/error heartbeat also stops after launch.
8. Full-grid validation must finish and containment exit must be verified before
   authoritative COMPLETE. Raw P2 status/analysis alone is insufficient.

The proposal's4GiB floor plus max(1025048576-byte burst,2s observed consumption)
is unchanged. It covers observed growth/free-space loss, not an unbounded
concurrent writer. Fixed-iteration1000 and wall-loop100000 ceilings remain
unchanged; storage forecasts do not truncate scientific iterations.

The controller's2700s deadline starts before wrapper preflight and covers its
setup as well as P2 execution/output/analysis. Its initial setup watchdog is120s.
The unchanged inner P2 monitor retains per-phase120s and per-search60s/RSS
controls; the original worker retains CUDA/RSS observations. A blocked disk
   sampler is handled by independent init heartbeat; loss of init/controller is
handled by kernel containment. Monitor load is disclosed, not time-corrected.

Before-launch failures are BLOCKED with no solver work; after-launch monitor,
disk, RSS, CUDA or deadline failures are RESOURCE-INCONCLUSIVE. Source/solver
correctness failures remain INCONCLUSIVE. First-failure monotonic timestamps
prevent cleanup from replacing an earlier cause. ENOSPC/quota trace evidence
is explicitly classified as disk stop rather than a scientific quality result.
If the final report cannot be persisted, absence is not interpreted as success;
external reconciliation is required. Evidence stays in the attempt directory.

Approval Gates1/2/3 are distinct and all remain UNAPPROVED in this task.
The generation permission must already be bound into the future seal; no
generator is supplied or invoked. The public fixture mode is rejected. The
internal fixed dummy backend cannot create a principal capability or invoke
arbitrary principal commands. Legacy execute/worker calls without a valid grant
fail before touching outputs/claim. Arbitrary edits by a source-owning operator
are outside the supported-path model; approved hashes must be reverified.
