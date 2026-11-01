# M2 registered execution tooling

No command generates principal coordinates. `python -m srr_p2.execute` without
arguments exits with usage. Do not execute principal until explicitly approved.
Run under the frozen WSL Python environment, from this worktree. Set
LD_LIBRARY_PATH to the R1 sysroot library directory and PYTHONPATH to this root.

After protocol approval, an external, separately authorized generation step must
save the single specified NPY draw and a JSON seal with keys `role` equal to
`M2_PRINCIPAL_SEALED`, `registration_sha256`, `file` (relative to seal), `sha256`.
Preserve generation and sealing timestamps/provenance alongside that seal.
The seal has no placeholder content hash: compute it only after generation.

The separately supplied authorization JSON must contain `action` equal to
`APPROVE_M2_PRINCIPAL_192`, `registration_sha256`, `dataset_manifest_sha256`,
`approved_by`, and `approved_at`. The operator supplies the approval; code cannot
authenticate a human signature. Archive the approval communication independently.

The sole principal entry point is `python -m srr_p2.execute --registration PATH
--authorization PATH --dataset-manifest PATH --output NEW_DIRECTORY`.
This is documentation of the future interface, not authorization to run it.
The worker rechecks the bound authorization, hashes and environment. The parent
holds an exclusive GPU lock; a persistent per-registration claim prevents repeat
launches even into another output directory. Failure never clears the claim.
No resume or automatic retry is implemented. Independent review is required
after a failed attempt; never delete the claim to replace a scientific case.

`completed_iterations.bin` is an append-only, unbuffered journal with dtype
`core.JOURNAL_DTYPE`: elapsed seconds, cumulative native calls/evaluations,
and the 1000-node global best array. Decode complete records only; a partial
last record is an interrupted write and never eligible. Its cost is charged
to the running search clock. It uses saved per-iteration state and adds no
move-level route-copy diagnostics. Hard-killed in-flight iterations have no
complete state/counter guarantee and render the whole study resource-inconclusive.
All files, completed prior runs, journal prefixes and manifests are retained.

Each run saves manifest before execution, raw NPZ and JSON, validation and hashes.
The parent watchdog checks RSS and phase deadlines every 10ms and kills the
worker process group even during CUDA/native blocking. This is an OS scheduling
bound, not a real-time zero-overshoot guarantee. Model initialization, warmup,
setup, serialization and final analysis remain under parent deadlines.
No unrelated GPU processes can be controlled by this advisory lock; the launch
operator must reserve the GPU and stop rather than compete with another job.
