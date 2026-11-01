# Proposed operational admission policy — review required

This policy does not alter the100000 loop ceiling,4s search deadline,1000 fixed
iterations,192 identities or scientific decisions. It replaces only the external
disk-admission assumption and adds an external supervisor. It is not active
principal-launch authorization.

## A. Expected storage

Use actual development files and observed wall attempts: approximately1.023 GB
for the full matrix. Compression and future iteration throughput may differ.
Only one development instance/seed supports this projection.

## B. Conservative admission under declared assumptions

Observed max wall attempt rate is1401/4=350.25/s. Provision for four times this
rate for4s plus one boundary iteration:5605. This engineering stress factor
allows a fourfold speedup on the same hardware; it is not a measured upper bound
or confidence limit. It was not selected by principal outcomes or reduced to
fit available space. Faster trajectories remain possible and require monitoring.

Provision10KiB per iteration rather than relying on compression:4024 journal,
4408 NPZ arrays, about80 JSON/hash bytes leave about20% representation slack.
Provision1MiB fixed per case (over7 times the140000-byte fixed array payload,
also covering headers/manifests/directory allocation);64MiB top-level for the
small dataset, registration references, reports, approvals, warmup and logs.
The resulting payload envelope is:

`96*(1000+5605)*10240 + 192*1048576 + 67108864 = 6761414656 bytes`.

Admission requires twice this payload plus a4GiB free-space floor:
`17817796608 bytes` (17.82 GB,16.59 GiB). The second payload-sized allowance
covers delivery/Git object or staging duplication without assuming compression.
The4GiB floor leaves room for the roughly1.025GB uncompressed maximal single-case
serialization provision plus emergency bookkeeping and an assumed1GiB unrelated
burst, rounded upward. None of these allowances guarantees protection against
unbounded unrelated writers or sudden filesystem failure.

## C. Theoretical ceiling

Keeping100000 for every wall case with the same10KiB/iteration and fixed
provisions gives99555475456 bytes before extra delivery reserve. The old exact
formula instead gave83.23GB before its factor2. Both are format/workload
envelopes, not a strict bound on total filesystem consumption. A practical
deterministic total-storage guarantee cannot be established from development
observations. Either use monitored-resource admission as explicitly approved,
or provision additional capacity for the theoretical workload and overhead.

## D. Runtime conditions

The external stdlib-only supervisor checks the actual output filesystem device,
free bytes, cumulative logical output bytes, growth rate and free-space loss
every0.25s. Admission remains conditional on continuous monitoring. Stop before
the4GiB floor if free bytes fall below
`floor + max(1025048576, 2 seconds * max(output growth, free-space loss rate))`.
This reserves a maximal-case serialization burst or the recent two-second rate,
whichever is larger. Growth beyond the estimated5605 iteration envelope alone
does not truncate a run. No new scientific iteration limit is introduced.

Low floor/growth reserve, ENOSPC or quota errors yield RESOURCE-INCONCLUSIVE
with DISK_FREE_FLOOR, DISK_GROWTH_RESERVE or DISK_WRITE_ERROR reasons. Monitor
I/O failures fail closed. The supervisor kills the launcher and separate worker
sessions found in its descendant tree, preserving all files and journal prefixes.
No partial case becomes valid; no retry, resume or claim deletion occurs.
Its own report supplements frozen P2 status rather than rewriting it.

Sampling and process-tree termination are best effort under OS scheduling;
unrelated bursts can exceed the reserve between samples. A partially written
last journal/NPZ is invalid evidence and must be retained, not repaired.
If even the operational report cannot be written, stderr is the fallback.
Future orchestration must archive that stderr on independent storage where
possible. External supervision adds host CPU/I/O load; disclose it equally
for both arms, without correcting observed timing. GPU and existing time/RSS
watchdogs remain authoritative alongside the additional disk stop.
