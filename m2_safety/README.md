# Contained M2 launch candidate — independent re-review required

No command generates datasets. No default command launches principal. This
implementation and its registration candidate are NOT approved for execution.

The sole public principal command is `python -m m2_safety.launch --spec PATH`.
The externally supplied spec must have mode `principal`, a NEW `attempt`,
absolute `record` and unused `output` paths, plus paths named `registration`,
`amendment`, `operational_approval`, `generation_permission`,
`execution_authorization`, `dataset_manifest`. Every referenced approval must
be supplied by the operator after actual human permission; none is generated.

All three approvals bind `attempt`, `amendment_sha256`, `implementation_sha256`,
`scientific_registration_sha256` (original scientific registration), and
`registration_sha256` (the independently reviewed NEW executable candidate).
They require truthful `approved_by`, timezone-aware `approved_at`, `basis_file`
and its `basis_sha256`. Actions in order:
ACCEPT_M2_OPERATIONAL_AMENDMENT, PERMIT_M2_DATASET_GENERATION,
APPROVE_M2_PRINCIPAL_192. The execution approval also binds
`dataset_manifest_sha256`. The seal must bind `generation_permission_sha256`
and a `sealed_at` timestamp between generation permission and execution approval.
This validates provenance content/ordering; it does not authenticate a human
signature. Preserve actual approval communications independently.

Old P2 execute/worker CLI commands now refuse without a sealed contained grant.
The guard revalidates approval metadata and implementation/version before each
entry. Original registration checks, GPU flock and persistent principal claim
remain. An extra per-attempt safety claim is never deleted on failure.
No automatic retry, resume, replacement or reset exists.

After interrupted controller death, `m2_safety.launch.reconcile(record_path)`
audits the stopped containment and writes authoritative_status.json if absent.
It never starts or resumes work and sends no signals. A stale/missing owner
result becomes BLOCKED before launch or RESOURCE-INCONCLUSIVE after work starts,
never COMPLETE. Already running containment returns RUNNING without finalizing.
Only authoritative_status.json after verified namespace exit may support a
COMPLETE interpretation; raw P2 analysis/status files alone are insufficient.

Fixture mode is available only to the internal library test harness. It runs
the fixed dummy module, cannot select arbitrary principal commands, and cannot
mint a principal grant. Public CLI refuses it. Synthetic test metadata is not
an execution authorization. Do not turn fixtures into real approval files.
