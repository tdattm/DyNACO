# Reproduction commands

From PowerShell; all paths refer to the isolated research worktree. Existing
WSL environment is reused read-only. Do not overwrite an archived experiment.
For an independent rerun, use a fresh worktree and a new versioned output path.

```powershell
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO-r3-final/r3/build.sh
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO-r3-final/r3/run.sh -m r3_final.validate
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO-r3-final/r3/run.sh -m r3_final.final_verify
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO-r3-final/r3/run.sh -m r3_final.diagnostics
# Commit implementation and protocol, then seal the dataset exactly once.
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO-r3-final/r3/run.sh -m r3_final.execute seal
# Commit sealed provenance before the first principal run.
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO-r3-final/r3/run.sh -m r3_final.execute supervise
wsl -d Ubuntu -- bash /mnt/d/GitRepository/DyNACO-r3-final/r3/run.sh -m r3_final.analyze
```

The last command alone reproduces CSV/JSON statistics, decision, PNG/SVG figures
and reports from archived raw records without new solver runs. It verifies raw
hashes, routes, paired initialization, directed slots, timestamps and official
float64 outcomes first. Scientific sources and native binary must match the seal.
The unchanged reference binary for the full1000 comparison was reused from
DyNACO-r3/.r3-build/native; its hash is archived in development/reference_identity.json.
Rebuild that binary from the reference commit if it is unavailable.

`executed_sources.zip` preserves literal scientific and analysis source bytes,
including the original Windows/WSL line-ending representations. Its per-file
SHA256 ledger is `executed_source_archive_manifest.json`. In a fresh reproduction
checkout, use those archived bytes if checkout line endings differ from the seal.
The exact new and original native binaries are also archived as `.so.gz`, with
compressed/uncompressed hashes in `native_archive_manifest.json`.

Final delivery checks: `python -m r3_final.delivery_audit` (no solver execution).
Analysis reporting fixes only normalized NumPy Boolean JSON serialization,
clarified report text and improved plot layout; the frozen scientific source,
dataset, allocation, budgets, pairing, bootstrap RNG and decision thresholds did
not change during or after collection. The principal repository HEAD was the
seal commit `de828bb`; the scientific implementation was frozen in `6742353`.

No paid services, training, reserved confirmation data or two-pass SRR are used.
