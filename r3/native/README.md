# Independent native diagnostic

Build with `bash r3/build.sh` in the audited WSL environment. The separately
built `r3_diagnostic` module uses C++17/pybind11 and does not include the original
solver header or mutate solver state. Build flags and source/binary hashes are
preserved in R3 artifacts. No OpenMP is used inside the diagnostic scan; the
original solver retains its unchanged OpenMP path.

For fixed policy/detail counts, scan time is O(N²), storage O(NK + PN + cap*P)
where P=15 and cap=32. It stores no NxN matrix or complete move list. Raw
nonrobust signed-gain records are bounded samples closest to zero, not an
exhaustive distribution. Candidate ranking is checked separately against the
native backend using O(N) distance scratch space per row.

The scanner preserves double gain calculations and native float32 pruning
prefixes separately. Fused enumeration/classification total, time inside robust
move classification, remaining enumeration/bounded sampling, setup and result
packaging have explicit separate counters; timing instrumentation overhead is
included. Python separately records candidate construction/audit, edge counting,
copied-tour gain verification, state extraction and report generation.
