# Review of all22 reported S1 tests

The22 PASS labels were read from the immutable S1 safety_tests.json. This review
does not rerun S1's review.py, which would try to recreate its evidence directory.
Reported tests were inspected against code; evidence reproduced here is explicit.

| S1 checks | Count | S2 assessment |
|---|---:|---|
| Original formula/development sizes |2| Independently recomputed from struct layout and actual files |
| Simulated admission pass/reject |2| Source comparisons at required±1 are correct, but do not prove real launch admission |
| Stop/prefix/nested-worker for floor/growth/ENOSPC |9| Mocked sampler and tiny sleeping children; useful unit cases. S2 independently reprobed floor stop/prefix, not all9 |
| Logged write-error classification/prefix |2| Fake log contains literal Errno28; does not demonstrate a real serialization OSError or status arbitration |
| Four frozen protection hashes |4| Independently rehashed all108 registered entries |
| Exclusive lock mechanics |1| Generic temporary flock contention; not integrated GPU reservation testing |
| Historical bytes/no-principal checks |2| Independently verified1216 historical hashes and reviewed prior zero-run state/absent claims |

Counts sum to22. The checks cover arithmetic, mocked branches, generic locks
and preservation, not the integrated new operational launch. Their PASS status
does not cover missing wiring. S2 added two previously absent adversarial cases:
non-OSError monitor failure and launcher-exit/orphan-worker survival. Both
exposed unsafe behavior; their synthetic children were explicitly cleaned up.

Missing evidence: real filesystem sampler tied to actual output, enforced
prelaunch reject before claim/data generation, serialization growth, independent
sampler heartbeat/death handling, unrelated-writer pressure, report persistence,
one official disk-stop status and bound amendment approval. No principal or CUDA
job is needed to test those future integration requirements safely.
