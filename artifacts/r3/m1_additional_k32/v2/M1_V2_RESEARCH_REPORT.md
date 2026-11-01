# M1 v2 technical amendment and controlled restart

**M1 EXPLORATORY SIGNAL**. 32/32 principal cases passed. 8/8 instances contain a robustly improved case; all32 correct.
M1 v1 remains INCONCLUSIVE due to a harness error before any second pass;
it was not a negative mechanics finding. Stage A remains NO-GO, secondary
review INCONCLUSIVE, and M0 VALIDATED. These historical artifacts are unchanged.

## Registration, comparator and control-only preflight

The scientific fields of v1 were copied unchanged into the v2 protocol before
any replay. The technical amendment records strict complete JSON serialization
with nonfinite values rejected, focused comparator tests, and an all32 first-pass
preflight. No fields are dropped, converted to strings, or compared approximately.
Sorting dictionary keys affects only representation; event/route/checklist array
order and float values remain exact. Tests detect route/checklist/gain mutations,
event deletion/order/ranks, unexpected fields and nonfinite values, and cover a
complete representative M0 event record with native tuple-shaped route updates.
See comparator_tests.json and preflight_verdict.json for actual gate results.
The principal stage is entered only after all32 preflight cases pass. Every
attempted preflight has a persisted control object and per-case validation.

Principal execution revalidates pass1 against the whole M0 object, then runs one
unchanged K32 pass2 from its final route/checklist. The unlogged pass2 is timed;
the logged verification starts from the same pass1 state. All32 principal records
are retained, including failure/NOT_RUN statuses if any. Independent validate_trace
checks float32 evaluations, selection order, every accepted move, float64 edge/cost
delta, permutation and checklist evolution. The unchanged logged C++ path checks
inverse positions after accepted moves. Correctness flags in the case records
describe only checks actually reached. No solver source, M0 code or binary changed.

## Paired results (instance means)

| Instance | Robust cases /4 | Absolute gain | Relative gain | Extra evaluations | Extra/baseline ratio | Pass2 accepted | Offline K32 residual |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 0.036431601 | 0.001528727 | 328.75 | 0.460 | 15.75 | 10.25 |
| 1 | 4 | 0.119436266 | 0.005020709 | 240.00 | 0.404 | 13.50 | 5.50 |
| 2 | 4 | 0.053685596 | 0.002266254 | 345.00 | 0.596 | 15.25 | 6.00 |
| 3 | 4 | 0.086881940 | 0.003608079 | 351.50 | 0.541 | 19.50 | 14.25 |
| 4 | 4 | 0.063215334 | 0.002662107 | 386.25 | 0.542 | 16.00 | 10.00 |
| 5 | 3 | 0.029674881 | 0.001275063 | 284.00 | 0.443 | 14.00 | 5.00 |
| 6 | 4 | 0.100426881 | 0.004254308 | 351.75 | 0.522 | 19.50 | 10.50 |
| 7 | 4 | 0.047520021 | 0.002002465 | 321.50 | 0.442 | 20.50 | 7.00 |

paired_metrics.csv retains all32 cases. instance_summary.json averages two phases
within seed, then two seeds within each independent TSP instance, and reports
mean/median/range/sample SD across eight instance means. Moves, seeds and phases
are not independent problem instances. Float64 robust improvement exceeds
max(1e-8,1e-6*baseline length). Native accepted moves use the unchanged positive
float32 threshold and need not exceed that robust threshold. Offline final-tour
K32 residuals use robust canonical float64 2-opt moves and are not accepted events.

## Resource audit and interpretation

Elapsed experiment wall time including comparator tests, preflight, principal
calls, logging verification, checks and trace serialization: 13.167146s.
Peak process RSS: 151834624 bytes (Linux ru_maxrss *1024, process-lifetime peak).
One CPU worker and a conservative1GB virtual-memory ceiling were configured.
The audit accounts for all physical calls: 32 control-only preflight calls plus
three principal invocations per completed case (two sequential passes and a
pass2 logging duplicate from identical pass1 state). No third sequential pass
is allowed. Path limits remain2N anchors and4NK evaluations; all work is charged
against the300s wall budget and1GB RSS cap. Unlogged pass2 timing excludes logging
and serialization but includes pybind copies/conversion. It is not a speed benchmark.
See test_resource_audit.json for measured counts and failures. Final report and
delivery hashing are recorded after the experiment timer, outside search timing.

Cursor reset revisits all final-checklist nodes on an evolved tour. Any measured
gain cannot be attributed specifically to duplicate suppression, pruning,
candidate limitations, neural guidance or original-paper claims. These selected
ant0 records are not best-of-100 source trajectories and establish no end-to-end
DyNACO effect. No Stage B, solver-level experiment, expansion, M2 or new research
branch was implemented or run.

**Recommendation: Conditionally design a new independently registered equal-compute controlled study on new independent data.** If a new study is designed, register
new independent instances and paired solver seeds, original K32 control and an
equal-compute revisit arm with common evaluation/time budgets. Freeze final and
anytime best-of-100 quality metrics, resource accounting, instance-level uncertainty
and scientific GO/NO-GO criteria before data. Keep the reserved confirmation set sealed.
