# Telemetry compatibility and scope

Python `MFACO_TSP(..., lightweight_srr=True)` enables counters without detailed
mutation accounting. Default false. Existing `measure_srr=True` and native
`configure_srr(extra, measure)` retain detailed behavior. The optional third
native argument is `lightweight=False`; both measurement modes together are
rejected. Scheduling flag and K32/NLS restrictions are unchanged from P0.

`srr_metrics[0]` counts actual original `two_opt_nn` calls. `[1]` counts gain
formula evaluations reached after native distance pruning in both orientations,
including degenerate or nonpositive results. These are not unique moves or
offline potential counts. Both counters are relaxed atomics, reset once at the
start of a measured sample, aggregated across ants and read after workers join.
Test-only direct scheduling does not reset counters between calls.

Detailed `[2]` accepted-positive-float32 and `[3]` actual mutation counters are
preserved. In lightweight mode these fields are not measured and remain zero;
they must never be interpreted as zero accepted/mutating events. Counters in an
unmeasured sample are unspecified/stale and must not be reported as observations.
Lightweight mode never takes the detailed `before=route` or route-comparison
branches. It adds no random draws, move logs or float arithmetic changes.

The ordinary and traced scheduling paths share the same unchanged calls. Their
different original RNG initialization contracts remain; parallel-traced and
normal are compared directly, serial traced against its own frozen baseline.
The final protocol must pin the production binary and launch environment.
