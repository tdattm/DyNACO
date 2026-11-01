# M1.5 saved-event audit

{
  "totals": {
    "accepted": 536,
    "empty": 396,
    "mutations": 140,
    "robust": 140,
    "noop_extensions": 73
  },
  "expected_counts_match": true,
  "noop_degenerate": 396,
  "noop_gain_range": [
    1.1641532182693481e-10,
    7.450580596923828e-09
  ],
  "noop_nonzero_float64_changes": 0
}

All32 second-pass traces were applied as exact indexed route patches from saved
pass1 routes, with independent float64 lengths on source float32 coordinates.
Native positive-float32 accepted events, nonempty mutations and robust float64
reductions are distinct. Degeneracy/shared endpoint status is recorded per event,
not assumed as a universal cause. Tiny native positives can reflect cancellation
on overlapping edges. Empty patches have zero actual tour-length change.
Insertion attempts and observed checklist extensions are recorded exactly.
An extension changes the observed later node schedule; its causal effect on
quality cannot be inferred without an intervention. Historical M1 conclusions
remain unchanged; counts of accepted events must not be described as counts of
actual tour improvements. No native 2-opt correction or replay was performed.
