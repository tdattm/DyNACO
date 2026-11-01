# M2-S1 storage audit

Prior outcome stays **M2 RESOURCE-INCONCLUSIVE**, 0/192 valid cases, no principal
dataset/authorization/launch claim. The failure was an admission rejection,
not an observed disk-full error. The former 166.46 GB threshold was introduced
by `m2_principal/preflight.py`, outside the registered scientific protocol.

## Exact reconstruction

`JOURNAL_DTYPE.itemsize = 4024`: float64 elapsed time (8), two int64 cumulative
counters (16), int32 tour of1000 nodes (4000). No dtype padding is present.
The old formula adds4000 for tours,400 for100 float32 ant costs,160 for metadata:
`per_iteration=8584`; `96*(1000+100000)*8584=83,230,464,000`;
`required=2*estimate=166,460,928,000` bytes (166.46 decimal GB,155.03 GiB).

The 100000 value is the unchanged P2 wall-loop safety ceiling, not the number
of iterations promised or required during a4s run. P1 observed1401/1218 attempted
iterations in its two wall runs. Multiplying every wall case by100000 produces
the dominant discrepancy. The factor2 was generic headroom, not a measured
compression factor or a registered scientific rule.

## Persisted representation versus RAM

Each completed iteration appends one uncompressed4024-byte journal record.
`raw.npz` later persists float64 times (8/iteration), int32 global-best tours
(4000/iteration), float32 ant costs (400/iteration) using `savez_compressed`.
`raw.json` persists one64-character guidance hash per iteration (roughly72
bytes including indentation/separators); remaining scalar fields are fixed.
The journal and NPZ intentionally persist the same best-tour state for two
purposes. This is real duplication, not a term that can be removed from the
storage model. JSON hashes are persisted too. The old160-byte allowance
approximately covers those hashes and timestamps; it is not an exact byte count.

NPZ fixed arrays are initial/source/best int32 tours (3*4000) and float32
1000x32 pheromone (128000):140000 bytes before compression. Manifest and
validated JSON, ZIP/NPY headers, directories and filesystem allocation overhead
are also fixed per case. Dataset NPY (only after future approval), approval,
warmup/startup/analysis and supervisor records are top-level overhead.
No principal dataset is present in this recovery.

The in-memory lists `times/tours/costs/priors` and their final NumPy arrays are
transient RAM representations; list/object overhead and temporary serialization
copies do not themselves become files. Graphs, candidate lists, per-ant native
routes, neural tensors and route-position arrays are RAM/GPU objects, not
additional per-iteration persisted files. Their resource limits remain separate.

Thus the old formula approximates uncompressed linear payload, intentionally
includes journal/NPZ duplication, ignores compression savings and omits explicit
fixed overhead. It is a conservative estimate, not a proven strict filesystem
maximum. The frozen iteration ceiling also does not bound unrelated writers,
logs, filesystem metadata or Git working space.

## Independently recomputed observed evidence

P2 control1000-iteration files: JSON72652, NPZ340560, journal4024000 bytes;
total4437212. Treatment: JSON72653, NPZ304539, journal4024000; total4401192.
The raw NPZ array payload is4548000 bytes before compression in each case;
different compression sizes are observed, not guaranteed for future tours.
The earlier crude192-times-largest-case figure is851944704 bytes.
Accounting for the observed1401 wall attempts for all96 wall cases, plus
96 fixed1000-iteration cases, scales to about1022759618 bytes.
These projections omit uncertainty and do not establish a hard bound.
Exact sizes, shapes and dtypes are in storage_model.json; no tour-quality
selection, coordinate generation or principal solver call was performed.

Current and historical free-space observations are separate in
resource_review.json. Only a fresh measurement on the actual future output
filesystem can admit a new launch.
