# R3 source audit at original baseline

Audit owner: `cd16e2a68fd035f94a8916d41463904f31483d7f`. References below are
one-based lines in the unchanged baseline files. The proposal is a hypothesis,
not evidence. No R1 scheduler, search modification, or R1 tree was copied.

| Question | Verified implementation and consequence |
|---|---|
| Directed semantics | `src/mfaco_train.h:225-232` and `src/mfaco_train.cpp:358-405`: each row is an independent directed list; reciprocal membership is not guaranteed. Sparse state is indexed `u*k+j`. Pheromone route-membership is undirected, but storage remains directed (`:265-299`). |
| Ranking/ties | Coordinates are float32 copied into the solver (`src/mfaco_train.cpp:46-49`). KD points promote those exact values to double (`:367-376`); repeated nearest queries exclude self and delete each selected neighbor (`src/kd_tree.h:259-270`, `:291-339`). No node-index tie-break is specified: strict distance improvement retains traversal order. Delete/undelete changes bucket traversal. Constructed duplicated-coordinate tests demonstrate that different total query counts can reorder later rows' exact ties; native K32 is authoritative. The audit rejects prefix mismatches rather than silently replacing lists. Every pilot/principal row must pass exact K-prefix checks. |
| K32/40/48/64 generation | Same constructor and KD procedure, with `k=min(requested,n-1)` and `bl=min(requested_backup,n-1-k)` (`src/mfaco_train.cpp:25-43`). K is limited to 64 (`src/mfaco_train.h:25`, `src/mfaco_train.cpp:41-43`). Independent native constructions verify all prefixes on actual datasets. K128 in the proposal is unsupported by this baseline. |
| Backup ordering | Backup is the subsequent KD query sequence after the primary list (`src/mfaco_train.cpp:385-402`), not an independent random neighborhood. At n=1000 K32+backup32 covers the first 64 queries; exact equality to another K64 construction is checked, not assumed under ties. |
| Backup use | `src/mfaco_train.cpp:879-906`: first unvisited backup only after all primary candidates are visited; then global nearest unvisited. `build_initial_tour` uses primary then global fallback (`:441-459`). Backup is absent from SRR and the neural edge graph. |
| Pheromone influence | Every primary directed slot evaporates/targets a trail bound using adjacency in the supplied iteration-best ant (`src/mfaco_train.cpp:251-307`). An off-primary tour edge influences total route cost/global incumbent, but has no direct pheromone slot. Smooth bounds are 1 and 1/K (`:493-505`). Source is always replaced by the iteration-best supplied tour; global best changes only for lower cost (`:253-256`, `:302-307`). Source and global best are different observables. |
| Neural alignment/K | `faco.py:126-129` caches N*K tensors. `utils.py:306-365` constructs directed PyG edges in row-major order, with per-row mean distance, pheromone mean/std, relative log pheromone, and source adjacency features. `net.py:35-45`, `:120-134`, `:195-210`: fixed feature widths, variable edge count, one scalar per edge. K32 weights load strictly and finite K40/48/64 forward shapes are tested. Tensor compatibility does not validate learned generalization to another K; K changes normalization, pooling neighborhoods, trail minimum, sampling and SRR together. |
| Both 2-opt orientations | `src/mfaco_train.cpp:959-1044`: successor scan removes `(a,succ(a))` and `(b,succ(b))`; predecessor scan removes `(pred(a),a)` and `(pred(b),b)`. Gain is the sum of removed lengths minus the sum of added lengths. For a canonical tour-edge pair `(a,b),(c,d)`, possible directed representations are `(a,c),(c,a),(b,d),(d,b)`. A union across these representations counts the move once. |
| Pruning | In each orientation the first candidate for which `dist(anchor,adjacent) > dist(anchor,candidate)` is false ends the scan (`src/mfaco_train.cpp:980-1023`). It uses float32 edge distances, strict comparison and actual list order. Candidate membership alone is not evaluation coverage. R3 computes literal reachable prefixes for all nodes; it does not label them actual execution traces. |
| Checklist | Perturbation starts with start node and adds current, chosen and the chosen node's pre-relocation predecessor for edges absent from the original source (`src/mfaco_train.cpp:580-622`, `:684-740`). `contains_edge` ignores its local positions argument and reads source positions (`:1169-1177`). Relocation order matters, including array rotation/order in target-after-node case (`:912-953`). R3 independently reconstructs the array and checks it against native raw output exactly. |
| Revisiting/convergence | `two_opt_nn` increments a monotone `checklist_pos` and appends accepted-move endpoints only when absent anywhere in the vector (`src/mfaco_train.cpp:964-967`, `:1031-1039`). Already processed nodes cannot be re-enqueued in that call. Each distinct node is processed at most once. This property follows from actual control flow; residual improving moves are independently measured rather than presupposed. An all-node frozen-tour scan is not a replacement run of native SRR. |
| Initial/updated optimality | Initialization chooses the best of up to eight greedy NN tours, without a call to 2-opt (`src/mfaco_train.cpp:423-480`). Updated source is an ant refined once on its checklist; neither initialization nor updates guarantee KNN-2-opt optimality. |
| State APIs/ownership | Public C++ arrays/state/RNG object are in `src/mfaco_train.h:225-301`; RNG words themselves are private (`:100-105`). Python exposes source/best tours/costs, candidates, pheromone, heuristic, positions, seed and snapshot load (`src/binding.cpp:903-926`), but not private SRR/evaluation hooks or complete RNG state. Views have no owning native object base (`src/binding.cpp:22-46`); retain solver ownership and immediately copy saved views. R3 tests caught and corrected temporary-owner misuse in its own fixtures; original solver unchanged. |
| Snapshot omissions | Native load copies routes/costs, pheromone, bounds and lists, rebuilds raw heuristic and source positions (`src/mfaco_train.cpp:310-347`). It does not restore RNG words or Python caches/model/prior/schedule/global NumPy RNG. Binding does not fully validate buffer dimensions (`src/binding.cpp:312-334`). R3 uses passive capture and full reruns, not branching from this API. |
| Heuristic cache mismatch | `faco.py:119-129` normalizes heuristic at construction and creates cached tensors. Native `load_snapshot` rebuilds unnormalized `1/d` values; wrapper has no TSP snapshot-load wrapper to renormalize/refresh neighbor and heuristic tensors. The explicit test confirms raw native heuristic plus stale normalized Torch cache after load. `sync_pheromone_to_torch` refreshes pheromone only (`faco.py:237-239`). |
| Defaults/CLI | C++ header/binding: rho=.9, min_new_edges=8, extend_ls=false, smooth=false (`src/mfaco_train.h:254-260`, `src/binding.cpp:134-150`). Python constructor: rho=.1 but other defaults still min_new_edges=8, extend=false, smooth=false, normalized=false (`faco.py:62-81`). CLI inference defaults: min_new_edges=12, normalization/extension/smooth enabled via negated flags (`test.py:110-143`, `utils.py:1392-1406`). Python sample defaults parallel_traced=true (`faco.py:185-190`) whereas native binding default is false (`src/binding.cpp:920-922`). `invtemp` is unused by native binding (`src/binding.cpp:195`). Checkpoint config can override unspecified CLI fields (`test.py:213-239`); all R3 fields are explicit. |
| Seeds/tracing | `utils.py:1434-1436`, `:1482-1489` constructs twice, discards the first explicitly seeded solver and seeds the active solver from the global NumPy stream. Fast and parallel-traced sampling consume per-ant seeds; serial traced uses the shared RNG (`src/mfaco_train.cpp:95-139`, `:159-184`, `:230-245`). R3 proves matched fast/parallel-traced behavior; never assumes serial equivalence. |

## Paper versus executable source

The paper's appendix B.2 (PDF page 11) assumes a 2-opt-optimal initial incumbent
and describes reactivation after an accepted move. Those assumptions do not
describe the actual greedy initialization and once-per-distinct-node checklist
in this commit. The stated certificate therefore cannot automatically be
assigned to baseline output. This is a code/paper mismatch, not a new fix.
Appendix Table A6 is reported paper evidence about uniform K expansion at 5K;
it does not establish selective allocation gains at 1K.

The chosen baseline is explicit Model(anneal), H=10/S=100, no phased Mix
injection. Original per-block annealing is preserved (`utils.py:1515-1598`).
`experiment.py:11-36` and CLI contain different evaluation switches, including
Mix/warmup; those are distinct methods. No numerical paper reproduction is
claimed on the new synthetic datasets or laptop.

## SRR observation boundary

Native `touched_list` is empty for every ant (`src/binding.cpp:233-235`).
Perturbation traces expose decisions and pre-SRR arrays, not the sequence of
evaluated or accepted SRR moves. The initial checklist can be reconstructed and
validated. Initial-checklist potential coverage is reported separately, because
SRR can extend that checklist and change the tour while executing. Actual
dynamic evaluated/accepted booleans remain null. No passive production hook was
needed for the all-node offline graph gate, so no native instrumentation was
introduced. Conclusions must remain within this observation boundary.

## Reuse audit

R1 SHA inspected: `41d9a15b5f311d85118592550c7bdeaf71b3fb9e`.
Read-only snapshots of its README, datasets utility, environment scripts,
environment manifest, baseline validation and reports are under
`artifacts/r3/references/`. Only the existing interpreter/compiler dependency
directories are reused read-only, with an isolated native rebuild. The R3
native binary hash exactly matches the independently recorded R1 rebuild.
Pilot coordinates are read from R1 Git blobs, rehashed against its manifest
and regenerated independently. The dataset utility's PCG64/SeedSequence scheme
is algorithm-independent; R3's generator is separately implemented. R1 runtime
results and strict replay timings are historical context and not new R3 data.
