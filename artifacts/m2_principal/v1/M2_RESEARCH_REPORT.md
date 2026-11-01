# Registered M2 study: pre-launch resource stop

**Official decision: M2 RESOURCE-INCONCLUSIVE.** Completed principal runs:0/192;
valid principal runs:0/192. No dataset was generated or sealed, no principal
launch claim was created, and no registered execution authorization was issued.
The actual user instruction authorizing the workflow is archived verbatim with
its session record and SHA256; no human identity/signature was invented.

## Research question and planned method

Does one additional unchanged K32 SRR pass improve end-to-end DyNACO under matched
computational budgets? The frozen design specified16 uniform Euclidean TSP1K
instances, PCG64 seed202612070201, solver seeds31415/27182/16180,100 ants,
one frozen checkpoint, original seed/guidance/candidate/pheromone semantics and
two arms. Planned modes were1000 iterations and4.0s instrumented search,
192 runs total, two CPU threads and sequential RTX3050 execution.
Registered limits were1GiB allocated CUDA,3GiB RSS,60s search,120s setup,
2700s aggregate. Co-primary screening required both modes to satisfy median
instance gain>=0.1%, positive mean-bootstrap lower bound and>=12/16 positive
instances;10000 paired instance resamples use PCG64 seed202612070202.

Registration SHA256: `5d2d8942f900d4d1a4ef89e069668c3d0290d17c628d058150ffe6998d5bca45`.
Protocol SHA256: `574cd6781e7a026951521e12bfdef28f5790752289c029f21b8db0f4e5e5dedd`.
Dataset SHA256: **NOT CREATED**. Authoritative protocol/source/binary/checkpoint
hashes passed. Frozen Python3.13.18, NumPy2.5.4, Torch2.7.1+cu126,
CUDA12.6, GCC13.3.0, exact RTX3050 identity and two CPU threads passed.
The GPU compute-process query reported no conflicting compute workload.

## Mandatory gate failure

The preflight stopped on conservative disk headroom. Available bytes:
22120095744; admission maximum-output estimate:83230464000;
required twofold headroom:166460928000. The bound allocates
96 iteration-mode cases at1000 iterations plus96 wall cases at the frozen
100000-iteration loop ceiling, with8584 bytes per iteration for journals,
tour/cost states and metadata. Actual disk exhaustion did not occur.

The new admission check was deliberately conservative and was not itself a
frozen scientific threshold. For transparency, actual P2 development artifacts
have at most4437212 bytes per1000-iteration case;
scaling that observed footprint to192 cases gives
851944704 bytes before headroom and variable
wall-run iteration counts. Thus the blocked ceiling estimate must not be
described as expected consumption. No threshold was relaxed after the failed
mandatory gate, no alternative storage/dataset was selected, and no retry ran.

Preflight elapsed:14.473s; peak process RSS:
568557568 bytes. CUDA free memory:3463865959 bytes of
4294443008. Available system RAM was
7247237120 bytes. These are admission observations,
not principal study resource use. Per-run search, setup, warmup, allocated/
reserved CUDA peaks and principal evaluation counters are NOT RUN/NOT MEASURED.
Checks after the failed disk gate, including the fresh watchdog smoke test,
were not executed. Prior P2 tests passed, but were not substituted for new tests.

## Results and statistical interpretation

Equal iterations: NOT RUN; no quality, runtime or evaluation comparison.
Equal wall time: NOT RUN; no eligible principal checkpoints or outcomes.
No official mean, median, uncertainty, anytime curve or scientific GO/NO-GO is
computed from this incomplete grid. A technical/resource failure is not zero
improvement and is not a scientific NO-GO. The existing M1v2 exploratory signal
is neither confirmed nor contradicted by this admission failure.

## Limitations and recommendation

The planned scientific inference would cover only TSP1K,16 independent instances,
three seeds, one checkpoint and the RTX3050 environment. Instrumented equal-time
results would include telemetry overhead (P1 approximately19.9% control and
32.2% treatment), without subtraction. The prospective co-primary screening
rule is not formally calibrated. No causal isolation of native duplicate
suppression or evidence for adaptive candidate expansion is provided.

Recommendation: prepare a separately reviewed future launch with sufficient
audited storage capacity, or prospectively justify a practical output admission
bound using observed iteration rates. Do not silently rerun this stopped study
or alter the frozen scientific budgets. No follow-up is executed here.

Historical decisions remain Stage A NO-GO, secondary INCONCLUSIVE, M0 VALIDATED,
M1v1 technical INCONCLUSIVE, M1v2 EXPLORATORY SIGNAL, M1.5 536/396/140,
P0/P1/P2 readiness. Frozen source and historical evidence remain unchanged.
