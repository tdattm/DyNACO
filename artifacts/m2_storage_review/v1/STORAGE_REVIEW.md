# Independently recomputed storage

Journal layout: little-endian float64 timestamp8 + two int64 counters16 +
1000 int32 nodes4000 =4024 bytes. No compression/padding in the journal.
Old model:4024+4000+400+160=8584 bytes/iteration;
96*(1000+100000)*8584=83230464000; doubled=166460928000.

The100000 wall-loop ceiling is still unchanged and is not the4s completion
count. Using it for every wall case explains the old large estimate. It was
never a proven total-filesystem bound; fixed metadata and competing writes
are separate. Intentional journal plus NPZ tour duplication is persisted twice.
Transient Python object/list/graph/native/GPU memory is not extra disk output.

Independently observed P2 control files:72652-byte JSON,340560-byte compressed
NPZ,4024000-byte journal. Treatment:72653,304539,4024000. Each raw NPZ array
payload is4548000 bytes before compression:4408 bytes/iteration plus140000
fixed bytes. JSON prior hashes grow linearly, while fixed scalar metadata,
initial/source/best/pheromone and ZIP headers add fixed overhead. Compression
depends on tour/cost states; these tiny observations are not universal ratios.

P1 observed1401 and1218 attempted wall iterations. The proposal uses350.25/s,
multiplies by4 and budgets4s plus one boundary iteration:5605. It adds10KiB per
iteration,1MiB per case and64MiB top-level. Exact payload:
96*(1000+5605)*10240+192*1048576+67108864=6761414656 bytes.
Admission=2*6761414656+4294967296=17817796608 bytes.
The unchanged100000 ceiling with the new provisions would instead be
99555475456 bytes before delivery reserve. Expected observed-size scaling
is approximately1022759618 bytes, much smaller than either provisioning bound.

The arithmetic is sound. The engineering safety inference remains conditional:
fourfold rate is not a tested ceiling across16 new instances and three seeds;
1MiB/64MiB fixed allowances do not bound arbitrary logs or packaging; factor2
is not measured peak Git temporary usage; and a4GiB floor cannot protect
against unbounded concurrent writes. No deterministic hard bound follows.

The runtime rule adds max(1025048576, two seconds of recent output growth or
filesystem free-space loss) to the4GiB floor. It can react to serialization and
external writers only while active and sampling successfully. Recursive stat
scans are synchronous;0.25s is a sleep interval, not a guaranteed maximum
detection latency. Mount/device identity is recorded but not checked against
the registered output destination at an enforced entry point. Logical file
sizes omit block allocation; actual free bytes must be the safety authority.

S1 margin:21037568000-17817796608=3219771392 bytes (about3.22 GB or3.00 GiB).
Original failed-preflight free bytes would leave4302299136 under the proposal,
but that historical counterfactual neither completes the experiment nor proves
runtime safety. Current available space must be measured at each future launch.
No adjustment to either threshold is made by this review.
