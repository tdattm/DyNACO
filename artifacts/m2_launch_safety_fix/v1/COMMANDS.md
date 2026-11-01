# Executed commands and evidence

All commands ran from `/mnt/d/GitRepository/DyNACO-m2-safety-s2f` under WSL Ubuntu.
Python executable: `/mnt/d/GitRepository/DyNACO/.r1-env/venv/bin/python`.
Base environment: `PYTHONPATH=$PWD`, `OPENBLAS_NUM_THREADS=1`; native checks
also set `LD_LIBRARY_PATH=/mnt/d/GitRepository/DyNACO/.r1-env/sysroot/usr/lib/x86_64-linux-gnu`.

Initial environment probes: `id`, `uname -a`, `cat /proc/self/cgroup`,
`findmnt -t cgroup2`, `unshare --user --map-root-user --pid --fork --mount-proc
sh -c 'printf namespace-ok; cat /proc/self/status'`, `unshare --help`.
They established unprivileged namespace support; no cgroup was created/modified.

`python -m m2_safety.tests` ran as the dummy harness in attempts1/2/3/4/5/6.
Logs: safety_test_attempt1.log, safety_test_attempt2.log,
safety_test_attempt3.log, safety_test_final.log, safety_test_validation5.log and safety_test_validation6.log.
Early readiness harness failures and passing versions remain preserved. Final
run uses dummy_run_v6; every spec, bootstrap/controller log, identity, journal
prefix, disk heartbeat, init/final status and per-case snapshot remains there.
summary_attempt3/4/5 retain earlier passing summaries without overwriting them.
Full command arrays for child creation are in m2_safety/tests.py and fixture specs.

`python artifacts/m2_launch_safety_fix/v1/regression.py`:
source guard stripping/AST comparison and existing saved-state prefix of
srr_p1/gates.py, excluding its old output writes and sampling worker portion.
Output: regression.log, SOURCE_INVARIANCE.json, SAVED_STATE_REGRESSION.json.

With `PYTHONPATH=$PWD:/mnt/d/GitRepository/DyNACO-srr-p1/.srr-build/test:$PWD/artifacts/srr_p2/v1/native`:
`python -m pytest -q tests/test_r3_gate.py tests/test_r3_two_opt.py`
→12 PASS, existing_pytest.log.

With that same PYTHONPATH, `OMP_NUM_THREADS=2`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`:
`timeout 120 unshare --user --map-root-user --pid --fork --mount-proc
--kill-child=KILL python -m pytest -q tests/test_r3_regression.py tests/test_r3_state_capture.py`
→5 PASS in77.43s, one existing PyG deprecation warning;
namespace_development_regression.log. Inputs are the existing deterministic
small TSP100 development tests, not principal data/identities.

`python artifacts/m2_launch_safety_fix/v1/bootstrap_pin_smoke.py`
→PINNED_BOOTSTRAP_TESTS.json, bootstrap_pin_smoke.log and
pinned_bootstrap_checks/ (normal, early-controller kill, controller kill).

`python artifacts/m2_launch_safety_fix/v1/prepare_registration.py`
packages a candidate only, verifies its hashes and scans for live test workers.
It does not call a principal entry point or create approval/authorization.

Exact inner and host PIDs, PPIDs, PGIDs, SIDs, start ticks, namespace IDs and
exit statuses are in ORPHAN_PROCESS_TESTS.json/per-case records. Signals target
owned pidfds; no destructive disk-filling command or unrelated deletion ran.
