#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse audited interpreter/compiler dependencies read-only, rebuild R3 native separately.
R1_ENV=/mnt/d/GitRepository/DyNACO/.r1-env
export LD_LIBRARY_PATH="$R1_ENV/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD:$PWD/.r3-build/native:$PWD/r3/native${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
# Exact same-device replay contract; timings are explicitly strict-CUDA timings.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$R1_ENV/venv/bin/python" "$@"
