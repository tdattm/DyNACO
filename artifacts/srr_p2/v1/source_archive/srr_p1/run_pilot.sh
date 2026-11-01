#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:/mnt/d/GitRepository/DyNACO-r3/.r3-build/native"
export LD_LIBRARY_PATH="/mnt/d/GitRepository/DyNACO/.r1-env/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /mnt/d/GitRepository/DyNACO/.r1-env/venv/bin/python -m srr_p1.pilot
