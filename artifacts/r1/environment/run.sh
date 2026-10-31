#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/GitRepository/DyNACO
export LD_LIBRARY_PATH="$PWD/.r1-env/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD:$PWD/.r1-env/native:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MPLBACKEND=Agg
exec .r1-env/venv/bin/python "$@"
