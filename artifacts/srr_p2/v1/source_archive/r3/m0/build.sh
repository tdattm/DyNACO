#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
R1_ENV=/mnt/d/GitRepository/DyNACO/.r1-env
export LD_LIBRARY_PATH="$R1_ENV/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$R1_ENV/sysroot/usr/bin/x86_64-linux-gnu-gcc-13"
export CXX="$R1_ENV/sysroot/usr/bin/x86_64-linux-gnu-g++-13"
export CFLAGS="--sysroot=$R1_ENV/sysroot"
export CXXFLAGS="$CFLAGS"
export LDFLAGS="--sysroot=$R1_ENV/sysroot -L$R1_ENV/sysroot/usr/lib/x86_64-linux-gnu"
export PYTHONDONTWRITEBYTECODE=1
"$R1_ENV/venv/bin/python" r3/m0/build.py
