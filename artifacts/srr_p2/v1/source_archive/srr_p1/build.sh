#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=/mnt/d/GitRepository/DyNACO/.r1-env
MODE=${1:-native}
export LD_LIBRARY_PATH="$ENV/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$ENV/sysroot/usr/bin/x86_64-linux-gnu-gcc-13"
export CXX="$ENV/sysroot/usr/bin/x86_64-linux-gnu-g++-13"
export CFLAGS="--sysroot=$ENV/sysroot"
if [[ "$MODE" == test ]]; then export CFLAGS="$CFLAGS -DSRR_TEST_API"; fi
export CXXFLAGS="$CFLAGS"
export LDFLAGS="--sysroot=$ENV/sysroot -L$ENV/sysroot/usr/lib/x86_64-linux-gnu"
cd src
"$ENV/venv/bin/python" setup.py build_ext --force --build-lib ../.srr-build/$MODE --build-temp ../.srr-build/objects-$MODE
