#!/usr/bin/env bash
# Rebuild to a separate directory; retain every upstream binary/object unchanged.
set -euo pipefail
cd /mnt/d/GitRepository/DyNACO
for pkg in .r1-env/downloads/*.deb; do dpkg-deb -x "$pkg" .r1-env/sysroot; done
# Link the host's matching libc runtime into the local compiler sysroot.
python3 - <<'PY'
from pathlib import Path
root = Path('.r1-env/sysroot')
for folder in ('lib/x86_64-linux-gnu', 'lib64'):
    target = root / folder
    target.mkdir(parents=True, exist_ok=True)
    for source in Path('/' + folder).glob('*.so*'):
        dest = target / source.name
        if source.is_file() and not dest.exists() and not dest.is_symlink():
            dest.symlink_to(source)
PY
export LD_LIBRARY_PATH="$PWD/.r1-env/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$PWD/.r1-env/sysroot/usr/bin/x86_64-linux-gnu-gcc-13"
export CXX="$PWD/.r1-env/sysroot/usr/bin/x86_64-linux-gnu-g++-13"
export CFLAGS="--sysroot=$PWD/.r1-env/sysroot"
export CXXFLAGS="$CFLAGS"
export LDFLAGS="--sysroot=$PWD/.r1-env/sysroot -L$PWD/.r1-env/sysroot/usr/lib/x86_64-linux-gnu"
"$CXX" --version
"$CXX" --sysroot="$PWD/.r1-env/sysroot" -std=c++17 -fopenmp \
    artifacts/r1/environment/openmp_probe.cpp -o .r1-env/openmp_probe
.r1-env/openmp_probe
cd src
../.r1-env/venv/bin/python setup.py build_ext --force \
    --build-lib ../.r1-env/native --build-temp ../.r1-env/build
