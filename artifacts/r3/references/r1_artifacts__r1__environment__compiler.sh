#!/usr/bin/env bash
# Extract distro compiler packages locally; never install system packages.
set -euo pipefail
cd /mnt/d/GitRepository/DyNACO
cd .r1-env/downloads
apt-get download gcc-13-x86-64-linux-gnu g++-13-x86-64-linux-gnu cpp-13-x86-64-linux-gnu \
    libgcc-13-dev libstdc++-13-dev libc6-dev linux-libc-dev=6.8.0-31.31 libmpc3 libisl23
cd ../..
for pkg in .r1-env/downloads/*.deb; do dpkg-deb -x "$pkg" .r1-env/sysroot; done
export LD_LIBRARY_PATH="$PWD/.r1-env/sysroot/usr/lib/x86_64-linux-gnu"
.r1-env/sysroot/usr/bin/x86_64-linux-gnu-g++-13 --version
