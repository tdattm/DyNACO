#!/usr/bin/env bash
# Project-local only: no sudo, shell-profile edits, drivers or system packages.
set -euo pipefail
cd /mnt/d/GitRepository/DyNACO
export UV_CACHE_DIR="$PWD/.r1-env/cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.r1-env/python"
for pkg in .r1-env/downloads/*.deb; do
    dpkg-deb -x "$pkg" .r1-env/sysroot
done
.r1-env/bin/uv venv --python 3.13 .r1-env/venv
.r1-env/bin/uv pip install --python .r1-env/venv/bin/python \
    torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.r1-env/bin/uv pip install --python .r1-env/venv/bin/python \
    torch-geometric==2.7.0 numpy scipy pybind11 setuptools psutil matplotlib tqdm pytest
