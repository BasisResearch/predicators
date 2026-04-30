#!/bin/bash
set -e
git submodule update --init --recursive

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected: building PyBullet from source (workaround for macOS compatibility)..."

    # Initialize the virtual environment first so we can use its Python
    uv venv

    VENV_PYTHON="$(pwd)/.venv/bin/python"
    BULLET_TMP=$(mktemp -d)
    trap 'rm -rf "$BULLET_TMP"' EXIT

    git clone https://github.com/bulletphysics/bullet3 "$BULLET_TMP/bullet3"
    git -C "$BULLET_TMP/bullet3" checkout 3.25

    # Comment out the line that causes build failure on recent macOS
    sed -i '' \
        's|^#define fdopen(fd, mode) NULL|// #define fdopen(fd, mode) NULL|' \
        "$BULLET_TMP/bullet3/examples/ThirdPartyLibs/zlib/zutil.h"

    uv pip install setuptools
    pushd "$BULLET_TMP/bullet3"
    "$VENV_PYTHON" setup.py build
    "$VENV_PYTHON" setup.py install
    popd

    # Install everything else; pybullet 3.2.5 is already installed from source
    # above so pip will skip it
    uv pip install -e .
else
    uv pip install -e .
fi
