#!/bin/bash
set -e
git submodule update --init --recursive

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected: building PyBullet from source (workaround for macOS compatibility)..."

    BULLET_TMP=$(mktemp -d)
    trap 'rm -rf "$BULLET_TMP"' EXIT

    git clone https://github.com/bulletphysics/bullet3 "$BULLET_TMP/bullet3"

    # Comment out the line that causes build failure on recent macOS
    sed -i '' \
        's|^#define fdopen(fd, mode) NULL|// #define fdopen(fd, mode) NULL|' \
        "$BULLET_TMP/bullet3/examples/ThirdPartyLibs/zlib/zutil.h"

    pip install setuptools
    pushd "$BULLET_TMP/bullet3"
    python setup.py build
    python setup.py install
    popd

    # Install everything else; pybullet 3.2.7 is already installed from source
    # above so pip will skip it
    pip install -e .
else
    pip install -e .
fi
