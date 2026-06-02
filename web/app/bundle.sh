#!/usr/bin/env bash
# Build everything the browser needs into web/wheels/:
#   - predicators-0.1.0-py3-none-any.whl       (pure-Python wheel)
#   - gym-0.26.2-py3-none-any.whl              (tiny gym shim)
#   - assets.tar.gz                            (envs/assets/ for FS unpack)
# Requires:
#   - a Python venv (uses ../.venv if it exists, else "python")
#   - pybullet wasm wheel already in web/wheels/ (committed-out;
#     fetch from github.com/BasisResearch/pybullet-pyodide if missing)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WHEELS_DIR="$REPO_ROOT/web/wheels"
mkdir -p "$WHEELS_DIR"

PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then PY="python"; fi
echo "Using Python: $PY"

echo "[1/3] Building predicators wheel…"
cd "$REPO_ROOT"
"$PY" -m build --wheel --no-isolation --outdir /tmp/_predicators_wheels . >/dev/null
cp /tmp/_predicators_wheels/predicators-0.1.0-py3-none-any.whl "$WHEELS_DIR/"

echo "[2/3] Building gym shim…"
"$PY" "$HERE/gym_shim_setup.py"

echo "[3/3] Packing envs/assets/ tarball (~30 MB)…"
cd "$REPO_ROOT/predicators/envs"
tar czf "$WHEELS_DIR/assets.tar.gz" assets

echo
echo "Done. web/wheels/ now contains:"
ls -lh "$WHEELS_DIR"
echo
echo "If the pybullet wasm wheel is missing, fetch it from:"
echo "  https://github.com/BasisResearch/pybullet-pyodide/releases"
echo "and drop it into web/wheels/."
