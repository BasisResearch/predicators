#!/usr/bin/env bash
# Cross-build pybullet for Pyodide using a curated headless setup.py.
#
# Usage:
#   cd pyodide-bullet
#   source .venv/bin/activate
#   source "$HOME/.cache/.pyodide-xbuildenv-0.34.3/0.27.7/emsdk/emsdk_env.sh"
#   ./recipes/build.sh
#
# Output wheel ends up in pyodide-bullet/out/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/src/pybullet-3.2.7"
OUT_DIR="$ROOT/out"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source tree missing at $SRC_DIR. Run: tar -C $ROOT/src -xzf $ROOT/src/pybullet-3.2.7.tar.gz" >&2
  exit 1
fi

# Snapshot the original setup.py once; replace with the emscripten one.
if [[ ! -f "$SRC_DIR/setup.py.orig" ]]; then
  cp "$SRC_DIR/setup.py" "$SRC_DIR/setup.py.orig"
fi
cp "$ROOT/recipes/setup_emscripten.py" "$SRC_DIR/setup.py"
cp "$ROOT/recipes/pyproject.toml" "$SRC_DIR/pyproject.toml"
python "$ROOT/recipes/source_patches.py" "$SRC_DIR"

# Strip pybullet_data bulk that MANIFEST.in would otherwise sweep in.
# MANIFEST.in does `recursive-include examples/pybullet/gym *.*`, which
# overrides our setup.py's package_data filter. Physically deleting the
# unwanted paths is the most reliable way to enforce the trim.
DATA_DIR="$SRC_DIR/examples/pybullet/gym/pybullet_data"
if [[ -d "$DATA_DIR" ]]; then
  # RL policies / mocap / large samplers — never used at runtime by us.
  rm -rf \
    "$DATA_DIR/data/policies" \
    "$DATA_DIR/data/motions" \
    "$DATA_DIR/policies" \
    "$DATA_DIR/MPL" \
    "$DATA_DIR/samurai_monastry.obj" \
    "$DATA_DIR/random_urdfs" \
    "$DATA_DIR/random_urdfs.zip" \
    "$DATA_DIR/toys" \
    "$DATA_DIR/heightmaps" \
    "$DATA_DIR/testdata"
  # Robot models — predicators ships its own under
  # predicators/envs/assets/urdf/, so the pybullet_data copies are dead weight.
  rm -rf \
    "$DATA_DIR/laikago" \
    "$DATA_DIR/atlas" \
    "$DATA_DIR/quadruped" \
    "$DATA_DIR/humanoid" \
    "$DATA_DIR/biped" \
    "$DATA_DIR/franka_panda" \
    "$DATA_DIR/xarm" \
    "$DATA_DIR/a1" \
    "$DATA_DIR/mini_cheetah" \
    "$DATA_DIR/aliengo" \
    "$DATA_DIR/roboschool" \
    "$DATA_DIR/racecar" \
    "$DATA_DIR/kuka_iiwa" \
    "$DATA_DIR/bicycle" \
    "$DATA_DIR/husky" \
    "$DATA_DIR/differential" \
    "$DATA_DIR/dinnerware" \
    "$DATA_DIR/duck.obj" \
    "$DATA_DIR/duck.dae" \
    "$DATA_DIR/duck_vhacd.obj" \
    "$DATA_DIR/duck_vhacd.urdf" \
    "$DATA_DIR/r2d2_concave.urdf" \
    "$DATA_DIR/r2d2.urdf" \
    "$DATA_DIR/kiva_shelf" \
    "$DATA_DIR/jenga" \
    "$DATA_DIR/lego" \
    "$DATA_DIR/torus.vtk" \
    "$DATA_DIR/soccerball.obj" \
    "$DATA_DIR/pickup2.zip" \
    "$DATA_DIR/table" \
    "$DATA_DIR/tray"
fi
# Stop pip from picking up the upstream README.md as long_description; it's huge.
[[ -f "$SRC_DIR/README.md" ]] && [[ ! -f "$SRC_DIR/README.md.orig" ]] && \
  cp "$SRC_DIR/README.md" "$SRC_DIR/README.md.orig" && \
  printf "pybullet for Pyodide (headless)\n" > "$SRC_DIR/README.md"

cd "$SRC_DIR"
# Clean any stale artifacts so source-list changes take effect.
rm -rf build
LOG="$LOG_DIR/build-$(date +%Y%m%d-%H%M%S).log"
echo "Building. Log: $LOG"
# pyodide build runs pypa/build inside the cross-compile environment.
pyodide build "$SRC_DIR" --outdir "$OUT_DIR" 2>&1 | tee "$LOG"
echo "Built artifacts:"
ls -la "$OUT_DIR"
