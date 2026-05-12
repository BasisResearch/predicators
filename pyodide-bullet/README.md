# pyodide-bullet

Cross-build of [PyBullet](https://pybullet.org/) for [Pyodide](https://pyodide.org/),
so that the standard `import pybullet as p` works inside a browser.

This **replaces** the Ammo.js adapter approach for getting predicators envs
to run in a browser. Trade-offs:

| | Ammo.js + adapter | pyodide-bullet (this) |
|---|---|---|
| URDF loading | Have to write a Python URDF parser | Free (PyBullet has it) |
| IK | Have to wire up ikfast/DLS | Free (`p.calculateInverseKinematics`) |
| Existing predicators envs | Need refactor through adapter | Run unchanged |
| `getCameraImage` | Need three.js mirror | Free (TinyRenderer compiled in) |
| Build complexity | Adapter + per-env shim | One curated `setup.py`, one rebuild |
| Determinism vs server | PyBullet vs Ammo — different Bullet builds | Same Bullet source, both sides |

## Smoke test result

```
pybullet build time: May 11 2026 22:52:34
pybullet version: 202010061
connected, client id: 0
loaded plane, body id: 0
loaded cube at z=1.0, body id: 1
after 1s sim, cube z = 0.0250 (should be near 0)
disconnected
```

A cube was loaded from URDF, dropped under gravity, and landed at the
expected height — all inside Pyodide. Real PyBullet, real Bullet
physics, real URDF parsing.

## What's in the wheel

77 MB. Includes:

- Bullet core (LinearMath, Collision, Dynamics, SoftBody).
- PyBullet Python binding (`pybullet.c`).
- SharedMemory subsystem in `DIRECT` mode (no UDP/TCP/Win32/Posix shm).
- URDF / MJCF / Collada importers, OBJ mesh loader.
- TinyRenderer (CPU-only) for `getCameraImage` in `DIRECT` mode.
- BussIK analytical IK.
- BulletInverseDynamics extras.
- pybullet_data assets (plane/cube/duck/sphere URDFs, textures, etc.).

Explicitly excluded — would not work in a browser anyway:

- `p.connect(p.GUI)` (OpenGL window, X11 / Cocoa / Win32).
- ExampleBrowser.
- Gwen GUI library.
- enet, clsocket (network shared-memory).
- EGL plugin.
- VHACD.
- Multi-threading bits (Bullet built with `BT_THREADSAFE=0`).

## Layout

```
pyodide-bullet/
├── recipes/
│   ├── setup_emscripten.py   # Curated setup.py (drops GL/networking)
│   ├── pyproject.toml        # Build deps (numpy)
│   ├── source_patches.py     # Tiny in-place source patches
│   ├── retag.py              # Wheel-tag rewriter (see below)
│   └── build.sh              # End-to-end build script
├── src/
│   ├── pybullet-3.2.7.tar.gz # Upstream sdist (80 MB)
│   └── pybullet-3.2.7/       # Extracted source
├── out/
│   └── pybullet-3.2.7-cp312-cp312-emscripten_3_1_58_wasm32.whl
├── test/
│   ├── package.json
│   └── smoke.mjs             # Node-based Pyodide smoke test
└── README.md (this file)
```

## How to rebuild

```bash
cd pyodide-bullet
source .venv/bin/activate
source "$HOME/.cache/.pyodide-xbuildenv-0.34.3/0.27.7/emsdk/emsdk_env.sh"
recipes/build.sh
python recipes/retag.py out/pybullet-3.2.7-cp312-cp312-pyemscripten_2024_0_wasm32.whl
```

First-time setup (already done):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install pyodide-build[resolve] pip
pyodide xbuildenv install
pyodide xbuildenv install-emscripten
```

## How to smoke-test

```bash
cd test/
node smoke.mjs
```

Requires `npm install` to have run there (already done; pins
`pyodide@0.27.7` which matches our build).

## The retag step

`pyodide-build` 0.34.3 tags wheels as `pyemscripten_2024_0_wasm32`
(newer Pyodide ABI convention). Pyodide 0.27.x runtime expects the older
`emscripten_3_1_58_wasm32` tag. The binary is identical — only the
filename and the `WHEEL` metadata file inside differ. `recipes/retag.py`
rewrites both.

If we eventually upgrade to Pyodide 0.28+, the retag step can be
dropped.

## What was tricky

1. **Source list curation.** pybullet's upstream `setup.py` compiles
   ~250 files including OpenGL/X11/Cocoa/Win32 windows, the
   ExampleBrowser, Gwen GUI, enet networking, VHACD, etc. Most won't
   compile under Emscripten and none are needed for headless physics.
   `setup_emscripten.py` curates a minimal subset.

2. **`SimpleCamera`.** Lives under `examples/OpenGLWindow/` but is pure
   math. Pulled in transitively by `TinyRendererVisualShapeConverter`.
   Had to add it back.

3. **`Wavefront2GLInstanceGraphicsShape`.** Pure data conversion (OBJ
   → mesh struct) but its source file `#include`s
   `SimpleOpenGL3App.h`, which pulls in OpenGL. `source_patches.py`
   drops the unused include and adds `Bullet3Common/b3MinMax.h` to
   replace the implicit `b3Max` it brought in.

4. **Soft body.** I excluded `BulletSoftBody/*` initially, but
   `PhysicsServerCommandProcessor` unconditionally links
   `btSoftBodyRigidBodyCollisionConfiguration`. Had to put soft body
   back in.

5. **numpy headers.** PyBullet's `pybullet.c` includes
   `<numpy/arrayobject.h>`. Solved by declaring `numpy>=1.23` in the
   recipe's `pyproject.toml [build-system].requires` so the
   cross-build env has numpy headers on the include path.

## Known gaps / TODO

- **PyBullet 3.2.7** is the upstream version, but predicators uses
  `pybullet-arm64>=3.2.8`. The arm64 fork has identical sources for
  WASM purposes (the "arm64" name refers to a macOS build target). If
  any predicators code paths require an arm64-only API change, we'd
  need to compare them. For now this build should be a drop-in.
- **No GUI.** `p.connect(p.GUI)` will fail. Use `p.connect(p.DIRECT)`
  and pipe `getCameraImage` output to Three.js / canvas for
  visualization. (Predicators headless tests already use `DIRECT`.)
- **No native threads.** Bullet is built with `BT_THREADSAFE=0`. We
  could turn on Pyodide's pthread support later if needed.
- **Wheel is 77 MB** uncompressed; ~22 MB gzipped. Large but
  acceptable for a one-time download.

## Next step

Wire this wheel into the browser host in `web/` to replace the Ammo.js
path. Concretely:

1. In `web/app.js`, replace `physics.init()` (Ammo) with
   `pyodide.loadPackage("numpy")` + `micropip.install(wheel)`.
2. Drop `web/backend/`, `web/envs/`, `web/physics.js` — no longer
   needed.
3. Replace with code that imports `predicators.envs.pybullet_blocks`
   directly. (Will need to slim predicators' top-level import graph;
   torch and friends won't load in Pyodide.)
