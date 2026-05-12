# pybullet recipe for Pyodide

A [Pyodide package recipe](https://pyodide.org/en/0.27.7/development/new-packages.html)
that cross-builds `pybullet` for WebAssembly so it can be loaded inside a
Pyodide runtime with `pyodide.loadPackage(...)` or `micropip.install(...)`.

This is the **canonical** way to (re)build the wheel. The older scripts
under `../recipes/` are kept as legacy reference; everything they do is
encoded here in declarative form.

## Layout

```
packages/pybullet/
├── meta.yaml                                       Recipe spec.
├── patches/
│   └── 0001-wavefront-drop-opengl-include.patch    Drops one GL header.
└── extras/
    ├── CMakeLists.txt                              Curated source list + flags.
    └── pyproject.toml                              scikit-build-core backend.
```

`meta.yaml` references the patch via `source/patches` and overlays the
curated `CMakeLists.txt` / `pyproject.toml` via `source/extras`, so the
upstream PyPI tarball is consumed unmodified except for those edits and
the data-trim run in `build/script`.

The build chain is **scikit-build-core → CMake → emcmake → Emscripten**:
`pyproject.toml` declares scikit-build-core as the PEP 517 backend;
scikit-build-core invokes CMake; Pyodide's xbuildenv supplies the
Emscripten toolchain file via `CMAKE_TOOLCHAIN_FILE`; `find_package(Python3)`
inside the CMakeLists resolves to the cross-Python and cross-numpy that
Pyodide installs as host requirements.

## Building the wheel

The recipe expects to be built with `pyodide build-recipes`. From a
clone of Pyodide (or via the out-of-tree build flow):

```bash
# One-time toolchain setup if you don't already have it.
python -m venv .venv && source .venv/bin/activate
pip install "pyodide-build>=0.27,<0.28"
pyodide xbuildenv install 0.27.7
pyodide xbuildenv install-emscripten

# Build the recipe. Point --recipe-dir at this directory's parent.
pyodide build-recipes pybullet \
    --recipe-dir /path/to/predicators/pyodide-bullet/packages \
    --install-dir ./dist
```

The wheel will be written under `./dist/` with the canonical filename
`pybullet-3.2.7-cp312-cp312-emscripten_3_1_58_wasm32.whl` (filename
will reflect whatever Pyodide / Emscripten versions are pinned by the
build environment you installed above).

## Releasing as a standalone WASM package

The wheel produced above is the artifact you'd publish. Three viable
distribution channels:

1. **CDN / static host.** Upload the wheel to S3, jsDelivr, or any
   static origin. Consumers point `micropip.install(<url>)` at it.
2. **GitHub Release.** Attach the wheel to a tagged release of this
   repo; same `micropip.install` flow.
3. **Pyodide distribution.** If the upstream Pyodide project adopts the
   recipe, the wheel ships inside the standard Pyodide bundle and can
   be loaded by name with `pyodide.loadPackage("pybullet")`.

For (1) and (2), the consumer side looks like:

```js
const pyodide = await loadPyodide();
await pyodide.loadPackage("numpy");          // host runtime dep
await pyodide.loadPackage("micropip");
await pyodide.runPythonAsync(`
  import micropip
  await micropip.install("https://example.com/pybullet-...-wasm32.whl")
  import pybullet as p
  c = p.connect(p.DIRECT)
`);
```

## Tag mismatch caveat

Older releases of `pyodide-build` (≤0.34) emit the legacy
`pyemscripten_2024_0_wasm32` platform tag while the 0.27.x runtime
expects `emscripten_3_1_58_wasm32`. When `pyodide build-recipes` is run
from inside a Pyodide checkout pinned to the matching version, the tags
agree and no fixup is needed. For out-of-tree builds with mismatched
toolchains, see `../recipes/retag.py` for a wheel-tag rewriter.

## What this recipe excludes

The curated `CMakeLists.txt` compiles only the headless physics
surface area:

- Bullet core (LinearMath, Collision, Dynamics, SoftBody).
- `pybullet.c` Python binding.
- SharedMemory in DIRECT mode (no UDP/TCP/Win32/Posix shm).
- URDF / MJCF / Collada / OBJ importers.
- TinyRenderer (CPU rasterizer for `getCameraImage`).
- BussIK analytical IK.
- BulletInverseDynamics extras.

Deliberately dropped — none of these would work in a browser anyway:

- `p.connect(p.GUI)` (OpenGL/X11/Cocoa/Win32 windows).
- ExampleBrowser, Gwen GUI library.
- enet, clsocket (network shared memory).
- EGL plugin, VHACD.

`build/script` further strips `pybullet_data` of RL policy checkpoints,
mocap reference motions, and robot meshes that the embedded examples
ship but most consumers never load. This drops the wheel from ~80 MB to
~5 MB compressed.
