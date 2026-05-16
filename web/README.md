# predicators in the browser (Pyodide POC)

A proof-of-concept that runs a PyBullet environment from `predicators`
inside a web browser via Pyodide + a WASM build of `pybullet`. No
Python server, no native binaries — the entire stack ships as static
files to the user's browser, where Pyodide loads it.

## What works today

- Booting Pyodide in the page.
- Installing the WASM `pybullet` wheel + the pure-Python `predicators`
  wheel (with a tiny `gym.spaces.Box` shim).
- Unpacking the `predicators/envs/assets/` tarball into Pyodide's
  emulated FS.
- Constructing a PyBullet env (`pybullet_blocks` etc.), rendering a
  320×240 RGBA frame through `p.ER_TINY_RENDERER`, and drawing it
  onto a `<canvas>`.
- **Option-level interaction:** after Reset, the option `<select>`
  is populated from `get_gt_options(env_name)`; per-argument
  selectors filter the env's objects by type; clicking Execute
  grounds the option, runs its policy in the env until termination
  (or 200 steps), and re-renders. Verified in headless Chromium with
  `Pick(robby, block0)` — the robot arm visibly moves to the block.
- Verified end-to-end in Node Pyodide via `web/app/node_smoke.mjs`
  and in headless Chromium via `web/app/browser_smoke.mjs`
  (~10 seconds cold to Reset+Execute).

## Layout

```
web/
  app/
    index.html       minimal UI (env picker + Render + Step buttons)
    main.js          JS bootstrap (boots Pyodide, installs wheels, etc.)
    setup.py         Pyodide-side bridge (reset / render / step)
    bundle.sh        builds wheels + assets tarball into web/wheels/
    serve.sh         tiny `python -m http.server` for local dev
    node_smoke.mjs   the same flow as the browser, runnable in Node
    gym_shim_setup.py  builds the 50-line `gym` shim wheel
  wheels/            generated; not tracked (see .gitignore)
    pybullet-…wasm32.whl   pre-built WASM pybullet (Emscripten 3.1.58)
    predicators-…whl       built from this repo
    gym-…whl               shim
    assets.tar.gz          predicators/envs/assets/ packed
```

## Setup

```
# 1) Build the wheels + asset tarball
web/app/bundle.sh

# 2) (One-time) drop a matching pybullet wasm wheel into web/wheels/
# Get one from https://github.com/BasisResearch/pybullet-pyodide

# 3) Serve and open
web/app/serve.sh        # serves on http://localhost:8080
# Open http://localhost:8080/app/
```

## Iterating without a browser

```
cd web/app && node node_smoke.mjs
```

Same flow as the browser. Expected output ends with:
```
RESET RESULT: {'task_idx': 0, 'num_objects': 6, 'action_dim': 9}
RENDER RESULT: 320x240, 307200 bytes
```

## What's not done

- Per-step rendering during option execution — currently we only
  render once when the option finishes. A loop with a yield/render
  callback would let you watch the arm move frame-by-frame.
- The `human_option_control_approach.py` interactive prompt-loop
  isn't wired up directly — `setup.py`'s `execute_option` is a
  simpler thing that takes the ground-truth option + object args
  from JS instead of calling Python's `input()`.
- Asset slimming — we ship all 32 MB of `envs/assets/` packed. The
  blocks env only needs a few MB; a targeted manifest would be
  faster to boot.
- `gymnasium` and `imageio` aren't installable in Pyodide alongside
  pybullet (both fail with `bad export type for 'gSharedMemoryKey'`
  when installed after pybullet). `RoboDiscoEnv` raises a clear
  error if anyone tries to use it; the `save_video` etc. helpers
  live in the full `predicators.utils` and are never imported on
  the env path. May be a pyodide-vs-pybullet linker conflict worth
  reporting upstream.
- The OpenGL camera backend is unavailable in WASM pybullet, so the
  bridge monkey-patches `env.render()` to use `p.ER_TINY_RENDERER`
  (CPU rasterizer). It's slower than the HW path but works in-browser.
- The auto-discovery loader (`utils.import_submodules`) now skips
  submodules whose optional deps are missing (`gym_sokoban`,
  `gymnasium-robotics`, torch in `*/processes.py`, etc.). The flag
  defaults to off; only `predicators/envs/__init__.py` and
  `predicators/ground_truth_models/__init__.py` opt in.

## How the env import path is kept torch-free

This POC relies on the `utils.py` / `utils_lite.py` split (see commit
`986bd2c13`). All env-side modules import `predicators.utils_lite`,
which has no heavy ML deps. The full `predicators.utils` re-exports
the lite API and adds the torch / scipy / imageio / OpenAI / Gemini
helpers, so non-env code is unchanged.
