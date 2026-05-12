# web/ — predicators envs in a browser, on real PyBullet

This directory runs PyBullet-style demo envs in the browser via
**Pyodide + the pyodide-bullet wheel**. The same `import pybullet as p`
that runs on a server runs here, unchanged.

This is no longer an adapter shim — it's the real thing.

## Run locally

```bash
cd web/
python -m http.server 8000
# or:
darkhttpd . --port 8000 --addr 127.0.0.1
# open http://localhost:8000 in a browser
```

First load is slow (~10 s Pyodide + ~74 MB pybullet wheel). After that
the wheel is cached by the browser, so reloads are fast.

To smoke-test headlessly (Node):

```bash
cd web/
node test_envs.mjs
```

That runs each env for ~1 simulated second and prints how much each
scene moved.

## Layout

```
web/
├── envs/
│   ├── base.py         # tiny BaseDemoEnv (wraps pybullet calls,
│   │                    #  tracks render metadata for Three.js)
│   ├── blocks.py       # scripted gripper stacks cubes
│   ├── bowling.py      # rolling ball into a triangle of pins
│   ├── cover.py        # slide blocks onto target zones
│   ├── domino.py       # cascading chain
│   ├── newton.py       # Newton's cradle — pendulum + elastic contacts
│   └── wrecking.py     # heavy pendulum smashes a tower
├── wheels/
│   └── pybullet-3.2.7-cp312-cp312-emscripten_3_1_58_wasm32.whl  (5.4 MB)
├── app.js              # Pyodide loader + Three.js renderer
├── demo.py             # Pyodide-side entry: load_env / step / poses
├── index.html          # host page
├── test_envs.mjs       # Node-based headless smoke test
└── README.md
```

## The envs and what each one exercises

| name       | physics features                                                  |
|------------|-------------------------------------------------------------------|
| `blocks`   | gravity, contact, fixed-constraint grasping, stacking             |
| `cover`    | sliding friction, kinematic vs dynamic interaction                |
| `domino`   | cascading contact, dynamic finger + velocity transfer             |
| `newton`   | point-to-point constraints (pendulum), near-elastic restitution   |
| `wrecking` | heavy pendulum momentum, tower collapse, debris scatter           |
| `bowling`  | rolling friction, ball spin, 10-body contact cascade              |

## How the pieces fit together

1. `index.html` pulls in Three.js (rendering) + Pyodide (CPython on
   WASM).
2. `app.js` boots Pyodide, loads NumPy, installs the pybullet wheel
   via `micropip`, then writes the small `web/` Python tree into the
   Pyodide filesystem and imports `web.demo`.
3. `web/envs/base.py` exposes simple body-spawning helpers
   (`spawn_box`, `spawn_plane`, `grasp`, `release`) that call
   PyBullet directly and record per-body render metadata.
4. Each env (`blocks`, `cover`, `domino`) implements `_build()` and
   `_policy()`. The policies are scripted state machines that
   manipulate a kinematic floating gripper.
5. `web/demo.py` is the thin Pyodide-callable wrapper:
   `load_env(name)` → render manifest, `step(dt)` → tick physics,
   `poses()` → body transforms.
6. `app.js` polls `poses()` once per animation frame and writes the
   transforms into Three.js meshes.

## Why this is different from the earlier Ammo.js plan

Earlier work in this directory (now deleted) wrapped Ammo.js (Bullet
compiled to JS) behind an adapter. That required reimplementing URDF
loading, IK, and per-env shims to bridge to the predicators codebase.

Approach B — building PyBullet itself for Pyodide — turned out to work
([`../pyodide-bullet/`](../pyodide-bullet/README.md)). Now the same
PyBullet that runs on the server runs in the browser. No adapter, no
URDF parser, no IK shim.

## Scope

These three demos still use a **scripted kinematic gripper** rather
than a robot arm + IK. That's an env-design choice, not a runtime
limitation — IK and URDF-based robots are available in the wheel and
can be added when needed.

The next step is to load actual `predicators.envs.pybullet_*` envs
here, which requires slimming the predicators top-level import graph
(today `predicators.__init__` and `predicators.utils` pull in torch,
gym, etc.). See `../pyodide-bullet/README.md` for the rebuild details.

## Determinism note

Server PyBullet and the browser's PyBullet are built from the same
Bullet source. Floating-point behaviour under Emscripten can still
diverge slightly from a native build, so bit-identical rollouts
across server and browser aren't guaranteed — but you're at least
running the same algorithm, not two different physics engines.
