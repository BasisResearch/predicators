// Node-side smoke test: mirrors what main.js does in the browser, so we
// can iterate on the Pyodide bridge without spinning up a real browser.
//
// Run from web/app/ via:  node --experimental-fetch node_smoke.mjs
// (or with regular node if your version has fetch built in).

import { loadPyodide } from "../node_modules/pyodide/pyodide.mjs";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WHEELS = resolve(HERE, "../wheels");

const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s]`, ...a);

log("Loading Pyodide…");
const pyodide = await loadPyodide({ stdout: log, stderr: log });
log("Pyodide ready");

await pyodide.loadPackage(["micropip", "numpy", "matplotlib", "pillow"]);
log("Base packages loaded");

// Pyodide's micropip can install from local files via file:// URIs,
// but it's easier to read the wheel bytes and hand it to micropip.
function readWheel(name) {
  return readFileSync(resolve(WHEELS, name));
}

// Pyodide's micropip parses the wheel filename out of the URL, so we
// must keep the canonical names rather than abbreviating.
const PYBULLET_WHEEL = "pybullet-3.2.7-cp313-cp313-pyemscripten_2025_0_wasm32.whl";
const PREDICATORS_WHEEL = "predicators-0.1.0-py3-none-any.whl";
const GYM_SHIM_WHEEL = "gym-0.26.2-py3-none-any.whl";
pyodide.FS.writeFile(`/tmp/${PYBULLET_WHEEL}`, readWheel(PYBULLET_WHEEL));
pyodide.FS.writeFile(`/tmp/${PREDICATORS_WHEEL}`, readWheel(PREDICATORS_WHEEL));
pyodide.FS.writeFile(`/tmp/${GYM_SHIM_WHEEL}`, readWheel(GYM_SHIM_WHEEL));

try {
  await pyodide.runPythonAsync(`
import sys, traceback
print("=== start install block ===", flush=True)
try:
    import micropip
    print("micropip loaded", flush=True)
    await micropip.install("emfs:/tmp/${PYBULLET_WHEEL}")
    print("pybullet installed", flush=True)
    # Try to import pybullet right away to see if it loads.
    try:
        import pybullet as p
        print("pybullet imported; version =", getattr(p, "__version__", "?"), flush=True)
        cid = p.connect(p.DIRECT)
        print("pybullet connected, cid =", cid, flush=True)
    except Exception as e:
        print("pybullet import/connect FAILED:", type(e).__name__, e, flush=True)
        traceback.print_exc()
        raise
    # Install our gym shim first so micropip doesn't try the real gym.
    await micropip.install("emfs:/tmp/${GYM_SHIM_WHEEL}")
    print("gym shim installed", flush=True)
    # The predicators wheel's install_requires is now the env-runtime
    # slim set, so let micropip do the resolution. keep_going=True
    # skips platform-specific dead-ends (pybullet-arm64 on PyPI,
    # numpy/matplotlib/pillow version pins vs. Pyodide-shipped).
    await micropip.install("emfs:/tmp/${PREDICATORS_WHEEL}",
                           deps=True, keep_going=True)
    print("predicators installed", flush=True)
except Exception as e:
    print("INSTALL ERROR:", type(e).__name__, e, flush=True)
    traceback.print_exc()
    raise
`);
} catch (e) {
  log("install threw: " + (e.message || e));
  process.exit(1);
}
log("Wheels installed");

// Mount the env asset dir at the path predicators expects so that
// `os.path.exists(envs/assets/urdf/plane.urdf)` etc. work without
// baking 141 MB of meshes into the wheel.
const ASSET_SRC = resolve(HERE, "../../predicators/envs/assets");
const ASSET_DEST = "/lib/python3.13/site-packages/predicators/envs/assets";
try { pyodide.FS.rmdir(ASSET_DEST); } catch {}
pyodide.FS.mkdirTree(ASSET_DEST);
pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: ASSET_SRC }, ASSET_DEST);
log(`Mounted assets: ${ASSET_SRC} -> ${ASSET_DEST}`);

const setupSrc = readFileSync(resolve(HERE, "setup.py"), "utf8");
pyodide.FS.writeFile("/setup.py", setupSrc);
try {
  await pyodide.runPythonAsync(`
import traceback
try:
    exec(open('/setup.py').read(), globals())
    print('setup.py loaded ok', flush=True)
except SystemExit as e:
    print('SystemExit during setup:', e, flush=True)
except Exception as e:
    print('SETUP ERROR:', type(e).__name__, e, flush=True)
    traceback.print_exc()
    raise
`);
} catch (e) {
  log("setup threw: " + (e.message || e));
  process.exit(1);
}
log("Bridge ready");

log("Trying bridge.reset…");
try {
  await pyodide.runPythonAsync(`
import traceback
try:
    info = bridge.reset("pybullet_coffee")
    print(f"MANIFEST ({len(info['manifest'])} bodies):", flush=True)
    for e in info['manifest']:
        kind = e.get('kind', '?')
        url = e.get('url', '')
        shapes = e.get('shapes', [])
        print(f"  body_id={e['body_id']} name={e['name']!r:20s} kind={kind} "
              f"url={url or ''}", flush=True)
        for s in shapes:
            print(f"      link={s['link']:>2} geom={s['geom']:8s} "
                  f"dims={s['dims']} rgba={[round(c,2) for c in s['rgba']]} "
                  f"local_pos={[round(c,3) for c in s['local_pos']]}",
                  flush=True)
    import sys; sys.exit(0)

    # Inspect grow PickJug behaviour in detail.
    import pybullet as p
    from predicators.ground_truth_models import get_gt_options, get_gt_nsrts
    env = bridge.env
    state = env._current_observation
    print("---- pre-pick state ----", flush=True)
    for o in sorted(env._objects, key=lambda o: o.name):
        feats = {f: state.get(o, f) for f in o.type.feature_names
                 if f in {'x','y','z','rot','fingers','tilt','wrist'}}
        print(f"  {o.name:12s} {o.type.name:10s} {feats}", flush=True)

    options = {o.name: o for o in get_gt_options('pybullet_grow')}
    nsrts = {n.name: n for n in get_gt_nsrts('pybullet_grow', env.predicates, set(options.values()))}
    print("NSRTs:", list(nsrts.keys()), flush=True)

    opt = options['PickJug']
    print(f"PickJug params_space: low={opt.params_space.low} high={opt.params_space.high}", flush=True)
    name_to_obj = {o.name: o for o in env._objects}
    chosen = [name_to_obj['robot'], name_to_obj['jug1']]
    import numpy as np
    rng = np.random.default_rng(0)
    g = bridge._sample_ground_option(opt, chosen, state, rng)
    print(f"Grounded PickJug params: {g.params}", flush=True)
    print(f"Initiable: {g.initiable(state)}", flush=True)
    # Run PickJug step-by-step and print every N steps.
    import logging; logging.basicConfig(level=logging.DEBUG)
    for step in range(200):
        if g.terminal(state):
            print(f"TERMINATED at step={step}", flush=True)
            break
        act = g.policy(state)
        state = env.step(act)
        r = name_to_obj['robot']
        j = name_to_obj['jug1']
        mem = g.memory if hasattr(g, 'memory') else {}
        phase_idx = mem.get('phase_idx', '?')
        print(f"  step={step:3d} phase_idx={phase_idx} "
              f"ee=({state.get(r,'x'):.3f},{state.get(r,'y'):.3f},{state.get(r,'z'):.3f}) "
              f"fingers={state.get(r,'fingers'):.3f} "
              f"jug=({state.get(j,'x'):.3f},{state.get(j,'y'):.3f},{state.get(j,'z'):.3f})", flush=True)
    else:
        print("Did not terminate within 200 steps", flush=True)
    r = name_to_obj['robot']; j = name_to_obj['jug1']
    print(f"POST-PickJug ee=({state.get(r,'x'):.3f},{state.get(r,'y'):.3f},{state.get(r,'z'):.3f}) "
          f"fingers={state.get(r,'fingers'):.3f} "
          f"jug=({state.get(j,'x'):.3f},{state.get(j,'y'):.3f},{state.get(j,'z'):.3f})", flush=True)
    # Holding predicate check
    Holding = next((pp for pp in env.predicates if pp.name == 'Holding'), None)
    if Holding is not None:
        print(f"Holding(robot, jug1)? {Holding.holds(state, [r, j])}", flush=True)

    # Dump colors so we can pick a matching cup for pour.
    print("Colors:", flush=True)
    for o in env._objects:
        if o.type.name in {'cup', 'jug'}:
            cr = state.get(o, 'r'); cg = state.get(o, 'g'); cb = state.get(o, 'b')
            print(f"  {o.name}: ({cr:.2f}, {cg:.2f}, {cb:.2f})", flush=True)

    # Find a cup that matches jug1's color, fall back to cup0.
    jug_obj = name_to_obj['jug1']
    jr, jg, jb = state.get(jug_obj,'r'), state.get(jug_obj,'g'), state.get(jug_obj,'b')
    matching = None
    for o in env._objects:
        if o.type.name == 'cup':
            if abs(state.get(o,'r')-jr)<0.01 and abs(state.get(o,'g')-jg)<0.01 and abs(state.get(o,'b')-jb)<0.01:
                matching = o.name
                break
    if matching is None:
        matching = 'cup0'
    print(f"Pouring into matching cup: {matching}", flush=True)

    # Continue with Pour(robot, jug1, matching) + Place
    for op_name, obj_names in [
        ('Pour', ['robot', 'jug1', matching]),
        ('Place', ['robot', 'jug1']),
    ]:
        opt = options[op_name]
        chosen = [name_to_obj[n] for n in obj_names]
        g = bridge._sample_ground_option(opt, chosen, state, rng)
        print(f"=== Try {op_name}{tuple(obj_names)} params={list(g.params)} initiable={g.initiable(state)} ===", flush=True)
        if not g.initiable(state):
            continue
        for step in range(400):
            if g.terminal(state):
                print(f"{op_name} TERMINATED at step={step}", flush=True)
                break
            try:
                act = g.policy(state)
            except Exception as e:
                print(f"{op_name} policy threw at step {step}: {type(e).__name__}: {e}", flush=True)
                break
            state = env.step(act)
            if step % 20 == 0 or step < 3:
                cup_obj = name_to_obj.get(matching if op_name == 'Pour' else 'cup0')
                growth = state.get(cup_obj, 'growth') if cup_obj else -1
                mem = g.memory if hasattr(g, 'memory') else {}
                phase_idx = mem.get('phase_idx', '?')
                print(f"  {op_name} step={step:3d} phase={phase_idx} "
                      f"ee=({state.get(r,'x'):.3f},{state.get(r,'y'):.3f},{state.get(r,'z'):.3f}) "
                      f"tilt={state.get(r,'tilt'):.2f} "
                      f"jug=({state.get(j,'x'):.3f},{state.get(j,'y'):.3f},{state.get(j,'z'):.3f}) "
                      f"growth={growth:.3f}", flush=True)
        else:
            print(f"{op_name} did not terminate within 400 steps", flush=True)
except Exception as e:
    print("BRIDGE ERROR:", type(e).__name__, e, flush=True)
    traceback.print_exc()
`);
} catch (e) {
  log("bridge call threw: " + (e.message || e));
}

log("DONE");
process.exit(0);
