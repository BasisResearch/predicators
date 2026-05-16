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
    for pkg in ["dill", "tabulate", "pyperplan", "colorlog", "imageio", "gymnasium"]:
        try:
            await micropip.install(pkg, deps=True, keep_going=True)
            print(f"{pkg} installed", flush=True)
        except Exception as e:
            print(f"{pkg} FAILED: {type(e).__name__} {e}", flush=True)
    print("deps installed (some maybe skipped)", flush=True)
    await micropip.install("emfs:/tmp/${PREDICATORS_WHEEL}", deps=False)
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
    info = bridge.reset("pybullet_blocks")
    print("RESET RESULT:", info, flush=True)
    out = bridge.render()
    print(f"RENDER RESULT: {out['width']}x{out['height']}, {len(out['pixels'])} bytes", flush=True)
except Exception as e:
    print("BRIDGE ERROR:", type(e).__name__, e, flush=True)
    traceback.print_exc()
`);
} catch (e) {
  log("bridge call threw: " + (e.message || e));
}

log("DONE");
process.exit(0);
