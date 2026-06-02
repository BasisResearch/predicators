// Cheap CI gate: boots Pyodide in Node, installs the three wheels
// (pybullet WASM, gym shim, predicators), and verifies that
// `predicators.envs` loads and exposes the env names we expect. Does
// not construct any env, render anything, or fetch assets. ~30 s wall.
//
//   node web/app/import_check.mjs
//
// Exits 0 iff every expected env name shows up in the registered
// subclasses of BaseEnv. Otherwise exits 1.

import { loadPyodide } from "../node_modules/pyodide/pyodide.mjs";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WHEELS = resolve(HERE, "../wheels");
const HTML = readFileSync(resolve(HERE, "index.html"), "utf8");
const EXPECTED_ENVS = Array.from(HTML.matchAll(/value="(pybullet_[a-z_]+)"/g))
  .map((m) => m[1]);

const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s]`, ...a);

// WASM aborts inside Pyodide land on Node's default unhandled-error path
// (stack frame on stderr, immediate exit) — and GitHub Actions log
// buffers eat the message, leaving only `pyodide.asm.js:8` in the CI
// log. Capture everything so the next CI run has something to read.
process.on("unhandledRejection", (e) => {
  console.error("[unhandledRejection]", e && (e.stack || e.message || e));
  process.exit(1);
});
process.on("uncaughtException", (e) => {
  console.error("[uncaughtException]", e && (e.stack || e.message || e));
  process.exit(1);
});

const PYBULLET_WHEEL = "pybullet-3.2.7-cp313-cp313-pyemscripten_2025_0_wasm32.whl";
const PREDICATORS_WHEEL = "predicators-0.1.0-py3-none-any.whl";
const GYM_SHIM_WHEEL = "gym-0.26.2-py3-none-any.whl";

async function main() {
  log("Loading Pyodide…");
  const pyodide = await loadPyodide({
    stdout: (m) => log("py:", m),
    stderr: (m) => log("py!:", m),
  });
  log("Pyodide ready");

  await pyodide.loadPackage(["micropip", "numpy", "matplotlib", "pillow"]);
  log("Base packages loaded");

  const readWheel = (name) => readFileSync(resolve(WHEELS, name));
  pyodide.FS.writeFile(`/tmp/${PYBULLET_WHEEL}`, readWheel(PYBULLET_WHEEL));
  pyodide.FS.writeFile(`/tmp/${PREDICATORS_WHEEL}`, readWheel(PREDICATORS_WHEEL));
  pyodide.FS.writeFile(`/tmp/${GYM_SHIM_WHEEL}`, readWheel(GYM_SHIM_WHEEL));

  await pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:/tmp/${PYBULLET_WHEEL}")
await micropip.install("emfs:/tmp/${GYM_SHIM_WHEEL}")
await micropip.install("emfs:/tmp/${PREDICATORS_WHEEL}",
                       deps=True, keep_going=True)
`);
  log("Wheels installed");

  // Trigger env subclass auto-discovery. import_submodules will skip
  // modules whose optional deps (torch, gym_sokoban) aren't present,
  // which is exactly what we expect in this slim Pyodide runtime.
  // Wrap the import in Python try/except so any traceback lands on
  // our captured stderr instead of crashing the WASM runtime.
  const found = await pyodide.runPythonAsync(`
import json, sys, traceback
_names = []
try:
    from predicators import utils_lite
    from predicators.envs import BaseEnv
    _names = sorted(
        cls.get_name() for cls in utils_lite.get_all_subclasses(BaseEnv)
        if not cls.__abstractmethods__
    )
except BaseException as e:
    sys.stderr.write("PYTHON-IMPORT-ERROR: " + repr(e) + "\\n")
    traceback.print_exc()
    sys.stderr.flush()
json.dumps(_names)
`);
  const registeredArr = JSON.parse(found);
  log(`predicators.envs loaded, ${registeredArr.length} envs registered`);
  const registered = new Set(registeredArr);
  const missing = EXPECTED_ENVS.filter((e) => !registered.has(e));
  const extra = registeredArr.filter((e) => !EXPECTED_ENVS.includes(e));
  if (extra.length) {
    log(`note: ${extra.length} envs registered but not in the dropdown:`, extra);
  }
  if (missing.length) {
    log(`FAIL: ${missing.length} dropdown envs failed to register:`, missing);
    process.exitCode = 1;
    return;
  }
  log(`OK: all ${EXPECTED_ENVS.length} dropdown envs registered`);
}

try {
  await main();
} catch (e) {
  console.error("[main] threw:", e && (e.stack || e.message || e));
  process.exitCode = 1;
}
