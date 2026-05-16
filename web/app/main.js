// predicators-in-the-browser bootstrap.
//
// Boots Pyodide, installs the WASM pybullet wheel + predicators wheel +
// a tiny gym shim, unpacks the env asset tarball into the Pyodide FS,
// then exposes Reset / Render / Step on env names like `pybullet_blocks`.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const infoEl = document.getElementById("info");
const envSelect = document.getElementById("env-select");
const bootBtn = document.getElementById("boot-env");
const renderBtn = document.getElementById("render");
const stepBtn = document.getElementById("step-zero");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const WHEELS_BASE = "../wheels";
const PYBULLET_WHEEL = "pybullet-3.2.7-cp312-cp312-emscripten_3_1_58_wasm32.whl";
const PREDICATORS_WHEEL = "predicators-0.1.0-py3-none-any.whl";
const GYM_SHIM_WHEEL = "gym-0.26.2-py3-none-any.whl";
const ASSETS_TARBALL = "assets.tar.gz";

const t0 = performance.now();
function log(msg) {
  const t = ((performance.now() - t0) / 1000).toFixed(1);
  const line = `[${t}s] ${msg}`;
  console.log(line);
  logEl.textContent += line + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}
function setStatus(s) { statusEl.textContent = s; }

let pyodide = null;

async function fetchBytes(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function boot() {
  setStatus("Loading Pyodide runtime…");
  pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/",
    stdout: log,
    stderr: log,
  });
  log("Pyodide loaded");

  setStatus("Loading base packages (numpy, matplotlib, pillow)…");
  await pyodide.loadPackage(["micropip", "numpy", "matplotlib", "pillow"]);

  setStatus("Staging wheels into Pyodide FS…");
  pyodide.FS.writeFile(`/tmp/${PYBULLET_WHEEL}`, await fetchBytes(`${WHEELS_BASE}/${PYBULLET_WHEEL}`));
  pyodide.FS.writeFile(`/tmp/${GYM_SHIM_WHEEL}`, await fetchBytes(`${WHEELS_BASE}/${GYM_SHIM_WHEEL}`));
  pyodide.FS.writeFile(`/tmp/${PREDICATORS_WHEEL}`, await fetchBytes(`${WHEELS_BASE}/${PREDICATORS_WHEEL}`));

  setStatus("Installing pybullet wasm wheel…");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:/tmp/${PYBULLET_WHEEL}")
print("pybullet installed")
await micropip.install("emfs:/tmp/${GYM_SHIM_WHEEL}")
print("gym shim installed")
for pkg in ["dill", "tabulate", "pyperplan", "colorlog", "imageio"]:
    try:
        await micropip.install(pkg, deps=True, keep_going=True)
        print(f"{pkg} installed")
    except Exception as e:
        print(f"{pkg} FAILED: {e}")
await micropip.install("emfs:/tmp/${PREDICATORS_WHEEL}", deps=False)
print("predicators installed")
`);

  setStatus("Fetching + unpacking env assets (~30 MB)…");
  const assetsBuf = await fetchBytes(`${WHEELS_BASE}/${ASSETS_TARBALL}`);
  // Tarball contains a top-level `assets/` dir; unpack inside the
  // installed package so the env code finds it at the expected path.
  pyodide.FS.mkdirTree("/lib/python3.12/site-packages/predicators/envs");
  await pyodide.runPythonAsync(
    `import os; os.chdir("/lib/python3.12/site-packages/predicators/envs")`);
  pyodide.unpackArchive(assetsBuf, "tar.gz");
  log("Assets unpacked");

  setStatus("Loading bridge…");
  const setupSrc = await (await fetch("./setup.py")).text();
  pyodide.FS.writeFile("/setup.py", setupSrc);
  await pyodide.runPythonAsync("exec(open('/setup.py').read(), globals())");

  setStatus("Ready. Pick an env and hit Reset.");
  envSelect.disabled = false;
  bootBtn.disabled = false;
}

const optionRow = document.getElementById("option-row");
const optionSelect = document.getElementById("option-select");
const optionArgs = document.getElementById("option-args");
const executeBtn = document.getElementById("execute-option");

let currentOptions = [];
let currentObjects = [];

async function resetEnv() {
  const envName = envSelect.value;
  setStatus(`Constructing ${envName}…`);
  try {
    const infoJson = await pyodide.runPythonAsync(`
import json
json.dumps(bridge.reset("${envName}"))
`);
    const info = JSON.parse(infoJson);
    infoEl.textContent = `task=${info.task_idx} objects=${info.num_objects} action_dim=${info.action_dim}`;
    log(`Reset ${envName} -> ${infoJson}`);

    // Pull options + objects.
    const optsJson = await pyodide.runPythonAsync("import json; json.dumps(bridge.list_options())");
    const objsJson = await pyodide.runPythonAsync("import json; json.dumps(bridge.list_objects())");
    currentOptions = JSON.parse(optsJson);
    currentObjects = JSON.parse(objsJson);
    populateOptionPicker();

    setStatus("Env ready. Pick an option or hit Render.");
    renderBtn.disabled = false;
    stepBtn.disabled = false;
    optionRow.style.display = "";
    await renderFrame();
  } catch (e) {
    setStatus("Reset failed — see log.");
    log("ERROR: " + e.message);
  }
}

function populateOptionPicker() {
  optionSelect.innerHTML = "";
  for (const opt of currentOptions) {
    const o = document.createElement("option");
    o.value = opt.name;
    o.textContent = `${opt.name}(${opt.type_names.join(", ")})`;
    optionSelect.appendChild(o);
  }
  optionSelect.addEventListener("change", renderOptionArgs);
  renderOptionArgs();
}

function renderOptionArgs() {
  optionArgs.innerHTML = "";
  const opt = currentOptions.find((o) => o.name === optionSelect.value);
  if (!opt) return;
  // For each type, build a <select> of objects of that type.
  for (const tname of opt.type_names) {
    const sel = document.createElement("select");
    sel.className = "opt-arg";
    sel.dataset.typeName = tname;
    for (const obj of currentObjects.filter((o) => o.type_name === tname)) {
      const o = document.createElement("option");
      o.value = obj.name;
      o.textContent = obj.name;
      sel.appendChild(o);
    }
    optionArgs.appendChild(sel);
  }
}

async function executeOption() {
  const name = optionSelect.value;
  const args = Array.from(optionArgs.querySelectorAll("select.opt-arg"))
    .map((s) => s.value);
  setStatus(`Executing ${name}(${args.join(", ")})…`);
  try {
    const argList = JSON.stringify(args);
    const steps = await pyodide.runPythonAsync(`
import json
json.dumps(bridge.execute_option("${name}", ${argList}))
`);
    log(`Executed ${name}(${args.join(", ")}) -> ${steps} steps`);
    setStatus(`${name} done in ${steps} steps.`);
    await renderFrame();
  } catch (e) {
    setStatus("Execute failed — see log.");
    log("ERROR: " + e.message);
  }
}

executeBtn.addEventListener("click", executeOption);

async function renderFrame() {
  // Call render on the Python side, return a binary buffer + dims via
  // a small helper so we can avoid going through PyProxy.toJs (which
  // doesn't always preserve the bytes shape we want).
  const out = await pyodide.runPythonAsync(`
import json
r = bridge.render()
(r["width"], r["height"], r["pixels"])
`);
  // out is a Python tuple of (int, int, bytes); convert to JS.
  const [width, height, pixels] = out.toJs({ create_proxies: false });
  canvas.width = width;
  canvas.height = height;
  const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
  ctx.putImageData(imageData, 0, 0);
}

async function stepZero() {
  try {
    await pyodide.runPythonAsync("bridge.step_zero()");
    await renderFrame();
  } catch (e) {
    log("Step error: " + e.message);
  }
}

bootBtn.addEventListener("click", resetEnv);
renderBtn.addEventListener("click", renderFrame);
stepBtn.addEventListener("click", stepZero);

boot().catch((e) => {
  setStatus("Boot failed — see log.");
  log("FATAL: " + e.message);
  console.error(e);
});
