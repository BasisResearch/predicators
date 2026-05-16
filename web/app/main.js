// predicators-in-the-browser bootstrap.
//
// Loads Pyodide, the WASM pybullet wheel, and our predicators wheel,
// then exposes a tiny JS<->Python bridge for resetting an env, stepping
// it, and rendering frames into a <canvas>. Talks to setup.py for the
// Python side of the work.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.mjs";

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const infoEl = document.getElementById("info");
const envSelect = document.getElementById("env-select");
const bootBtn = document.getElementById("boot-env");
const renderBtn = document.getElementById("render");
const stepBtn = document.getElementById("step-zero");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function log(msg) {
  console.log(msg);
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(s) {
  statusEl.textContent = s;
}

let pyodide = null;
let bridge = null;

async function boot() {
  setStatus("Loading Pyodide runtime…");
  pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
    stdout: log,
    stderr: log,
  });
  log("Pyodide loaded");

  setStatus("Loading micropip + base packages…");
  await pyodide.loadPackage(["micropip", "numpy", "matplotlib", "pillow"]);
  log("Base packages loaded");

  setStatus("Installing pybullet (wasm) + predicators wheel…");
  // Note: relative URLs are resolved against the page origin, so the wheels
  // must be served by the same HTTP server.
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("../wheels/pybullet-3.2.7-cp312-cp312-emscripten_3_1_58_wasm32.whl")
await micropip.install([
    "gym==0.26.2",
    "gymnasium",
    "dill",
    "tabulate",
    "pyperplan",
    "colorlog",
    "imageio",
], deps=True)
await micropip.install("../wheels/predicators-0.1.0-py3-none-any.whl", deps=False)
print("Installed predicators + deps")
`);

  setStatus("Loading setup.py…");
  const setupSrc = await (await fetch("./setup.py")).text();
  pyodide.FS.writeFile("/setup.py", setupSrc);
  await pyodide.runPythonAsync("exec(open('/setup.py').read())");
  bridge = pyodide.globals.get("bridge");
  log("Bridge ready");

  setStatus("Ready. Pick an env and hit Reset.");
  envSelect.disabled = false;
  bootBtn.disabled = false;
}

async function resetEnv() {
  const envName = envSelect.value;
  setStatus(`Constructing ${envName}…`);
  try {
    const info = bridge.reset(envName).toJs({ dict_converter: Object.fromEntries });
    infoEl.textContent = `task=${info.task_idx} objects=${info.num_objects} action_dim=${info.action_dim}`;
    log(`Reset ${envName} -> ${JSON.stringify(info)}`);
    setStatus("Env ready. Try Render or Step.");
    renderBtn.disabled = false;
    stepBtn.disabled = false;
    await renderFrame();
  } catch (e) {
    setStatus("Reset failed — see log.");
    log("ERROR: " + e.message);
  }
}

async function renderFrame() {
  const out = bridge.render().toJs({ dict_converter: Object.fromEntries });
  // out.pixels is a Uint8Array of length width*height*4 (RGBA).
  const { width, height, pixels } = out;
  canvas.width = width;
  canvas.height = height;
  const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
  ctx.putImageData(imageData, 0, 0);
}

async function stepZero() {
  try {
    bridge.step_zero();
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
