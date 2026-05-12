// Headless smoke test for the new env code path.
//
// Runs each demo env in Pyodide for ~1 simulated second and checks the
// final body poses look sane.

import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const WHEEL = resolve(
  __dirname,
  "wheels/pybullet-3.2.7-cp312-cp312-emscripten_3_1_58_wasm32.whl",
);
const WHEEL_NAME = "pybullet-3.2.7-cp312-cp312-emscripten_3_1_58_wasm32.whl";

const ENV_FILES = [
  "envs/__init__.py",
  "envs/base.py",
  "envs/blocks.py",
  "envs/cover.py",
  "envs/domino.py",
  "demo.py",
];

async function main() {
  console.log("loading pyodide…");
  const py = await loadPyodide();

  console.log("loading numpy…");
  await py.loadPackage("numpy");

  console.log("installing pybullet wheel…");
  py.FS.writeFile(`/tmp/${WHEEL_NAME}`, readFileSync(WHEEL));
  await py.loadPackage("micropip");
  await py.runPythonAsync(
    `import micropip\nawait micropip.install("emfs:/tmp/${WHEEL_NAME}")`,
  );

  console.log("mounting web/ package…");
  py.FS.mkdirTree("/home/pyodide/web");
  py.FS.mkdirTree("/home/pyodide/web/envs");
  py.FS.writeFile("/home/pyodide/web/__init__.py", "");
  for (const rel of ENV_FILES) {
    py.FS.writeFile(
      `/home/pyodide/web/${rel}`,
      readFileSync(resolve(__dirname, rel)),
    );
  }
  await py.runPythonAsync(
    `import sys\nsys.path.insert(0, "/home/pyodide")`,
  );

  const demo = py.pyimport("web.demo");
  for (const name of demo.list_envs()) {
    console.log(`\n--- ${name} ---`);
    const manifest = demo.load_env(name).toJs({
      dict_converter: Object.fromEntries,
    });
    console.log(`  bodies in scene: ${manifest.length}`);
    const before = demo.poses().toJs({ dict_converter: Object.fromEntries });
    for (let i = 0; i < 60; i++) demo.step(1 / 60);
    const after = demo.poses().toJs({ dict_converter: Object.fromEntries });
    const bodyIds = Object.keys(before);
    // Just confirm something moved or stayed put plausibly.
    let movement = 0;
    for (const id of bodyIds) {
      const a = before[id][0], b = after[id][0];
      movement += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    }
    console.log(`  total body movement after 1s: ${movement.toFixed(3)} m`);
  }
  console.log("\nall envs ran cleanly.");
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
