// Smoke test: load the cross-compiled pybullet wheel into Pyodide
// (running under Node) and exercise the core API.
//
// Run from this directory:  node smoke.mjs

import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const WHEEL = resolve(
  __dirname,
  "../out/pybullet-3.2.7-cp312-cp312-emscripten_3_1_58_wasm32.whl",
);

async function main() {
  console.log("Loading pyodide…");
  const py = await loadPyodide();

  console.log("Loading numpy (pybullet build links libnpymath)…");
  await py.loadPackage("numpy");

  console.log("Loading our pybullet wheel…");
  // micropip parses the wheel name to derive package metadata, so the
  // path must end with the canonical wheel filename.
  const wheelName = "pybullet-3.2.7-cp312-cp312-emscripten_3_1_58_wasm32.whl";
  const wheelBytes = readFileSync(WHEEL);
  py.FS.writeFile(`/tmp/${wheelName}`, wheelBytes);
  await py.loadPackage("micropip");
  await py.runPythonAsync(`
import micropip
await micropip.install("emfs:/tmp/${wheelName}")
print("pybullet installed")
`);

  console.log("\n--- smoke test ---");
  const out = await py.runPythonAsync(`
import pybullet as p
import pybullet_data as pd
print("pybullet version:", getattr(p, "getAPIVersion", lambda: "?")())

cid = p.connect(p.DIRECT)
print("connected, client id:", cid)

p.setAdditionalSearchPath(pd.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
print("loaded plane, body id:", plane_id)

cube_id = p.loadURDF("cube_small.urdf", basePosition=[0, 0, 1.0])
print("loaded cube at z=1.0, body id:", cube_id)

for _ in range(240):
    p.stepSimulation()

pos, _ = p.getBasePositionAndOrientation(cube_id)
print(f"after 1s sim, cube z = {pos[2]:.4f} (should be near 0)")

p.disconnect()
print("disconnected")
"OK"
`);
  console.log("\nresult:", out);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
