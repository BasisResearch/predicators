// app.js — Pyodide + real PyBullet + Three.js
//
// Responsibilities:
//   1. Load Pyodide and the cross-compiled pybullet wheel.
//   2. Mount the web/ Python package into Pyodide's FS.
//   3. Drive a fixed-step physics loop and mirror body poses into Three.js.

const STATUS = document.getElementById("status");
const CANVAS = document.getElementById("canvas");
const SELECTOR = document.getElementById("env-selector");
const RESET = document.getElementById("reset-btn");

function setStatus(msg) {
  console.log("[demo]", msg);
  if (STATUS) STATUS.textContent = msg;
}

// ---- Three.js scene -------------------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf2f4f8);

const camera = new THREE.PerspectiveCamera(
  50, CANVAS.clientWidth / CANVAS.clientHeight, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(0.7, -0.6, 0.55);
camera.lookAt(0, 0, 0.05);

const renderer = new THREE.WebGLRenderer({ canvas: CANVAS, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(CANVAS.clientWidth, CANVAS.clientHeight, false);
renderer.shadowMap.enabled = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(0.5, -0.4, 1.5);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
scene.add(sun);

// Body-id → THREE.Object3D.
const meshes = new Map();

function addBoxMesh(id, halfExtents, color) {
  const geom = new THREE.BoxGeometry(
    halfExtents[0] * 2, halfExtents[1] * 2, halfExtents[2] * 2);
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(color[0], color[1], color[2]),
    opacity: color[3], transparent: color[3] < 1.0,
    roughness: 0.55, metalness: 0.05,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);
  meshes.set(id, mesh);
}

function addPlaneMesh(id) {
  const geom = new THREE.PlaneGeometry(2.0, 2.0);
  const mat = new THREE.MeshStandardMaterial({
    color: 0xdde2eb, roughness: 0.9, metalness: 0.0,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.receiveShadow = true;
  scene.add(mesh);
  meshes.set(id, mesh);
}

function addSphereMesh(id, radius, color) {
  const geom = new THREE.SphereGeometry(radius, 24, 16);
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(color[0], color[1], color[2]),
    opacity: color[3], transparent: color[3] < 1.0,
    roughness: 0.4, metalness: 0.1,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);
  meshes.set(id, mesh);
}

function clearMeshes() {
  for (const m of meshes.values()) {
    scene.remove(m);
    m.geometry.dispose();
    m.material.dispose();
  }
  meshes.clear();
}

function applyManifest(manifest) {
  clearMeshes();
  for (const entry of manifest) {
    if (entry.kind === "box") {
      addBoxMesh(entry.id, entry.half_extents, entry.color);
    } else if (entry.kind === "sphere") {
      addSphereMesh(entry.id, entry.radius, entry.color);
    } else if (entry.kind === "plane") {
      addPlaneMesh(entry.id);
    }
  }
}

// ---- Pyodide bootstrap ----------------------------------------------------
let pyodide = null;
let demoMod = null;
let running = false;
let lastT = 0;

const WHEEL_URL = "./wheels/pybullet-3.2.7-cp312-cp312-emscripten_3_1_58_wasm32.whl";

async function bootstrap() {
  setStatus("Loading Pyodide…");
  pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/",
  });

  setStatus("Loading numpy (pybullet links libnpymath)…");
  await pyodide.loadPackage("numpy");

  setStatus("Loading micropip…");
  await pyodide.loadPackage("micropip");

  setStatus("Installing pybullet wheel (~74 MB, one-time)…");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("${WHEEL_URL}")
`);

  setStatus("Mounting browser env code…");
  await mountWebPackage(pyodide);

  setStatus("Importing demo module…");
  demoMod = pyodide.pyimport("web.demo");

  // Populate env selector from the Python side.
  const names = demoMod.list_envs().toJs();
  SELECTOR.innerHTML = "";
  for (const n of names) {
    const opt = document.createElement("option");
    opt.value = n;
    opt.textContent = n;
    SELECTOR.appendChild(opt);
  }

  await loadEnv(SELECTOR.value);
  setStatus("Running.");
  running = true;
  lastT = performance.now();
  requestAnimationFrame(tick);
}

async function mountWebPackage(py) {
  const files = [
    "envs/__init__.py",
    "envs/base.py",
    "envs/blocks.py",
    "envs/bowling.py",
    "envs/cover.py",
    "envs/domino.py",
    "envs/newton.py",
    "envs/wrecking.py",
    "demo.py",
  ];
  py.FS.mkdirTree("/home/pyodide/web");
  py.FS.mkdirTree("/home/pyodide/web/envs");
  py.FS.writeFile("/home/pyodide/web/__init__.py", "");
  for (const rel of files) {
    const resp = await fetch(`./${rel}`);
    if (!resp.ok) throw new Error(`fetch ${rel}: ${resp.status}`);
    const text = await resp.text();
    py.FS.writeFile(`/home/pyodide/web/${rel}`, text);
  }
  await py.runPythonAsync(`
import sys
if "/home/pyodide" not in sys.path:
    sys.path.insert(0, "/home/pyodide")
`);
}

async function loadEnv(name) {
  // demoMod.load_env returns a PyProxy List of dicts; convert to JS.
  const manifestPy = demoMod.load_env(name);
  const manifest = manifestPy.toJs({ dict_converter: Object.fromEntries });
  manifestPy.destroy();
  applyManifest(manifest);
}

// ---- Main loop ------------------------------------------------------------
function tick(now) {
  if (!running) return;
  const dt = Math.min((now - lastT) / 1000.0, 1 / 30);
  lastT = now;
  try {
    demoMod.step(dt);
    syncPoses();
    renderer.render(scene, camera);
  } catch (err) {
    setStatus("Error: " + err.message);
    console.error(err);
    running = false;
    return;
  }
  requestAnimationFrame(tick);
}

function syncPoses() {
  // poses() returns a dict { id: ([x,y,z], [qx,qy,qz,qw]) }.
  const posesPy = demoMod.poses();
  const poses = posesPy.toJs({ dict_converter: Object.fromEntries });
  posesPy.destroy();
  for (const [idStr, val] of Object.entries(poses)) {
    const id = Number(idStr);
    const mesh = meshes.get(id);
    if (!mesh) continue;
    const p = val[0], q = val[1];
    mesh.position.set(p[0], p[1], p[2]);
    mesh.quaternion.set(q[0], q[1], q[2], q[3]);
  }
}

// ---- UI -------------------------------------------------------------------
SELECTOR.addEventListener("change", () => loadEnv(SELECTOR.value));
RESET.addEventListener("click", () => loadEnv(SELECTOR.value));

window.addEventListener("resize", () => {
  camera.aspect = CANVAS.clientWidth / CANVAS.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(CANVAS.clientWidth, CANVAS.clientHeight, false);
});

bootstrap().catch((err) => {
  setStatus("Fatal: " + err.message);
  console.error(err);
});
