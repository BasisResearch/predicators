// predicators-in-the-browser bootstrap.
//
// Pyodide runs predicators + pybullet headless; Three.js renders the
// scene client-side from a manifest extracted by the Python bridge.
// urdf-loader handles real URDF parsing (Fetch robot meshes, plane,
// table, etc.); primitive boxes/spheres are built directly from
// THREE.{Box,Sphere}Geometry. Each option execution returns a list of
// {body_id: {pos, orn, joints}} snapshots that we replay onto the
// scene via requestAnimationFrame.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/pyodide.mjs";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { ColladaLoader } from "three/addons/loaders/ColladaLoader.js";
import URDFLoader from "urdf-loader";

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const infoEl = document.getElementById("info");
const envSelect = document.getElementById("env-select");
const bootBtn = document.getElementById("boot-env");
const sceneHost = document.getElementById("scene-host");

const WHEELS_BASE = "../wheels";
const PYBULLET_WHEEL = "pybullet-3.2.7-cp313-cp313-pyemscripten_2025_0_wasm32.whl";
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

// ----------------------------------------------------------------------------
// Three.js scene setup
// ----------------------------------------------------------------------------

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a2030);

// World convention: pybullet uses Z-up. So does this scene.
scene.up = new THREE.Vector3(0, 0, 1);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(1.7, -1.4, 1.3);
camera.lookAt(0.75, 0.75, 0.5);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
sceneHost.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.75, 0.75, 0.5);
controls.update();

// Lights: soft sky-ish ambient + a directional "sun" that casts shadows.
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 1.05);
sun.position.set(2, -2, 3);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -3; sun.shadow.camera.right = 3;
sun.shadow.camera.top = 3; sun.shadow.camera.bottom = -3;
sun.shadow.camera.near = 0.1; sun.shadow.camera.far = 10;
sun.shadow.bias = -0.0005;
scene.add(sun);

function resize() {
  const w = sceneHost.clientWidth;
  const h = sceneHost.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// Per-body state: body_id -> { root: THREE.Object3D, joints: {name: URDFJoint or null} }
let bodyMap = new Map();

function clearScene() {
  for (const [, b] of bodyMap) {
    scene.remove(b.root);
    b.root.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        for (const m of mats) m.dispose();
      }
    });
  }
  bodyMap.clear();
}

const urdfLoader = new URDFLoader();
// urdf-loader needs the absolute base URL so package:// resolution works.
urdfLoader.packages = (url) => url;
urdfLoader.parseVisual = true;
urdfLoader.parseCollision = false;
// Default loader only handles .stl / .dae. Some envs reference .obj
// (e.g. plane.urdf -> plane.obj) — plug in OBJLoader explicitly.
urdfLoader.loadMeshCb = (path, manager, done) => {
  if (/\.obj$/i.test(path)) {
    new OBJLoader(manager).load(path, (obj) => done(obj),
      undefined, (err) => done(null, err));
  } else if (/\.stl$/i.test(path)) {
    new STLLoader(manager).load(path, (geom) => {
      done(new THREE.Mesh(geom, new THREE.MeshStandardMaterial({
        color: 0xb8c0cc, roughness: 0.6, metalness: 0.1,
      })));
    }, undefined, (err) => done(null, err));
  } else if (/\.dae$/i.test(path)) {
    new ColladaLoader(manager).load(path, (dae) => done(dae.scene),
      undefined, (err) => done(null, err));
  } else {
    console.warn(`URDFLoader: no loader for ${path}`);
    done(null, new Error(`no loader for ${path}`));
  }
};

function loadUrdfBody(entry) {
  return new Promise((resolve) => {
    urdfLoader.load(entry.url, (robot) => {
      if (!robot) {
        log(`URDF returned null for ${entry.url} — using placeholder`);
        makePlaceholder(entry);
        return resolve();
      }
      robot.up.set(0, 0, 1);
      robot.rotation.order = "ZYX";
      robot.traverse((o) => {
        if (o.isMesh) {
          o.castShadow = true;
          o.receiveShadow = true;
        }
      });
      scene.add(robot);
      const joints = {};
      for (const jn of entry.joint_names) {
        joints[jn] = robot.joints?.[jn] || null;
      }
      bodyMap.set(entry.body_id, { root: robot, joints, kind: "urdf" });
      resolve();
    }, undefined, (err) => {
      log(`URDF load failed (${entry.url}): ${err?.message || err}`);
      makePlaceholder(entry);
      resolve();
    });
  });
}

function makePlaceholder(entry) {
  const root = new THREE.Group();
  let mesh;
  if (entry.name && entry.name.toLowerCase().includes("plane")) {
    // Real-looking ground plane — receives shadows, not pink.
    const geom = new THREE.PlaneGeometry(20, 20);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xdde2eb, roughness: 0.95, metalness: 0.0,
    });
    mesh = new THREE.Mesh(geom, mat);
    mesh.receiveShadow = true;
  } else {
    mesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.1, 0.1, 0.1),
      new THREE.MeshStandardMaterial({ color: 0xff00ff, roughness: 0.6 })
    );
  }
  root.add(mesh);
  scene.add(root);
  bodyMap.set(entry.body_id, { root, joints: {}, kind: "placeholder" });
}

function makePrimitive(entry) {
  // entry.shapes is a list of {geom, dims, mesh_url, local_pos,
  //   local_orn, rgba, link}. For multi-shape primitives we group them
  // under one parent.
  const root = new THREE.Group();
  for (const s of entry.shapes) {
    let geom = null;
    const dims = s.dims;
    switch (s.geom) {
      case "box":
        geom = new THREE.BoxGeometry(dims[0]*2, dims[1]*2, dims[2]*2);
        break;
      case "sphere":
        geom = new THREE.SphereGeometry(dims[0], 24, 16);
        break;
      case "cylinder":
        // pybullet cylinder dims: [radius, ?, length] varies; assume
        // [length, radius, _]
        geom = new THREE.CylinderGeometry(dims[1], dims[1], dims[0], 24);
        break;
      case "plane":
        geom = new THREE.PlaneGeometry(5, 5);
        break;
      case "mesh":
        // No mesh URL handling for primitives yet; draw a box stand-in
        // sized to the reported dims.
        geom = new THREE.BoxGeometry(dims[0]*2, dims[1]*2, dims[2]*2);
        break;
      default:
        continue;
    }
    const rgba = s.rgba;
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      opacity: rgba[3],
      transparent: rgba[3] < 0.999,
      roughness: 0.55, metalness: 0.05,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // Local visual frame offset (from URDF visual origin).
    mesh.position.fromArray(s.local_pos);
    mesh.quaternion.fromArray(s.local_orn);
    root.add(mesh);
  }
  scene.add(root);
  bodyMap.set(entry.body_id, { root, joints: {}, kind: "primitive" });
}

async function buildSceneFromManifest(manifest) {
  clearScene();
  const urdfPromises = [];
  for (const entry of manifest) {
    try {
      if (entry.kind === "urdf") {
        urdfPromises.push(loadUrdfBody(entry));
      } else {
        makePrimitive(entry);
      }
    } catch (e) {
      log(`Skipping body ${entry.body_id} (${entry.name}): ${e.message}`);
      makePlaceholder(entry);
    }
  }
  await Promise.all(urdfPromises);
}

function applyFrame(frame) {
  for (const [idStr, state] of Object.entries(frame)) {
    const id = Number(idStr);
    const b = bodyMap.get(id);
    if (!b) continue;
    b.root.position.fromArray(state.pos);
    b.root.quaternion.fromArray(state.orn);
    if (b.kind === "urdf") {
      for (const [jname, angle] of Object.entries(state.joints)) {
        const j = b.joints[jname];
        if (j && typeof j.setJointValue === "function") {
          j.setJointValue(angle);
        }
      }
    }
  }
}

let rafRunning = false;
function renderLoop() {
  controls.update();
  renderer.render(scene, camera);
  if (rafRunning) requestAnimationFrame(renderLoop);
}
function startRenderLoop() {
  if (rafRunning) return;
  rafRunning = true;
  requestAnimationFrame(renderLoop);
}

// ----------------------------------------------------------------------------
// Pyodide bootstrap
// ----------------------------------------------------------------------------

async function boot() {
  setStatus("Loading Pyodide runtime…");
  pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/",
    stdout: log, stderr: log,
  });
  log("Pyodide loaded");

  setStatus("Loading base packages (numpy, matplotlib, pillow)…");
  await pyodide.loadPackage(["micropip", "numpy", "matplotlib", "pillow"]);

  setStatus("Staging wheels into Pyodide FS…");
  pyodide.FS.writeFile(`/tmp/${PYBULLET_WHEEL}`, await fetchBytes(`${WHEELS_BASE}/${PYBULLET_WHEEL}`));
  pyodide.FS.writeFile(`/tmp/${GYM_SHIM_WHEEL}`, await fetchBytes(`${WHEELS_BASE}/${GYM_SHIM_WHEEL}`));
  pyodide.FS.writeFile(`/tmp/${PREDICATORS_WHEEL}`, await fetchBytes(`${WHEELS_BASE}/${PREDICATORS_WHEEL}`));

  setStatus("Installing wheels…");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:/tmp/${PYBULLET_WHEEL}")
await micropip.install("emfs:/tmp/${GYM_SHIM_WHEEL}")
for pkg in ["dill", "tabulate", "pyperplan", "colorlog", "imageio", "gymnasium"]:
    try:
        await micropip.install(pkg, deps=True, keep_going=True)
    except Exception as e:
        print(f"{pkg} FAILED: {e}")
await micropip.install("emfs:/tmp/${PREDICATORS_WHEEL}", deps=False)
print("predicators installed")
`);

  setStatus("Fetching + unpacking env assets (~30 MB)…");
  const assetsBuf = await fetchBytes(`${WHEELS_BASE}/${ASSETS_TARBALL}`);
  pyodide.FS.mkdirTree("/lib/python3.13/site-packages/predicators/envs");
  await pyodide.runPythonAsync(
    `import os; os.chdir("/lib/python3.13/site-packages/predicators/envs")`);
  pyodide.unpackArchive(assetsBuf, "tar.gz");
  log("Assets unpacked");

  setStatus("Loading bridge…");
  const setupSrc = await (await fetch("./setup.py")).text();
  pyodide.FS.writeFile("/setup.py", setupSrc);
  await pyodide.runPythonAsync("exec(open('/setup.py').read(), globals())");

  setStatus("Ready. Pick an env and hit Reset.");
  envSelect.disabled = false;
  bootBtn.disabled = false;
  startRenderLoop();
}

// ----------------------------------------------------------------------------
// Env reset + option execution
// ----------------------------------------------------------------------------

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
    const outProxy = await pyodide.runPythonAsync(
      `bridge.reset("${envName}")`);
    const info = outProxy.toJs({ dict_converter: Object.fromEntries });
    outProxy.destroy();
    infoEl.textContent = `task=${info.task_idx} objects=${info.num_objects} action_dim=${info.action_dim} bodies=${info.manifest.length}`;
    log(`Reset ${envName} -> ${info.manifest.length} bodies`);

    setStatus("Building Three.js scene from manifest…");
    await buildSceneFromManifest(info.manifest);
    // Initial pose snapshot.
    const stateProxy = await pyodide.runPythonAsync(`bridge.get_all_body_states()`);
    applyFrame(stateProxy.toJs({ dict_converter: Object.fromEntries }));
    stateProxy.destroy();

    const optsProxy = await pyodide.runPythonAsync(`bridge.list_options()`);
    const objsProxy = await pyodide.runPythonAsync(`bridge.list_objects()`);
    currentOptions = optsProxy.toJs({ dict_converter: Object.fromEntries });
    currentObjects = objsProxy.toJs({ dict_converter: Object.fromEntries });
    optsProxy.destroy(); objsProxy.destroy();
    populateOptionPicker();

    setStatus("Env ready. Drag canvas to orbit. Pick an option to execute.");
    optionRow.style.display = "";
  } catch (e) {
    setStatus("Reset failed — see log.");
    log("ERROR: " + (e.message || e));
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
    const outProxy = await pyodide.runPythonAsync(
      `bridge.execute_option("${name}", ${argList})`);
    const result = outProxy.toJs({ dict_converter: Object.fromEntries });
    outProxy.destroy();
    log(`Executed ${name}(${args.join(", ")}) -> ${result.steps} steps, ${result.frames.length} frames`);
    setStatus(`${name}: ${result.steps} steps. Playing ${result.frames.length} frames…`);
    await playFrames(result.frames);
    setStatus(`${name} done in ${result.steps} steps.`);
  } catch (e) {
    setStatus("Execute failed — see log.");
    log("ERROR: " + (e.message || e));
  }
}

async function playFrames(frames, intervalMs = 50) {
  for (const f of frames) {
    applyFrame(f);
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

executeBtn.addEventListener("click", executeOption);
bootBtn.addEventListener("click", resetEnv);

boot().catch((e) => {
  setStatus("Boot failed — see log.");
  log("FATAL: " + (e.message || e));
  console.error(e);
});
