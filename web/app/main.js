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
  const w = sceneHost.clientWidth || 1;
  const h = sceneHost.clientHeight || 1;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
new ResizeObserver(resize).observe(sceneHost);
resize();

// Position the Three.js camera using the env's pybullet camera
// params. PyBullet uses (yaw, pitch, distance) around a target with
// Z-up, yaw measured from +X axis, pitch negative looking down.
function applyEnvCamera(cam) {
  const target = new THREE.Vector3(cam.target[0], cam.target[1], cam.target[2]);
  const yawRad = THREE.MathUtils.degToRad(cam.yaw);
  const pitchRad = THREE.MathUtils.degToRad(cam.pitch);
  // PyBullet's `computeViewMatrixFromYawPitchRoll` constructs the
  // forward vector (where the camera looks) as
  //   forward = (-cos(p)sin(y), cos(p)cos(y), sin(p))
  // and places the camera at `target - forward * distance`.
  const cp = Math.cos(pitchRad), sp = Math.sin(pitchRad);
  const cy = Math.cos(yawRad),   sy = Math.sin(yawRad);
  const forward = new THREE.Vector3(-cp * sy, cp * cy, sp);
  // Bump the camera back a bit. The env-author's distance is tuned
  // for pybullet's 320x240 TinyRenderer view; in Three.js with our
  // larger viewport we want more of the robot in frame.
  const offset = forward.clone().multiplyScalar(-cam.distance * 1.7);
  camera.position.copy(target).add(offset);
  camera.fov = cam.fov;
  camera.updateProjectionMatrix();
  camera.lookAt(target);
  controls.target.copy(target);
  controls.update();
  // Adjust shadow camera so it covers the workspace.
  const m = Math.max(cam.distance * 1.5, 2.0);
  sun.shadow.camera.left = -m; sun.shadow.camera.right = m;
  sun.shadow.camera.top = m; sun.shadow.camera.bottom = -m;
  sun.shadow.camera.far = cam.distance * 8;
  sun.shadow.camera.updateProjectionMatrix();
  // Reposition the sun relative to the target for nicer shadows.
  sun.position.set(target.x + 2, target.y - 2, target.z + 3);
  sun.target.position.copy(target);
  sun.target.updateMatrixWorld();
}

// Fit the camera to the scene by computing a bounding box over all
// rendered objects. Called after the manifest finishes loading.
function fitCameraToScene() {
  const box = new THREE.Box3();
  let hasContents = false;
  // Predicators stashes unused objects at world coords like (10,10).
  // Skip anything more than 5 m from the workspace centroid; the
  // 0.75/0.75 origin is roughly where the Fetch base + table sit.
  const STASH_THRESHOLD = 5.0;
  const tmp = new THREE.Vector3();
  for (const [, b] of bodyMap) {
    if (b.root.userData.isGround) continue;
    b.root.updateMatrixWorld(true);
    const objBox = new THREE.Box3().setFromObject(b.root);
    if (objBox.isEmpty()) continue;
    objBox.getCenter(tmp);
    if (Math.abs(tmp.x) > STASH_THRESHOLD || Math.abs(tmp.y) > STASH_THRESHOLD
        || Math.abs(tmp.z) > STASH_THRESHOLD) {
      log(`  skipping body (stashed at ${tmp.x.toFixed(1)},${tmp.y.toFixed(1)},${tmp.z.toFixed(1)})`);
      continue;
    }
    if (!hasContents) {
      box.copy(objBox);
      hasContents = true;
    } else {
      box.union(objBox);
    }
  }
  if (!hasContents) { log("fitCameraToScene: no content bodies"); return; }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.5;
  log(`fitCameraToScene: center=(${center.x.toFixed(2)},${center.y.toFixed(2)},${center.z.toFixed(2)}) size=(${size.x.toFixed(2)},${size.y.toFixed(2)},${size.z.toFixed(2)}) bodies=${bodyMap.size}`);
  // Place camera at a 45-deg azimuth angle, elevated, distance ~3x radius.
  const distance = Math.max(radius * 3.0, 1.0);
  const dir = new THREE.Vector3(1, -1, 0.9).normalize();
  camera.position.copy(center).addScaledVector(dir, distance);
  camera.lookAt(center);
  controls.target.copy(center);
  controls.update();
  // Adjust shadow camera extents to cover the scene + a margin.
  const m = Math.max(radius * 2.0, 2.0);
  sun.shadow.camera.left = -m; sun.shadow.camera.right = m;
  sun.shadow.camera.top = m; sun.shadow.camera.bottom = -m;
  sun.shadow.camera.far = distance * 4;
  sun.shadow.camera.updateProjectionMatrix();
}

// Per-body state: body_id -> { root: THREE.Object3D, joints: {name: URDFJoint or null} }
let bodyMap = new Map();
// Expose for ad-hoc browser-console debugging: window.predBodies(), etc.
window.predBodies = () => Array.from(bodyMap.entries()).map(([id, b]) => {
  b.root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(b.root);
  const c = box.getCenter(new THREE.Vector3());
  const s = box.getSize(new THREE.Vector3());
  let meshCount = 0;
  b.root.traverse((o) => { if (o.isMesh) meshCount++; });
  return { id, kind: b.kind, meshCount,
           center: [c.x.toFixed(2), c.y.toFixed(2), c.z.toFixed(2)],
           size: [s.x.toFixed(2), s.y.toFixed(2), s.z.toFixed(2)] };
});
window.predScene = scene;

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
    new ColladaLoader(manager).load(path, (dae) => {
      // URDF is Z-up; ColladaLoader unhelpfully rotates Z-up Collada
      // assets to Y-up. Clear that so meshes stay aligned with their
      // URDF link frames.
      dae.scene.rotation.set(0, 0, 0);
      // Blender-exported DAEs include leftover Camera/Lamp nodes at
      // distant world positions; they're invisible but inflate any
      // bbox computation. Drop them.
      const stash = [];
      dae.scene.traverse((o) => {
        if (o.isCamera || o.isLight) stash.push(o);
      });
      for (const o of stash) o.parent?.remove(o);
      done(dae.scene);
    }, undefined, (err) => done(null, err));
  } else {
    console.warn(`URDFLoader: no loader for ${path}`);
    done(null, new Error(`no loader for ${path}`));
  }
};

// Some predicators URDFs (fetch_description/robots/fetch.urdf) use
// undefined XML namespace prefixes like `<sensor:camera>` inside
// `<gazebo>` blocks. Chromium's DOMParser tolerates them; Firefox's
// (and the spec-strict path) aborts and returns a <parsererror>. We
// don't render gazebo plugins anyway, so just strip them. Returns
// the cleaned URDF text.
function sanitizeUrdf(text) {
  return text.replace(/<gazebo[\s\S]*?<\/gazebo>/g, "");
}

function loadUrdfBody(entry) {
  return new Promise((resolve) => {
    // Fetch the URDF ourselves so we can sanitize before parsing,
    // then hand the cleaned text to urdf-loader. No JS-side
    // caching: clone + LoadingManager interactions made it too
    // brittle (cloned-before-meshes-load, concurrent waiters
    // racing on a single onLoad slot, etc). The browser still
    // disk-caches the URDF + mesh fetches across env switches, so
    // re-visits aren't full cold loads.
    const workingPath = entry.url.substring(0, entry.url.lastIndexOf("/") + 1);
    fetch(entry.url).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.text();
    }).then((text) => {
      const cleaned = sanitizeUrdf(text);
      urdfLoader.workingPath = workingPath;
      const robot = urdfLoader.parse(cleaned);
      urdfLoader.workingPath = "";
      if (!robot) {
        log(`URDF returned null for ${entry.url} — using placeholder`);
        makePlaceholder(entry);
        return resolve();
      }
      robot.up.set(0, 0, 1);
      robot.rotation.order = "ZYX";
      // pybullet loadURDF supports a globalScaling kwarg (e.g.
      // pybullet_coffee passes 0.09 for kettle.urdf). urdf-loader
      // doesn't know about it, so apply it client-side.
      if (entry.scale && entry.scale !== 1.0) {
        robot.scale.setScalar(entry.scale);
      }
      robot.traverse((o) => {
        if (o.isMesh) {
          o.castShadow = true;
          o.receiveShadow = true;
        }
      });
      // Apply per-link RGBA from pybullet (overrides the URDF's
      // parsed material colors — e.g. grow's cup/jug URDFs have no
      // <material> blocks, so urdf-loader renders them white; the
      // env tints them via p.changeVisualShape).
      if (entry.link_colors && robot.links) {
        for (const [linkName, rgba] of Object.entries(entry.link_colors)) {
          const link = robot.links[linkName];
          if (!link) continue;
          link.traverse((o) => {
            if (!o.isMesh || !o.material) return;
            // Clone so we don't mutate a shared default material.
            const mat = o.material.clone();
            mat.color?.setRGB?.(rgba[0], rgba[1], rgba[2]);
            mat.opacity = rgba[3];
            mat.transparent = rgba[3] < 1;
            o.material = mat;
          });
        }
      }
      scene.add(robot);
      const joints = {};
      for (const jn of entry.joint_names) {
        joints[jn] = robot.joints?.[jn] || null;
      }
      bodyMap.set(entry.body_id, { root: robot, joints, kind: "urdf" });
      resolve();
    }).catch((err) => {
      log(`URDF load failed (${entry.url}): ${err?.message || err}`);
      makePlaceholder(entry);
      resolve();
    });
  });
}

function makePlaceholder(entry) {
  log(`PLACEHOLDER for body ${entry.body_id} (${entry.name}, kind=${entry.kind}, url=${entry.url || '<none>'})`);
  const root = new THREE.Group();
  let mesh;
  if (entry.name && entry.name.toLowerCase().includes("plane")) {
    // Real-looking ground plane — receives shadows, not pink.
    // Kept modest size (5x5) so it doesn't blow up the scene bbox
    // used by fitCameraToScene().
    const geom = new THREE.PlaneGeometry(5, 5);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xdde2eb, roughness: 0.95, metalness: 0.0,
    });
    mesh = new THREE.Mesh(geom, mat);
    mesh.receiveShadow = true;
    root.userData.isGround = true;
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
    // pybullet's getVisualShapeData returns FULL extents for BOX
    // (not the halfExtents passed to createCollisionShape) — verified
    // against pybullet_ants where food_half_extents=(0.03,0.03,0.03)
    // gets reported as dims=(0.06,0.06,0.06). So no *2 here.
    switch (s.geom) {
      case "box":
        geom = new THREE.BoxGeometry(dims[0], dims[1], dims[2]);
        break;
      case "sphere":
        geom = new THREE.SphereGeometry(dims[0], 24, 16);
        break;
      case "cylinder":
        // pybullet cylinder dims: [length, radius, _].
        geom = new THREE.CylinderGeometry(dims[1], dims[1], dims[0], 24);
        break;
      case "plane":
        geom = new THREE.PlaneGeometry(5, 5);
        break;
      case "mesh":
        // No mesh URL handling for primitives yet; draw a box stand-in.
        geom = new THREE.BoxGeometry(dims[0], dims[1], dims[2]);
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
# Pybullet + gym shim first; these are local emfs wheels that wouldn't
# resolve from PyPI (no Pyodide-targeted pybullet on PyPI; gym 0.26.2
# has no pure-Python wheel — see web/app/gym_shim_setup.py).
await micropip.install("emfs:/tmp/${PYBULLET_WHEEL}")
await micropip.install("emfs:/tmp/${GYM_SHIM_WHEEL}")
# predicators' install_requires is now the env-runtime slim set
# (matches the dep audit in setup.py). Let micropip pull each dep
# transitively. keep_going=True so platform-specific dead-ends
# (pybullet-arm64 on PyPI, version pins that conflict with the
# Pyodide-shipped numpy/matplotlib/pillow) get skipped instead of
# aborting the install.
await micropip.install("emfs:/tmp/${PREDICATORS_WHEEL}",
                       deps=True, keep_going=True)
print("predicators installed")
`);

  // Probe the tarball size up-front so the status reflects reality
  // rather than a stale hard-coded number. Falls back gracefully if
  // the server doesn't return Content-Length.
  const assetsUrl = `${WHEELS_BASE}/${ASSETS_TARBALL}`;
  let assetSizeLabel = "";
  try {
    const head = await fetch(assetsUrl, { method: "HEAD" });
    const len = parseInt(head.headers.get("content-length") || "", 10);
    if (Number.isFinite(len) && len > 0) {
      assetSizeLabel = ` (${(len / 1024 / 1024).toFixed(1)} MB)`;
    }
  } catch { /* fall through, show no size */ }
  setStatus(`Fetching + unpacking env assets${assetSizeLabel}…`);
  const assetsBuf = await fetchBytes(assetsUrl);
  pyodide.FS.mkdirTree("/lib/python3.13/site-packages/predicators/envs");
  await pyodide.runPythonAsync(
    `import os; os.chdir("/lib/python3.13/site-packages/predicators/envs")`);
  pyodide.unpackArchive(assetsBuf, "tar.gz");
  log("Assets unpacked");

  setStatus("Loading bridge…");
  const setupSrc = await (await fetch("./setup.py")).text();
  pyodide.FS.writeFile("/setup.py", setupSrc);
  await pyodide.runPythonAsync("exec(open('/setup.py').read(), globals())");

  setStatus("Ready. Pick an env from the dropdown.");
  envSelect.disabled = false;
  // Reset button stays disabled until the user picks an env, so it
  // can't be ambiguous about "start" vs "reset".
  bootBtn.disabled = true;
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
  if (!envName) return;  // placeholder option selected
  bootBtn.disabled = false;
  setStatus(`Constructing ${envName}…`);
  const t = (() => { const s = performance.now(); return () => ((performance.now() - s) / 1000).toFixed(2); });
  try {
    const tBridge = t();
    const outProxy = await pyodide.runPythonAsync(
      `bridge.reset("${envName}")`);
    const dtBridge = tBridge();
    const info = outProxy.toJs({ dict_converter: Object.fromEntries });
    outProxy.destroy();
    infoEl.textContent = `task=${info.task_idx} objects=${info.num_objects} action_dim=${info.action_dim} bodies=${info.manifest.length}`;
    log(`Reset ${envName} -> ${info.manifest.length} bodies (bridge.reset: ${dtBridge}s)`);

    setStatus("Building Three.js scene from manifest…");
    const tScene = t();
    await buildSceneFromManifest(info.manifest);
    log(`  buildSceneFromManifest: ${tScene()}s`);
    // Initial pose snapshot.
    const stateProxy = await pyodide.runPythonAsync(`bridge.get_all_body_states()`);
    applyFrame(stateProxy.toJs({ dict_converter: Object.fromEntries }));
    stateProxy.destroy();
    if (info.camera) {
      applyEnvCamera(info.camera);
    } else {
      fitCameraToScene();
    }

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
    // Arm the option: bridge stashes the grounded option + initial
    // state. Returns the initial-frame body states so JS can paint
    // step 0 before any pybullet step has happened.
    const argList = JSON.stringify(args);
    const initProxy = await pyodide.runPythonAsync(
      `bridge.begin_option("${name}", ${argList})`);
    const init = initProxy.toJs({ dict_converter: Object.fromEntries });
    initProxy.destroy();
    if (init.error) {
      setStatus(`${name}: ${init.error}`);
      log(`${name}: ${init.error}`);
      return;
    }
    applyFrame(init.initial_frame);

    // Drive the rollout via requestAnimationFrame: one pybullet step
    // per rAF tick, so the renderer paints each frame at its natural
    // pace (no batched "sim everything then play back" choppiness,
    // and no UI freeze during sim).
    let lastSteps = 0;
    let finalError = null;
    await new Promise((resolve) => {
      const stepCall = `bridge.step_option()`;
      function tick() {
        let r;
        try {
          const proxy = pyodide.runPython(stepCall);
          r = proxy.toJs({ dict_converter: Object.fromEntries });
          proxy.destroy();
        } catch (e) {
          log("ERROR during step: " + (e.message || e));
          resolve();
          return;
        }
        // Reconcile mid-rollout bodies *before* applying the frame —
        // otherwise applyFrame can't find the body in bodyMap yet.
        if (r.added_bodies?.length || r.removed_body_ids?.length) {
          // reconcileBodies is async (URDFs load); for primitives it's
          // sync. Don't await — primitives mount synchronously and
          // appear next frame anyway. URDFs will pop in a tick late,
          // which is fine.
          reconcileBodies(r.added_bodies || [],
                          r.removed_body_ids || []);
        }
        applyFrame(r.frame);
        lastSteps = r.steps;
        if (r.done) {
          if (r.color_updates) applyColorUpdates(r.color_updates);
          finalError = r.error;
          resolve();
        } else {
          // Update status periodically (cheap text DOM write) so the
          // user sees progress. requestAnimationFrame yields to the
          // event loop so clicks aren't blocked.
          if (r.steps % 10 === 0) {
            setStatus(`Executing ${name}(${args.join(", ")}) — step ${r.steps}`);
          }
          requestAnimationFrame(tick);
        }
      }
      requestAnimationFrame(tick);
    });

    if (finalError) {
      setStatus(`${name}: ${finalError}`);
      log(`${name}(${args.join(", ")}) stopped after ${lastSteps} steps: ${finalError}`);
    } else {
      setStatus(`${name} done in ${lastSteps} steps.`);
      log(`Executed ${name}(${args.join(", ")}) -> ${lastSteps} steps`);
    }
  } catch (e) {
    setStatus("Execute failed — see log.");
    log("ERROR: " + (e.message || e));
  }
}

function applyColorUpdates(updates) {
  for (const [bidStr, linkMap] of Object.entries(updates)) {
    const bid = Number(bidStr);
    const b = bodyMap.get(bid);
    if (!b) continue;
    if (b.kind === "primitive") {
      // Primitive bodies are a flat group: one child Mesh per shape,
      // in the same order the manifest produced them. We don't track
      // which mesh maps to which link, but our envs all repaint with
      // a single uniform color per body. Push that color onto every
      // child mesh.
      const rgba = Object.values(linkMap)[0];
      if (!rgba) continue;
      b.root.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        o.material.color?.setRGB?.(rgba[0], rgba[1], rgba[2]);
        o.material.opacity = rgba[3];
        o.material.transparent = rgba[3] < 1;
        o.material.needsUpdate = true;
      });
    } else if (b.kind === "urdf") {
      for (const [linkIdxStr, rgba] of Object.entries(linkMap)) {
        // urdf-loader keys links by name, not index. linkIdx -1 is the
        // base. We don't have a robust index→name map, so just paint
        // every mesh in the body — matches the envs that recolor a
        // whole URDF (e.g. grow's cup/jug tints).
        void linkIdxStr;
        b.root.traverse((o) => {
          if (!o.isMesh || !o.material) return;
          o.material.color?.setRGB?.(rgba[0], rgba[1], rgba[2]);
          o.material.opacity = rgba[3];
          o.material.transparent = rgba[3] < 1;
          o.material.needsUpdate = true;
        });
      }
    }
  }
}

async function reconcileBodies(added, removedIds) {
  for (const bid of removedIds) {
    const b = bodyMap.get(bid);
    if (!b) continue;
    scene.remove(b.root);
    bodyMap.delete(bid);
  }
  const urdfPromises = [];
  for (const entry of added) {
    if (entry.kind === "urdf") {
      urdfPromises.push(loadUrdfBody(entry));
    } else {
      makePrimitive(entry);
    }
  }
  await Promise.all(urdfPromises);
}

executeBtn.addEventListener("click", executeOption);
bootBtn.addEventListener("click", resetEnv);
envSelect.addEventListener("change", resetEnv);

boot().catch((e) => {
  setStatus("Boot failed — see log.");
  log("FATAL: " + (e.message || e));
  console.error(e);
});
