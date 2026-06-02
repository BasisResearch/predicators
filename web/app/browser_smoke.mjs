// Drives a headless Chromium against the local web/app page and
// waits for the predicators bridge to come online, then asks for a
// reset + render and reports back. Run alongside ./serve.sh.
//
//   web/app/serve.sh &
//   node web/app/browser_smoke.mjs
//
// Exit code 0 on success.

import puppeteer from "puppeteer-core";

const URL = process.env.URL || "http://localhost:8765/app/";
const CHROMIUM = process.env.CHROMIUM || "/usr/bin/chromium";

const browser = await puppeteer.launch({
  executablePath: CHROMIUM,
  headless: "new",
  // Software WebGL via SwiftShader for headless Three.js rendering.
  args: [
    "--no-sandbox", "--disable-dev-shm-usage",
    "--use-gl=swiftshader", "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
  ],
});

const page = await browser.newPage();
page.on("console", (msg) => console.log("[page]", msg.type(), msg.text()));
page.on("pageerror", (err) => console.error("[pageerror]", err.message));
page.on("response", (r) => {
  if (r.status() >= 400) console.error("[HTTP]", r.status(), r.url());
});

const ENV = process.env.ENV || "pybullet_blocks";

await page.goto(URL, { waitUntil: "load" });

console.log("Page loaded. Waiting for bridge to be ready (env-select enabled)…");
await page.waitForFunction(
  () => !document.getElementById("env-select").disabled,
  { timeout: 180000 },
);

console.log(`Bridge ready. Selecting ${ENV} and triggering Reset…`);
await page.select("#env-select", ENV);
await page.click("#boot-env");
await page.waitForFunction(
  () => document.getElementById("info").textContent.includes("action_dim="),
  { timeout: 60000 },
);

const info = await page.$eval("#info", (el) => el.textContent);
console.log("Info:", info);

const status = await page.$eval("#status", (el) => el.textContent);
console.log("Status:", status);

const canvasSize = await page.$eval("#scene-host canvas", (el) => ({ w: el.width, h: el.height }));
console.log("Canvas:", canvasSize);

// Inspect option picker state.
const options = await page.$$eval("#option-select option", (els) =>
  els.map((e) => e.value));
console.log(`Option picker (${options.length} entries):`, options.slice(0, 5),
  options.length > 5 ? "…" : "");

async function execOpt(name, args) {
  console.log(`Triggering Execute on ${name}(${args.join(",")})…`);
  await page.select("#option-select", name);
  // Set arg selects.
  const argSelects = await page.$$("select.opt-arg");
  for (let i = 0; i < argSelects.length && i < args.length; i++) {
    await page.evaluate((el, v) => { el.value = v; el.dispatchEvent(new Event("change")); }, argSelects[i], args[i]);
  }
  await page.click("#execute-option");
  try {
    await page.waitForFunction(
      () => {
        const s = document.getElementById("status").textContent;
        // "X done in N steps." is the final post-playback status.
        // Errors surface as "X: <reason>" without "Playing".
        if (/done in \d+ steps/.test(s)) return true;
        if (/Playing \d+ frames/.test(s)) return false;
        return /^[^:]+:/.test(s) && !s.startsWith("Executing");
      },
      { timeout: 60000 });
    const status = await page.$eval("#status", (el) => el.textContent);
    console.log(`${name} status:`, status);
  } catch (e) {
    const status = await page.$eval("#status", (el) => el.textContent);
    console.log(`${name} timed out. Status:`, status);
  }
}

if (options.length) {
  if (ENV === "pybullet_grow") {
    // jug1 / cup0 are both yellow in the default test task; jug0 is
    // blue and has no matching cup so pouring it never grows anything.
    await execOpt("PickJug", ["robot", "jug1"]);
    await page.screenshot({ path: "/tmp/predicators_browser_pickjug.png", fullPage: true });
    await execOpt("Pour", ["robot", "jug1", "cup0"]);
    await page.screenshot({ path: "/tmp/predicators_browser_pour.png", fullPage: true });
    await execOpt("Place", ["robot", "jug1"]);
    await page.screenshot({ path: "/tmp/predicators_browser_place.png", fullPage: true });
  } else if (ENV === "pybullet_balance") {
    // Oracle plan for seed=0/num_test_tasks=1 (6 blocks split 1:5):
    //   Pick(block5) -> Stack(block0) -> Pick(block4) -> Stack(block5)
    //   -> TurnMachineOn(plate1, plate3)
    // After both moves the count is 3:3 = balanced, and pressing the
    // button while balanced flips the machine on.
    const steps = [
      ["Pick",          ["robby", "block5"]],
      ["Stack",         ["robby", "block0"]],
      ["Pick",          ["robby", "block4"]],
      ["Stack",         ["robby", "block5"]],
      ["TurnMachineOn", ["plate1", "plate3"]],
    ];
    for (const [i, [name, args]] of steps.entries()) {
      await execOpt(name, args);
      await page.screenshot({
        path: `/tmp/predicators_browser_balance_${i+1}_${name}.png`,
        fullPage: true,
      });
    }
  } else {
    await execOpt(options[0], []);
  }
}

const bodies = await page.evaluate(() => window.predBodies());
console.log("Bodies in scene:");
for (const b of bodies) {
  console.log(`  id=${b.id} kind=${b.kind} meshes=${b.meshCount} `
    + `center=[${b.center}] size=[${b.size}]`);
}

// Save a screenshot for inspection.
await page.screenshot({ path: "/tmp/predicators_browser.png", fullPage: true });
console.log("Screenshot -> /tmp/predicators_browser.png");

await browser.close();
console.log("DONE");
