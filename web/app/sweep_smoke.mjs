// Boots a headless browser + Pyodide once, then resets the bridge for
// each env in the dropdown in turn. Much faster than running
// browser_smoke.mjs per env (Pyodide cold-boot is the dominant cost).
//
//   web/app/serve.sh &
//   node web/app/sweep_smoke.mjs
//
// Exits 0 iff every env reaches `action_dim=...` (i.e. the bridge
// returned a non-error reset). Saves /tmp/predicators_sweep_<env>.png
// after each successful reset.

import puppeteer from "puppeteer-core";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const URL = process.env.URL || "http://localhost:8765/app/";
const CHROMIUM = process.env.CHROMIUM || "/usr/bin/chromium";
const PER_ENV_TIMEOUT_MS = Number(process.env.PER_ENV_TIMEOUT_MS || 60000);

// Parse the dropdown from index.html so the env list stays in lockstep
// with what users see in the UI. Strips the trailing " (no options)"
// label hints — we only need the option `value`.
const HERE = dirname(fileURLToPath(import.meta.url));
const HTML = readFileSync(resolve(HERE, "index.html"), "utf8");
const ENVS = Array.from(HTML.matchAll(/value="(pybullet_[a-z_]+)"/g))
  .map((m) => m[1]);
if (!ENVS.length) {
  console.error("No pybullet envs found in index.html dropdown");
  process.exit(2);
}
console.log(`Sweeping ${ENVS.length} envs:`, ENVS.join(" "));

const browser = await puppeteer.launch({
  executablePath: CHROMIUM,
  headless: "new",
  args: [
    "--no-sandbox", "--disable-dev-shm-usage",
    "--use-gl=swiftshader", "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
  ],
});

const page = await browser.newPage();
page.on("pageerror", (err) => console.error("[pageerror]", err.message));
// Surface Python tracebacks (logged via console.log from Pyodide).
page.on("console", (msg) => {
  const t = msg.text();
  if (t.includes("ERROR") || t.includes("Traceback")
      || t.includes("ModuleNotFound") || t.includes("FATAL")) {
    console.log("[page]", msg.type(), t);
  }
});

await page.goto(URL, { waitUntil: "load" });

console.log("Waiting for Pyodide bridge…");
await page.waitForFunction(
  () => !document.getElementById("env-select").disabled,
  { timeout: 180000 },
);
console.log("Bridge ready.");

const results = [];
for (const env of ENVS) {
  process.stdout.write(`  ${env.padEnd(36)}`);
  // Reset the info text so the next waitForFunction polls *this* run.
  await page.evaluate(() => {
    document.getElementById("info").textContent = "";
  });
  await page.select("#env-select", env);
  await page.click("#boot-env");

  let info = null;
  let ok = false;
  try {
    await page.waitForFunction(
      () => document.getElementById("info").textContent.includes("action_dim="),
      { timeout: PER_ENV_TIMEOUT_MS },
    );
    info = await page.$eval("#info", (el) => el.textContent);
    ok = true;
  } catch (e) {
    info = await page.$eval("#status", (el) => el.textContent)
      .catch(() => "<no status>");
  }
  results.push({ env, ok, info });
  console.log(ok ? `OK    ${info}` : `FAIL  ${info}`);

  if (ok) {
    await page.screenshot({
      path: `/tmp/predicators_sweep_${env}.png`,
      fullPage: false,
    }).catch(() => {});
  }
}

await browser.close();

const failed = results.filter((r) => !r.ok);
console.log("");
console.log(`=== summary: ${results.length - failed.length}/${results.length} OK ===`);
if (failed.length) {
  for (const r of failed) console.log(`  FAIL ${r.env}: ${r.info}`);
  process.exit(1);
}
process.exit(0);
