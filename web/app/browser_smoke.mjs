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
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
});

const page = await browser.newPage();
page.on("console", (msg) => console.log("[page]", msg.type(), msg.text()));
page.on("pageerror", (err) => console.error("[pageerror]", err.message));

await page.goto(URL, { waitUntil: "load" });

console.log("Page loaded. Waiting for bridge to be ready (env-select enabled)…");
await page.waitForFunction(
  () => !document.getElementById("env-select").disabled,
  { timeout: 180000 },
);

console.log("Bridge ready. Triggering Reset…");
await page.click("#boot-env");
await page.waitForFunction(
  () => document.getElementById("info").textContent.includes("action_dim="),
  { timeout: 60000 },
);

const info = await page.$eval("#info", (el) => el.textContent);
console.log("Info:", info);

const status = await page.$eval("#status", (el) => el.textContent);
console.log("Status:", status);

const canvasSize = await page.$eval("#canvas", (el) => ({ w: el.width, h: el.height }));
console.log("Canvas:", canvasSize);

// Save a screenshot for inspection.
await page.screenshot({ path: "/tmp/predicators_browser.png", fullPage: true });
console.log("Screenshot -> /tmp/predicators_browser.png");

await browser.close();
console.log("DONE");
