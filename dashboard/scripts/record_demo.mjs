// Records a scripted walk through the dashboard (or any page) with headless Chrome's screencast, for the demo
// video. No extra packages: Node >= 22 (global WebSocket) and the DevTools protocol, as scripts/screenshot.mjs.
//
//   node scripts/record_demo.mjs <storyboard.json> <out_dir>
//
// storyboard.json: {"width": 1920, "height": 1080, "badge": "...", "steps": [step, ...]}; a step is
//   {"url": "...", "hold": ms, "caption": "...", "settle": ms}       open a page, wait `settle`, record `hold`
//   {"url": ".../#search", "type": "white SUV", ...}                  ... and type a query into the search box
//   {"identity": "<api base>", "url": "<base>/#live", ...}            open the identity with the most tracks
// Only the `hold` windows are recorded (page loads are cut). Writes out_dir/frames/NNNNN.jpg and
// out_dir/frames.ffconcat (each frame with its duration) for ffmpeg.
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [boardPath, outDir] = process.argv.slice(2);
if (!boardPath || !outDir) {
  console.error("usage: node scripts/record_demo.mjs <storyboard.json> <out_dir>");
  process.exit(2);
}
const board = JSON.parse(readFileSync(boardPath, "utf8"));
const W = board.width ?? 1920, H = board.height ?? 1080;
mkdirSync(join(outDir, "frames"), { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(process.env.CHROME ?? "google-chrome", [
  "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars", "--allow-file-access-from-files",
  `--remote-debugging-port=${port}`, `--window-size=${W},${H}`, "about:blank",
], { stdio: "ignore" });

const CAPTION_JS = (caption, badge) => `(() => {
  for (const id of ["demo-caption", "demo-badge"]) document.getElementById(id)?.remove();
  const mk = (id, css, text) => { const d = document.createElement("div"); d.id = id; d.textContent = text;
    d.style.cssText = css; document.body.appendChild(d); };
  const font = "font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;";
  if (${JSON.stringify(caption)}) mk("demo-caption", font + "position:fixed;left:50%;bottom:34px;transform:translateX(-50%);" +
    "max-width:1560px;padding:14px 26px;border-radius:12px;background:rgba(11,11,11,.84);color:#fff;" +
    "font-size:27px;font-weight:600;line-height:1.3;text-align:center;z-index:2147483647;box-shadow:0 4px 24px rgba(0,0,0,.3)",
    ${JSON.stringify(caption)});
  if (${JSON.stringify(badge)}) mk("demo-badge", font + "position:fixed;right:18px;top:14px;padding:6px 12px;border-radius:8px;" +
    "background:rgba(11,11,11,.72);color:#fff;font-size:15px;font-weight:500;z-index:2147483647", ${JSON.stringify(badge)});
})()`;

try {
  let target;
  for (let i = 0; i < 80 && !target; i++) {
    try {
      target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page");
    } catch {
      await sleep(100);
    }
  }
  if (!target) throw new Error("Chrome did not start");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const pending = new Map();
  const frames = [];        // {file, t} in recording time (ms since the first window started)
  let recording = false, clock = 0, windowStart = 0;
  const send = (method, params = {}) => new Promise((r) => {
    const n = ++id; pending.set(n, r); ws.send(JSON.stringify({ id: n, method, params }));
  });
  const save = (b64) => {
    const file = `${String(frames.length).padStart(5, "0")}.jpg`;
    writeFileSync(join(outDir, "frames", file), Buffer.from(b64, "base64"));
    frames.push({ file, t: clock + (performance.now() - windowStart) });
  };
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); return; }
    if (msg.method === "Page.screencastFrame") {
      if (recording) save(msg.params.data);
      ws.send(JSON.stringify({ id: ++id, method: "Page.screencastFrameAck", params: { sessionId: msg.params.sessionId } }));
    }
  };
  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: board.theme ?? "light" }] });
  await send("Page.startScreencast", { format: "jpeg", quality: 88, maxWidth: W, maxHeight: H, everyNthFrame: 1 });
  const key = async (k, code, vk, text) => {
    await send("Input.dispatchKeyEvent", { type: "keyDown", key: k, code, windowsVirtualKeyCode: vk, text });
    await send("Input.dispatchKeyEvent", { type: "keyUp", key: k, code, windowsVirtualKeyCode: vk });
  };

  for (const [i, step] of board.steps.entries()) {
    let url = step.url;
    if (step.identity) { // the single-camera identity with the most tracks: its "journey" drawer (the demo's
      // cameras show unrelated scenes, so a cross-camera link there could only be a false match)
      const list = await (await fetch(`${step.identity}/api/identities?min_cameras=1&limit=2000`)).json();
      const best = list.filter((x) => x.cameras.length === 1).sort((a, b) => b.n_tracks - a.n_tracks)[0];
      if (best) url = url.replace("#", `?identity=${best.id}#`);
      console.log(`step ${i}: identity ${best?.id} (${best?.n_tracks} tracks, cameras ${best?.cameras})`);
    }
    if (url.startsWith("http")) { // a fresh page load per step (a same-URL hash change would keep the old state)
      const [path, hash = ""] = url.split("#");
      url = `${path}${path.includes("?") ? "&" : "?"}demo_step=${i}${hash ? `#${hash}` : ""}`;
    }
    await send("Page.navigate", { url });
    await sleep(step.settle ?? 2500);
    await send("Runtime.evaluate", { expression: CAPTION_JS(step.caption ?? "", board.badge ?? "") });
    await sleep(150);
    windowStart = performance.now();
    recording = true;
    const shot = await send("Page.captureScreenshot", { format: "jpeg", quality: 88 });
    save(shot.result.data);
    if (step.type) { // type into the search box like a person would, then submit
      await sleep(700);
      const found = await send("Runtime.evaluate", { returnByValue: true, expression:
        "(() => { const i = document.querySelector('form.searchbox input'); if (!i) return false; i.focus(); i.select(); return true; })()" });
      if (!found.result?.result?.value) throw new Error(`step ${i}: no search box on ${url}`);
      for (const ch of step.type) {
        await send("Input.insertText", { text: ch });
        await sleep(70);
      }
      await sleep(400);
      await key("Enter", "Enter", 13, "\r");
    }
    const left = step.hold - (performance.now() - windowStart);
    if (left > 0) await sleep(left);
    recording = false;
    clock += performance.now() - windowStart;
    console.log(`step ${i}: ${url.slice(0, 80)} ${Math.round(step.hold / 1000)} s, ${frames.length} frames so far`);
  }
  await send("Page.stopScreencast");
  // ffconcat: each frame lasts until the next one (the last until the end of the recording)
  const lines = ["ffconcat version 1.0"];
  frames.forEach((f, k) => {
    const end = k + 1 < frames.length ? frames[k + 1].t : clock;
    lines.push(`file 'frames/${f.file}'`, `duration ${Math.max(end - f.t, 1) / 1000}`);
  });
  lines.push(`file 'frames/${frames[frames.length - 1].file}'`);
  writeFileSync(join(outDir, "frames.ffconcat"), lines.join("\n") + "\n");
  console.log(`recorded ${(clock / 1000).toFixed(1)} s, ${frames.length} frames -> ${outDir}`);
  ws.close();
} finally {
  chrome.kill();
}
