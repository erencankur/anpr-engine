import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { once } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { createServer as createNetServer } from "node:net";
import test from "node:test";
import sharp from "sharp";

async function freePort() {
  const socket = createNetServer();
  socket.listen(0, "127.0.0.1");
  await once(socket, "listening");
  const port = socket.address().port;
  socket.close();
  await once(socket, "close");
  return port;
}

async function waitFor(url, process) {
  for (let index = 0; index < 100; index++) {
    if (process.exitCode !== null) throw new Error("frontend exited before readiness");
    try { const response = await fetch(url); if (response.ok) return response; }
    catch { /* starting */ }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("frontend did not become ready");
}

for (const provider of ["project", "paddleocr", "hyperlpr640"]) {
test(`bilingual raw-image BFF boundary and safe lifecycle (${provider})`, async () => {
  const usesPaddle = provider !== "project";
  const learned = provider === "hyperlpr640";
  const appPort = await freePort();
  const privatePort = await freePort();
  const origin = `http://127.0.0.1:${appPort}`;
  const credential = "test-private-credential-32-characters-minimum";
  const quotaDir = mkdtempSync(join(tmpdir(), "anpr-web-quota-test-"));
  let mode = "prediction";
  let delay = 0;
  const calls = [];
  const cropJpeg = await sharp({ create: { width: 80, height: 20, channels: 3, background: "white" } }).jpeg().toBuffer();
  const cropPreview = { media_type: "image/jpeg", encoded_data: cropJpeg.toString("base64"), width: 80, height: 20, metadata_removed: true };
  const privateServer = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    calls.push({ headers: request.headers, bytes: Buffer.concat(chunks) });
    if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
    if (mode === "timeout") { response.writeHead(504).end(); return; }
    if (mode === "disconnect") { response.destroy(); return; }
    const noPlate = mode === "no-plate";
    const payload = {
      schema_version: "anpr-private-inference-response-v1",
      request_id: request.headers["x-anpr-request-id"],
      admission_id: request.headers["x-anpr-admission-id"],
      status: noPlate ? "no_plate_detected" : mode === "multiple" ? "multiple_detections" : mode === "crop-failed" ? "crop_refinement_failed" : "prediction_available",
      prediction: noPlate || mode === "crop-failed" ? null : mode === "malicious" ? "private/path/SENSITIVE" : "34ABC123",
      prediction_source: usesPaddle ? "paddleocr" : "existing_project_decoder",
      decoding_changed_raw: false,
      structural_validity: noPlate || mode === "crop-failed" ? "not_run" : "valid",
      informational_confidence: noPlate ? null : 0.82,
      detection: { candidate_count: noPlate ? 0 : mode === "multiple" ? 2 : 1, selection_rule: "highest_confidence_then_box_coordinates", box_xyxy_normalized: noPlate ? null : [0.2, 0.2, 0.8, 0.8], informational_confidence: noPlate ? null : 0.91 },
      models: {
        detection: { model_id: "detection-v3-20260827", sha256: "0d5e9b52fae5638e0ef1badafdeb53fe0771d056e43c4953d13cceed520e71ea" },
        recognition: { model_id: "m3-v4-stroke-20260914", sha256: "25b6e4af757b87da75162293a28274e3a0ecbb7f29bc55489b7132e353ad5ff7" }
      },
      crop_policy: "BLUE_BAND_GLYPH_QUAD_V1",
      timings_ms: { total: 8, validation: 1, detection: 4, crop_refinement: 1, recognition: 2 },
      error_code: noPlate ? "NO_PLATE_DETECTED" : mode === "multiple" ? "MULTIPLE_DETECTIONS_TOP1_SELECTED" : mode === "crop-failed" ? "CROP_REFINEMENT_FAILED" : null
    };
    if (learned) {
      payload.models.detection = {model_id:"hyperlpr-y5fu-640-corners-v1",sha256:"0306de937471b87f56eb3f5620815e7e7058f8ab7428e0734fc64627cc4d716c"};
      payload.crop_policy = "HYPERLPR_FOUR_CORNERS_V1";
    }
    if (usesPaddle) payload.models.recognition = {
      model_id: "paddleocr-latin-finetuned-v1",
      sha256: "5fffa451df2efa707bafd3d1da7e94aaebf83c5d365d2c4b8a6d439b4ac48964"
    };
    if (["geometry", "geometry-crossed", "geometry-invalid-format", "geometry-unresolved", "geometry-bad-preview", "geometry-candidate"].includes(mode)) {
      payload.plate_geometry = {
        state: "estimated", method: learned ? "HYPERLPR_FOUR_CORNERS_V1" : "PLATE_FACE_CONTOUR_V1", rectification_applied: true,
        quadrilateral_xy_normalized: [[0.2, 0.3], [0.8, 0.2], [0.8, 0.6], [0.2, 0.7]]
      };
      if (mode === "geometry-crossed") payload.plate_geometry.quadrilateral_xy_normalized.reverse();
      if (mode === "geometry-unresolved") payload.plate_geometry = { state: "unresolved", method: null, rectification_applied: false, quadrilateral_xy_normalized: null };
      if (mode === "geometry-invalid-format") { payload.prediction = "82TR37"; payload.structural_validity = "invalid"; }
      if (mode !== "geometry-unresolved") payload.plate_crops = { rectified: cropPreview, recognition_input: cropPreview };
      if (mode === "geometry-bad-preview") payload.plate_crops.rectified = { ...cropPreview, width: 10000 };
      if (mode === "geometry-candidate") { payload.candidate_prediction = "34ABC123"; payload.prediction = null; payload.status = "recognition_unresolved"; }
    }
    response.setHeader("Content-Type", "application/json");
    response.end(JSON.stringify(payload));
  });
  privateServer.listen(privatePort, "127.0.0.1");
  await once(privateServer, "listening");
  const next = spawn(process.execPath, ["node_modules/next/dist/bin/next", "start", "--hostname", "127.0.0.1", "-p", String(appPort)], {
    cwd: process.cwd(),
    env: { ...process.env, ANPR_RECOGNITION_PROVIDER: usesPaddle ? "paddleocr" : "project", ANPR_DETECTION_PROVIDER: learned ? "hyperlpr640" : "project", ANPR_PUBLIC_ORIGIN: origin, ANPR_PRIVATE_INFERENCE_URL: `http://127.0.0.1:${privatePort}`, ANPR_INTERNAL_SERVICE_CREDENTIAL: credential, ANPR_TRUSTED_PROXY_IP_HEADER: "x-real-ip", ANPR_DAILY_QUOTA_PATH: join(quotaDir, "quota.sqlite"), ANPR_RATE_LIMIT_EXEMPT_IPS: "198.51.100.77" },
    stdio: ["ignore", "pipe", "pipe"]
  });
  let operationalOutput = "";
  for (const stream of [next.stdout, next.stderr]) stream.on("data", (chunk) => { operationalOutput = (operationalOutput + chunk.toString()).slice(-64_000); });
  try {
    const english = await waitFor(origin, next);
    assert.match(english.headers.get("set-cookie") ?? "", /;\s*Secure/i);
    assert.equal((await fetch(`${origin}/api/proxy-proof`)).status, 404);
    const englishHtml = await english.text();
    assert.match(englishHtml, /Tek görüntülü ANPR gösterimi/);
    assert.match(englishHtml, /<html lang="tr"/);
    assert.match(englishHtml, /data-theme="dark"/);
    assert.match(englishHtml, /Açık temaya geç/);
    assert.match(englishHtml, /aria-label="Türkçe" aria-pressed="true"/);
    assert.match(englishHtml, /aria-label="English" aria-pressed="false"/);
    assert.match(englishHtml, /aria-label="Koyu temaya geç"[^>]*aria-pressed="true"/);
    assert.match(englishHtml, /icon\.png/);
    assert.match(englishHtml, /İletişime Geç/);
    assert.match(englishHtml, /https:\/\/vibesofters\.com/);
    const visibleCopy = (html) => html.replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1>/gi, "").replace(/<[^>]+>/g, " ");
    assert.doesNotMatch(visibleCopy(englishHtml), /vibesofters|vision lab|eren\s+cankur|erncnkr@gmail\.com|©/i);
    assert.doesNotMatch(englishHtml, /erncnkr@gmail\.com|mailto:/i);
    assert.equal((englishHtml.match(/href="https:\/\/vibesofters\.com"/g) ?? []).length, 2);
    for (const topic of ["privacy", "responsible-use", "limitations"]) {
      const information = await fetch(`${origin}/information/${topic}`);
      assert.equal(information.status, 200);
      const html = await information.text();
      assert.doesNotMatch(visibleCopy(html), /vibesofters|vision lab|eren\s+cankur|erncnkr@gmail\.com|©/i);
      assert.doesNotMatch(html, /erncnkr@gmail\.com|mailto:/i);
      assert.equal((html.match(/href="https:\/\/vibesofters\.com"/g) ?? []).length, 2);
    }
    assert.doesNotMatch(englishHtml, /Eren Cankur · M\. Sırac Özer/);
    assert.match(englishHtml, /https:\/\/github\.com\/erencankur\/anpr-engine/);
    assert.match(englishHtml, /4 CORNERS/);
    assert.match(englishHtml, /class="hero-logo"/);
    assert.doesNotMatch(englishHtml, /class="plate-study"/);
    assert.match(englishHtml, /id="workspace"/);
    assert.match(englishHtml, /id="how-it-works"/);
    assert.doesNotMatch(englishHtml, /hreflang="tr"/i);
    assert.ok(!englishHtml.includes(credential));
    for (const oldPath of ["/en", "/tr"]) {
      const legacy = await fetch(`${origin}${oldPath}`);
      assert.equal(legacy.url, `${origin}/`);
      assert.match(await legacy.text(), /Tek görüntülü ANPR gösterimi/);
    }
    const light = await fetch(`${origin}/information/privacy`, { headers: { Cookie: "anpr_theme=light" } });
    const lightHtml = await light.text();
    assert.match(lightHtml, /data-theme="light"/);
    assert.match(lightHtml, /Koyu temaya geç/);
    const cookie = english.headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    assert.ok(cookie);
    const image = await sharp({ create: { width: 40, height: 20, channels: 3, background: "#708090" } }).png().toBuffer();
    const headers = { Origin: origin, Referer: `${origin}/`, Cookie: cookie, "X-Real-IP": "127.0.0.1", "Content-Type": "image/png", "X-Privacy-Acknowledged": "true" };
    const post = (body, change = {}) => fetch(`${origin}/api/inference`, { method: "POST", headers: { ...headers, ...change }, body });
    assert.equal((await post(image, { "X-Privacy-Acknowledged": "false" })).status, 400);
    assert.equal((await post(image, { Origin: "https://example.invalid" })).status, 403);
    assert.equal((await post(image, { Cookie: "" })).status, 403);
    assert.equal((await post(image, { "X-Real-IP": "" })).status, 503);
    assert.equal((await post(image, { "X-Real-IP": "127.0.0.1, 198.51.100.1" })).status, 503);
    assert.equal((await post(image, { "X-Real-IP": "not-an-ip" })).status, 503);
    assert.equal((await post(image, { "Content-Type": "image/gif" })).status, 415);
    assert.equal((await post(Buffer.from("not an image"))).status, 415);
    assert.equal((await post(image.subarray(0, image.length - 8))).status, 422);
    assert.equal(calls.length, 0);
    headers.Cookie = (await fetch(origin)).headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    assert.ok(headers.Cookie);
    const success = await post(image);
    assert.equal(success.status, 200);
    assert.match(success.headers.get("cache-control"), /no-store/);
    assert.equal(success.headers.get("access-control-allow-origin"), null);
    const result = await success.json();
    assert.equal(result.status, "unresolved");
    assert.equal(result.recognition.prediction, "34ABC123");
    assert.deepEqual(result.detection.box_xyxy_normalized, [0.2, 0.2, 0.8, 0.8]);
    assert.equal(result.annotated_image, null);
    assert.equal(result.crop_refinement.state, "unavailable");
    assert.equal(calls.length, 1);
    assert.equal(calls[0].headers.authorization, `Bearer ${credential}`);
    assert.equal(calls[0].headers["x-anpr-locale"], "en");
    assert.deepEqual(calls[0].bytes, image);
    assert.ok(!JSON.stringify(result).includes(credential));

    delay = 400;
    const second = post(image);
    while (calls.length < 2) await new Promise((resolve) => setTimeout(resolve, 10));
    const concurrent = await post(image);
    assert.equal(concurrent.status, 409);
    assert.equal((await concurrent.json()).error.code, "SESSION_INFERENCE_ACTIVE");
    assert.equal((await second).status, 200);
    delay = 0;
    assert.equal((await post(image)).status, 200);
    assert.equal((await post(image)).status, 200);
    assert.equal((await post(image)).status, 200);
    assert.equal((await post(image)).status, 429);
    const quota = new DatabaseSync(join(quotaDir, "quota.sqlite"), { readOnly: true });
    const entries = quota.prepare("SELECT key, used FROM quota").all();
    assert.equal(entries.length, 1);
    assert.equal(entries[0].used, 5);
    assert.match(entries[0].key, /^[a-f0-9]{64}$/);
    quota.close();
    const alternateCookie = (await fetch(origin)).headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    assert.ok(alternateCookie);
    assert.equal((await post(image, { Cookie: alternateCookie })).status, 429);
    assert.equal((await post(image, { Cookie: alternateCookie, "X-Forwarded-For": "198.51.100.77" })).status, 429);

    const exemptCookie = (await fetch(origin)).headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    assert.ok(exemptCookie);
    for (let attempt = 0; attempt < 26; attempt += 1) {
      assert.equal((await post(image, { Cookie: exemptCookie, "X-Real-IP": "198.51.100.77" })).status, 200);
    }
    const afterExemption = new DatabaseSync(join(quotaDir, "quota.sqlite"), { readOnly: true });
    assert.equal(afterExemption.prepare("SELECT key, used FROM quota").all().length, 1);
    afterExemption.close();
    for (const geometryMode of ["geometry", "geometry-crossed", "geometry-invalid-format", "geometry-unresolved", "geometry-bad-preview", "geometry-candidate"]) {
      mode = geometryMode;
      const response = await post(image, { Cookie: exemptCookie, "X-Real-IP": "198.51.100.77" });
      if (mode === "geometry-crossed" || mode === "geometry-bad-preview") { assert.equal(response.status, 500); continue; }
      assert.equal(response.status, 200);
      const mapped = await response.json();
      assert.equal(mapped.plate_geometry.state, mode === "geometry-unresolved" ? "unresolved" : "estimated");
      assert.equal(mapped.crop_refinement.state, mode === "geometry-unresolved" ? "unavailable" : "applied");
      if (mode === "geometry-unresolved") assert.equal(mapped.plate_crops, undefined);
      else assert.deepEqual(mapped.plate_crops, {rectified: cropPreview, recognition_input: cropPreview});
      if (mode === "geometry-candidate") {
        assert.equal(mapped.recognition.prediction, null);
        assert.equal(mapped.recognition.candidate_prediction, "34ABC123");
        assert.ok(mapped.warnings.includes("RECOGNITION_UNRESOLVED"));
      }
      if (mode === "geometry-invalid-format") {
        assert.equal(mapped.recognition.prediction, "82TR37");
        assert.equal(mapped.structural_validation, "invalid");
        assert.ok(mapped.warnings.includes("STRUCTURAL_VALIDATION_FAILED"));
      }
    }

    const newCookie = (await fetch(origin)).headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    assert.ok(newCookie);
    mode = "no-plate"; delay = 0;
    const noPlateResponse = await post(image, { Cookie: newCookie, Referer: `${origin}/`, "X-Real-IP": "127.0.0.5" });
    assert.equal(noPlateResponse.status, 200);
    assert.equal((await noPlateResponse.json()).status, "no_plate_detected");
    assert.equal(calls.at(-1).headers["x-anpr-locale"], "en");
    mode = "timeout";
    assert.equal((await post(image, { Cookie: newCookie, "X-Real-IP": "127.0.0.5" })).status, 504);
    mode = "malicious";
    const lastCookie = (await fetch(origin)).headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    const invalidPrivate = await post(image, { Cookie: lastCookie, "X-Real-IP": "127.0.0.6" });
    assert.equal(invalidPrivate.status, 500);
    assert.ok(!JSON.stringify(await invalidPrivate.json()).includes("SENSITIVE"));
    mode = "disconnect";
    const disconnectedCookie = (await fetch(origin)).headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    const disconnected = await post(image, { Cookie: disconnectedCookie, "X-Real-IP": "127.0.0.7" });
    assert.equal(disconnected.status, 503);
    assert.equal((await disconnected.json()).error.code, "MODEL_NOT_READY");
    assert.ok(!operationalOutput.includes(credential));
    assert.ok(!operationalOutput.includes("SENSITIVE"));
    const oversizedCookie = (await fetch(origin)).headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    assert.equal((await post(Buffer.alloc(10 * 1024 * 1024 + 1), { Cookie: oversizedCookie, "X-Real-IP": "127.0.0.4" })).status, 413);
    mode = "multiple";
    const multipleCookie = (await fetch(origin)).headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    const multiple = await post(image, { Cookie: multipleCookie, "X-Real-IP": "127.0.0.2" });
    assert.equal(multiple.status, 200);
    assert.ok((await multiple.json()).warnings.includes("MULTIPLE_DETECTIONS_TOP1_SELECTED"));
    mode = "crop-failed";
    const cropCookie = (await fetch(origin)).headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    const crop = await post(image, { Cookie: cropCookie, "X-Real-IP": "127.0.0.3" });
    assert.equal(crop.status, 200);
    assert.ok((await crop.json()).warnings.includes("CROP_REFINEMENT_FAILED"));
  } finally {
    next.kill("SIGTERM");
    privateServer.close();
    await once(privateServer, "close");
    rmSync(quotaDir, { recursive: true, force: true });
  }
});

}

test("edge proof stays closed by default and reveals no address", async () => {
  const port = await freePort();
  const origin = `http://127.0.0.1:${port}`;
  const next = spawn(process.execPath, ["node_modules/next/dist/bin/next", "start", "--hostname", "127.0.0.1", "-p", String(port)], {
    cwd: process.cwd(),
    env: { ...process.env, ANPR_PUBLIC_ORIGIN: origin, ANPR_PROXY_PROOF_ENABLED: "1", ANPR_TRUSTED_PROXY_IP_HEADER: "" },
    stdio: "ignore"
  });
  try {
    const page = await waitFor(origin, next);
    const html = await page.text();
    assert.match(html, /Görüntü işleme geçici olarak kullanılamıyor/);
    const response = await fetch(`${origin}/api/proxy-proof`, { headers: { "X-Real-IP": "198.51.100.77" } });
    assert.equal(response.status, 200);
    const body = await response.text();
    assert.deepEqual(JSON.parse(body), {
      header_present: true, forged_ipv4_replaced: false, forged_ipv6_replaced: true
    });
    assert.ok(!body.includes("198.51.100.77"));
  } finally { next.kill("SIGTERM"); }
});

test("loopback owner bypasses the daily quota without trusting forwarded IPs", async () => {
  const port = await freePort();
  const privatePort = await freePort();
  const origin = `http://127.0.0.1:${port}`;
  const quotaDir = mkdtempSync(join(tmpdir(), "anpr-local-exemption-test-"));
  let calls = 0;
  const privateServer = createServer(async (request, response) => {
    for await (const chunk of request) void chunk;
    calls++;
    response.writeHead(503).end();
  });
  privateServer.listen(privatePort, "127.0.0.1");
  await once(privateServer, "listening");
  const next = spawn(process.execPath, ["node_modules/next/dist/bin/next", "start", "--hostname", "127.0.0.1", "-p", String(port)], {
    cwd: process.cwd(),
    env: { ...process.env, ANPR_PUBLIC_ORIGIN: origin, ANPR_LOCAL_DIRECT_MODE: "1", ANPR_RATE_LIMIT_EXEMPT_IPS: "127.0.0.1", ANPR_PRIVATE_INFERENCE_URL: `http://127.0.0.1:${privatePort}`, ANPR_INTERNAL_SERVICE_CREDENTIAL: "test-loopback-credential-32-characters-minimum", ANPR_DAILY_QUOTA_PATH: join(quotaDir, "quota.sqlite") },
    stdio: "ignore"
  });
  try {
    const page = await waitFor(origin, next);
    const cookie = page.headers.get("set-cookie")?.match(/anpr_session=[a-f0-9]{64}/)?.[0];
    assert.ok(cookie);
    const image = await sharp({ create: { width: 40, height: 20, channels: 3, background: "white" } }).png().toBuffer();
    for (let index = 0; index < 7; index++) {
      const response = await fetch(`${origin}/api/inference`, { method: "POST", headers: { Origin: origin, Cookie: cookie, "Content-Type": "image/png", "X-Privacy-Acknowledged": "true", "X-Real-IP": "203.0.113.123" }, body: image });
      assert.equal(response.status, 503); // Test stub unavailable, never quota 429.
      assert.equal((await response.json()).error.code, "CAPACITY_UNAVAILABLE");
    }
    assert.equal(calls, 7);
  } finally {
    next.kill("SIGTERM");
    privateServer.close();
    await once(privateServer, "close");
    rmSync(quotaDir, {recursive: true, force: true});
  }
});

test("local HTTP mode keeps HttpOnly session cookies without HTTPS-only flag", async () => {
  const port = await freePort();
  const origin = `http://127.0.0.1:${port}`;
  const next = spawn(process.execPath, ["node_modules/next/dist/bin/next", "start", "--hostname", "127.0.0.1", "-p", String(port)], {
    cwd: process.cwd(),
    env: { ...process.env, ANPR_PUBLIC_ORIGIN: origin, ANPR_LOCAL_DIRECT_MODE: "1" },
    stdio: "ignore"
  });
  try {
    const response = await waitFor(origin, next);
    const cookie = response.headers.get("set-cookie") ?? "";
    assert.match(cookie, /anpr_session=[a-f0-9]{64}/);
    assert.match(cookie, /;\s*HttpOnly/i);
    assert.match(cookie, /SameSite=strict/i);
    assert.doesNotMatch(cookie, /;\s*Secure/i);
  } finally { next.kill("SIGTERM"); }
});
