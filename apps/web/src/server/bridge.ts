import { randomBytes } from "node:crypto";
import type {
  AnprImageInferenceResponseV1,
  AnprPrivateInferenceResponseV1
} from "../../../../packages/api-contract/generated/typescript/contracts";
import { ImageError, type MediaType } from "./image";

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function jpegSize(bytes: Buffer): [number, number] | null {
  let offset = 2;
  while (offset + 4 <= bytes.length) {
    if (bytes[offset++] !== 0xff) return null;
    while (bytes[offset] === 0xff) offset++;
    const marker = bytes[offset++];
    if (marker === 0xda || marker === 0xd9 || offset + 2 > bytes.length) return null;
    const length = bytes.readUInt16BE(offset);
    if (length < 2 || offset + length > bytes.length) return null;
    if (marker === 0xc0 || marker === 0xc2) {
      if (length < 8) return null;
      return [bytes.readUInt16BE(offset + 5), bytes.readUInt16BE(offset + 3)];
    }
    offset += length;
  }
  return null;
}

export function privateResult(value: unknown, requestId: string, admissionId: string): AnprPrivateInferenceResponseV1 {
  if (!isRecord(value) || value.schema_version !== "anpr-private-inference-response-v1" ||
      value.request_id !== requestId || value.admission_id !== admissionId || !isRecord(value.detection) ||
      !isRecord(value.models) || !isRecord(value.timings_ms) ||
      !["prediction_available", "recognition_unresolved", "no_plate_detected", "multiple_detections", "crop_refinement_failed"].includes(String(value.status))) {
    throw new ImageError("INFERENCE_FAILED");
  }
  const models = value.models;
  const provider = process.env.ANPR_RECOGNITION_PROVIDER ?? "project";
  if (!["project", "paddleocr"].includes(provider)) throw new ImageError("MODEL_NOT_READY");
  const paddle = provider === "paddleocr";
  const detector = process.env.ANPR_DETECTION_PROVIDER ?? "project";
  if (!["project", "hyperlpr", "hyperlpr640"].includes(detector)) throw new ImageError("MODEL_NOT_READY");
  const learned = detector !== "project";
  if (!isRecord(models.detection) || !isRecord(models.recognition) ||
      models.detection.model_id !== (learned ? (detector === "hyperlpr640" ? "hyperlpr-y5fu-640-corners-v1" : "hyperlpr-y5fu-320-corners-v1") : "detection-v3-20260827") ||
      models.recognition.model_id !== (paddle ? "paddleocr-latin-finetuned-v1" : "m3-v4-stroke-20260914") ||
      value.prediction_source !== (paddle ? "paddleocr" : "existing_project_decoder") ||
      models.detection.sha256 !== (learned ? (detector === "hyperlpr640" ? "0306de937471b87f56eb3f5620815e7e7058f8ab7428e0734fc64627cc4d716c" : "2a985dc63a5cc947ec36d18503d6fc0fd54525b9dba17f2fe29a71e86b44456d") : "0d5e9b52fae5638e0ef1badafdeb53fe0771d056e43c4953d13cceed520e71ea") ||
      models.recognition.sha256 !== (paddle ? "5fffa451df2efa707bafd3d1da7e94aaebf83c5d365d2c4b8a6d439b4ac48964" : "25b6e4af757b87da75162293a28274e3a0ecbb7f29bc55489b7132e353ad5ff7")) {
    throw new ImageError("INFERENCE_FAILED");
  }
  const box = value.detection.box_xyxy_normalized;
  if (value.candidate_prediction !== undefined && (value.prediction !== null ||
      typeof value.candidate_prediction !== "string" || !/^[0-9A-Z ]{1,16}$/.test(value.candidate_prediction) ||
      value.informational_confidence === null)) throw new ImageError("INFERENCE_FAILED");
  if (value.plate_crops !== undefined) {
    const crops = value.plate_crops;
    if (!isRecord(crops) || !isRecord(value.plate_geometry) || value.plate_geometry.state !== "estimated" ||
        Object.keys(crops).some((key) => !["rectified", "recognition_input"].includes(key))) throw new ImageError("INFERENCE_FAILED");
    for (const key of ["rectified", "recognition_input"]) {
      const preview = crops[key];
      if (preview === null) continue;
      if (!isRecord(preview) || preview.media_type !== "image/jpeg" || preview.metadata_removed !== true ||
          Object.keys(preview).some((field) => !["media_type", "encoded_data", "width", "height", "metadata_removed"].includes(field)) ||
          typeof preview.encoded_data !== "string" || preview.encoded_data.length < 8 || preview.encoded_data.length > 16000 ||
          preview.encoded_data.length % 4 !== 0 || !/^[A-Za-z0-9+/]+={0,2}$/.test(preview.encoded_data) ||
          typeof preview.width !== "number" || !Number.isInteger(preview.width) || preview.width < 1 || preview.width > 640 ||
          typeof preview.height !== "number" || !Number.isInteger(preview.height) || preview.height < 1 || preview.height > 192) throw new ImageError("INFERENCE_FAILED");
      const bytes = Buffer.from(preview.encoded_data, "base64");
      if (bytes[0] !== 0xff || bytes[1] !== 0xd8 || bytes.at(-2) !== 0xff || bytes.at(-1) !== 0xd9) throw new ImageError("INFERENCE_FAILED");
      const dimensions = jpegSize(bytes);
      if (!dimensions || dimensions[0] !== preview.width || dimensions[1] !== preview.height) throw new ImageError("INFERENCE_FAILED");
    }
  }
  if (value.plate_geometry !== undefined) {
    const geometry = value.plate_geometry;
    if (!isRecord(geometry) || Object.keys(geometry).some((key) => !["state", "method", "quadrilateral_xy_normalized", "rectification_applied"].includes(key))) throw new ImageError("INFERENCE_FAILED");
    const quad = geometry.quadrilateral_xy_normalized;
    if (geometry.state === "estimated") {
      if (box === null || geometry.method !== (learned ? "HYPERLPR_FOUR_CORNERS_V1" : "PLATE_FACE_CONTOUR_V1") || geometry.rectification_applied !== true ||
          !Array.isArray(quad) || quad.length !== 4 || !quad.every((point) => Array.isArray(point) && point.length === 2 && point.every((n) => typeof n === "number" && Number.isFinite(n) && n >= 0 && n <= 1))) throw new ImageError("INFERENCE_FAILED");
      // Clockwise image-space convex polygon: reject crossed, duplicate or degenerate corners.
      const points = quad as number[][];
      if (points.some((a, i) => { const b = points[(i + 1) % 4]; const c = points[(i + 2) % 4]; return (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]) <= 1e-10; })) throw new ImageError("INFERENCE_FAILED");
    } else if (!["unresolved", "not_run"].includes(String(geometry.state)) || quad !== null || geometry.method !== null || geometry.rectification_applied !== false) throw new ImageError("INFERENCE_FAILED");
  }
  const timings = value.timings_ms as Record<string, unknown>;
  const confidence = (item: unknown) => item === null || (typeof item === "number" && Number.isFinite(item) && item >= 0 && item <= 1);
  if (!(box === null || (Array.isArray(box) && box.length === 4 && box.every((item) => typeof item === "number" && Number.isFinite(item) && item >= 0 && item <= 1) && box[0] <= box[2] && box[1] <= box[3])) ||
      !(value.prediction === null || (typeof value.prediction === "string" && /^[0-9A-Z ]{1,16}$/.test(value.prediction))) ||
      !confidence(value.informational_confidence) || !confidence(value.detection.informational_confidence) ||
      ["total", "validation", "detection", "crop_refinement", "recognition"].some((key) => { const item = timings[key]; return typeof item !== "number" || !Number.isFinite(item) || item < 0 || item > 120_000; }) ||
      !["valid", "invalid", "not_run"].includes(String(value.structural_validity)) ||
      value.crop_policy !== (learned ? "HYPERLPR_FOUR_CORNERS_V1" : "BLUE_BAND_GLYPH_QUAD_V1")) throw new ImageError("INFERENCE_FAILED");
  return value as unknown as AnprPrivateInferenceResponseV1;
}

export function publicResult(privateValue: AnprPrivateInferenceResponseV1): AnprImageInferenceResponseV1 {
  const found = privateValue.detection.box_xyxy_normalized !== null;
  const noPlate = privateValue.status === "no_plate_detected";
  const warnings: AnprImageInferenceResponseV1["warnings"][number][] = [];
  if (privateValue.structural_validity === "invalid") warnings.push("STRUCTURAL_VALIDATION_FAILED");
  if (privateValue.status === "multiple_detections") warnings.push("MULTIPLE_DETECTIONS_TOP1_SELECTED");
  if (privateValue.status === "crop_refinement_failed") warnings.push("CROP_REFINEMENT_FAILED");
  if (privateValue.status === "recognition_unresolved") warnings.push("RECOGNITION_UNRESOLVED");
  return {
    schema_version: "anpr-image-inference-response-v1",
    request_id: privateValue.request_id,
    // A model prediction is never owner-resolved until the active-page review.
    status: noPlate ? "no_plate_detected" : "unresolved",
    models: privateValue.models,
    detection: {
      state: found ? "found" : "not_found",
      box_xyxy_normalized: privateValue.detection.box_xyxy_normalized,
      confidence: privateValue.detection.informational_confidence
    },
    ...(privateValue.plate_geometry ? { plate_geometry: privateValue.plate_geometry } : {}),
    ...(privateValue.plate_crops ? { plate_crops: privateValue.plate_crops } : {}),
    crop_refinement: privateValue.plate_geometry?.state === "estimated"
      ? { state: "applied", method: privateValue.plate_geometry.method }
      : { state: noPlate ? "not_run" : "unavailable", method: found ? privateValue.crop_policy : null },
    recognition: { prediction: privateValue.prediction, confidence: privateValue.informational_confidence,
      ...(privateValue.candidate_prediction !== undefined ? {candidate_prediction: privateValue.candidate_prediction} : {}) },
    structural_validation: privateValue.structural_validity,
    human_review_required: true,
    annotated_image: null,
    timings_ms: privateValue.timings_ms,
    warnings
  };
}

export async function callPrivate(bytes: Buffer, type: MediaType, locale: "en" | "tr", requestId: string): Promise<AnprImageInferenceResponseV1> {
  const rawUrl = process.env.ANPR_PRIVATE_INFERENCE_URL;
  const credential = process.env.ANPR_INTERNAL_SERVICE_CREDENTIAL;
  if (!rawUrl || !credential || credential.length < 32) throw new ImageError("MODEL_NOT_READY");
  let url: URL;
  try { url = new URL("/api/v1/inference", rawUrl); } catch { throw new ImageError("MODEL_NOT_READY"); }
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new ImageError("MODEL_NOT_READY");
  const admissionId = randomBytes(24).toString("base64url");
  let response: Response;
  const deadline = AbortSignal.timeout(125_000);
  try {
    response = await fetch(url, {
      method: "POST",
      cache: "no-store",
      signal: deadline,
      headers: {
        "Content-Type": type,
        "Content-Length": String(bytes.length),
        "Authorization": `Bearer ${credential}`,
        "X-ANPR-Request-ID": requestId,
        "X-ANPR-Admission-ID": admissionId,
        "X-ANPR-Locale": locale,
        "X-ANPR-Privacy-Acknowledged": "true"
      },
      body: bytes as unknown as BodyInit
    });
  } catch {
    throw new ImageError(deadline.aborted ? "INFERENCE_DEADLINE_EXCEEDED" : "MODEL_NOT_READY");
  }
  if (!response.ok) {
    const mapping: Record<number, ConstructorParameters<typeof ImageError>[0]> = {
      409: "SESSION_INFERENCE_ACTIVE", 413: "IMAGE_TOO_LARGE", 415: "UNSUPPORTED_MEDIA_TYPE",
      422: "IMAGE_DECODE_FAILED", 429: "RATE_LIMITED", 503: "CAPACITY_UNAVAILABLE", 504: "INFERENCE_DEADLINE_EXCEEDED"
    };
    throw new ImageError(mapping[response.status] ?? "INFERENCE_FAILED");
  }
  const declaredLength = Number(response.headers.get("content-length") ?? 0);
  if (!Number.isFinite(declaredLength) || declaredLength < 0 || declaredLength > 64_000) throw new ImageError("INFERENCE_FAILED");
  if (!response.body) throw new ImageError("INFERENCE_FAILED");
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for await (const chunk of response.body) {
      size += chunk.byteLength;
      if (size > 64_000) {
        throw new ImageError("INFERENCE_FAILED");
      }
      chunks.push(chunk);
    }
  } catch (error) {
    if (error instanceof ImageError) throw error;
    throw new ImageError(deadline.aborted ? "INFERENCE_DEADLINE_EXCEEDED" : "MODEL_NOT_READY");
  }
  const text = Buffer.concat(chunks).toString("utf8");
  let parsed: unknown;
  try { parsed = JSON.parse(text); } catch { throw new ImageError("INFERENCE_FAILED"); }
  return publicResult(privateResult(parsed, requestId, admissionId));
}
