"use client";
/* eslint-disable @next/next/no-img-element -- The current-session blob URL cannot use server image optimization. */

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import type { ChangeEvent, FormEvent } from "react";
import { useLocale } from "./site-chrome";
import type {
  AnprApiErrorV1, AnprImageInferenceResponseV1, AnprSessionResultV1, Notice
} from "../../../../packages/api-contract/generated/typescript/contracts";

type Review = "OWNER_CONFIRMED_PROJECT_MODEL" | "OWNER_CORRECTED" | "UNRESOLVED" | null;
const LIMIT = 10 * 1024 * 1024;
const CLIENT_DEADLINE_MS = 140_000;
const labels = {
  en: {
    heroLabel: "ANPR ENGINE / PUBLIC DEMONSTRATION", limits: "JPEG / PNG · ≤ 10 MiB · ≤ 20 MP · ≤ 8,192 px per side", home: "ANPR Engine home", information: "Information",
    tagline: "A single-image ANPR demonstration", intro: "Upload one vehicle photo to detect the plate and read its text. Check the result before using it.",
    stepNotice: "01 / Review the notice", stepUpload: "02 / Choose one image", stepResult: "03 / Review the result",
    select: "Choose a JPEG or PNG", selected: "Selected image", submit: "Process image", processing: "Processing one image…", clear: "Clear current session",
    original: "Original image with plate boundary overlay", detection: "Detection", recognition: "Recognition prediction", structure: "Structural validation", confidence: "Informational confidence, not certainty",
    noPlate: "No plate was detected. Try a different authorized image.", unresolved: "This result is unresolved until you review it.",
    confirm: "Confirm prediction", correct: "Correct plate text", correction: "Corrected plate text", applyCorrection: "Use correction", leaveUnresolved: "Leave unresolved",
    confirmed: "You confirmed the model prediction.", corrected: "You corrected the text.", reviewUnresolved: "The result remains unresolved.",
    json: "Download versioned JSON", annotated: "Download annotated image", unavailable: "An annotated image needs a Detection box.",
    found: "Plate area detected", notFound: "No plate area detected", valid: "Format check passed", invalid: "Format check failed", notRun: "Format check not run",
    privacy: "Privacy", responsible: "Responsible use", limitations: "Limitations", language: "Language", contact: "Contact", repo: "Repository",
    invalidFile: "Choose one complete, single-frame JPEG or PNG within 10 MiB, 20 megapixels and 8,192 pixels per side.",
    acknowledge: "Acknowledge the notice before processing.", busy: "Only one image can be processed at a time.", error: "The image could not be processed. Review it or try again manually.",
    rate: "Limit reached: at most 5 valid processing attempts per IP each day (Istanbul time). Shared networks share this limit.", capacity: "The demonstration is busy or temporarily unavailable. Try again later.", timeout: "Processing timed out. Please try again later.", unavailableService: "Image processing is temporarily unavailable.",
    originalPanel: "Original image / estimated corners", rectifiedPanel: "Perspective-corrected plate", ocrPanel: "Crop sent to OCR", previewNote: "Small previews of the actual processing images; not stored on the server.", previewUnavailable: "The corrected plate preview is unavailable.", noRectified: "No reliable quadrilateral was found; no corrected plate is shown.",
    learnedPrecision: "Four corners predicted directly by the detector; the same corners rectify the OCR image.", geometryPrecision: "Estimated plate face boundary; this same quadrilateral was used to rectify the OCR image.", geometryWarning: "Plate corners could not be established reliably. The dashed rectangle is only the approximate detector area.",
    precision: "The box marks the Detection result on your original image; it is not the model's refined crop.",
    warning: "Neither confidence nor format validation proves that the plate text is correct.",
    multipleWarning: "Multiple plate areas were found. Only the highest-ranked Detection was processed.", cropWarning: "The detected area could not be refined or read. This result is unresolved.", candidateLabel: "OCR candidate / review required", candidateWarning: "OCR produced the candidate below its acceptance threshold. Check it against the image; it is not an accepted result.", recognitionWarning: "Recognition did not produce a usable prediction. This result is unresolved.",
    downloadNote: "Downloads are created only when you click and are not saved by this site."
  },
  tr: {
    heroLabel: "ANPR ENGINE / KAMUYA AÇIK GÖSTERİM", limits: "JPEG / PNG · ≤ 10 MiB · ≤ 20 MP · kenar başına ≤ 8.192 piksel", home: "ANPR Engine ana sayfa", information: "Bilgilendirme",
    tagline: "Tek görüntülü ANPR gösterimi", intro: "Bir araç fotoğrafı yükleyin; plaka alanını tespit edip yazısını okuyalım. Kullanmadan önce sonucu kontrol edin.",
    stepNotice: "01 / Bildirimi inceleyin", stepUpload: "02 / Bir görüntü seçin", stepResult: "03 / Sonucu inceleyin",
    select: "JPEG veya PNG seçin", selected: "Seçilen görüntü", submit: "Görüntüyü işle", processing: "Bir görüntü işleniyor…", clear: "Geçerli oturumu temizle",
    original: "Plaka sınırı işaretli özgün görüntü", detection: "Tespit", recognition: "Tanıma tahmini", structure: "Yapısal doğrulama", confidence: "Bilgilendirici güven puanı; kesinlik değil",
    noPlate: "Plaka tespit edilmedi. Yetkili başka bir görüntü deneyin.", unresolved: "Siz inceleyene kadar bu sonuç çözümlenmemiştir.",
    confirm: "Tahmini onayla", correct: "Plaka metnini düzelt", correction: "Düzeltilmiş plaka metni", applyCorrection: "Düzeltmeyi uygula", leaveUnresolved: "Çözümlenmemiş bırak",
    confirmed: "Model tahminini onayladınız.", corrected: "Metni düzelttiniz.", reviewUnresolved: "Sonuç çözümlenmemiş olarak kaldı.",
    json: "Sürümlü JSON indir", annotated: "İşaretli görüntüyü indir", unavailable: "İşaretli görüntü için Tespit kutusu gereklidir.",
    found: "Plaka alanı tespit edildi", notFound: "Plaka alanı tespit edilmedi", valid: "Biçim denetimi geçti", invalid: "Biçim denetimi başarısız", notRun: "Biçim denetimi yapılmadı",
    privacy: "Gizlilik", responsible: "Sorumlu kullanım", limitations: "Sınırlamalar", language: "Dil", contact: "İletişim", repo: "Depo",
    invalidFile: "10 MiB, 20 megapiksel ve kenar başına 8.192 piksel sınırları içinde tek, tam, tek kareli JPEG veya PNG seçin.",
    acknowledge: "İşlemeden önce bildirimi onaylayın.", busy: "Aynı anda yalnızca bir görüntü işlenebilir.", error: "Görüntü işlenemedi. İnceleyin veya elle yeniden deneyin.",
    rate: "Günlük sınır doldu: IP adresi başına en fazla 5 geçerli işleme denemesi (İstanbul saati). Ortak ağlarda bu sınır paylaşılır.", capacity: "Gösterim meşgul veya geçici olarak kullanılamıyor. Daha sonra deneyin.", timeout: "İşlem zaman aşımına uğradı. Daha sonra tekrar deneyin.", unavailableService: "Görüntü işleme geçici olarak kullanılamıyor.",
    originalPanel: "Özgün görüntü / tahmin edilen köşeler", rectifiedPanel: "Perspektifi düzeltilmiş plaka", ocrPanel: "OCR’a verilen kırpım", previewNote: "Gerçek işlem görüntülerinin küçük önizlemeleri; sunucuda saklanmaz.", previewUnavailable: "Düzeltilmiş plaka önizlemesi alınamadı.", noRectified: "Güvenilir dörtgen bulunamadı; düzeltilmiş plaka gösterilemiyor.",
    learnedPrecision: "Detector’ın doğrudan tahmin ettiği dört köşe; OCR görüntüsü aynı köşelerle düzleştirildi.", geometryPrecision: "Görüntüden tahmin edilen plaka sınırı; OCR görüntüsü aynı dörtgenle perspektif düzeltildi.", geometryWarning: "Plaka köşeleri güvenilir biçimde belirlenemedi. Kesikli dikdörtgen yalnızca yaklaşık tespit alanıdır.",
    precision: "Kutu, özgün görüntüdeki Tespit sonucunu gösterir; modelin iyileştirilmiş kırpması değildir.",
    warning: "Güven puanı veya biçim denetimi plaka metninin doğru olduğunu kanıtlamaz.",
    multipleWarning: "Birden fazla plaka alanı bulundu. Yalnızca en üst sıradaki Tespit işlendi.", cropWarning: "Tespit edilen alan iyileştirilemedi veya okunamadı. Sonuç çözümlenmemiştir.", candidateLabel: "OCR adayı / kontrol gerekli", candidateWarning: "OCR bir aday okudu fakat kabul eşiğini geçmedi. Görüntüden doğrulayın; kabul edilmiş sonuç değildir.", recognitionWarning: "Tanıma kullanılabilir bir tahmin üretmedi. Sonuç çözümlenmemiştir.",
    downloadNote: "İndirmeler yalnızca tıklamanızla oluşturulur ve bu site tarafından saklanmaz."
  }
} as const;

function downloadable(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function errorMessage(code: string | undefined, t: typeof labels.en | typeof labels.tr): string {
  if (code === "RATE_LIMITED") return t.rate;
  if (code === "CAPACITY_UNAVAILABLE" || code === "SESSION_INFERENCE_ACTIVE") return t.capacity;
  if (code === "INFERENCE_DEADLINE_EXCEEDED") return t.timeout;
  if (code === "MODEL_NOT_READY" || code === "MAINTENANCE") return t.unavailableService;
  if (code === "ORIGIN_REJECTED") return t.unavailableService;
  if (code === "PRIVACY_ACKNOWLEDGEMENT_REQUIRED") return t.acknowledge;
  if (code?.startsWith("IMAGE_") || code === "UNSUPPORTED_MEDIA_TYPE") return t.invalidFile;
  return t.error;
}

export function Workspace({ notices, inferenceAvailable }: { notices: { en: Notice; tr: Notice }; inferenceAvailable: boolean }) {
  const locale = useLocale();
  const notice = notices[locale];
  const t = labels[locale];
  const [acknowledged, setAcknowledged] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [cropPreviews, setCropPreviews] = useState<{rectified: string | null; recognition: string | null}>({rectified: null, recognition: null});
  const cropUrls = useRef<string[]>([]);
  const [result, setResult] = useState<AnprImageInferenceResponseV1 | null>(null);
  const [review, setReview] = useState<Review>(null);
  const [corrected, setCorrected] = useState("");
  const [reviewedValue, setReviewedValue] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const busyRef = useRef(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const workspaceRef = useRef<HTMLElement>(null);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    const root = workspaceRef.current;
    if (!root) return;
    const stage = root.querySelector(".hero-art");
    if (!stage) return;
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const visible = new IntersectionObserver(([entry]) => {
      root.dataset.motion = entry.isIntersecting && !document.hidden && !motion.matches ? "on" : "off";
    }, { threshold: 0 });
    const update = () => {
      const bounds = stage.getBoundingClientRect();
      root.dataset.motion = bounds.bottom > 0 && bounds.top < window.innerHeight && !document.hidden && !motion.matches ? "on" : "off";
    };
    visible.observe(stage);
    document.addEventListener("visibilitychange", update);
    motion.addEventListener("change", update);
    update();
    return () => {
      visible.disconnect();
      document.removeEventListener("visibilitychange", update);
      motion.removeEventListener("change", update);
    };
  }, []);

  useEffect(() => () => { controller.current?.abort(); if (preview) URL.revokeObjectURL(preview); }, [preview]);

  useEffect(() => () => { cropUrls.current.forEach((url) => URL.revokeObjectURL(url)); }, []);

  function releaseCropPreviews() {
    cropUrls.current.forEach((url) => URL.revokeObjectURL(url));
    cropUrls.current = [];
    setCropPreviews({rectified: null, recognition: null});
  }

  function cropUrl(encoded: string | undefined): string | null {
    if (!encoded) return null;
    const bytes = Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], {type: "image/jpeg"}));
    cropUrls.current.push(url);
    return url;
  }

  function clear(preserveAcknowledgement = false) {
    if (busyRef.current) return;
    if (preview) URL.revokeObjectURL(preview);
    releaseCropPreviews();
    setFile(null); setPreview(null); setResult(null); setReview(null); setCorrected(""); setReviewedValue(null); setMessage("");
    if (!preserveAcknowledgement) setAcknowledged(false);
  }

  async function select(event: ChangeEvent<HTMLInputElement>) {
    const chosen = event.target.files?.[0];
    event.target.value = "";
    if (!chosen || busyRef.current) return;
    clear(true);
    const extension = chosen.name.split(".").pop()?.toLowerCase();
    if (!((chosen.type === "image/jpeg" && ["jpg", "jpeg"].includes(extension ?? "")) ||
          (chosen.type === "image/png" && extension === "png")) || chosen.size === 0 || chosen.size > LIMIT) {
      setMessage(t.invalidFile); return;
    }
    try {
      const bitmap = await createImageBitmap(chosen);
      const valid = bitmap.width <= 8192 && bitmap.height <= 8192 && bitmap.width * bitmap.height <= 20_000_000;
      bitmap.close();
      if (!valid) { setMessage(t.invalidFile); return; }
      setFile(chosen); setPreview(URL.createObjectURL(chosen));
    } catch { setMessage(t.invalidFile); }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busyRef.current) { setMessage(t.busy); return; }
    if (!acknowledged) { setMessage(t.acknowledge); return; }
    if (!file) { setMessage(t.invalidFile); return; }
    if (!inferenceAvailable) { setMessage(t.unavailableService); return; }
    releaseCropPreviews();
    busyRef.current = true; setBusy(true); setMessage(""); setResult(null); setReview(null); setReviewedValue(null);
    const abort = new AbortController(); controller.current = abort;
    const deadline = window.setTimeout(() => abort.abort(), CLIENT_DEADLINE_MS);
    try {
      const response = await fetch("/api/inference", {
        method: "POST", credentials: "same-origin", cache: "no-store", signal: abort.signal,
        headers: { "Content-Type": file.type, "X-Privacy-Acknowledged": "true" }, body: file
      });
      const body: unknown = await response.json();
      if (!response.ok) {
        const code = (body as AnprApiErrorV1 | null)?.error?.code;
        setMessage(errorMessage(code, t)); return;
      }
      const value = body as AnprImageInferenceResponseV1;
      if (value.schema_version !== "anpr-image-inference-response-v1" || value.human_review_required !== true) {
        setMessage(t.error); return;
      }
      setCropPreviews({rectified: cropUrl(value.plate_crops?.rectified?.encoded_data), recognition: cropUrl(value.plate_crops?.recognition_input?.encoded_data)});
      setResult(value);
      if (value.status === "no_plate_detected") setReview("UNRESOLVED");
    } catch { setMessage(abort.signal.aborted ? t.timeout : t.error); }
    finally { window.clearTimeout(deadline); busyRef.current = false; setBusy(false); controller.current = null; }
  }

  function downloadJson() {
    if (!result) return;
    const payload: AnprSessionResultV1 = {
      schema_version: "anpr-session-result-v1", result_schema_version: "anpr-image-inference-response-v1",
      request_id: result.request_id, review_outcome: review ?? "UNRESOLVED",
      model_prediction: result.recognition.prediction ?? result.recognition.candidate_prediction ?? null,
      reviewed_value: review === "OWNER_CORRECTED" ? reviewedValue : review === "OWNER_CONFIRMED_PROJECT_MODEL" ? result.recognition.prediction : null,
      warnings: result.warnings
    };
    downloadable(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }), "anpr-result-v1.json");
  }

  function downloadAnnotated() {
    const box = result?.detection.box_xyxy_normalized;
    const image = imageRef.current;
    if (!box || box.length !== 4 || !image?.complete || !image.naturalWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    context.strokeStyle = "#e85d3f"; context.lineWidth = Math.max(3, canvas.width / 300);
    const corners = result?.plate_geometry?.quadrilateral_xy_normalized;
    if (corners) {
      context.beginPath();
      corners.forEach((point, i) => { if (i === 0) context.moveTo(point[0]*canvas.width, point[1]*canvas.height); else context.lineTo(point[0]*canvas.width, point[1]*canvas.height); });
      context.closePath(); context.stroke();
    } else {
      context.setLineDash([context.lineWidth*3, context.lineWidth*2]);
      context.strokeRect(box[0] * canvas.width, box[1] * canvas.height, (box[2] - box[0]) * canvas.width, (box[3] - box[1]) * canvas.height);
    }
    canvas.toBlob((blob) => { if (blob) downloadable(blob, "anpr-detection-overlay-v1.png"); }, "image/png");
  }

  const box = result?.detection.box_xyxy_normalized;
  const corners = result?.plate_geometry?.quadrilateral_xy_normalized;
  const geometryCaption = result?.plate_geometry?.method === "HYPERLPR_FOUR_CORNERS_V1" ? t.learnedPrecision : t.geometryPrecision;
  const crops = result?.plate_crops;
  const resultMessage = !result ? "" : result.status === "no_plate_detected"
    ? (locale === "tr" ? "Plaka bulunamadı. Başka bir fotoğraf deneyin." : "No plate found. Try another photo.")
    : result.warnings.includes("CROP_REFINEMENT_FAILED")
      ? (locale === "tr" ? "Plaka köşeleri belirlenemedi. Başka bir fotoğraf deneyin." : "Plate corners could not be found. Try another photo.")
      : result.warnings.includes("MULTIPLE_DETECTIONS_TOP1_SELECTED")
        ? (locale === "tr" ? "Birden fazla plaka bulundu; en belirgin olan gösteriliyor. Sonucu kontrol edin." : "Multiple plates found; showing the highest-ranked one. Check the result.")
        : !result.recognition.prediction && !result.recognition.candidate_prediction
          ? (locale === "tr" ? "Plaka okunamadı. Daha net bir fotoğraf deneyin." : "The plate could not be read. Try a clearer photo.")
          : (locale === "tr" ? "Tahmini görüntüyle karşılaştırıp doğrulayın." : "Check the prediction against the image.");
  return <main className="shell premium-workspace" ref={workspaceRef} data-motion="off" aria-busy={busy}>
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-spark" aria-hidden="true">✳</span>VIBESOFTERS <span className="eyebrow-divider">/</span> VISION LAB</p>
          <p className="hero-kicker">{t.tagline}</p>
          <h1><span>{locale === "en" ? "Find the plate." : "Plakaya odaklan."}</span><span className="chrome-text">{locale === "en" ? "Read the detail." : "Detayı yakala."}</span></h1>
          <p className="lead">{t.intro}</p>
          <div className="hero-actions"><a className="primary hero-cta" href="#workspace">{locale === "en" ? "Explore the engine" : "Motoru deneyin"}<span aria-hidden="true">↗</span></a><a className="text-link" href="#how-it-works">{locale === "en" ? "How it works" : "Nasıl çalışır"}<span aria-hidden="true">↓</span></a></div>
          <div className="hero-detail"><span className="status-dot" />{locale === "en" ? "One image at a time · Human review required" : "Bir seferde bir görüntü · İnsan incelemesi gerekli"}</div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="hero-art-halo" />
          <div className="scanner-heading"><span className="status-dot" />ANPR ENGINE<span>↗</span></div>
          <div className="logo-stage">
            <div className="orbital orbital-outer" /><div className="orbital orbital-inner" />
            <span className="stage-cross cross-top">+</span><span className="stage-cross cross-bottom">+</span>
            <div className="logo-plinth"><Image className="hero-logo" src="/icon.png" alt="" width={230} height={230} priority /><div className="logo-sheen" /></div>
            <div className="stage-chip chip-corners"><span>⌖</span><div><strong>4 CORNERS</strong><small>{locale === "en" ? "Perspective geometry" : "Perspektif geometrisi"}</small></div></div>
            <div className="stage-chip chip-ocr"><span>≋</span><div><strong>OCR</strong><small>{locale === "en" ? "From pixels to text" : "Pikselden metne"}</small></div></div>
          </div>
          <div className="scanner-caption"><span>DETECT <b>→</b> RECTIFY <b>→</b> READ</span><span>01 — VISION SYSTEM</span></div>
        </div>
      </section>
      <ol className="journey" id="how-it-works" aria-label={locale === "en" ? "How it works" : "Nasıl çalışır"}>
        <li><span className="journey-number">01</span><div><strong>{locale === "en" ? "Locate the boundary" : "Sınırları belirle"}</strong><p>{locale === "en" ? "Four corners, not just a box." : "Yalnızca kutu değil, dört köşe."}</p></div><span className="journey-arrow" aria-hidden="true">↗</span></li>
        <li><span className="journey-number">02</span><div><strong>{locale === "en" ? "Correct the perspective" : "Perspektifi düzelt"}</strong><p>{locale === "en" ? "A straightened view for OCR." : "OCR için düzleştirilmiş görüntü."}</p></div><span className="journey-arrow" aria-hidden="true">↗</span></li>
        <li><span className="journey-number">03</span><div><strong>{locale === "en" ? "Read. Review. Decide." : "Oku. İncele. Karar ver."}</strong><p>{locale === "en" ? "The final check belongs to you." : "Son kontrol her zaman sizde."}</p></div><span className="journey-arrow" aria-hidden="true">↗</span></li>
      </ol>
      <div className="workspace-heading" id="workspace"><div><p className="eyebrow">{locale === "en" ? "YOUR IMAGE. YOUR WORKSPACE." : "SİZİN GÖRÜNTÜNÜZ. SİZİN ALANINIZ."}</p><h2>{locale === "en" ? "A closer look starts here." : "Detaya buradan yaklaşın."}</h2></div><span className="workspace-badge"><span className="status-dot" />JPEG / PNG <span>·</span> MAX 10 MiB</span></div>
      <div className="workflow">
        <section className="panel notice-panel" aria-labelledby="notice-title"><div className="panel-topline"><p className="step">{t.stepNotice}</p><span className="panel-symbol" aria-hidden="true">↗</span></div><h2 id="notice-title">{locale === "en" ? "Processing & responsible use" : "İşleme ve sorumlu kullanım"}</h2><div className="notice-text"><p>{notice.purpose}</p><p>{notice.authorization}</p><p>{notice.accuracy} {notice.human_review}</p><p>{notice.retention} {notice.training}</p><p>{notice.prohibited_uses}</p></div><label className="checkbox-row"><input type="checkbox" checked={acknowledged} disabled={busy} onChange={(event) => setAcknowledged(event.target.checked)} /><span>{notice.acknowledgement_label}</span></label></section>
      <section className={`panel upload-panel ${busy ? "is-processing" : ""}`} aria-labelledby="upload-title">
        <div className="panel-topline"><p className="step">{t.stepUpload}</p><span className="panel-symbol" aria-hidden="true">＋</span></div>
        <h2 id="upload-title">{t.select}</h2><p className="subtle">{t.limits}</p>
        <form onSubmit={submit}>
          <label className={`file-picker ${busy ? "disabled" : ""} ${file ? "has-file" : ""}`}>
            {preview ? <img className="selected-thumbnail" src={preview} alt="" /> : <span className="upload-illustration" aria-hidden="true"><span /><span /><svg viewBox="0 0 48 48" fill="none"><path d="M24 32V10m0 0-8 8m8-8 8 8M10 30v8h28v-8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg></span>}
            <span className="file-copy"><strong>{file ? `${t.selected}: ${file.name}` : t.select}</strong><small>{locale === "en" ? "Click to browse your files" : "Dosyalarınıza göz atmak için tıklayın"}</small></span>
            <span className="file-badges" aria-hidden="true"><span>JPEG</span><span>PNG</span><span>10 MiB</span></span>
            <input type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" disabled={busy || !inferenceAvailable} onChange={select} aria-label={t.select} />
          </label>
          <div className="action-row"><button type="submit" className="primary" disabled={busy || !file || !acknowledged || !inferenceAvailable}>{busy && <span className="processing-spinner" aria-hidden="true" />}{busy ? t.processing : t.submit}<span aria-hidden="true">↗</span></button><button type="button" className="secondary" onClick={() => clear()} disabled={busy || (!file && !result)}>{t.clear}</button></div>
        </form>
        <p role="status" aria-live="polite" className="feedback">{!inferenceAvailable ? t.unavailableService : message || (busy ? t.processing : "")}</p>
        <div className="upload-footnote"><span aria-hidden="true">⌑</span>{locale === "en" ? "Session-only processing. Your images are not used for training." : "Oturumluk işleme. Görüntüleriniz eğitim için kullanılmaz."}</div>
      </section>
        <section className={`panel result-panel ${busy ? "is-processing" : ""}`} aria-labelledby="result-title"><div className="panel-topline"><p className="step">{t.stepResult}</p><span className="result-badge">{busy ? (locale === "en" ? "PROCESSING" : "İŞLENİYOR") : result ? (locale === "en" ? "REVIEW REQUIRED" : "İNCELEME GEREKLİ") : (locale === "en" ? "AWAITING IMAGE" : "GÖRÜNTÜ BEKLENİYOR")}</span></div><h2 id="result-title">{locale === "en" ? "Result & human review" : "Sonuç ve insan incelemesi"}</h2>{!result ? <div className="empty-state"><div className="empty-symbol" aria-hidden="true"><span /><span /><span /><span /><span className="empty-symbol-center">⌖</span></div><strong>{busy ? t.processing : (locale === "en" ? "Your next perspective." : "Sıradaki bakış açınız.")}</strong><p>{locale === "en" ? "Your current result will appear here after processing." : "Geçerli sonuç işlemden sonra burada görünecek."}</p><div className="empty-pipeline" aria-hidden="true"><span>DETECT</span><i /><span>RECTIFY</span><i /><span>OCR</span></div></div> : <>
          {preview && <div className={`geometry-comparison ${!cropPreviews.rectified ? "geometry-single" : ""}`}><figure><h3>{locale === "tr" ? "Görüntü" : "Image"}</h3><div className="image-frame"><img ref={imageRef} src={preview} alt={t.original} />{corners ? <svg className="plate-geometry" aria-label={geometryCaption} viewBox="0 0 1 1" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}><polygon points={corners.map((p) => p.join(",")).join(" ")} fill="none" stroke="#e85d3f" strokeWidth="3" vectorEffect="non-scaling-stroke" /></svg> : box?.length === 4 && <span className="detection-box" aria-label={t.geometryWarning} style={{ borderStyle: "dashed", left: `${box[0]*100}%`, top: `${box[1]*100}%`, width: `${(box[2]-box[0])*100}%`, height: `${(box[3]-box[1])*100}%` }} />}</div></figure>{crops?.rectified && cropPreviews.rectified && <div className="rectified-panel"><h3>{t.rectifiedPanel}</h3><img className="rectified-plate" src={cropPreviews.rectified} width={crops.rectified.width} height={crops.rectified.height} alt={t.rectifiedPanel} /></div>}</div>}
          <div className="readout"><div><span className="readout-label">{t.detection}</span><strong>{result.detection.state === "found" ? t.found : t.notFound}</strong></div><div><span className="readout-label">{result.recognition.candidate_prediction ? (locale === "tr" ? "Tahmin · kontrol gerekli" : "Candidate · review required") : t.recognition}</span><strong className="plate-text">{result.recognition.prediction ?? result.recognition.candidate_prediction ?? "—"}</strong></div></div>
          <p className="feedback">{resultMessage}</p>
          <details className="result-details"><summary>{locale === "tr" ? "Teknik ayrıntılar" : "Technical details"}</summary><div className="technical-content"><p className="subtle">{t.structure}: {result.structural_validation === "valid" ? t.valid : result.structural_validation === "invalid" ? t.invalid : t.notRun}</p><p className="subtle">{t.confidence}: {result.recognition.confidence === null ? "—" : `${Math.round(result.recognition.confidence * 100)}%`}</p><p className="subtle">{t.warning}</p>{crops?.recognition_input && cropPreviews.recognition && <figure><h4>{t.ocrPanel}</h4><img className="ocr-input-plate" src={cropPreviews.recognition} width={crops.recognition_input.width} height={crops.recognition_input.height} alt={t.ocrPanel} /></figure>}</div></details>
          <details className="result-details"><summary>{locale === "tr" ? "Sonucu doğrula, düzelt veya indir" : "Confirm, correct or download"}</summary><div className="review"><p>{review === "OWNER_CONFIRMED_PROJECT_MODEL" ? t.confirmed : review === "OWNER_CORRECTED" ? t.corrected : review === "UNRESOLVED" ? t.reviewUnresolved : t.unresolved}</p><div className="action-row"><button type="button" className="secondary" onClick={() => setReview("OWNER_CONFIRMED_PROJECT_MODEL")} disabled={!result.recognition.prediction}>{t.confirm}</button><button type="button" className="secondary" onClick={() => setReview("UNRESOLVED")}>{t.leaveUnresolved}</button></div><label className="correction-label">{t.correction}<input type="text" inputMode="text" maxLength={16} value={corrected} onChange={(event) => { setCorrected(event.target.value.toUpperCase().replace(/[^0-9A-Z ]/g, "")); if (review === "OWNER_CORRECTED") setReview(null); }} aria-label={t.correction} /></label><button type="button" className="secondary" disabled={!/^[0-9A-Z](?:[0-9A-Z ]{0,14}[0-9A-Z])?$/.test(corrected)} onClick={() => { setReviewedValue(corrected); setReview("OWNER_CORRECTED"); }}>{t.applyCorrection}</button></div>
          <div className="downloads"><button type="button" className="primary" onClick={downloadJson}>{t.json}</button><button type="button" className="secondary" onClick={downloadAnnotated} disabled={!box}>{t.annotated}</button>{!box && <p className="subtle">{t.unavailable}</p>}<p className="subtle">{t.downloadNote}</p></div></details>
        </>}</section>
      </div>
    </main>;
}
