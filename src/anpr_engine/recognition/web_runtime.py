"""Single CPU OCR session with lazy loading and automatic idle release."""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from anpr_engine.configuration import StrictConfigModel
from anpr_engine.domain import ArtifactIdentity, RecognitionResult
from anpr_engine.model_manifest import artifact_path_sha256
from anpr_engine.recognition.config import CropConfig
from anpr_engine.recognition.crop import PlateCrop, _aspect_fit
from anpr_engine.recognition.normalization import PlateNormalizationError, normalize_plate_text
from anpr_engine.recognition.validation import validate_turkish_plate
from anpr_engine.recognition.worker_protocol import ManagedIsolatedPaddleOcrSession


def _gray_range(rgb: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], float, float]:
    gray = cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    # Ignore a few isolated highlights/shadows without thresholding characters.
    lo, hi = np.percentile(gray, [2, 98])
    return gray, float(lo), float(hi)


def verification_views(rgb: NDArray[np.uint8]) -> tuple[str, str]:
    """Bounded, pixel-driven policy, independent of text, province and file name."""
    _, lo, hi = _gray_range(rgb)
    # Near-flat inputs must not have sensor/JPEG noise amplified into glyphs.
    return ("contrast" if 16 <= hi - lo < 96 else "gray", "white_border")


def recognition_view_rgb(rgb: NDArray[np.uint8], view: str) -> NDArray[np.uint8]:
    """Photometric/context views only: never cut glyphs, invent detail or rewrite text."""
    if view == "contrast":
        gray, lo, hi = _gray_range(rgb)
        if 16 <= hi - lo < 96:
            gray = np.rint(
                np.clip((gray.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255)
            ).astype(np.uint8)
        return np.ascontiguousarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    if view == "gray":
        gray = cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        return np.ascontiguousarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    if view == "white_border":
        border = max(2, min(24, round(rgb.shape[0] * 0.15)))
        return np.ascontiguousarray(
            cv2.copyMakeBorder(
                rgb, border, border, border, border, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        )
    if view != "base":
        raise ValueError("unknown recognition view")
    return rgb


def _normalized_read(result: RecognitionResult) -> str:
    try:
        return normalize_plate_text(result.raw_text).normalized_text
    except PlateNormalizationError:
        return ""


def _consensus_read(
    reads: list[tuple[str, RecognitionResult]],
) -> tuple[str, RecognitionResult, float | None, bool]:
    """Whole-string votes, no province/character substitutions or confidence inflation."""
    groups: dict[str, list[tuple[str, RecognitionResult]]] = {}
    for view, result in reads:
        text = _normalized_read(result)
        if re.fullmatch(r"[0-9A-Z]{1,16}", text) and result.recognition_confidence is not None:
            groups.setdefault(text, []).append((view, result))
    support = max(groups.values(), key=len) if groups else [reads[0]]
    agreed = len(support) >= 2
    # A disputed first read is not silently replaced by one speculative alternative.
    if not agreed:
        view, result = reads[0]
        return view, result, result.recognition_confidence, False
    view, result = support[0]
    scores = [r.recognition_confidence for _, r in support if r.recognition_confidence is not None]
    confidence = min(scores) if scores else None
    return view, result, confidence, agreed


class WebOcrConfig(StrictConfigModel):
    python: Path
    worker: Path
    model_dir: Path
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cpu_threads: int = Field(default=1, ge=1, le=2)
    backend: Literal["paddleocr", "paddle_inference"] = "paddleocr"
    estimate_plate_geometry: bool = True
    corner_model: Path | None = None
    corner_model_size: Literal[320, 640] = 320
    idle_seconds: float = Field(default=60, ge=1, le=3600, allow_inf_nan=False)
    timeout_seconds: float = Field(default=30, ge=1, le=60, allow_inf_nan=False)
    # Provisional operating point, not a calibrated accuracy guarantee.
    minimum_confidence: float = Field(default=0.99, ge=0, le=1, allow_inf_nan=False)


class WebOcrRuntime:
    def __init__(self, config: WebOcrConfig) -> None:
        if not config.python.is_file() or not config.worker.is_file():
            raise ValueError("OCR executable or worker missing")
        if artifact_path_sha256(config.model_dir) != config.model_sha256:
            raise ValueError("OCR model hash mismatch")
        self.config = config
        self.identity = ArtifactIdentity(
            name="latin_PP-OCRv5_mobile_rec_finetuned",
            version="paddleocr-latin-finetuned-v1",
            sha256=config.model_sha256,
        )
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._generation = 0
        self._closed = False
        self._session = ManagedIsolatedPaddleOcrSession(
            (
                str(config.python),
                str(config.worker),
                "--model-dir",
                str(config.model_dir),
                "--model-sha256",
                config.model_sha256,
                "--cpu-threads",
                str(config.cpu_threads),
                "--device",
                "cpu",
                "--engine",
                "paddle_static",
                "--backend",
                config.backend,
            ),
            identity=self.identity,
            timeout_seconds=config.timeout_seconds,
        )

    @property
    def worker_process_id(self) -> int | None:
        """Expose the live worker PID for bounded, read-only resource sampling."""
        return self._session.process_id

    def _expire(self, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self._session.close()
                self._timer = None

    def recognize(self, crop: PlateCrop) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("OCR runtime closed")
            self._generation += 1
            if self._timer:
                self._timer.cancel()
            try:
                result = self._session.recognize(crop)
                score = result.recognition_confidence
                view, view_count, verified = "base", 1, True
                # Small text and uncertain reads are sensitive to borders/colour.
                # Reuse ONE worker, at most three serial calls. No extra network.
                if (
                    score is None
                    or score < self.config.minimum_confidence
                    or crop.rgb.shape[0] < 32
                ):
                    reads = [("base", result)]
                    seen_pixels = [crop.rgb]
                    for variant in verification_views(crop.rgb):
                        pixels = recognition_view_rgb(crop.rgb, variant)
                        # Identical pixels are not a second vote, and need no inference.
                        if any(np.array_equal(pixels, previous) for previous in seen_pixels):
                            continue
                        try:
                            alternative = self._session.recognize(
                                replace(
                                    crop,
                                    rgb=pixels,
                                    reference=crop.reference.model_copy(
                                        update={
                                            "crop_id": "view-"
                                            + hashlib.sha256(
                                                f"{crop.reference.crop_id}:{variant}".encode()
                                            ).hexdigest(),
                                            "preprocessing_signature": (
                                                crop.reference.preprocessing_signature
                                                + f"|ocr-view-v1:{variant}"
                                            ),
                                        }
                                    ),
                                    prepared_rgb=_aspect_fit(
                                        pixels,
                                        CropConfig(
                                            target_height=max(16, crop.prepared_rgb.shape[0]),
                                            target_width=max(32, crop.prepared_rgb.shape[1]),
                                        ),
                                    ),
                                )
                            )
                        except Exception:
                            self._session.close()
                            break
                        reads.append((variant, alternative))
                        seen_pixels.append(pixels)
                    view, result, score, verified = _consensus_read(reads)
                    view_count = len(reads)
                text = _normalized_read(result)
                validation = validate_turkish_plate(text)
                has_candidate = bool(re.fullmatch(r"[0-9A-Z]{1,16}", text)) and score is not None
                accepted = (
                    has_candidate
                    and verified
                    and score is not None
                    and score >= self.config.minimum_confidence
                )
                return {
                    "raw_text": result.raw_text,
                    "raw_confidence": result.recognition_confidence,
                    "tr_text": text if accepted and validation.valid else None,
                    "display_text": text if accepted else None,
                    "candidate_text": text if has_candidate and not accepted else None,
                    "decoder_confidence": score,
                    "decoding_changed_raw": False,
                    "structural_validation": validation.model_dump(mode="json"),
                    "decoder_status": "ACCEPTED" if accepted else "LOW_CONFIDENCE",
                    "decoder_reason": None if accepted else "RECOGNITION_UNRESOLVED",
                    "decoder_version": result.decoder_version
                    + ("/consensus-v2" if view_count > 1 else ""),
                    "profile_id": "tr-plate-validation",
                    "ocr_view": view,
                    "ocr_view_count": view_count,
                    "ocr_consensus": verified,
                }
            except Exception:
                self._session.close()
                return {
                    "tr_text": None,
                    "decoder_status": "FAILED",
                    "decoder_reason": "RECOGNITION_UNRESOLVED",
                }
            finally:
                self._timer = threading.Timer(
                    self.config.idle_seconds, self._expire, (self._generation,)
                )
                self._timer.daemon = True
                self._timer.start()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._generation += 1
            if self._timer:
                self._timer.cancel()
            self._session.close()
