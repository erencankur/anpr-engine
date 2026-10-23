"""Explicit CPU-only corner + isolated native OCR path; no legacy bundle/Torch."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from anpr_engine.detection.hyperlpr_corners import (
    METHOD,
    HyperLprCornerDetector,
    rectify_predicted_face,
)
from anpr_engine.domain import ArtifactIdentity, BBoxXYXY
from anpr_engine.integration.plate_face_geometry import plate_crop_preview
from anpr_engine.recognition.config import CropConfig
from anpr_engine.recognition.crop import (
    InvalidCropError,
    PlateCrop,
    _aspect_fit,
    generate_plate_crop,
)
from anpr_engine.recognition.web_runtime import WebOcrRuntime, recognition_view_rgb


class LeanCornerPipeline:
    """Same approved networks and geometry, without unused training/runtime code."""

    def __init__(self, web_ocr: WebOcrRuntime) -> None:
        started = time.perf_counter()
        settings = web_ocr.config
        if (
            settings.corner_model is None
            or settings.corner_model_size != 640
            or settings.backend != "paddle_inference"
            or settings.cpu_threads != 1
            or not settings.estimate_plate_geometry
        ):
            raise ValueError("lean runtime requires native OCR, one CPU thread and 640 corners")
        self.corner_detector = HyperLprCornerDetector(settings.corner_model, 640)
        self.detector_identity = ArtifactIdentity(
            name="hyperlpr_yolov5_four_corners",
            version=self.corner_detector.model_id,
            sha256=self.corner_detector.model_sha,
            reference="private-runtime:learned-corners",
        )
        self.web_ocr = web_ocr
        self.recognizer_identity = web_ocr.identity
        # Matches the prior frozen crop preprocessing, not its unused network config.
        self.crop_config = CropConfig(padding_x=0, padding_y=0, interpolation="area")
        self.load_count = 1
        self.startup_ms = (time.perf_counter() - started) * 1000
        self._lock = threading.Lock()

    def close(self) -> None:
        self.web_ocr.close()

    def health(self) -> dict[str, Any]:
        return {
            "status": "HEALTHY",
            "device": "cpu",
            "runtime_profile": "lean_corners",
            "model_load_count": self.load_count,
            "startup_model_load_ms": self.startup_ms,
            "detector": self.detector_identity.model_dump(mode="json"),
            "recognizer": self.recognizer_identity.model_dump(mode="json"),
        }

    def infer(
        self,
        image: NDArray[np.uint8],
        *,
        batch_id: str,
        request_id: str,
        input_sha256: str,
        filename: str,
        output_directory: Path | None,
    ) -> dict[str, Any]:
        if output_directory is not None:
            raise ValueError("lean service does not export images")
        with self._lock:
            started = time.perf_counter()
            candidates = self.corner_detector.detect(image)
            candidates.sort(key=lambda row: (-row["confidence"], row["bbox_xyxy"]))
            timings = {
                "detection_inference_ms": (time.perf_counter() - started) * 1000,
                "crop_preprocessing_ms": 0.0,
                "recognition_inference_ms": 0.0,
            }
            result: dict[str, Any] = {
                "schema_version": "anpr-v1-browser-result-v1",
                "batch_id": batch_id,
                "request_id": request_id,
                "input": {
                    "filename": filename,
                    "sha256": input_sha256,
                    "width": image.shape[1],
                    "height": image.shape[0],
                },
                "active_device": "cpu",
                "crop_policy": METHOD,
                "prediction_source": "paddleocr",
                "models": {
                    "detector": self.detector_identity.model_dump(mode="json"),
                    "recognizer": self.recognizer_identity.model_dump(mode="json"),
                },
                "detection": {
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "selected": candidates[0] if candidates else None,
                },
                "annotated_image": None,
                "recognition": None,
                "crop_reference": None,
                "errors": [],
                "timings_ms": timings,
                "pipeline_status": "NO_PLATE_DETECTED",
            }
            if candidates:
                stage = time.perf_counter()
                selected = candidates[0]
                face = rectify_predicted_face(image, selected["corners_xy"])
                result["pipeline_status"] = "INVALID_CROP"
                result["plate_geometry"] = {
                    "state": "unresolved",
                    "method": None,
                    "quadrilateral_xy_normalized": None,
                    "rectification_applied": False,
                }
                try:
                    if face.rgb is None or face.corners is None:
                        raise InvalidCropError("unusable learned corners")
                    bbox = BBoxXYXY(
                        **dict(zip(("x1", "y1", "x2", "y2"), selected["bbox_xyxy"], strict=True))
                    )
                    crop = generate_plate_crop(
                        image, bbox, source_image_id=request_id, config=self.crop_config
                    )
                    result["crop_reference"] = {
                        **crop.reference.model_dump(mode="json"),
                        "artifact": None,
                    }
                    result["plate_geometry"] = {
                        "state": "estimated",
                        "method": METHOD,
                        "quadrilateral_xy_normalized": (
                            face.corners / [image.shape[1], image.shape[0]]
                        ).tolist(),
                        "rectification_applied": True,
                    }
                    result["plate_crops"] = {
                        "rectified": plate_crop_preview(face.rgb),
                        "recognition_input": plate_crop_preview(face.rgb),
                    }
                    timings["crop_preprocessing_ms"] = (time.perf_counter() - stage) * 1000
                    stage = time.perf_counter()
                    result["recognition"] = self.web_ocr.recognize(
                        PlateCrop(
                            rgb=np.ascontiguousarray(face.rgb),
                            prepared_rgb=_aspect_fit(face.rgb, self.crop_config),
                            reference=crop.reference.model_copy(
                                update={"preprocessing_signature": METHOD}
                            ),
                        )
                    )
                    result["plate_crops"]["recognition_input"] = plate_crop_preview(
                        recognition_view_rgb(
                            face.rgb, result["recognition"].get("ocr_view", "base")
                        )
                    )
                    result["pipeline_status"] = (
                        "MULTIPLE_PLATE_CANDIDATES"
                        if len(candidates) > 1
                        else result["recognition"]["decoder_status"]
                    )
                    timings["recognition_inference_ms"] = (time.perf_counter() - stage) * 1000
                except InvalidCropError:
                    timings["crop_preprocessing_ms"] = (time.perf_counter() - stage) * 1000
            timings["total_request_ms"] = (time.perf_counter() - started) * 1000
            return result
