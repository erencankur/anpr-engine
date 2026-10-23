"""Frozen V1 Detection-to-Recognition pipeline with fail-closed provenance."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from anpr_engine.detection.data import letterbox
from anpr_engine.detection.hyperlpr_corners import (
    METHOD as CORNER_METHOD,
)
from anpr_engine.detection.hyperlpr_corners import (
    HyperLprCornerDetector,
    rectify_predicted_face,
)
from anpr_engine.detection.model import decode_multiscale_outputs, non_maximum_suppression
from anpr_engine.detection.runtime import (
    load_checkpoint,
    resolve_device,
    sha256_file,
)
from anpr_engine.domain import ArtifactIdentity, BBoxXYXY
from anpr_engine.integration.image_input import decode_upload as decode_upload
from anpr_engine.integration.model_ids import (
    DETECTOR_SHA as DETECTOR_SHA,
)
from anpr_engine.integration.model_ids import (
    DETECTOR_VERSION as DETECTOR_VERSION,
)
from anpr_engine.integration.model_ids import (
    RECOGNIZER_SHA as RECOGNIZER_SHA,
)
from anpr_engine.integration.model_ids import (
    RECOGNIZER_VERSION as RECOGNIZER_VERSION,
)
from anpr_engine.integration.plate_boundary_refinement import (
    PlateBoundaryRefinement,
    refine_plate_boundary,
)
from anpr_engine.integration.plate_face_geometry import METHOD as FACE_METHOD
from anpr_engine.integration.plate_face_geometry import (
    PlateFaceGeometry,
    estimate_plate_face,
    plate_crop_preview,
)
from anpr_engine.recognition.adapters import adapt_fixed_slot_batch
from anpr_engine.recognition.charset import V1_CHARSET
from anpr_engine.recognition.checkpointing import load_recognition_checkpoint
from anpr_engine.recognition.crop import (
    InvalidCropError,
    PlateCrop,
    _aspect_fit,
    generate_plate_crop,
)
from anpr_engine.recognition.decoder import ProbabilityAwareTrDecoder
from anpr_engine.recognition.profiles import TR_PROFILE
from anpr_engine.recognition.registry import M3_CAPABILITY
from anpr_engine.recognition.transformer_slot_config import (
    load_transformer_fixed_slot_foundation_config,
)
from anpr_engine.recognition.transformer_slot_model import (
    build_transformer_fixed_slot_recognizer,
)
from anpr_engine.recognition.web_runtime import WebOcrRuntime

RECOGNIZER_CONFIG = Path("configs/recognition/cnn-transformer-dual-slot-tr-v1-foundation.yaml")
RESULT_SCHEMA = "anpr-v1-browser-result-v1"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class FrozenBundle:
    root: Path
    detector: Path
    recognizer: Path
    manifest: Path

    @classmethod
    def open(cls, root: Path) -> FrozenBundle:
        """Open a separately provisioned private bundle without modifying it."""
        resolved = root.expanduser().resolve()
        return cls(
            resolved,
            resolved / "detection.pt",
            resolved / "recognition.pt",
            resolved / "bundle-manifest.json",
        )

    def verify(self) -> None:
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "anpr-engine-private-model-bundle-v1":
            raise ValueError("private model bundle manifest mismatch")
        if payload.get("classification") != "PRIVATE_OWNER_ONLY":
            raise ValueError("private model bundle classification mismatch")
        models = {str(item.get("role")): item for item in payload.get("models", [])}
        if models.get("detection", {}).get("sha256") != DETECTOR_SHA:
            raise ValueError("detector manifest identity mismatch")
        if models.get("recognition", {}).get("sha256") != RECOGNIZER_SHA:
            raise ValueError("recognizer manifest identity mismatch")
        if sha256_file(self.detector) != DETECTOR_SHA:
            raise ValueError("detector bundle hash mismatch")
        if sha256_file(self.recognizer) != RECOGNIZER_SHA:
            raise ValueError("recognizer bundle hash mismatch")


class AnprV1Pipeline:
    """Load the two frozen models once and execute the governed sequence."""

    def __init__(
        self,
        repository: Path,
        *,
        model_bundle: Path,
        device_name: str = "auto",
        web_ocr: WebOcrRuntime | None = None,
    ) -> None:
        started = time.perf_counter()
        self.repository = repository.resolve()
        self.bundle = FrozenBundle.open(model_bundle)
        self.bundle.verify()
        self.device = resolve_device(device_name)
        self.corner_detector = (
            HyperLprCornerDetector(web_ocr.config.corner_model, web_ocr.config.corner_model_size)
            if web_ocr is not None and web_ocr.config.corner_model is not None
            else None
        )
        if self.corner_detector is not None:
            self.detector_identity = ArtifactIdentity(
                name="hyperlpr_yolov5_four_corners",
                version=self.corner_detector.model_id,
                sha256=self.corner_detector.model_sha,
                reference="private-runtime:learned-corners",
            )
        else:
            self.detector, detector_metadata = load_checkpoint(self.bundle.detector, self.device)
            if detector_metadata.get("architecture") != "yolox_multiscale_v1":
                raise ValueError("detector architecture identity mismatch")
            self.detector.eval()
            self.detector_identity = ArtifactIdentity(
                name="yolox_multiscale_v1",
                version=DETECTOR_VERSION,
                sha256=DETECTOR_SHA,
                reference="private-runtime:detection",
            )
        self.foundation = load_transformer_fixed_slot_foundation_config(
            self.repository / RECOGNIZER_CONFIG
        )
        self.web_ocr = web_ocr
        self._lock = threading.Lock()
        self.load_count = 1
        if web_ocr is not None:
            self.recognizer_identity = web_ocr.identity
            self.startup_ms = (time.perf_counter() - started) * 1000
            return
        if M3_CAPABILITY.max_slots is None:
            raise ValueError("M3 fixed-slot capability is missing max_slots")
        self.recognizer = build_transformer_fixed_slot_recognizer(
            self.foundation.model, max_slots=M3_CAPABILITY.max_slots
        )
        load_recognition_checkpoint(
            self.bundle.recognizer,
            model=self.recognizer,
            expected_capability=M3_CAPABILITY,
            expected_sha256=RECOGNIZER_SHA,
        )
        self.recognizer.to(self.device).eval()
        self.decoder = ProbabilityAwareTrDecoder(
            charset=V1_CHARSET, profile=TR_PROFILE, config=self.foundation.decoder
        )
        self.recognizer_identity = ArtifactIdentity(
            name="cnn_transformer_dual_slot_tr_v1",
            version=RECOGNIZER_VERSION,
            sha256=RECOGNIZER_SHA,
            reference="private-runtime:recognition",
        )
        self._lock = threading.Lock()
        self.load_count = 1
        self.startup_ms = (time.perf_counter() - started) * 1000

    def close(self) -> None:
        if self.web_ocr is not None:
            self.web_ocr.close()

    def health(self) -> dict[str, Any]:
        self.bundle.verify()
        return {
            "status": "HEALTHY",
            "device": self.device.type,
            "model_load_count": self.load_count,
            "startup_model_load_ms": self.startup_ms,
            "detector": self.detector_identity.model_dump(mode="json"),
            "recognizer": self.recognizer_identity.model_dump(mode="json"),
            "freeze_state": "FROZEN_FOR_V1_INTEGRATION",
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
        total_start = time.perf_counter()
        timings: dict[str, float] = {}
        if output_directory is not None:
            output_directory.mkdir(parents=True, exist_ok=True)
        with self._lock, torch.inference_mode():
            stage = time.perf_counter()
            if self.corner_detector is not None:
                candidates = self.corner_detector.detect(image)
            else:
                prepared, transform = letterbox(image, 384)
                tensor = (
                    torch.from_numpy(
                        np.ascontiguousarray(
                            cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                        )
                    )
                    .float()
                    .div(255)
                    .unsqueeze(0)
                    .to(self.device)
                )
                native = self.detector(tensor)
                objectness, boxes = decode_multiscale_outputs(native)
                scores = objectness[0].sigmoid().detach().cpu()
                candidate_boxes = boxes[0].clamp(0, 1).detach().cpu()
                mask = scores >= 0.15
                threshold_boxes, threshold_scores = candidate_boxes[mask], scores[mask]
                candidates = []
                if len(threshold_scores):
                    keep = non_maximum_suppression(threshold_boxes, threshold_scores, 0.30)
                    for index in keep.tolist():
                        x1, y1, x2, y2 = transform.to_original_xyxy(threshold_boxes[index])
                        candidates.append(
                            {
                                "bbox_xyxy": [x1, y1, x2, y2],
                                "confidence": float(threshold_scores[index]),
                            }
                        )
            candidates.sort(key=lambda row: (-row["confidence"], row["bbox_xyxy"]))
            timings["detection_inference_ms"] = (time.perf_counter() - stage) * 1000
            if output_directory is not None:
                annotated = image.copy()
                for index, item in enumerate(candidates):
                    x1, y1, x2, y2 = (round(value) for value in item["bbox_xyxy"])
                    color = (0, 70, 255) if index == 0 else (0, 210, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3 if index == 0 else 2)
                    cv2.putText(
                        annotated,
                        f"{'TOP-1 ' if index == 0 else ''}{item['confidence']:.3f}",
                        (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                annotated_path = output_directory / "annotated.jpg"
                cv2.imwrite(annotated_path.as_posix(), annotated)
            base: dict[str, Any] = {
                "crop_policy": CORNER_METHOD if self.corner_detector else "BLUE_BAND_GLYPH_QUAD_V1",
                "prediction_source": "paddleocr" if self.web_ocr else "existing_project_decoder",
                "schema_version": RESULT_SCHEMA,
                "batch_id": batch_id,
                "request_id": request_id,
                "input": {
                    "filename": filename,
                    "sha256": input_sha256,
                    "width": image.shape[1],
                    "height": image.shape[0],
                },
                "active_device": self.device.type,
                "models": {
                    "detector": self.detector_identity.model_dump(mode="json"),
                    "recognizer": self.recognizer_identity.model_dump(mode="json"),
                },
                "detection": {
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "selected": candidates[0] if candidates else None,
                },
                "annotated_image": "annotated.jpg" if output_directory is not None else None,
                "crop_reference": None,
                "recognition": None,
                "manual_review": {
                    "detection_assessment": "UNREVIEWED",
                    "recognition_assessment": "NOT_APPLICABLE" if not candidates else "UNREVIEWED",
                },
                "errors": [],
                "timings_ms": timings,
            }
            if not candidates:
                base["pipeline_status"] = "NO_PLATE_DETECTED"
                timings["total_request_ms"] = (time.perf_counter() - total_start) * 1000
                return base
            stage = time.perf_counter()
            selected = candidates[0]
            try:
                bbox = BBoxXYXY(
                    x1=selected["bbox_xyxy"][0],
                    y1=selected["bbox_xyxy"][1],
                    x2=selected["bbox_xyxy"][2],
                    y2=selected["bbox_xyxy"][3],
                )
                zero_crop = self.foundation.crop.model_copy(
                    update={"padding_x": 0.0, "padding_y": 0.0}
                )
                crop = generate_plate_crop(
                    image, bbox, source_image_id=request_id, config=zero_crop
                )
            except (ValueError, InvalidCropError) as error:
                base["pipeline_status"] = "INVALID_CROP"
                base["errors"] = [str(error)]
                timings["total_request_ms"] = (time.perf_counter() - total_start) * 1000
                return base
            if output_directory is not None:
                crop_path = output_directory / "crop.jpg"
                cv2.imwrite(crop_path.as_posix(), cv2.cvtColor(crop.rgb, cv2.COLOR_RGB2BGR))
            base["crop_reference"] = {
                **crop.reference.model_dump(mode="json"),
                "artifact": "crop.jpg" if output_directory is not None else None,
            }
            base["detection_crop_reference"] = base["crop_reference"]
            refinement = (
                PlateBoundaryRefinement(crop.rgb, False, {"status": "NOT_RUN_LEARNED_CORNERS"})
                if self.corner_detector is not None
                else refine_plate_boundary(crop.rgb)
            )
            if output_directory is not None:
                refined_path = output_directory / "refined-crop.jpg"
                cv2.imwrite(
                    refined_path.as_posix(), cv2.cvtColor(refinement.rgb, cv2.COLOR_RGB2BGR)
                )
            base["recognition_crop_reference"] = {
                "artifact": (
                    ("refined-crop.jpg" if refinement.applied else "crop.jpg")
                    if output_directory is not None
                    else None
                ),
                "refinement_applied": refinement.applied,
                **refinement.provenance,
            }
            timings["crop_preprocessing_ms"] = (time.perf_counter() - stage) * 1000
            stage = time.perf_counter()
            prepared_recognition_crop = (
                _aspect_fit(refinement.rgb, self.foundation.crop)
                if refinement.applied
                else crop.prepared_rgb
            )
            if self.web_ocr is not None:
                effective_rgb = refinement.rgb if refinement.applied else crop.rgb
                learned = self.corner_detector is not None
                face = (
                    rectify_predicted_face(image, selected["corners_xy"])
                    if learned
                    else estimate_plate_face(image, tuple(selected["bbox_xyxy"]))
                    if self.web_ocr.config.estimate_plate_geometry
                    else PlateFaceGeometry(None, None, "GEOMETRY_DISABLED")
                )
                if face.rgb is not None and face.corners is not None:
                    # Rectify the full face first, then apply the existing pixel-only
                    # glyph/band framing. No OCR output selects geometry or retries.
                    face_reading = (
                        PlateBoundaryRefinement(
                            face.rgb, False, {"status": "NOT_RUN_LEARNED_CORNERS"}
                        )
                        if learned
                        else refine_plate_boundary(face.rgb)
                    )
                    effective_rgb = face_reading.rgb if face_reading.applied else face.rgb
                    base["plate_crops"] = {
                        "rectified": plate_crop_preview(face.rgb),
                        "recognition_input": plate_crop_preview(effective_rgb),
                    }
                    prepared_recognition_crop = _aspect_fit(effective_rgb, self.foundation.crop)
                    base["plate_geometry"] = {
                        "state": "estimated",
                        "method": CORNER_METHOD if learned else FACE_METHOD,
                        "quadrilateral_xy_normalized": (
                            face.corners / np.array([image.shape[1], image.shape[0]])
                        ).tolist(),
                        "rectification_applied": True,
                    }
                    base["recognition_crop_reference"] = {
                        "artifact": None,
                        "refinement_applied": True,
                        "method_id": CORNER_METHOD if learned else FACE_METHOD,
                        "quadrilateral_xy_original": face.corners.tolist(),
                        "post_rectification_refinement": face_reading.provenance,
                    }
                    if output_directory is not None:
                        cv2.imwrite(
                            (output_directory / "face-crop.jpg").as_posix(),
                            cv2.cvtColor(effective_rgb, cv2.COLOR_RGB2BGR),
                        )
                        base["recognition_crop_reference"]["artifact"] = "face-crop.jpg"
                else:
                    base["plate_geometry"] = {
                        "state": "unresolved",
                        "method": None,
                        "quadrilateral_xy_normalized": None,
                        "rectification_applied": False,
                    }
                    base["plate_geometry_reason"] = face.reason
                    if self.web_ocr.config.estimate_plate_geometry:
                        # Geometry is required in the perspective-aware path. Do not
                        # read a possibly clipped bbox and present it as plate OCR.
                        base["recognition"] = None
                        base["pipeline_status"] = "INVALID_CROP"
                        timings["crop_preprocessing_ms"] += (time.perf_counter() - stage) * 1000
                        timings["recognition_inference_ms"] = 0.0
                        timings["total_request_ms"] = (time.perf_counter() - total_start) * 1000
                        return base
                timings["crop_preprocessing_ms"] = (time.perf_counter() - stage) * 1000 + timings[
                    "crop_preprocessing_ms"
                ]
                stage = time.perf_counter()
                base["recognition"] = self.web_ocr.recognize(
                    PlateCrop(
                        rgb=np.ascontiguousarray(effective_rgb),
                        prepared_rgb=prepared_recognition_crop,
                        reference=crop.reference.model_copy(
                            update={
                                "preprocessing_signature": CORNER_METHOD if learned else FACE_METHOD
                            }
                        )
                        if face.rgb is not None
                        else crop.reference,
                    )
                )
                base["prediction_source"] = "paddleocr"
                if "plate_crops" in base:
                    from anpr_engine.recognition.web_runtime import recognition_view_rgb

                    base["plate_crops"]["recognition_input"] = plate_crop_preview(
                        recognition_view_rgb(
                            effective_rgb, base["recognition"].get("ocr_view", "base")
                        )
                    )
                base["pipeline_status"] = (
                    "MULTIPLE_PLATE_CANDIDATES"
                    if len(candidates) > 1
                    else base["recognition"]["decoder_status"]
                )
                timings["recognition_inference_ms"] = (time.perf_counter() - stage) * 1000
                timings["total_request_ms"] = (time.perf_counter() - total_start) * 1000
                return base
            recognition_tensor = (
                torch.from_numpy(np.ascontiguousarray(prepared_recognition_crop.transpose(2, 0, 1)))
                .float()
                .div(255)
                .unsqueeze(0)
                .to(self.device)
            )
            native_recognition = self.recognizer(recognition_tensor)
            common = adapt_fixed_slot_batch(
                native_recognition.character_logits,
                native_recognition.type_logits,
                capability=M3_CAPABILITY,
                decoder_config=self.foundation.decoder,
            )[0]
            decoded = self.decoder.decode_common(common)
            timings["recognition_inference_ms"] = (time.perf_counter() - stage) * 1000
            base["recognition"] = {
                "raw_text": decoded.raw_text,
                "raw_confidence": decoded.raw_confidence,
                "tr_text": decoded.decoded_text,
                "decoder_confidence": decoded.decoder_confidence,
                "decoding_changed_raw": decoded.decoding_changed_raw,
                "structural_validation": decoded.structural_validation.model_dump(mode="json")
                if decoded.structural_validation
                else None,
                "profile_id": decoded.profile_id,
                "decoder_status": decoded.status.value,
                "decoder_reason": decoded.reason,
                "decoder_version": decoded.decoder_version,
            }
            base["pipeline_status"] = (
                "MULTIPLE_PLATE_CANDIDATES" if len(candidates) > 1 else decoded.status.value
            )
            timings["artifact_writing_ms"] = 0.0
            timings["total_request_ms"] = (time.perf_counter() - total_start) * 1000
            return base


def content_sha256(content: bytes) -> str:
    return _sha_bytes(content)
