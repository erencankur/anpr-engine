"""Pinned HyperLPR YOLOv5 multitask detector, OpenCV DNN CPU only.

Native output: cx,cy,w,h,objectness,TL/TR/BR/BL (8 values),two layout scores.
Weights remain private; no contour fitting, corner reordering or OCR-based selection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from anpr_engine.detection.quadrilateral import Quadrilateral, validate_normalized_quadrilateral

MODEL_ID = "hyperlpr-y5fu-320-corners-v1"
MODEL_SHA = "2a985dc63a5cc947ec36d18503d6fc0fd54525b9dba17f2fe29a71e86b44456d"
MODEL_ID_640 = "hyperlpr-y5fu-640-corners-v1"
MODEL_SHA_640 = "0306de937471b87f56eb3f5620815e7e7058f8ab7428e0734fc64627cc4d716c"
METHOD = "HYPERLPR_FOUR_CORNERS_V1"


@dataclass(frozen=True)
class PredictedFace:
    rgb: NDArray[np.uint8] | None
    corners: NDArray[np.float32] | None
    reason: str | None


def decode_predictions(
    output: NDArray[np.float32],
    width: int,
    height: int,
    scale: float,
    left: int,
    top: int,
    threshold: float = 0.25,
) -> list[dict[str, Any]]:
    if output.shape not in ((1, 6300, 15), (1, 25200, 15)) or not np.isfinite(output).all():
        raise ValueError("unexpected learned-corner tensor")
    rows = output[0]
    scores = rows[:, 4] * rows[:, 13:15].max(axis=1)
    ordered = np.argsort(-scores, kind="stable")[:100]
    candidates: list[dict[str, Any]] = []
    for index in ordered:
        score = float(scores[index])
        if score < threshold:
            break
        row = rows[index]
        if not 0 <= score <= 1 or row[13:15].argmax() != 0:
            continue  # Double-row OCR layout is not supported by this integration.
        points = (row[5:13].reshape(4, 2) - [left, top]) / scale
        normalized = points / [width, height]
        try:
            validate_normalized_quadrilateral(
                cast(Quadrilateral, tuple(tuple(map(float, p)) for p in normalized))
            )
        except ValueError:
            continue
        # The reporting box encloses the very same predicted corners, not a crop head.
        lo, hi = points.min(axis=0), points.max(axis=0)
        box = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
        duplicate = False
        for previous in candidates:
            a = np.array(previous["bbox_xyxy"])
            intersection = float(
                np.maximum(0, np.minimum(hi, a[2:]) - np.maximum(lo, a[:2])).prod()
            )
            union = float((hi - lo).prod() + (a[2:] - a[:2]).prod()) - intersection
            if intersection / max(union, 1) > 0.5:
                duplicate = True
                break
        if not duplicate:
            candidates.append(
                {
                    "bbox_xyxy": box,
                    "confidence": score,
                    "corners_xy": points.astype(np.float32).tolist(),
                }
            )
        if len(candidates) == 5:
            break
    return candidates


class HyperLprCornerDetector:
    def __init__(self, path: Path, input_size: int = 320) -> None:
        if input_size not in (320, 640):
            raise ValueError("unsupported learned-corner input")
        self.input_size = input_size
        self.model_id = MODEL_ID if input_size == 320 else MODEL_ID_640
        self.model_sha = MODEL_SHA if input_size == 320 else MODEL_SHA_640
        if (
            path.stat().st_size > 8_000_000
            or hashlib.sha256(path.read_bytes()).hexdigest() != self.model_sha
        ):
            raise ValueError("learned-corner weight identity mismatch")
        cv2.setNumThreads(1)
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, image: NDArray[np.uint8]) -> list[dict[str, Any]]:
        self.last_passes = 1
        candidates = self._detect_once(image)
        if candidates:
            return candidates
        # A close plate can fill a scale the training distribution rarely sees.
        # Add context in the fixed network canvas, NEVER by enlarging the source.
        # This is a global scale pass, not a location/colour/plate-text heuristic.
        self.last_passes += 1
        contextual = self._detect_once(image, context_scale=1 / 3)
        if contextual:
            return [{**candidate, "context_scale": 1 / 3} for candidate in contextual]
        # Bounded orientation recovery, same network/weights, only on a miss.
        # Coordinates are inverse-transformed, never fitted from colour/glyphs.
        height, width = image.shape[:2]
        scale = min(1.0, 1280 / max(width, height))
        sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
        small = cv2.resize(image, (sw, sh))
        for angle in (-30, 30):
            matrix = cast(NDArray[np.float64], cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1))
            cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
            rw, rh = int(np.ceil(sw * cosine + sh * sine)), int(np.ceil(sh * cosine + sw * sine))
            matrix[:, 2] += [(rw - sw) / 2, (rh - sh) / 2]
            rotated = cast(NDArray[np.uint8], cv2.warpAffine(small, matrix, (rw, rh)))
            self.last_passes += 1
            inverse = cv2.invertAffineTransform(matrix)
            restored = []
            for candidate in self._detect_once(rotated):
                points = np.asarray(candidate["corners_xy"])
                points = (np.column_stack((points, np.ones(4))) @ inverse.T) / [
                    sw / width,
                    sh / height,
                ]
                try:
                    validate_normalized_quadrilateral(
                        cast(
                            Quadrilateral,
                            tuple(tuple(map(float, p)) for p in points / [width, height]),
                        )
                    )
                except ValueError:
                    continue
                lo, hi = points.min(axis=0), points.max(axis=0)
                restored.append(
                    {
                        **candidate,
                        "corners_xy": points.astype(np.float32).tolist(),
                        "bbox_xyxy": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
                        "orientation_degrees": angle,
                    }
                )
            if restored:
                return restored
        return []

    def _detect_once(
        self, image: NDArray[np.uint8], *, context_scale: float = 1.0
    ) -> list[dict[str, Any]]:
        if not np.isfinite(context_scale) or not 0 < context_scale <= 1:
            raise ValueError("invalid learned-corner context scale")
        height, width = image.shape[:2]
        size = self.input_size
        scale = min(size / width, size / height) * context_scale
        nw, nh = max(1, int(width * scale)), max(1, int(height * scale))
        left, top = (size - nw) // 2, (size - nh) // 2
        resized = cv2.resize(image, (nw, nh))
        padded = cv2.copyMakeBorder(
            resized,
            top,
            size - nh - top,
            left,
            size - nw - left,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        self.net.setInput(cv2.dnn.blobFromImage(padded, 1 / 255, swapRB=True))
        output = cast(NDArray[np.float32], self.net.forward())
        return decode_predictions(output, width, height, scale, left, top)


def rectify_predicted_face(image: NDArray[np.uint8], corners: list[list[float]]) -> PredictedFace:
    points = np.asarray(corners, np.float32)
    if points.shape != (4, 2):
        return PredictedFace(None, None, "INVALID_LEARNED_CORNERS")
    height, width = image.shape[:2]
    try:
        validate_normalized_quadrilateral(
            cast(Quadrilateral, tuple(tuple(map(float, p)) for p in points / [width, height]))
        )
    except ValueError:
        return PredictedFace(None, None, "INVALID_LEARNED_CORNERS")
    lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    ow, oh = float((lengths[0] + lengths[2]) / 2), float((lengths[1] + lengths[3]) / 2)
    if ow < 24 or oh < 8 or not 2 <= ow / oh <= 10:
        return PredictedFace(None, None, "UNSUPPORTED_LEARNED_FACE_GEOMETRY")
    scale = min(1, 1280 / ow, 384 / oh)
    w, h = max(2, round(ow * scale)), max(2, round(oh * scale))
    target = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    rgb = cv2.cvtColor(cv2.warpPerspective(image, matrix, (w, h)), cv2.COLOR_BGR2RGB)
    return PredictedFace(cast(NDArray[np.uint8], rgb), points, None)
