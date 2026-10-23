"""Bounded pixel-based plate face quadrilateral estimation (not a corner model)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from anpr_engine.integration.plate_boundary_refinement import _blue_band, _character_components

METHOD = "PLATE_FACE_CONTOUR_V1"


@dataclass(frozen=True)
class PlateFaceGeometry:
    rgb: NDArray[np.uint8] | None
    corners: NDArray[np.float32] | None
    reason: str | None


def _fit_supported_edges(
    points: NDArray[np.float32], hull: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Fit observed edge support; avoid cutting a dark corner by polygon simplification."""
    cloud = hull.reshape(-1, 2)
    minimum_edge = float(np.min(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)))
    lines: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
    for a, b in zip(points, np.roll(points, -1, axis=0), strict=True):
        delta = b - a
        length = float(np.linalg.norm(delta))
        direction = delta / length
        projection = ((cloud - a) @ direction) / length
        distance = np.abs((cloud[:, 0] - a[0]) * direction[1] - (cloud[:, 1] - a[1]) * direction[0])
        support = cloud[
            (projection > 0.08) & (projection < 0.92) & (distance < 0.20 * minimum_edge)
        ]
        if len(support) >= 3:
            fitted = cv2.fitLine(support, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
            direction, a = fitted[:2], fitted[2:]
        lines.append((direction, a))
    result = points.copy()
    for i in range(4):
        u, a = lines[(i - 1) % 4]
        v, b = lines[i]
        matrix = np.column_stack((u, -v))
        if abs(float(np.linalg.det(matrix))) < 1e-5:
            return points
        corner = a + np.linalg.solve(matrix, b - a)[0] * u
        if np.linalg.norm(corner - points[i]) > 0.25 * minimum_edge:
            return points
        result[i] = corner
    return result


def estimate_plate_face(
    image_bgr: NDArray[np.uint8], box: tuple[float, float, float, float]
) -> PlateFaceGeometry:
    """Search around the box, reject ambiguous contours, and warp the chosen face.

    CPU work is capped at a 960px ROI side and at most 48 candidate contours.
    Selection uses pixels/geometry only; OCR text/confidence never select corners.
    """
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if (
        not np.isfinite(box).all()
        or bw <= 0
        or bh <= 0
        or x2 <= 0
        or y2 <= 0
        or x1 >= width
        or y1 >= height
    ):
        return PlateFaceGeometry(None, None, "INVALID_DETECTION_BOX")
    left, top = max(0, int(x1 - bw * 0.25)), max(0, int(y1 - bh * 0.6))
    right, bottom = (
        min(width, int(np.ceil(x2 + bw * 0.25))),
        min(height, int(np.ceil(y2 + bh * 0.6))),
    )
    roi = image_bgr[top:bottom, left:right]
    if roi.size == 0 or min(roi.shape[:2]) < 12:
        return PlateFaceGeometry(None, None, "ROI_TOO_SMALL")
    scale = min(1.0, 960 / max(roi.shape[:2]))
    resized = cv2.resize(
        roi, (max(1, round(roi.shape[1] * scale)), max(1, round(roi.shape[0] * scale)))
    )
    sx, sy = resized.shape[1] / roi.shape[1], resized.shape[0] / roi.shape[0]
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([179, 110, 255]))
    primary_blue = cv2.inRange(hsv, np.array([85, 45, 25]), np.array([140, 255, 255]))
    primary_mask = cv2.morphologyEx(
        cv2.bitwise_or(white, primary_blue), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    primary_contours, _ = cv2.findContours(primary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dark blue vehicle/UI shadows must not join the plate to the entire ROI.
    blue = cv2.inRange(hsv, np.array([85, 180, 30]), np.array([140, 255, 255]))
    # Shadows can lower a neutral plate face well below the primary white gate.
    # Keep this separate fallback, supported by the saturated band and glyphs.
    shadow_face = cv2.inRange(hsv, np.array([0, 0, 90]), np.array([179, 110, 255]))
    mask = cv2.bitwise_or(shadow_face, blue)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Nested contours retain the plate boundary when surrounding white bodywork
    # joins the colour mask. Edge contours also cover non-uniform illumination.
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    edges_mask = cv2.Canny(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 70, 180)
    edges_mask = cv2.morphologyEx(edges_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edge_contours, _ = cv2.findContours(edges_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    primary_contours = sorted(primary_contours, key=cv2.contourArea, reverse=True)[:16]
    color_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:16]
    bounded_contours = (
        primary_contours
        + color_contours
        + sorted(edge_contours, key=cv2.contourArea, reverse=True)[:16]
    )
    area_box = bw * bh * sx * sy
    local_box = np.array([(x1 - left) * sx, (y1 - top) * sy, (x2 - left) * sx, (y2 - top) * sy])
    candidates: list[tuple[float, NDArray[np.float32], NDArray[np.uint8]]] = []
    for contour_index, contour in enumerate(bounded_contours):
        # Prefer supported complete external faces, then nested colour faces,
        # then holder/grille edges. Do not replace a supported face with an inner trace.
        if (
            contour_index in (len(primary_contours), len(primary_contours) + len(color_contours))
            and candidates
        ):
            break
        if not 0.25 * area_box <= cv2.contourArea(contour) <= 2.3 * area_box:
            continue
        # Glyphs touching the face boundary create deep concavities. Recover the
        # outer envelope, not a rotated bounding rectangle or fitted glyph box.
        hull = cv2.convexHull(contour)
        if cv2.contourArea(contour) < 0.6 * cv2.contourArea(hull):
            continue
        perimeter = cv2.arcLength(hull, True)
        for tolerance in (0.015, 0.025, 0.035):
            polygon = cv2.approxPolyDP(hull, tolerance * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            points: NDArray[np.float32] = polygon.reshape(4, 2).astype(np.float32)
            center = points.mean(axis=0)
            points = points[
                np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
            ]
            points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
            if contour_index >= len(primary_contours):
                points = _fit_supported_edges(points, hull.astype(np.float32))
                if not cv2.isContourConvex(points):
                    continue
            # A clipped contour cannot establish the complete physical plate face.
            if (
                np.any(points <= 1)
                or np.any(points[:, 0] >= rgb.shape[1] - 2)
                or np.any(points[:, 1] >= rgb.shape[0] - 2)
            ):
                continue
            lo, hi = points.min(axis=0), points.max(axis=0)
            overlap = np.maximum(
                0, np.minimum(hi, local_box[2:]) - np.maximum(lo, local_box[:2])
            ).prod()
            union = (hi - lo).prod() + area_box - overlap
            iou = float(overlap / max(float(union), 1))
            if iou < 0.35:
                continue
            edges = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
            ow, oh = (
                round(float((edges[0] + edges[2]) / 2)),
                round(float((edges[1] + edges[3]) / 2)),
            )
            if ow < 48 or oh < 12 or not 2.5 <= ow / oh <= 8:
                continue
            output_scale = min(1.0, 1280 / ow, 384 / oh)
            ow, oh = max(1, round(ow * output_scale)), max(1, round(oh * output_scale))
            target = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
            matrix = cast(NDArray[np.float64], cv2.getPerspectiveTransform(points, target))
            if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
                continue
            warped = cv2.warpPerspective(rgb, matrix, (ow, oh), flags=cv2.INTER_LINEAR)
            gray = cast(NDArray[np.uint8], cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY))
            glyphs = _character_components(gray)
            contrast_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if (
                not 4 <= len(glyphs) <= 12
                or float(np.mean(gray > contrast_threshold)) < 0.5
                or _blue_band(cast(NDArray[np.uint8], warped), glyphs) is None
            ):
                continue
            original = points / np.array([sx, sy], np.float32) + np.array([left, top], np.float32)
            candidate = (iou, cast(NDArray[np.float32], original), cast(NDArray[np.uint8], warped))
            # Inner/outer edge traces of the same face are not independent plates.
            near = next(
                (
                    i
                    for i, (_, other, _) in enumerate(candidates)
                    if np.max(np.linalg.norm(other - original, axis=1)) < 0.035 * np.hypot(bw, bh)
                ),
                None,
            )
            if near is None:
                candidates.append(candidate)
            elif iou > candidates[near][0]:
                candidates[near] = candidate
            break
    if not candidates:
        return PlateFaceGeometry(None, None, "NO_RELIABLE_FACE_CONTOUR")
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.1:
        return PlateFaceGeometry(None, None, "AMBIGUOUS_FACE_CONTOURS")
    _, corners, rectified = candidates[0]
    return PlateFaceGeometry(np.ascontiguousarray(rectified), corners, None)


def plate_crop_preview(rgb: NDArray[np.uint8]) -> dict[str, object] | None:
    """Small, metadata-free session preview; never the model's input or a disk export."""
    height, width = rgb.shape[:2]
    for attempt in range(4):
        scale = min(1.0, 640 / width, 192 / height) * 0.5**attempt
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 80 - attempt * 10],
        )
        if ok and encoded.nbytes <= 12_000:
            return {
                "media_type": "image/jpeg",
                "encoded_data": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "width": size[0],
                "height": size[1],
                "metadata_removed": True,
            }
    return None
