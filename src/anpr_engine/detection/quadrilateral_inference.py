"""Candidate-preserving quadrilateral decode, validation, and rectification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from anpr_engine.detection.data import LetterboxTransform
from anpr_engine.detection.model import (
    QuadrilateralDetectorOutput,
    decode_multiscale_quadrilateral_outputs,
    non_maximum_suppression,
)
from anpr_engine.detection.quadrilateral import (
    Quadrilateral,
    signed_polygon_area,
)


class QuadrilateralRejectionReason(StrEnum):
    NON_FINITE = "NON_FINITE"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    WRONG_ORDER_OR_DEGENERATE = "WRONG_ORDER_OR_DEGENERATE"
    NON_CONVEX_OR_SELF_INTERSECTING = "NON_CONVEX_OR_SELF_INTERSECTING"
    AREA_TOO_SMALL = "AREA_TOO_SMALL"
    EDGE_TOO_SHORT = "EDGE_TOO_SHORT"
    EXCESSIVE_PERSPECTIVE = "EXCESSIVE_PERSPECTIVE"
    HOMOGRAPHY_UNSTABLE = "HOMOGRAPHY_UNSTABLE"
    OUTPUT_SIZE_INVALID = "OUTPUT_SIZE_INVALID"


@dataclass(frozen=True)
class QuadrilateralGeometryPolicy:
    minimum_area_ratio: float
    minimum_edge_pixels: float
    maximum_opposite_edge_ratio: float
    maximum_homography_condition: float
    maximum_output_pixels: int

    def __post_init__(self) -> None:
        values = (
            self.minimum_area_ratio,
            self.minimum_edge_pixels,
            self.maximum_opposite_edge_ratio,
            self.maximum_homography_condition,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("geometry policy limits must be finite and positive")
        if self.maximum_opposite_edge_ratio < 1.0:
            raise ValueError("maximum_opposite_edge_ratio must be at least one")
        if self.maximum_output_pixels <= 0:
            raise ValueError("maximum_output_pixels must be positive")


@dataclass(frozen=True)
class QuadrilateralCandidates:
    boxes: Tensor
    corners: Tensor
    scores: Tensor


@dataclass(frozen=True)
class OriginalQuadrilateralCandidate:
    box_xyxy: tuple[float, float, float, float]
    corners: Quadrilateral
    score: float


@dataclass(frozen=True)
class QuadrilateralValidation:
    valid: bool
    reason: QuadrilateralRejectionReason | None


@dataclass(frozen=True)
class RectifiedPlate:
    rgb: NDArray[np.uint8]
    homography: NDArray[np.float64]
    output_width: int
    output_height: int


def decode_quadrilateral_predictions(
    output: QuadrilateralDetectorOutput,
    *,
    confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    corner_offset_scale: float,
) -> list[QuadrilateralCandidates]:
    logits, boxes, corners = decode_multiscale_quadrilateral_outputs(
        output, corner_offset_scale=corner_offset_scale
    )
    boxes = boxes.clamp(0.0, 1.0)
    scores = logits.sigmoid()
    decoded: list[QuadrilateralCandidates] = []
    for batch_index in range(scores.shape[0]):
        mask = scores[batch_index] >= confidence_threshold
        sample_scores = scores[batch_index][mask]
        sample_boxes = boxes[batch_index][mask]
        sample_corners = corners[batch_index][mask]
        if not sample_scores.numel():
            decoded.append(
                QuadrilateralCandidates(
                    sample_boxes.reshape(0, 4),
                    sample_corners.reshape(0, 4, 2),
                    sample_scores,
                )
            )
            continue
        candidate_count = min(sample_scores.numel(), max(100, max_detections * 20))
        sample_scores, candidate_indices = sample_scores.topk(candidate_count)
        sample_boxes = sample_boxes[candidate_indices]
        sample_corners = sample_corners[candidate_indices]
        keep = non_maximum_suppression(sample_boxes, sample_scores, nms_iou_threshold)
        keep = keep[:max_detections]
        decoded.append(
            QuadrilateralCandidates(sample_boxes[keep], sample_corners[keep], sample_scores[keep])
        )
    return decoded


def map_candidates_to_original(
    candidates: QuadrilateralCandidates,
    transform: LetterboxTransform,
) -> tuple[OriginalQuadrilateralCandidate, ...]:
    original_corners = transform.to_original_points(candidates.corners)
    mapped: list[OriginalQuadrilateralCandidate] = []
    for box, corners, score in zip(
        candidates.boxes, original_corners, candidates.scores, strict=True
    ):
        mapped.append(
            OriginalQuadrilateralCandidate(
                box_xyxy=transform.to_original_xyxy(box),
                corners=cast(
                    Quadrilateral,
                    tuple((float(point[0]), float(point[1])) for point in corners),
                ),
                score=float(score),
            )
        )
    return tuple(mapped)


def validate_pixel_quadrilateral(
    quadrilateral: Quadrilateral,
    *,
    image_width: int,
    image_height: int,
    policy: QuadrilateralGeometryPolicy,
) -> QuadrilateralValidation:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not all(math.isfinite(value) for point in quadrilateral for value in point):
        return QuadrilateralValidation(False, QuadrilateralRejectionReason.NON_FINITE)
    if any(x < 0.0 or x > image_width or y < 0.0 or y > image_height for x, y in quadrilateral):
        return QuadrilateralValidation(False, QuadrilateralRejectionReason.OUT_OF_BOUNDS)
    area = signed_polygon_area(quadrilateral)
    if area <= 0.0:
        return QuadrilateralValidation(
            False, QuadrilateralRejectionReason.WRONG_ORDER_OR_DEGENERATE
        )
    cross_products = tuple(
        (quadrilateral[(index + 1) % 4][0] - quadrilateral[index][0])
        * (quadrilateral[(index + 2) % 4][1] - quadrilateral[(index + 1) % 4][1])
        - (quadrilateral[(index + 1) % 4][1] - quadrilateral[index][1])
        * (quadrilateral[(index + 2) % 4][0] - quadrilateral[(index + 1) % 4][0])
        for index in range(4)
    )
    if any(value <= 0.0 for value in cross_products):
        return QuadrilateralValidation(
            False, QuadrilateralRejectionReason.NON_CONVEX_OR_SELF_INTERSECTING
        )
    if area / (image_width * image_height) < policy.minimum_area_ratio:
        return QuadrilateralValidation(False, QuadrilateralRejectionReason.AREA_TOO_SMALL)
    edges = tuple(
        math.dist(quadrilateral[index], quadrilateral[(index + 1) % 4]) for index in range(4)
    )
    if min(edges) < policy.minimum_edge_pixels:
        return QuadrilateralValidation(False, QuadrilateralRejectionReason.EDGE_TOO_SHORT)
    opposite_ratios = (
        max(edges[0], edges[2]) / min(edges[0], edges[2]),
        max(edges[1], edges[3]) / min(edges[1], edges[3]),
    )
    if max(opposite_ratios) > policy.maximum_opposite_edge_ratio:
        return QuadrilateralValidation(False, QuadrilateralRejectionReason.EXCESSIVE_PERSPECTIVE)
    return QuadrilateralValidation(True, None)


def select_first_valid_candidate(
    candidates: tuple[OriginalQuadrilateralCandidate, ...],
    *,
    image_width: int,
    image_height: int,
    policy: QuadrilateralGeometryPolicy,
) -> tuple[
    OriginalQuadrilateralCandidate | None,
    tuple[QuadrilateralRejectionReason, ...],
]:
    rejections: list[QuadrilateralRejectionReason] = []
    for candidate in candidates:
        validation = validate_pixel_quadrilateral(
            candidate.corners,
            image_width=image_width,
            image_height=image_height,
            policy=policy,
        )
        if validation.valid:
            return candidate, tuple(rejections)
        if validation.reason is None:
            raise RuntimeError("invalid quadrilateral is missing a rejection reason")
        rejections.append(validation.reason)
    return None, tuple(rejections)


def rectify_plate(
    rgb: NDArray[np.uint8],
    quadrilateral: Quadrilateral,
    *,
    policy: QuadrilateralGeometryPolicy,
) -> RectifiedPlate:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rectification requires uint8 RGB")
    validation = validate_pixel_quadrilateral(
        quadrilateral,
        image_width=rgb.shape[1],
        image_height=rgb.shape[0],
        policy=policy,
    )
    if not validation.valid:
        raise ValueError(f"quadrilateral rejected: {validation.reason}")
    points = np.asarray(quadrilateral, dtype=np.float32)
    width = max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3]))
    height = max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1]))
    output_width = max(1, round(float(width)))
    output_height = max(1, round(float(height)))
    pixel_count = output_width * output_height
    if pixel_count > policy.maximum_output_pixels:
        scale = math.sqrt(policy.maximum_output_pixels / pixel_count)
        output_width = max(1, math.floor(output_width * scale))
        output_height = max(1, math.floor(output_height * scale))
    if output_width < 2 or output_height < 2:
        raise ValueError(
            f"quadrilateral rejected: {QuadrilateralRejectionReason.OUTPUT_SIZE_INVALID}"
        )
    target = np.asarray(
        [
            [0.0, 0.0],
            [output_width - 1.0, 0.0],
            [output_width - 1.0, output_height - 1.0],
            [0.0, output_height - 1.0],
        ],
        dtype=np.float32,
    )
    homography = np.asarray(cv2.getPerspectiveTransform(points, target), dtype=np.float64)
    if not np.isfinite(homography).all() or np.linalg.cond(homography) > (
        policy.maximum_homography_condition
    ):
        raise ValueError(
            f"quadrilateral rejected: {QuadrilateralRejectionReason.HOMOGRAPHY_UNSTABLE}"
        )
    rectified = cast(
        NDArray[np.uint8],
        cv2.warpPerspective(
            rgb,
            homography,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ),
    )
    return RectifiedPlate(
        np.ascontiguousarray(rectified),
        homography,
        output_width,
        output_height,
    )
