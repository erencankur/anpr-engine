"""Provider-independent boundary for plate bbox and four-keypoint predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anpr_engine.detection.quadrilateral import Quadrilateral


class PoseProviderOutputError(ValueError):
    """A provider output cannot be mapped without clipping, reordering, or guessing."""


class BboxProviderOutputError(ValueError):
    """A bbox provider output cannot be mapped without guessing."""


@dataclass(frozen=True, slots=True)
class BboxProviderOutput:
    boxes_xyxy: NDArray[np.floating]
    detection_scores: NDArray[np.floating]
    class_ids: NDArray[np.integer]


@dataclass(frozen=True, slots=True)
class ProviderBboxCandidate:
    box_xyxy: tuple[float, float, float, float]
    detection_score: float
    class_id: int


@dataclass(frozen=True, slots=True)
class PoseProviderOutput:
    boxes_xyxy: NDArray[np.floating]
    detection_scores: NDArray[np.floating]
    corners_xy: NDArray[np.floating]
    corner_scores: NDArray[np.floating]
    class_ids: NDArray[np.integer]


@dataclass(frozen=True, slots=True)
class ProviderPoseCandidate:
    box_xyxy: tuple[float, float, float, float]
    corners: Quadrilateral
    detection_score: float
    corner_scores: tuple[float, float, float, float]
    class_id: int

    @property
    def minimum_corner_score(self) -> float:
        return min(self.corner_scores)


def adapt_bbox_provider_output(
    output: BboxProviderOutput,
) -> tuple[ProviderBboxCandidate, ...]:
    """Map native bbox arrays while preserving provider coordinate semantics."""
    if output.boxes_xyxy.ndim != 2 or output.boxes_xyxy.shape[1:] != (4,):
        raise BboxProviderOutputError("provider boxes must have shape [N, 4]")
    population = output.boxes_xyxy.shape[0]
    expected = {
        "detection_scores": (population,),
        "class_ids": (population,),
    }
    actual = {
        "detection_scores": output.detection_scores.shape,
        "class_ids": output.class_ids.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise BboxProviderOutputError(
                f"provider {name} must have shape {list(shape)}, got {list(actual[name])}"
            )

    candidates: list[ProviderBboxCandidate] = []
    for index in range(population):
        box = tuple(float(value) for value in output.boxes_xyxy[index])
        score = float(output.detection_scores[index])
        if not all(math.isfinite(value) for value in (*box, score)):
            raise BboxProviderOutputError(f"provider candidate {index} contains non-finite values")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise BboxProviderOutputError(f"provider candidate {index} has a degenerate bbox")
        if not 0.0 <= score <= 1.0:
            raise BboxProviderOutputError(
                f"provider candidate {index} detection score is outside [0, 1]"
            )
        candidates.append(
            ProviderBboxCandidate(
                box_xyxy=(box[0], box[1], box[2], box[3]),
                detection_score=score,
                class_id=int(output.class_ids[index]),
            )
        )
    return tuple(candidates)


def select_plate_bbox_candidates(
    candidates: tuple[ProviderBboxCandidate, ...],
    *,
    plate_class_id: int,
    minimum_detection_score: float,
    maximum_candidates: int,
) -> tuple[ProviderBboxCandidate, ...]:
    """Apply declared class/confidence policy without clipping provider geometry."""
    if not math.isfinite(minimum_detection_score) or not 0.0 <= minimum_detection_score <= 1.0:
        raise ValueError("provider confidence threshold must be finite and within [0, 1]")
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    selected = (
        candidate
        for candidate in candidates
        if candidate.class_id == plate_class_id
        and candidate.detection_score >= minimum_detection_score
    )
    return tuple(sorted(selected, key=lambda item: -item.detection_score)[:maximum_candidates])


def _validate_shapes(output: PoseProviderOutput) -> int:
    if output.boxes_xyxy.ndim != 2 or output.boxes_xyxy.shape[1:] != (4,):
        raise PoseProviderOutputError("provider boxes must have shape [N, 4]")
    population = output.boxes_xyxy.shape[0]
    expected = {
        "detection_scores": (population,),
        "corners_xy": (population, 4, 2),
        "corner_scores": (population, 4),
        "class_ids": (population,),
    }
    actual = {
        "detection_scores": output.detection_scores.shape,
        "corners_xy": output.corners_xy.shape,
        "corner_scores": output.corner_scores.shape,
        "class_ids": output.class_ids.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise PoseProviderOutputError(
                f"provider {name} must have shape {list(shape)}, got {list(actual[name])}"
            )
    return population


def adapt_pose_provider_output(output: PoseProviderOutput) -> tuple[ProviderPoseCandidate, ...]:
    """Map native arrays while deliberately preserving provider coordinate semantics."""
    population = _validate_shapes(output)
    candidates: list[ProviderPoseCandidate] = []
    for index in range(population):
        box = tuple(float(value) for value in output.boxes_xyxy[index])
        detection_score = float(output.detection_scores[index])
        corners = tuple((float(point[0]), float(point[1])) for point in output.corners_xy[index])
        corner_scores = tuple(float(value) for value in output.corner_scores[index])
        class_id = int(output.class_ids[index])
        numeric_values = (*box, detection_score, *corner_scores, *(v for p in corners for v in p))
        if not all(math.isfinite(value) for value in numeric_values):
            raise PoseProviderOutputError(f"provider candidate {index} contains non-finite values")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise PoseProviderOutputError(f"provider candidate {index} has a degenerate bbox")
        if not 0.0 <= detection_score <= 1.0:
            raise PoseProviderOutputError(
                f"provider candidate {index} detection score is outside [0, 1]"
            )
        if any(not 0.0 <= score <= 1.0 for score in corner_scores):
            raise PoseProviderOutputError(
                f"provider candidate {index} corner score is outside [0, 1]"
            )
        candidates.append(
            ProviderPoseCandidate(
                box_xyxy=(box[0], box[1], box[2], box[3]),
                corners=(corners[0], corners[1], corners[2], corners[3]),
                detection_score=detection_score,
                corner_scores=(
                    corner_scores[0],
                    corner_scores[1],
                    corner_scores[2],
                    corner_scores[3],
                ),
                class_id=class_id,
            )
        )
    return tuple(candidates)


def select_plate_pose_candidates(
    candidates: tuple[ProviderPoseCandidate, ...],
    *,
    plate_class_id: int,
    minimum_detection_score: float,
    minimum_corner_score: float,
    maximum_candidates: int,
) -> tuple[ProviderPoseCandidate, ...]:
    """Apply declared confidence/class policy without modifying prediction geometry."""
    thresholds = (minimum_detection_score, minimum_corner_score)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("provider confidence thresholds must be finite and within [0, 1]")
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    selected = (
        candidate
        for candidate in candidates
        if candidate.class_id == plate_class_id
        and candidate.detection_score >= minimum_detection_score
        and candidate.minimum_corner_score >= minimum_corner_score
    )
    return tuple(
        sorted(
            selected,
            key=lambda item: (-item.detection_score, -item.minimum_corner_score),
        )[:maximum_candidates]
    )
