"""Ultralytics Pose result adapter without an import-time Ultralytics dependency."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from anpr_engine.detection.provider_adapter import (
    PoseProviderOutput,
    ProviderPoseCandidate,
    adapt_pose_provider_output,
)


class UltralyticsBoxes(Protocol):
    xyxy: object
    conf: object
    cls: object


class UltralyticsKeypoints(Protocol):
    xy: object
    conf: object


class UltralyticsPoseResult(Protocol):
    boxes: UltralyticsBoxes | None
    keypoints: UltralyticsKeypoints | None


def _numpy(value: object, *, name: str) -> NDArray[np.generic]:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    convert = getattr(current, "numpy", None)
    if callable(convert):
        current = convert()
    try:
        array = np.asarray(current)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Ultralytics {name} cannot be converted to an array") from error
    if array.dtype.kind not in "fiu":
        raise ValueError(f"Ultralytics {name} must be numeric")
    return cast(NDArray[np.generic], array)


def adapt_ultralytics_pose_result(
    result: UltralyticsPoseResult,
) -> tuple[ProviderPoseCandidate, ...]:
    """Map one image result; four plate keypoints are mandatory."""
    if result.boxes is None:
        raise ValueError("Ultralytics Pose result is missing boxes")
    if result.keypoints is None:
        raise ValueError("Ultralytics Pose result is missing keypoints")
    if result.keypoints.conf is None:
        raise ValueError("Ultralytics Pose result is missing keypoint confidence")
    output = PoseProviderOutput(
        boxes_xyxy=_numpy(result.boxes.xyxy, name="boxes.xyxy").astype(np.float64),
        detection_scores=_numpy(result.boxes.conf, name="boxes.conf").astype(np.float64),
        corners_xy=_numpy(result.keypoints.xy, name="keypoints.xy").astype(np.float64),
        corner_scores=_numpy(result.keypoints.conf, name="keypoints.conf").astype(np.float64),
        class_ids=_numpy(result.boxes.cls, name="boxes.cls").astype(np.int64),
    )
    return adapt_pose_provider_output(output)
