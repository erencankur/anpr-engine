"""MMDetection bbox result adapter without an import-time provider dependency."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from anpr_engine.detection.provider_adapter import (
    BboxProviderOutput,
    ProviderBboxCandidate,
    adapt_bbox_provider_output,
)


class MmdetPredInstances(Protocol):
    bboxes: object
    scores: object
    labels: object


class MmdetDetectionResult(Protocol):
    pred_instances: MmdetPredInstances | None


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
        raise ValueError(f"MMDetection {name} cannot be converted to an array") from error
    if array.dtype.kind not in "fiu":
        raise ValueError(f"MMDetection {name} must be numeric")
    return cast(NDArray[np.generic], array)


def adapt_mmdetection_result(
    result: MmdetDetectionResult,
) -> tuple[ProviderBboxCandidate, ...]:
    """Map a DetDataSample prediction without importing MMDetection in the service runtime."""
    instances = result.pred_instances
    if instances is None:
        raise ValueError("MMDetection result is missing pred_instances")
    output = BboxProviderOutput(
        boxes_xyxy=_numpy(instances.bboxes, name="pred_instances.bboxes").astype(np.float64),
        detection_scores=_numpy(instances.scores, name="pred_instances.scores").astype(np.float64),
        class_ids=_numpy(instances.labels, name="pred_instances.labels").astype(np.int64),
    )
    return adapt_bbox_provider_output(output)
