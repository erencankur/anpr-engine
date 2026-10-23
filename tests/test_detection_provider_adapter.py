from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from anpr_engine.detection.provider_adapter import (
    BboxProviderOutput,
    BboxProviderOutputError,
    PoseProviderOutput,
    PoseProviderOutputError,
    adapt_bbox_provider_output,
    adapt_pose_provider_output,
    select_plate_bbox_candidates,
    select_plate_pose_candidates,
)


def test_bbox_adapter_preserves_geometry_and_selection_policy() -> None:
    candidates = adapt_bbox_provider_output(
        BboxProviderOutput(
            boxes_xyxy=np.asarray([[10.0, 20.0, 110.0, 60.0], [-2.0, 3.0, 9.0, 8.0]]),
            detection_scores=np.asarray([0.9, 0.8]),
            class_ids=np.asarray([0, 1]),
        )
    )
    assert candidates[0].box_xyxy == (10.0, 20.0, 110.0, 60.0)
    assert candidates[1].box_xyxy == (-2.0, 3.0, 9.0, 8.0)
    assert select_plate_bbox_candidates(
        candidates,
        plate_class_id=0,
        minimum_detection_score=0.85,
        maximum_candidates=2,
    ) == (candidates[0],)


def test_bbox_adapter_rejects_malformed_values() -> None:
    with pytest.raises(BboxProviderOutputError, match="degenerate"):
        adapt_bbox_provider_output(
            BboxProviderOutput(
                boxes_xyxy=np.asarray([[10.0, 20.0, 5.0, 60.0]]),
                detection_scores=np.asarray([0.9]),
                class_ids=np.asarray([0]),
            )
        )


def _output() -> PoseProviderOutput:
    return PoseProviderOutput(
        boxes_xyxy=np.asarray(
            [[10.0, 20.0, 110.0, 60.0], [5.0, 5.0, 25.0, 15.0]], dtype=np.float32
        ),
        detection_scores=np.asarray([0.9, 0.8], dtype=np.float32),
        corners_xy=np.asarray(
            [
                [[12.0, 22.0], [108.0, 21.0], [109.0, 59.0], [11.0, 58.0]],
                [[6.0, 6.0], [24.0, 6.0], [24.0, 14.0], [6.0, 14.0]],
            ],
            dtype=np.float32,
        ),
        corner_scores=np.asarray([[0.95, 0.94, 0.93, 0.92], [0.8, 0.8, 0.8, 0.8]]),
        class_ids=np.asarray([0, 1], dtype=np.int64),
    )


def test_adapter_preserves_bbox_corner_order_and_scores() -> None:
    candidates = adapt_pose_provider_output(_output())
    assert len(candidates) == 2
    assert candidates[0].box_xyxy == (10.0, 20.0, 110.0, 60.0)
    assert candidates[0].corners == (
        (12.0, 22.0),
        (108.0, 21.0),
        (109.0, 59.0),
        (11.0, 58.0),
    )
    assert candidates[0].minimum_corner_score == pytest.approx(0.92)


def test_selection_filters_class_and_confidence_without_clipping_geometry() -> None:
    candidates = adapt_pose_provider_output(_output())
    selected = select_plate_pose_candidates(
        candidates,
        plate_class_id=0,
        minimum_detection_score=0.85,
        minimum_corner_score=0.9,
        maximum_candidates=5,
    )
    assert selected == (candidates[0],)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("boxes_xyxy", np.zeros((2, 5), dtype=np.float32), "boxes"),
        ("detection_scores", np.zeros((3,), dtype=np.float32), "detection_scores"),
        ("corners_xy", np.zeros((2, 8), dtype=np.float32), "corners_xy"),
        ("corner_scores", np.zeros((2, 3), dtype=np.float32), "corner_scores"),
    ],
)
def test_adapter_rejects_shape_mismatch(
    field: str, value: NDArray[np.float32], message: str
) -> None:
    source = _output()
    values = {
        "boxes_xyxy": source.boxes_xyxy,
        "detection_scores": source.detection_scores,
        "corners_xy": source.corners_xy,
        "corner_scores": source.corner_scores,
        "class_ids": source.class_ids,
    }
    values[field] = value
    with pytest.raises(PoseProviderOutputError, match=message):
        adapt_pose_provider_output(PoseProviderOutput(**values))  # type: ignore[arg-type]


def test_adapter_rejects_nonfinite_or_invalid_native_values() -> None:
    source = _output()
    corners = source.corners_xy.copy()
    corners[0, 0, 0] = np.nan
    with pytest.raises(PoseProviderOutputError, match="non-finite"):
        adapt_pose_provider_output(
            PoseProviderOutput(
                boxes_xyxy=source.boxes_xyxy,
                detection_scores=source.detection_scores,
                corners_xy=corners,
                corner_scores=source.corner_scores,
                class_ids=source.class_ids,
            )
        )


def test_adapter_does_not_hide_out_of_bounds_geometry() -> None:
    source = _output()
    corners = source.corners_xy.copy()
    corners[0, 0] = (-5.0, 22.0)
    candidates = adapt_pose_provider_output(
        PoseProviderOutput(
            boxes_xyxy=source.boxes_xyxy,
            detection_scores=source.detection_scores,
            corners_xy=corners,
            corner_scores=source.corner_scores,
            class_ids=source.class_ids,
        )
    )
    assert candidates[0].corners[0] == (-5.0, 22.0)
