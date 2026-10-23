from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch

from anpr_engine.detection.ultralytics_adapter import (
    UltralyticsBoxes,
    UltralyticsKeypoints,
    adapt_ultralytics_pose_result,
)


class Boxes:
    def __init__(self) -> None:
        self.xyxy: object = torch.tensor([[10.0, 20.0, 110.0, 60.0]])
        self.conf: object = torch.tensor([0.9])
        self.cls: object = torch.tensor([0.0])


class Keypoints:
    def __init__(self) -> None:
        self.xy: object = torch.tensor([[[12.0, 22.0], [108.0, 21.0], [109.0, 59.0], [11.0, 58.0]]])
        self.conf: object = torch.tensor([[0.95, 0.94, 0.93, 0.92]])


class Result:
    def __init__(self) -> None:
        self.boxes: UltralyticsBoxes | None = Boxes()
        self.keypoints: UltralyticsKeypoints | None = Keypoints()


def test_ultralytics_tensors_map_to_provider_candidates() -> None:
    candidates = adapt_ultralytics_pose_result(Result())
    assert len(candidates) == 1
    assert candidates[0].class_id == 0
    assert candidates[0].detection_score == pytest.approx(0.9)
    assert candidates[0].corners[0] == (12.0, 22.0)
    assert candidates[0].minimum_corner_score == pytest.approx(0.92)


def test_numpy_outputs_are_supported_without_ultralytics_import() -> None:
    result = Result()
    assert result.boxes is not None and result.keypoints is not None
    result.boxes.xyxy = cast(torch.Tensor, result.boxes.xyxy).numpy()
    result.boxes.conf = cast(torch.Tensor, result.boxes.conf).numpy()
    result.boxes.cls = cast(torch.Tensor, result.boxes.cls).numpy()
    result.keypoints.xy = cast(torch.Tensor, result.keypoints.xy).numpy()
    result.keypoints.conf = cast(torch.Tensor, result.keypoints.conf).numpy()
    assert adapt_ultralytics_pose_result(result)[0].box_xyxy == (10.0, 20.0, 110.0, 60.0)


def test_missing_native_sections_fail_closed() -> None:
    result = Result()
    result.keypoints = None
    with pytest.raises(ValueError, match="missing keypoints"):
        adapt_ultralytics_pose_result(result)


def test_wrong_keypoint_count_is_rejected() -> None:
    result = Result()
    assert result.keypoints is not None
    result.keypoints.xy = np.zeros((1, 5, 2), dtype=np.float32)
    result.keypoints.conf = np.ones((1, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="corners_xy"):
        adapt_ultralytics_pose_result(result)
