from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from anpr_engine.detection.mmdetection_adapter import MmdetPredInstances, adapt_mmdetection_result


@dataclass
class _Instances:
    bboxes: object
    scores: object
    labels: object


@dataclass
class _Result:
    pred_instances: MmdetPredInstances | None


def test_mmdetection_adapter_maps_bbox_result_without_provider_import() -> None:
    candidates = adapt_mmdetection_result(
        _Result(
            pred_instances=_Instances(
                bboxes=np.asarray([[4.0, 5.0, 44.0, 25.0]], dtype=np.float32),
                scores=np.asarray([0.875], dtype=np.float32),
                labels=np.asarray([0], dtype=np.int64),
            )
        )
    )
    assert len(candidates) == 1
    assert candidates[0].box_xyxy == (4.0, 5.0, 44.0, 25.0)
    assert candidates[0].detection_score == pytest.approx(0.875)
    assert candidates[0].class_id == 0


def test_mmdetection_adapter_accepts_empty_prediction() -> None:
    assert (
        adapt_mmdetection_result(
            _Result(
                pred_instances=_Instances(
                    bboxes=np.empty((0, 4), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                    labels=np.empty((0,), dtype=np.int64),
                )
            )
        )
        == ()
    )


def test_mmdetection_adapter_rejects_missing_or_non_numeric_instances() -> None:
    with pytest.raises(ValueError, match="missing pred_instances"):
        adapt_mmdetection_result(_Result(pred_instances=None))
    with pytest.raises(ValueError, match="bboxes must be numeric"):
        adapt_mmdetection_result(
            _Result(
                pred_instances=_Instances(
                    bboxes=np.asarray([["bad", "box", "data", "here"]]),
                    scores=np.asarray([0.5]),
                    labels=np.asarray([0]),
                )
            )
        )
