from pathlib import Path
from typing import cast

import pytest
import torch
from torch.optim import AdamW

from anpr_engine.detection.config import DetectorConfig, QuadrilateralDetectorConfig
from anpr_engine.detection.model import (
    YoloXMultiScaleDetector,
    YoloXQuadrilateralDetector,
    decode_multiscale_quadrilateral_outputs,
)
from anpr_engine.detection.quadrilateral_data import QuadrilateralTarget
from anpr_engine.detection.quadrilateral_training import (
    load_quadrilateral_checkpoint,
    load_v1_warm_start,
    quadrilateral_detector_loss,
    save_quadrilateral_checkpoint,
)
from anpr_engine.detection.training import save_checkpoint

SHA256 = "a" * 64


def _config() -> QuadrilateralDetectorConfig:
    return QuadrilateralDetectorConfig.model_validate(
        {
            "dataset_id": "quad-fixture",
            "dataset_content_sha256": SHA256,
            "model": {
                "image_size": 256,
                "base_channels": 16,
                "head_channels": 64,
                "minimum_epochs": 1,
            },
            "training": {"batch_size": 2, "epochs": 1, "device": "cpu"},
        }
    )


def _v1_config() -> DetectorConfig:
    return DetectorConfig.model_validate(
        {
            "dataset_content_sha256": SHA256,
            "model": {
                "architecture": "yolox_multiscale_v1",
                "image_size": 256,
                "base_channels": 16,
                "head_channels": 64,
                "minimum_epochs": 1,
            },
            "training": {"batch_size": 1, "epochs": 1},
        }
    )


def test_quadrilateral_model_outputs_and_decodes_candidate_bound_corners() -> None:
    model = YoloXQuadrilateralDetector(base_channels=16, head_channels=64)
    output = model(torch.zeros((2, 3, 256, 256)))

    logits, boxes, corners = decode_multiscale_quadrilateral_outputs(
        output, corner_offset_scale=0.75
    )

    assert [level.shape for level in output] == [
        (2, 13, 32, 32),
        (2, 13, 16, 16),
        (2, 13, 8, 8),
    ]
    assert logits.shape == (2, 1344)
    assert boxes.shape == (2, 1344, 4)
    assert corners.shape == (2, 1344, 4, 2)


def test_quadrilateral_loss_supports_positive_and_negative_samples() -> None:
    model = YoloXQuadrilateralDetector(base_channels=16, head_channels=64)
    output = model(torch.zeros((2, 3, 256, 256)))
    positive = QuadrilateralTarget(
        boxes=torch.tensor([[0.20, 0.30, 0.70, 0.55]]),
        corners=torch.tensor([[[0.20, 0.32], [0.70, 0.30], [0.68, 0.55], [0.22, 0.53]]]),
        corner_valid=torch.ones((1, 4), dtype=torch.bool),
        rectification_usable=torch.ones((1,), dtype=torch.bool),
    )
    negative = QuadrilateralTarget(
        boxes=torch.empty((0, 4)),
        corners=torch.empty((0, 4, 2)),
        corner_valid=torch.empty((0, 4), dtype=torch.bool),
        rectification_usable=torch.empty((0,), dtype=torch.bool),
    )

    total, objectness, box, corner, positives = quadrilateral_detector_loss(
        output,
        (positive, negative),
        focal_alpha=0.75,
        focal_gamma=2.0,
        box_loss_weight=5.0,
        corner_loss_weight=5.0,
        corner_offset_scale=0.75,
        center_radius=2.5,
        assignment_top_k=10,
    )
    total.backward()  # type: ignore[no-untyped-call]

    assert total.isfinite()
    assert objectness.item() > 0.0
    assert box.item() >= 0.0
    assert corner.item() >= 0.0
    assert positives >= 1
    corners = cast(torch.nn.Module, model.heads[0].corners)
    assert any(parameter.grad is not None for parameter in corners.parameters())


def test_v1_warm_start_loads_every_shared_parameter_and_only_initializes_corners(
    tmp_path: Path,
) -> None:
    v1_config = _v1_config()
    v1_model = YoloXMultiScaleDetector(base_channels=16, head_channels=64)
    optimizer = AdamW(v1_model.parameters())
    checkpoint = tmp_path / "v1.pt"
    save_checkpoint(
        checkpoint,
        v1_model,
        optimizer,
        epoch=1,
        config=v1_config,
        run_id="v1-fixture",
        validation_metrics={"map_50": 0.0},
    )
    v2_model = YoloXQuadrilateralDetector(base_channels=16, head_channels=64)

    receipt = load_v1_warm_start(v2_model, checkpoint)

    assert receipt["source_architecture"] == "yolox_multiscale_v1"
    assert receipt["initialized_parameter_keys"]
    assert all(".corners." in key for key in receipt["initialized_parameter_keys"])
    v1_state = v1_model.state_dict()
    v2_state = v2_model.state_dict()
    assert all(torch.equal(value, v2_state[key]) for key, value in v1_state.items())


def test_quadrilateral_checkpoint_round_trip_is_separate_from_v1(tmp_path: Path) -> None:
    config = _config()
    model = YoloXQuadrilateralDetector(base_channels=16, head_channels=64)
    optimizer = AdamW(model.parameters())
    checkpoint = tmp_path / "quad.pt"
    save_quadrilateral_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=1,
        config=config,
        run_id="quad-fixture",
        validation_metrics={"total_loss": 1.0},
        initialization={"source_checkpoint_sha256": "b" * 64},
    )

    restored, metadata = load_quadrilateral_checkpoint(checkpoint, torch.device("cpu"))

    assert isinstance(restored, YoloXQuadrilateralDetector)
    assert metadata["format_version"] == "2.0"
    assert metadata["architecture"] == "yolox_multiscale_quad_v2"
    assert metadata["corner_offset_scale"] == 0.75

    with pytest.raises(ValueError, match="unsupported detector checkpoint"):
        from anpr_engine.detection.training import load_checkpoint

        load_checkpoint(checkpoint, torch.device("cpu"))
