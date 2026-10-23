import numpy as np
import pytest
import torch

from anpr_engine.detection.data import LetterboxTransform
from anpr_engine.detection.model import decode_multiscale_quadrilateral_outputs
from anpr_engine.detection.quadrilateral_inference import (
    QuadrilateralCandidates,
    QuadrilateralGeometryPolicy,
    QuadrilateralRejectionReason,
    decode_quadrilateral_predictions,
    map_candidates_to_original,
    rectify_plate,
    select_first_valid_candidate,
    validate_pixel_quadrilateral,
)


def _policy() -> QuadrilateralGeometryPolicy:
    return QuadrilateralGeometryPolicy(
        minimum_area_ratio=0.001,
        minimum_edge_pixels=2.0,
        maximum_opposite_edge_ratio=5.0,
        maximum_homography_condition=1e8,
        maximum_output_pixels=100_000,
    )


def test_decode_keeps_corners_attached_through_ranking_and_nms() -> None:
    outputs = (
        torch.zeros((1, 13, 1, 2)),
        torch.full((1, 13, 1, 1), -20.0),
        torch.full((1, 13, 1, 1), -20.0),
    )
    outputs[0][:, 0] = torch.tensor([[[8.0, 7.0]]])
    outputs[0][0, 5:, 0, 0] = -0.4
    outputs[0][0, 5:, 0, 1] = 0.4
    _, _, all_corners = decode_multiscale_quadrilateral_outputs(outputs, corner_offset_scale=0.75)

    decoded = decode_quadrilateral_predictions(
        outputs,
        confidence_threshold=0.5,
        nms_iou_threshold=0.5,
        max_detections=1,
        corner_offset_scale=0.75,
    )[0]

    assert decoded.scores.tolist() == pytest.approx([torch.sigmoid(torch.tensor(8.0)).item()])
    assert decoded.corners[0].flatten().tolist() == pytest.approx(
        all_corners[0, 0].flatten().tolist()
    )


def test_inverse_letterbox_does_not_clip_bad_corner_predictions() -> None:
    transform = LetterboxTransform(200, 100, 200, 1.0, 0, 50)
    candidates = QuadrilateralCandidates(
        boxes=torch.tensor([[0.1, 0.3, 0.8, 0.6]]),
        corners=torch.tensor([[[0.1, 0.3], [1.1, 0.3], [0.8, 0.6], [0.1, 0.6]]]),
        scores=torch.tensor([0.9]),
    )

    mapped = map_candidates_to_original(candidates, transform)

    assert mapped[0].corners[1][0] == pytest.approx(220.0)
    validation = validate_pixel_quadrilateral(
        mapped[0].corners, image_width=200, image_height=100, policy=_policy()
    )
    assert validation.reason is QuadrilateralRejectionReason.OUT_OF_BOUNDS


def test_selection_skips_invalid_higher_score_and_reports_reason() -> None:
    transform = LetterboxTransform(200, 100, 200, 1.0, 0, 50)
    candidates = QuadrilateralCandidates(
        boxes=torch.tensor([[0.1, 0.3, 0.8, 0.6], [0.1, 0.3, 0.8, 0.6]]),
        corners=torch.tensor(
            [
                [[0.1, 0.3], [1.1, 0.3], [0.8, 0.6], [0.1, 0.6]],
                [[0.1, 0.3], [0.8, 0.3], [0.75, 0.6], [0.15, 0.6]],
            ]
        ),
        scores=torch.tensor([0.95, 0.90]),
    )
    mapped = map_candidates_to_original(candidates, transform)

    selected, rejected = select_first_valid_candidate(
        mapped, image_width=200, image_height=100, policy=_policy()
    )

    assert selected is not None
    assert selected.score == pytest.approx(0.90)
    assert rejected == (QuadrilateralRejectionReason.OUT_OF_BOUNDS,)


def test_rectification_preserves_orientation_and_caps_pixel_budget() -> None:
    rgb = np.zeros((120, 240, 3), dtype=np.uint8)
    rgb[20:100, 30:210, 0] = np.tile(np.arange(180, dtype=np.uint8), (80, 1))
    quadrilateral = ((30.0, 20.0), (210.0, 25.0), (200.0, 100.0), (40.0, 95.0))
    policy = QuadrilateralGeometryPolicy(
        minimum_area_ratio=0.001,
        minimum_edge_pixels=2.0,
        maximum_opposite_edge_ratio=5.0,
        maximum_homography_condition=1e8,
        maximum_output_pixels=5_000,
    )

    rectified = rectify_plate(rgb, quadrilateral, policy=policy)

    assert rectified.output_width * rectified.output_height <= 5_000
    assert rectified.rgb.shape == (rectified.output_height, rectified.output_width, 3)
    assert rectified.rgb[:, -1, 0].mean() > rectified.rgb[:, 0, 0].mean()


@pytest.mark.parametrize(
    ("quadrilateral", "reason"),
    [
        (
            ((10.0, 10.0), (80.0, 50.0), (80.0, 10.0), (10.0, 50.0)),
            QuadrilateralRejectionReason.WRONG_ORDER_OR_DEGENERATE,
        ),
        (
            ((10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0)),
            QuadrilateralRejectionReason.AREA_TOO_SMALL,
        ),
    ],
)
def test_geometry_gate_rejects_invalid_shapes(
    quadrilateral: tuple[tuple[float, float], ...],
    reason: QuadrilateralRejectionReason,
) -> None:
    validation = validate_pixel_quadrilateral(
        quadrilateral,  # type: ignore[arg-type]
        image_width=100,
        image_height=100,
        policy=_policy(),
    )

    assert validation.reason is reason
