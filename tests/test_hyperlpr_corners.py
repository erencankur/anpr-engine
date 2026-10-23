from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from anpr_engine.detection.hyperlpr_corners import (
    HyperLprCornerDetector,
    decode_predictions,
    rectify_predicted_face,
)


def native_output() -> NDArray[np.float32]:
    rows = np.zeros((1, 6300, 15), np.float32)
    rows[0, 0] = [160, 160, 200, 70, 0.9, 60, 125, 260, 125, 260, 195, 60, 195, 0.95, 0.05]
    return rows


def test_letterbox_inverse_preserves_predicted_corner_roles_and_deduplicates() -> None:
    rows = native_output()
    rows[0, 1] = rows[0, 0]
    candidates = decode_predictions(rows, 640, 320, 0.5, 0, 80)
    assert len(candidates) == 1
    assert candidates[0]["corners_xy"] == [[120, 90], [520, 90], [520, 230], [120, 230]]
    assert candidates[0]["confidence"] == pytest.approx(0.855)


@pytest.mark.parametrize("failure", ["crossed", "outside", "double", "nan"])
def test_invalid_predictions_are_never_repaired_into_valid_corners(failure: str) -> None:
    rows = native_output()
    if failure == "crossed":
        rows[0, 0, 7:11] = [260, 195, 260, 125]
    elif failure == "outside":
        rows[0, 0, 5] = -1
    elif failure == "double":
        rows[0, 0, 13:15] = [0.05, 0.95]
    else:
        rows[0, 0, 5] = np.nan
        with pytest.raises(ValueError, match="tensor"):
            decode_predictions(rows, 320, 320, 1, 0, 0)
        return
    assert not decode_predictions(rows, 320, 320, 1, 0, 0)


def test_warp_uses_provided_corners_without_a_colour_or_blue_band_requirement() -> None:
    # Deliberately no blue band: learned geometry, not a Turkish colour heuristic.
    rgb = np.full((300, 600, 3), 180, np.uint8)
    corners = [[40.0, 100.0], [500.0, 30.0], [520.0, 130.0], [45.0, 210.0]]
    result = rectify_predicted_face(rgb, corners)
    assert result.rgb is not None and result.corners is not None
    assert result.corners.tolist() == corners
    assert result.rgb.shape[1] > result.rgb.shape[0]
    assert result.rgb.shape[1] <= 1280 and result.rgb.shape[0] <= 384


def test_unapproved_weight_fails_before_native_model_parsing(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not-approved")
    with pytest.raises(ValueError, match="identity"):
        HyperLprCornerDetector(path)


def test_perspective_warp_preserves_corner_roles_and_rgb_channels() -> None:
    image = np.zeros((300, 600, 3), np.uint8)
    corners = [[40.0, 100.0], [500.0, 30.0], [520.0, 130.0], [45.0, 210.0]]
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for point, colour in zip(corners, colours, strict=True):
        cv2.circle(image, (int(point[0]), int(point[1])), 16, colour[::-1], -1)
    result = rectify_predicted_face(image, corners)
    assert result.rgb is not None
    pixels = [
        result.rgb[2, 2],
        result.rgb[2, -3],
        result.rgb[-3, -3],
        result.rgb[-3, 2],
    ]
    np.testing.assert_array_equal(pixels, colours)


@pytest.mark.parametrize(
    "corners",
    [
        [[-1.0, 30.0], [200.0, 30.0], [200.0, 80.0], [0.0, 80.0]],
        [[40.0, 30.0], [200.0, 80.0], [200.0, 30.0], [40.0, 80.0]],
        [[40.0, 30.0], [200.0, 30.0], [200.0, 30.0], [40.0, 30.0]],
        [[40.0, 30.0], [200.0, 30.0], [200.0, 32.0], [40.0, 32.0]],
        [[40.0, 30.0], [80.0, 30.0], [80.0, 90.0], [40.0, 90.0]],
        [[float("nan"), 30.0], [200.0, 30.0], [200.0, 80.0], [40.0, 80.0]],
    ],
)
def test_unusable_predicted_face_never_produces_a_warp(corners: list[list[float]]) -> None:
    result = rectify_predicted_face(np.zeros((120, 240, 3), np.uint8), corners)
    assert result.rgb is None and result.corners is None and result.reason is not None


def test_rectified_output_is_bounded_even_for_a_large_plate() -> None:
    result = rectify_predicted_face(
        np.zeros((600, 2400, 3), np.uint8),
        [[20.0, 20.0], [2380.0, 20.0], [2380.0, 580.0], [20.0, 580.0]],
    )
    assert result.rgb is not None
    assert result.rgb.shape[1] == 1280
    assert result.rgb.shape[0] <= 384


def test_empty_output_is_not_a_plate() -> None:
    assert not decode_predictions(np.zeros((1, 6300, 15), np.float32), 640, 480, 0.5, 0, 40)


def test_640_layout_uses_the_same_decoder_contract() -> None:
    rows = cast(NDArray[np.float32], np.zeros((1, 25200, 15), np.float32))
    rows[0, 0] = native_output()[0, 0]
    assert len(decode_predictions(rows, 640, 640, 1, 0, 0)) == 1


def test_orientation_recovery_inverse_maps_network_corners(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = object.__new__(HyperLprCornerDetector)
    corners = np.array([[80, 80], [400, 80], [400, 160], [80, 160]], np.float32)
    calls = 0

    def predict(image: NDArray[np.uint8], *, context_scale: float = 1.0) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 1 or context_scale != 1:
            return []
        matrix = cast(NDArray[np.float64], cv2.getRotationMatrix2D((320, 160), -30, 1))
        rw = int(np.ceil(640 * abs(matrix[0, 0]) + 320 * abs(matrix[0, 1])))
        rh = int(np.ceil(320 * abs(matrix[0, 0]) + 640 * abs(matrix[0, 1])))
        matrix[:, 2] += [(rw - 640) / 2, (rh - 320) / 2]
        rotated = np.column_stack((corners, np.ones(4))).astype(np.float64) @ matrix.T
        return [{"corners_xy": rotated.tolist(), "confidence": 0.8}]

    monkeypatch.setattr(detector, "_detect_once", predict)
    result = detector.detect(np.zeros((320, 640, 3), np.uint8))
    assert detector.last_passes == 3
    np.testing.assert_allclose(result[0]["corners_xy"], corners, atol=1e-4)


def test_orientation_recovery_is_strictly_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = object.__new__(HyperLprCornerDetector)
    monkeypatch.setattr(detector, "_detect_once", lambda _, **kwargs: [])
    assert detector.detect(np.zeros((320, 640, 3), np.uint8)) == []
    assert detector.last_passes == 4


def test_context_recovery_only_runs_after_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = object.__new__(HyperLprCornerDetector)
    seen = []
    candidate = {"confidence": 0.7, "corners_xy": [[10, 10], [200, 10], [200, 60], [10, 60]]}

    def predict(image: NDArray[np.uint8], *, context_scale: float = 1) -> list[dict[str, object]]:
        seen.append(context_scale)
        return [candidate] if context_scale < 1 else []

    monkeypatch.setattr(detector, "_detect_once", predict)
    result = detector.detect(np.zeros((300, 600, 3), np.uint8))
    assert seen == [1, 1 / 3] and detector.last_passes == 2
    assert result[0]["corners_xy"] == candidate["corners_xy"]
    assert result[0]["context_scale"] == 1 / 3
    seen.clear()
    monkeypatch.setattr(detector, "_detect_once", lambda image: [candidate])
    assert detector.detect(np.zeros((300, 600, 3), np.uint8)) == [candidate]
    assert detector.last_passes == 1


def test_context_letterbox_is_fixed_size_and_maps_to_the_original_image() -> None:
    detector = object.__new__(HyperLprCornerDetector)
    detector.input_size = 640
    points = np.array([[200, 400], [1800, 400], [1800, 800], [200, 800]], np.float32)
    scale = 640 / 2048 / 3
    nw, nh = int(2048 * scale), int(1024 * scale)
    left, top = (640 - nw) // 2, (640 - nh) // 2
    rows = native_output()
    rows[0, 0, 5:13] = (points * scale + [left, top]).reshape(-1)

    class Net:
        def setInput(self, blob: NDArray[np.float32]) -> None:
            assert blob.shape == (1, 3, 640, 640)
            assert np.count_nonzero(blob) <= 3 * nw * nh

        def forward(self) -> NDArray[np.float32]:
            return rows

    detector.net = cast(cv2.dnn.Net, Net())
    result = detector._detect_once(np.full((1024, 2048, 3), 180, np.uint8), context_scale=1 / 3)
    np.testing.assert_allclose(result[0]["corners_xy"], points, atol=1e-3)


@pytest.mark.parametrize("scale", [0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_context_scale_is_rejected(scale: float) -> None:
    detector = object.__new__(HyperLprCornerDetector)
    with pytest.raises(ValueError, match="context scale"):
        detector._detect_once(np.zeros((10, 20, 3), np.uint8), context_scale=scale)
