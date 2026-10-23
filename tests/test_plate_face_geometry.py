from __future__ import annotations

import base64
from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from anpr_engine.integration.plate_face_geometry import estimate_plate_face, plate_crop_preview


def perspective_scene() -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    plate = np.full((100, 480, 3), 245, np.uint8)
    plate[:, :35] = (210, 65, 10)
    for x in range(65, 440, 50):
        cv2.rectangle(plate, (x, 20), (x + 22, 80), (15, 15, 15), -1)
    corners = np.array([[100, 180], [580, 80], [560, 180], [95, 290]], np.float32)
    matrix = cv2.getPerspectiveTransform(
        np.array([[0, 0], [479, 0], [479, 99], [0, 99]], np.float32), corners
    )
    scene = cv2.warpPerspective(plate, matrix, (700, 400), borderValue=(35, 35, 35))
    return cast(NDArray[np.uint8], scene), corners


def test_perspective_face_recovers_corner_outside_detector_box() -> None:
    scene, expected = perspective_scene()
    # Deliberately cut off the lower-left corner in the original detector box.
    result = estimate_plate_face(scene, (100, 80, 580, 240))
    assert result.reason is None
    assert result.corners is not None and result.rgb is not None
    assert np.max(np.linalg.norm(result.corners - expected, axis=1)) < 5
    assert result.corners[3, 1] > 240
    assert result.rgb.shape[1] / result.rgb.shape[0] > 3
    assert result.rgb.shape[1] <= 1280 and result.rgb.shape[0] <= 384
    # Rectification retains the blue band; it does not crop to the glyph region.
    assert result.rgb[:, :15, 2].mean() > result.rgb[:, :15, 0].mean()


@pytest.mark.parametrize("value", [0, 255])
def test_blank_image_never_fabricates_corners(value: int) -> None:
    result = estimate_plate_face(np.full((400, 700, 3), value, np.uint8), (100, 80, 580, 240))
    assert result.corners is None and result.rgb is None
    assert result.reason == "NO_RELIABLE_FACE_CONTOUR"


@pytest.mark.parametrize("box", [(1, 1, 1, 10), (float("nan"), 1, 100, 30)])
def test_invalid_box_fails_closed(box: tuple[float, float, float, float]) -> None:
    result = estimate_plate_face(np.zeros((100, 200, 3), np.uint8), box)
    assert result.reason == "INVALID_DETECTION_BOX"


def test_blank_plate_shaped_surface_is_not_accepted() -> None:
    scene = np.full((400, 700, 3), 35, np.uint8)
    cv2.rectangle(scene, (100, 100), (580, 200), (245, 245, 245), -1)
    assert estimate_plate_face(scene, (100, 100, 580, 200)).corners is None


def test_large_roi_is_bounded_and_maps_corners_back() -> None:
    scene, expected = perspective_scene()
    scaled = cast(NDArray[np.uint8], cv2.resize(scene, None, fx=4, fy=4))
    result = estimate_plate_face(scaled, (400, 320, 2320, 960))
    assert result.corners is not None and result.rgb is not None
    assert np.max(np.linalg.norm(result.corners - expected * 4, axis=1)) < 15
    assert result.rgb.shape[1] <= 1280 and result.rgb.shape[0] <= 384


def test_dark_blue_background_does_not_absorb_plate_boundary() -> None:
    scene, expected = perspective_scene()
    # The old blue threshold included this low-light background and joined the ROI.
    scene[np.all(scene == 35, axis=2)] = (60, 30, 20)
    result = estimate_plate_face(scene, (100, 80, 580, 240))
    assert result.corners is not None
    assert np.max(np.linalg.norm(result.corners - expected, axis=1)) < 5


def test_nested_plate_boundary_survives_white_bodywork() -> None:
    scene, expected = perspective_scene()
    scene[np.all(scene == 35, axis=2)] = (225, 225, 225)
    cv2.polylines(scene, [expected.astype(np.int32)], True, (10, 10, 10), 3)
    result = estimate_plate_face(scene, (100, 80, 580, 240))
    assert result.corners is not None
    assert np.max(np.linalg.norm(result.corners - expected, axis=1)) < 7


def test_plate_preview_is_small_jpeg_and_contains_the_processing_image() -> None:
    rgb = np.full((100, 480, 3), [245, 10, 15], np.uint8)
    preview = plate_crop_preview(rgb)
    assert preview is not None and preview["metadata_removed"] is True
    data = base64.b64decode(str(preview["encoded_data"]), validate=True)
    assert len(data) <= 12_000 and len(str(preview["encoded_data"])) <= 16_000
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (preview["height"], preview["width"])
    assert decoded[50, 200, 2] > 230  # RGB was encoded correctly, not channel-swapped.


def test_high_entropy_preview_stays_within_response_budget() -> None:
    rgb = np.random.default_rng(42).integers(0, 256, (384, 1280, 3), dtype=np.uint8)
    preview = plate_crop_preview(rgb)
    assert preview is not None
    assert len(str(preview["encoded_data"])) <= 16_000
    assert int(str(preview["width"])) <= 640 and int(str(preview["height"])) <= 192


def test_shadowed_face_and_dark_band_are_not_cut_to_glyph_region() -> None:
    scene, expected = perspective_scene()
    # Neutral face below the old 135 gate, with large glyphs near the bottom edge.
    face = np.all(scene > 130, axis=2)
    scene[face] = (115, 115, 115)
    result = estimate_plate_face(scene, (100, 80, 580, 240))
    assert result.corners is not None and result.rgb is not None
    assert np.max(np.linalg.norm(result.corners - expected, axis=1)) < 7
