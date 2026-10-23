"""Canonical quadrilateral geometry shared by annotation and inference paths."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Point = tuple[float, float]
Quadrilateral = tuple[Point, Point, Point, Point]


def signed_polygon_area(points: Sequence[Point]) -> float:
    """Return signed area; canonical TL/TR/BR/BL image coordinates are positive."""
    if len(points) < 3:
        return 0.0
    doubled = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0]), strict=True)
    )
    return doubled / 2.0


def quadrilateral_bbox(quadrilateral: Quadrilateral) -> tuple[float, float, float, float]:
    xs = tuple(point[0] for point in quadrilateral)
    ys = tuple(point[1] for point in quadrilateral)
    return min(xs), min(ys), max(xs), max(ys)


def _cross(origin: Point, first: Point, second: Point) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
        second[0] - origin[0]
    )


def validate_normalized_quadrilateral(
    quadrilateral: Quadrilateral,
    *,
    minimum_area: float = 1e-8,
) -> None:
    """Validate ordered, in-bounds, strictly convex normalized plate corners."""
    if not math.isfinite(minimum_area) or minimum_area <= 0.0:
        raise ValueError("minimum_area must be finite and positive")
    for point in quadrilateral:
        if not all(math.isfinite(value) for value in point):
            raise ValueError("quadrilateral coordinates must be finite")
        if not all(0.0 <= value <= 1.0 for value in point):
            raise ValueError("quadrilateral coordinates must be normalized within image bounds")
    area = signed_polygon_area(quadrilateral)
    if area <= minimum_area:
        raise ValueError("quadrilateral must have canonical order and positive usable area")
    cross_products = tuple(
        _cross(
            quadrilateral[index],
            quadrilateral[(index + 1) % 4],
            quadrilateral[(index + 2) % 4],
        )
        for index in range(4)
    )
    if any(value <= 0.0 for value in cross_products):
        raise ValueError("quadrilateral must be strictly convex and non-self-intersecting")


def horizontal_flip_quadrilateral(quadrilateral: Quadrilateral) -> Quadrilateral:
    """Flip normalized corners while preserving canonical TL/TR/BR/BL identities."""
    transformed = tuple((1.0 - x, y) for x, y in quadrilateral)
    return transformed[1], transformed[0], transformed[3], transformed[2]


def transform_quadrilateral(
    quadrilateral: Quadrilateral,
    matrix: NDArray[np.float32],
) -> Quadrilateral:
    """Apply a 3x3 homography to a pixel-coordinate quadrilateral."""
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("quadrilateral transform requires a finite 3x3 matrix")
    points = np.asarray(quadrilateral, dtype=np.float64)
    homogeneous = np.concatenate((points, np.ones((4, 1), dtype=np.float64)), axis=1)
    projected = homogeneous @ matrix.astype(np.float64).T
    denominator = projected[:, 2]
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("quadrilateral transform maps a point to infinity")
    transformed = projected[:, :2] / denominator[:, None]
    if not np.isfinite(transformed).all():
        raise ValueError("quadrilateral transform produced non-finite coordinates")
    return tuple((float(point[0]), float(point[1])) for point in transformed)  # type: ignore[return-value]
