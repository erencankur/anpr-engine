"""Deterministic, review-only pilot preparation for box-labelled plate datasets.

This module never promotes a proposal to ground truth.  It selects a coverage-oriented
pilot from an existing YOLO box dataset and uses the frozen pixel-only plate-boundary
heuristic to propose corners that a human must confirm or correct.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from anpr_engine.detection.quadrilateral import (
    Quadrilateral,
    validate_normalized_quadrilateral,
)
from anpr_engine.integration.plate_boundary_refinement import (
    METHOD_ID,
    refine_plate_boundary,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
PILOT_SCHEMA_VERSION = "plate-quadrilateral-pilot-queue-v1"
SELECTION_METHOD = "coverage-round-robin-quantile-v1"
YOLO_BOUNDARY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class BoxSample:
    sample_id: str
    split: Literal["train", "validation", "test"]
    cohort: str
    image_path: Path
    label_path: Path
    image_relative_path: str
    label_relative_path: str
    box_xywh_normalized: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        return self.box_xywh_normalized[2] * self.box_xywh_normalized[3]

    @property
    def aspect_ratio(self) -> float:
        return self.box_xywh_normalized[2] / self.box_xywh_normalized[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def _parse_box_label(path: Path) -> tuple[float, float, float, float]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError(f"pilot requires exactly one five-field YOLO label: {path}")
    class_id, *coordinates = rows[0]
    if class_id not in {"0", "0.0"}:
        raise ValueError(f"pilot requires class-0 plate labels: {path}")
    try:
        center_x, center_y, width, height = (float(value) for value in coordinates)
    except ValueError as error:
        raise ValueError(f"pilot label contains a non-numeric coordinate: {path}") from error
    values = center_x, center_y, width, height
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"pilot label coordinates must be finite: {path}")
    if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
        raise ValueError(f"pilot label center must be normalized: {path}")
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError(f"pilot label size must be normalized and positive: {path}")
    if (
        center_x - width / 2 < -YOLO_BOUNDARY_TOLERANCE
        or center_x + width / 2 > 1.0 + YOLO_BOUNDARY_TOLERANCE
    ):
        raise ValueError(f"pilot label extends outside the image horizontally: {path}")
    if (
        center_y - height / 2 < -YOLO_BOUNDARY_TOLERANCE
        or center_y + height / 2 > 1.0 + YOLO_BOUNDARY_TOLERANCE
    ):
        raise ValueError(f"pilot label extends outside the image vertically: {path}")
    return values


def _matching_image(images: Path, label: Path) -> Path:
    matches = [
        candidate
        for suffix in IMAGE_SUFFIXES
        if (candidate := images / f"{label.stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one image for label: {label}")
    return matches[0]


def discover_box_samples(dataset_root: Path) -> tuple[BoxSample, ...]:
    """Discover ``split/cohort/images|labels`` pairs without changing source data."""
    root = dataset_root.resolve()
    samples: list[BoxSample] = []
    for label in sorted(root.glob("*/*/labels/*.txt")):
        split, cohort = label.relative_to(root).parts[:2]
        if split not in {"train", "validation", "test"}:
            continue
        image = _matching_image(label.parent.parent / "images", label)
        image_relative = image.relative_to(root).as_posix()
        label_relative = label.relative_to(root).as_posix()
        samples.append(
            BoxSample(
                sample_id=_stable_id("quad-pilot", image_relative),
                split=cast(Literal["train", "validation", "test"], split),
                cohort=cohort,
                image_path=image,
                label_path=label,
                image_relative_path=image_relative,
                label_relative_path=label_relative,
                box_xywh_normalized=_parse_box_label(label),
            )
        )
    if not samples:
        raise ValueError(f"no canonical YOLO box samples found under {root}")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("pilot sample IDs are not unique")
    return tuple(samples)


def _quantile_bin(value: float, values: Sequence[float], bin_count: int = 4) -> int:
    boundaries = np.quantile(
        np.asarray(values, dtype=np.float64), np.arange(1, bin_count) / bin_count
    )
    return int(np.searchsorted(boundaries, value, side="right"))


def select_pilot_samples(
    samples: Sequence[BoxSample], *, maximum_samples: int, seed: int
) -> tuple[BoxSample, ...]:
    """Select deterministic coverage across split, cohort, area, and aspect quantiles."""
    if maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive")
    if maximum_samples >= len(samples):
        return tuple(samples)
    areas = [sample.area for sample in samples]
    aspects = [sample.aspect_ratio for sample in samples]
    strata: dict[tuple[Literal["train", "validation", "test"], str, int, int], list[BoxSample]] = (
        defaultdict(list)
    )
    for sample in samples:
        key = (
            sample.split,
            sample.cohort,
            _quantile_bin(sample.area, areas),
            _quantile_bin(sample.aspect_ratio, aspects),
        )
        strata[key].append(sample)
    for members in strata.values():
        members.sort(
            key=lambda sample: hashlib.sha256(f"{seed}\0{sample.sample_id}".encode()).hexdigest()
        )
    selected: list[BoxSample] = []
    ordered_keys = sorted(strata)
    index = 0
    while len(selected) < maximum_samples:
        progressed = False
        for key in ordered_keys:
            members = strata[key]
            if index < len(members):
                selected.append(members[index])
                progressed = True
                if len(selected) == maximum_samples:
                    break
        if not progressed:
            break
        index += 1
    return tuple(sorted(selected, key=lambda sample: sample.sample_id))


def _load_oriented_rgb(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(oriented, dtype=np.uint8)


def _box_pixels(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    center_x, center_y, box_width, box_height = box
    left = max(0, math.floor((center_x - box_width / 2) * width))
    top = max(0, math.floor((center_y - box_height / 2) * height))
    right = min(width, math.ceil((center_x + box_width / 2) * width))
    bottom = min(height, math.ceil((center_y + box_height / 2) * height))
    if right <= left or bottom <= top:
        raise ValueError("box becomes empty in pixel coordinates")
    return left, top, right, bottom


def propose_quadrilateral(sample: BoxSample) -> dict[str, Any]:
    """Return a review-only proposal record; never an approved annotation."""
    rgb = _load_oriented_rgb(sample.image_path)
    height, width = rgb.shape[:2]
    left, top, right, bottom = _box_pixels(sample.box_xywh_normalized, width, height)
    crop = np.ascontiguousarray(rgb[top:bottom, left:right])
    refinement = refine_plate_boundary(crop)
    proposal: dict[str, Any] | None = None
    reason = str(refinement.provenance.get("fallback_reason", "PROPOSAL_AVAILABLE"))
    if refinement.applied:
        raw_quad = refinement.provenance.get("quad_in_detection_crop_xy")
        if not isinstance(raw_quad, list) or len(raw_quad) != 4:
            raise ValueError("applied boundary refinement omitted its quadrilateral")
        normalized = cast(
            Quadrilateral,
            tuple(
                ((float(point[0]) + left) / width, (float(point[1]) + top) / height)
                for point in raw_quad
            ),
        )
        validate_normalized_quadrilateral(normalized)
        proposal = {
            "corners_xy_normalized": [[x, y] for x, y in normalized],
            "method_id": METHOD_ID,
        }
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "cohort": sample.cohort,
        "canonical_image_path": sample.image_relative_path,
        "canonical_label_path": sample.label_relative_path,
        "image_sha256": sha256_file(sample.image_path),
        "label_sha256": sha256_file(sample.label_path),
        "image_width": width,
        "image_height": height,
        "box_xywh_normalized": list(sample.box_xywh_normalized),
        "review_state": "REVIEW_REQUIRED",
        "proposal": proposal,
        "proposal_reason": reason,
    }


def build_pilot_queue(
    samples: Sequence[BoxSample],
    *,
    dataset_id: str,
    dataset_content_sha256: str,
    maximum_samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(dataset_content_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in dataset_content_sha256
    ):
        raise ValueError("dataset_content_sha256 must be a lowercase SHA-256")
    selected = select_pilot_samples(samples, maximum_samples=maximum_samples, seed=seed)
    records = [propose_quadrilateral(sample) for sample in selected]
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_content_sha256": dataset_content_sha256,
        "selection": {
            "method": SELECTION_METHOD,
            "seed": seed,
            "maximum_samples": maximum_samples,
            "selected_samples": len(records),
            "stratification": ["split", "cohort", "box_area_quantile", "aspect_quantile"],
        },
        "policy": {
            "proposals_are_ground_truth": False,
            "human_review_required": True,
            "bbox_corners_used_as_quad": False,
        },
        "summary": {
            "proposal_available": sum(record["proposal"] is not None for record in records),
            "proposal_unavailable": sum(record["proposal"] is None for record in records),
        },
        "records": records,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def queue_content_sha256(records: Iterable[dict[str, Any]]) -> str:
    """Hash review records independently of queue metadata."""
    return hashlib.sha256(canonical_json_bytes(list(records))).hexdigest()
