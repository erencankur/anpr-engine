"""Strict v2 quadrilateral annotation and manifest contracts."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self, cast

from pydantic import Field, field_validator, model_validator

from anpr_engine.configuration import StrictConfigModel
from anpr_engine.detection.quadrilateral import (
    Quadrilateral,
    quadrilateral_bbox,
    validate_normalized_quadrilateral,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
Coordinate = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class CornerVisibility(StrEnum):
    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    CLIPPED = "CLIPPED"
    AMBIGUOUS = "AMBIGUOUS"


class GeometryReason(StrEnum):
    FULLY_VISIBLE = "FULLY_VISIBLE"
    OCCLUDED = "OCCLUDED"
    CLIPPED = "CLIPPED"
    AMBIGUOUS = "AMBIGUOUS"


class NormalizedPoint(StrictConfigModel):
    x: Coordinate
    y: Coordinate


class PlateQuadrilateralInstance(StrictConfigModel):
    instance_id: Identifier
    class_id: Literal["license_plate"] = "license_plate"
    corners: tuple[NormalizedPoint, NormalizedPoint, NormalizedPoint, NormalizedPoint]
    corner_visibility: tuple[
        CornerVisibility,
        CornerVisibility,
        CornerVisibility,
        CornerVisibility,
    ]
    geometry_usable_for_rectification: bool
    geometry_reason: GeometryReason

    @property
    def quadrilateral(self) -> Quadrilateral:
        return cast(Quadrilateral, tuple((point.x, point.y) for point in self.corners))

    @property
    def derived_box_xyxy(self) -> tuple[float, float, float, float]:
        return quadrilateral_bbox(self.quadrilateral)

    @model_validator(mode="after")
    def validate_geometry_and_visibility(self) -> Self:
        validate_normalized_quadrilateral(self.quadrilateral)
        all_visible = all(state is CornerVisibility.VISIBLE for state in self.corner_visibility)
        if self.geometry_usable_for_rectification:
            if not all_visible or self.geometry_reason is not GeometryReason.FULLY_VISIBLE:
                raise ValueError(
                    "rectification-usable geometry requires four visible corners and "
                    "FULLY_VISIBLE reason"
                )
        elif all_visible and self.geometry_reason is GeometryReason.FULLY_VISIBLE:
            raise ValueError("fully visible geometry must be marked usable for rectification")
        return self


class QuadrilateralAnnotationDocument(StrictConfigModel):
    schema_version: Literal["plate-quadrilateral-annotation-v2"]
    image_sha256: Sha256
    image_width: Annotated[int, Field(gt=0)]
    image_height: Annotated[int, Field(gt=0)]
    instances: tuple[PlateQuadrilateralInstance, ...]

    @model_validator(mode="after")
    def instance_ids_must_be_unique(self) -> Self:
        identifiers = tuple(instance.instance_id for instance in self.instances)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("annotation instance IDs must be unique")
        return self


class QuadrilateralManifestSample(StrictConfigModel):
    sample_id: Identifier
    split: Literal["train", "validation", "test"]
    atomic_group_id: Identifier
    canonical_image_path: Annotated[str, Field(min_length=1)]
    canonical_annotation_path: Annotated[str, Field(min_length=1)]

    @field_validator("canonical_image_path", "canonical_annotation_path")
    @classmethod
    def paths_must_be_portable_and_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("manifest paths must be portable paths within the dataset root")
        return value


class QuadrilateralDatasetManifest(StrictConfigModel):
    schema_version: Literal["detection-quadrilateral-manifest-v2"]
    dataset_id: Identifier
    content_sha256: Sha256
    samples: tuple[QuadrilateralManifestSample, ...]

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        sample_ids = tuple(sample.sample_id for sample in self.samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("manifest sample IDs must be unique")
        group_splits: dict[str, str] = {}
        for sample in self.samples:
            previous = group_splits.setdefault(sample.atomic_group_id, sample.split)
            if previous != sample.split:
                raise ValueError("atomic groups cannot cross dataset splits")
        return self


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], payload)


def load_quadrilateral_annotation(path: Path) -> QuadrilateralAnnotationDocument:
    return QuadrilateralAnnotationDocument.model_validate(_load_json_object(path))


def load_quadrilateral_manifest(path: Path) -> QuadrilateralDatasetManifest:
    return QuadrilateralDatasetManifest.model_validate(_load_json_object(path))
