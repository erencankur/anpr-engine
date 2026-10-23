"""V2 quadrilateral dataset with synchronized image and corner transforms."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

from anpr_engine.detection.annotations import (
    CornerVisibility,
    QuadrilateralDatasetManifest,
    QuadrilateralManifestSample,
    load_quadrilateral_annotation,
)
from anpr_engine.detection.config import QuadrilateralAugmentationConfig
from anpr_engine.detection.data import letterbox
from anpr_engine.detection.quadrilateral import (
    Quadrilateral,
    quadrilateral_bbox,
    transform_quadrilateral,
    validate_normalized_quadrilateral,
)


@dataclass(frozen=True)
class QuadrilateralTarget:
    boxes: Tensor
    corners: Tensor
    corner_valid: Tensor
    rectification_usable: Tensor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_oriented_bgr(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as opened:
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8)
    return cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _as_pixel_quadrilateral(normalized: Quadrilateral, width: int, height: int) -> Quadrilateral:
    return cast(
        Quadrilateral,
        tuple((point[0] * width, point[1] * height) for point in normalized),
    )


def _geometry_matrix(
    width: int,
    height: int,
    config: QuadrilateralAugmentationConfig,
) -> NDArray[np.float32]:
    center = (width / 2.0, height / 2.0)
    angle = random.uniform(-config.rotation_degrees, config.rotation_degrees)
    scale = 1.0 + random.uniform(-config.scale_limit, config.scale_limit)
    affine = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    affine[:, 2] += (
        random.uniform(-config.translation_fraction, config.translation_fraction) * width,
        random.uniform(-config.translation_fraction, config.translation_fraction) * height,
    )
    affine3 = np.vstack((affine, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)))
    source = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    jitter = config.perspective_fraction * min(width, height)
    destination = source + np.asarray(
        [[random.uniform(-jitter, jitter), random.uniform(-jitter, jitter)] for _ in range(4)],
        dtype=np.float32,
    )
    perspective = cv2.getPerspectiveTransform(source, destination)
    return cast(NDArray[np.float32], perspective @ affine3)


def _pixel_quad_is_valid(quad: Quadrilateral, width: int, height: int) -> bool:
    normalized = cast(
        Quadrilateral,
        tuple((point[0] / width, point[1] / height) for point in quad),
    )
    try:
        validate_normalized_quadrilateral(normalized)
    except ValueError:
        return False
    return True


class QuadrilateralDetectionDataset(Dataset[tuple[Tensor, QuadrilateralTarget, str]]):
    def __init__(
        self,
        manifest: QuadrilateralDatasetManifest,
        dataset_root: Path,
        *,
        split: str,
        image_size: int,
        training: bool,
        augmentation: QuadrilateralAugmentationConfig | None = None,
    ) -> None:
        self.samples = tuple(sample for sample in manifest.samples if sample.split == split)
        if not self.samples:
            raise ValueError(f"quadrilateral manifest contains no {split} samples")
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.training = training
        self.augmentation = augmentation or QuadrilateralAugmentationConfig()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, QuadrilateralTarget, str]:
        sample = self.samples[index]
        image_path = self.dataset_root / sample.canonical_image_path
        annotation_path = self.dataset_root / sample.canonical_annotation_path
        annotation = load_quadrilateral_annotation(annotation_path)
        if _sha256_file(image_path) != annotation.image_sha256:
            raise ValueError(f"image identity differs from annotation: {sample.sample_id}")
        image = _decode_oriented_bgr(image_path)
        height, width = image.shape[:2]
        if (width, height) != (annotation.image_width, annotation.image_height):
            raise ValueError(
                f"oriented image dimensions differ from annotation: {sample.sample_id}"
            )
        quads = tuple(
            _as_pixel_quadrilateral(instance.quadrilateral, width, height)
            for instance in annotation.instances
        )
        visibility = cast(
            tuple[tuple[bool, bool, bool, bool], ...],
            tuple(
                tuple(state is CornerVisibility.VISIBLE for state in instance.corner_visibility)
                for instance in annotation.instances
            ),
        )
        usable = tuple(
            instance.geometry_usable_for_rectification for instance in annotation.instances
        )
        if self.training:
            image, quads, visibility = self._augment(image, quads, visibility)
        prepared, transform = letterbox(image, self.image_size)
        if quads:
            pixel_corners = torch.tensor(quads, dtype=torch.float32)
            corners = transform.to_input_points(pixel_corners)
            pixel_boxes = torch.tensor(
                [quadrilateral_bbox(quad) for quad in quads], dtype=torch.float32
            )
            boxes = torch.stack(
                [transform.to_input_xyxy(tuple(box.tolist())) for box in pixel_boxes]
            )
            corner_valid = torch.tensor(visibility, dtype=torch.bool)
            rectification_usable = torch.tensor(usable, dtype=torch.bool) & corner_valid.all(dim=1)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            corners = torch.empty((0, 4, 2), dtype=torch.float32)
            corner_valid = torch.empty((0, 4), dtype=torch.bool)
            rectification_usable = torch.empty((0,), dtype=torch.bool)
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div(255.0)
        )
        return (
            image_tensor,
            QuadrilateralTarget(boxes, corners, corner_valid, rectification_usable),
            sample.sample_id,
        )

    def _augment(
        self,
        image: NDArray[np.uint8],
        quads: tuple[Quadrilateral, ...],
        visibility: tuple[tuple[bool, bool, bool, bool], ...],
    ) -> tuple[
        NDArray[np.uint8],
        tuple[Quadrilateral, ...],
        tuple[tuple[bool, bool, bool, bool], ...],
    ]:
        height, width = image.shape[:2]
        transformed_image = image
        transformed_quads = quads
        if quads and any(
            (
                self.augmentation.rotation_degrees,
                self.augmentation.translation_fraction,
                self.augmentation.scale_limit,
                self.augmentation.perspective_fraction,
            )
        ):
            for _ in range(self.augmentation.maximum_geometry_attempts):
                matrix = _geometry_matrix(width, height, self.augmentation)
                candidates = tuple(transform_quadrilateral(quad, matrix) for quad in quads)
                if all(_pixel_quad_is_valid(quad, width, height) for quad in candidates):
                    transformed_image = cast(
                        NDArray[np.uint8],
                        cv2.warpPerspective(
                            image,
                            matrix,
                            (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(114, 114, 114),
                        ),
                    )
                    transformed_quads = candidates
                    break
        transformed_visibility = visibility
        if random.random() < self.augmentation.horizontal_flip_probability:
            transformed_image = np.ascontiguousarray(transformed_image[:, ::-1])
            transformed_quads = cast(
                tuple[Quadrilateral, ...],
                tuple(
                    tuple(
                        (width - point[0], point[1])
                        for point in (quad[1], quad[0], quad[3], quad[2])
                    )
                    for quad in transformed_quads
                ),
            )
            transformed_visibility = tuple(
                (states[1], states[0], states[3], states[2]) for states in transformed_visibility
            )
        augmented = transformed_image.astype(np.float32)
        if self.augmentation.brightness_limit:
            augmented *= 1.0 + random.uniform(
                -self.augmentation.brightness_limit, self.augmentation.brightness_limit
            )
        if self.augmentation.contrast_limit:
            mean = augmented.mean(axis=(0, 1), keepdims=True)
            factor = 1.0 + random.uniform(
                -self.augmentation.contrast_limit, self.augmentation.contrast_limit
            )
            augmented = (augmented - mean) * factor + mean
        return (
            np.clip(augmented, 0, 255).astype(np.uint8),
            transformed_quads,
            transformed_visibility,
        )


def collate_quadrilateral_batch(
    batch: list[tuple[Tensor, QuadrilateralTarget, str]],
) -> tuple[Tensor, tuple[QuadrilateralTarget, ...], tuple[str, ...]]:
    images, targets, sample_ids = zip(*batch, strict=True)
    return torch.stack(images), tuple(targets), tuple(sample_ids)


def manifest_samples_for_split(
    manifest: QuadrilateralDatasetManifest, split: str
) -> tuple[QuadrilateralManifestSample, ...]:
    return tuple(sample for sample in manifest.samples if sample.split == split)
